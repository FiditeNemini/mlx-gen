"""MFSWA parity: mflux :class:`ShiftedWindowSelfAttention` against ``WanShiftWindow2DInferProcessor``.

MFSWA is the *only* difference between SwiftVR's DiT and stock Wan 2.2 TI2V-5B. The
weights are identical; what changes is which tokens attend to which. A port that got the
window starts, the boundary clamping or the overlap ownership wrong would still produce
plausible output, so this is the highest-value comparison in the harness.

Both sides receive the same hidden states, the same rotary tables and the same weights,
and both are driven through their real entry points - the reference through
``WanAttention.forward`` with the processor installed, mflux through the installed
strategy - so the projections, QK norms and output projection are exercised too.

Guarding against a vacuous pass
-------------------------------
Windowed attention that silently degenerated to global attention would match a *global*
reference perfectly and mean nothing. :meth:`test_windowing_is_not_global_attention`
therefore pins the reference's own windowed output as materially different from its
global output, and a separate case shows the two coincide exactly when the window covers
the whole grid. Together these establish that the passing comparisons are comparing the
windowed path.

Tolerance
---------
Same reasoning as the ReAE suite: float32 on both sides, so only accumulation order
differs. Measured ``relative_max`` is 5.8e-07 at dim 64 and 3.2e-06 at the real dim 3072,
the growth being the longer reduction in the 3072-wide GEMMs and the softmax. The bound
is 1e-4, ~30x above the worst measurement. A routing bug is not a near miss: swapping the
shift parity moves the output by 0.11 against an absmax of 0.14, and using global
attention instead moves it by 0.06 - both four orders of magnitude above this bound.
"""

import numpy as np
import pytest

from mflux.models.swiftvr.model.swiftvr_transformer.mfswa_attention import (
    ShiftedWindowRuntime,
    ShiftedWindowSelfAttention,
)
from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention as MlxWanAttention
from tests.swiftvr.parity.parity_support import (
    TRANSFORMER_CHECKPOINT,
    assert_parity,
    compare,
    seeded_normal,
    torch_reference,
)

MAX_RELATIVE = 1e-4
MIN_COSINE = 0.999999

# The production geometry: 24 heads x 128 = 3072, 16x16 windows.
REAL_DIM, REAL_HEADS, REAL_HEAD_DIM = 3072, 24, 128
REAL_WINDOW = (16, 16)

# A post-patch grid on which the half-window shift actually changes the partition.
#
# This is not a free choice. Starts are clamped into ``[0, size - window]``, so a shift of
# ``window // 2`` is absorbed entirely whenever ``size - window <= window // 2`` - with a
# 16-wide window, every axis of 24 tokens or fewer partitions identically shifted or not.
# A grid like (20, 18) therefore cannot detect a shift bug at all. 34 is the real 1080p
# post-patch height (1088 / 16 / 2), and is sensitive; the production width, 60, is too.
REAL_TOKEN_GRID = (2, 34, 34)

# A small stand-in used where the point is the routing algebra, not the weights.
SMALL_DIM, SMALL_HEADS, SMALL_HEAD_DIM = 64, 4, 16


def _seeded_attention_weights(dim: int, heads: int, head_dim: int, *, seed: int) -> dict:
    """Deterministic torch weights for one self-attention module, in the reference's key names."""
    import torch

    inner = heads * head_dim
    rng = np.random.default_rng(seed)

    def draw(*shape, scale=0.05):
        return torch.from_numpy((scale * rng.standard_normal(shape)).astype(np.float32))

    return {
        "to_q.weight": draw(inner, dim),
        "to_q.bias": draw(inner),
        "to_k.weight": draw(inner, dim),
        "to_k.bias": draw(inner),
        "to_v.weight": draw(inner, dim),
        "to_v.bias": draw(inner),
        "to_out.0.weight": draw(dim, inner),
        "to_out.0.bias": draw(dim),
        "norm_q.weight": torch.from_numpy(
            np.ones(inner, dtype=np.float32) + 0.02 * rng.standard_normal(inner).astype(np.float32)
        ),
        "norm_k.weight": torch.from_numpy(
            np.ones(inner, dtype=np.float32) + 0.02 * rng.standard_normal(inner).astype(np.float32)
        ),
    }


def _real_attention_weights(block_index: int) -> dict:
    """The published ``blocks.{i}.attn1`` tensors, read without materializing the 20 GB file.

    ``safe_open`` seeks to each tensor, so this costs ~0.1 s and ~150 MB rather than the
    whole checkpoint.
    """
    from safetensors import safe_open

    prefix = f"blocks.{block_index}.attn1."
    with safe_open(str(TRANSFORMER_CHECKPOINT), framework="pt") as handle:
        weights = {key[len(prefix) :]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    if not weights:
        raise ValueError(f"No tensors found under '{prefix}' in {TRANSFORMER_CHECKPOINT}")
    return weights


def _build_torch_attention(state_dict: dict, *, dim: int, heads: int, head_dim: int, window_hw: tuple[int, int]):
    """The reference self-attention module with the shifted-window processor installed."""
    torch_reference()
    from swiftvr.models.transformer import WanAttention, WanShiftWindow2DInferProcessor

    attention = WanAttention(
        dim=dim,
        heads=heads,
        dim_head=head_dim,
        processor=WanShiftWindow2DInferProcessor(window_hw=window_hw),
    )
    attention.eval()
    attention.load_state_dict(state_dict, strict=True)
    return attention


def _build_mlx_attention(state_dict: dict, *, dim: int, heads: int, head_dim: int) -> MlxWanAttention:
    """The mflux self-attention module carrying the same weights.

    Both frameworks store ``nn.Linear`` weights as ``[out, in]``, so the copy is a
    straight transfer; a transposed copy here would show up immediately as an O(1)
    parity failure rather than a subtle one.
    """
    import mlx.core as mx

    attention = MlxWanAttention(dim=dim, heads=heads, dim_head=head_dim)
    attention.to_q.weight = mx.array(state_dict["to_q.weight"].numpy())
    attention.to_q.bias = mx.array(state_dict["to_q.bias"].numpy())
    attention.to_k.weight = mx.array(state_dict["to_k.weight"].numpy())
    attention.to_k.bias = mx.array(state_dict["to_k.bias"].numpy())
    attention.to_v.weight = mx.array(state_dict["to_v.weight"].numpy())
    attention.to_v.bias = mx.array(state_dict["to_v.bias"].numpy())
    attention.to_out[0].weight = mx.array(state_dict["to_out.0.weight"].numpy())
    attention.to_out[0].bias = mx.array(state_dict["to_out.0.bias"].numpy())
    attention.norm_q.weight = mx.array(state_dict["norm_q.weight"].numpy())
    attention.norm_k.weight = mx.array(state_dict["norm_k.weight"].numpy())
    mx.eval(attention.parameters())
    return attention


def _synthetic_inputs(token_grid: tuple[int, int, int], dim: int, head_dim: int, *, seed: int):
    """Deterministic hidden states and rotary tables for one chunk."""
    frames, height, width = token_grid
    tokens = frames * height * width
    return (
        seeded_normal((1, tokens, dim), seed=seed, scale=0.5),
        seeded_normal((1, tokens, 1, head_dim), seed=seed + 1),
        seeded_normal((1, tokens, 1, head_dim), seed=seed + 2),
    )


def _run_reference(attention, hidden_states, freqs_cos, freqs_sin, token_grid, *, do_shift: bool):
    import torch

    attention._thw = token_grid
    attention._do_shift = do_shift
    with torch.no_grad():
        return attention(
            torch.from_numpy(hidden_states.copy()),
            None,
            None,
            (torch.from_numpy(freqs_cos.copy()), torch.from_numpy(freqs_sin.copy())),
        )


def _run_mlx(attention, hidden_states, freqs_cos, freqs_sin, token_grid, window_hw, *, do_shift: bool):
    import mlx.core as mx

    rotary = (mx.array(freqs_cos), mx.array(freqs_sin))
    runtime = ShiftedWindowRuntime(token_grid=token_grid, rotary_emb=rotary, window_hw=window_hw)
    strategy = ShiftedWindowSelfAttention(runtime, do_shift=do_shift)
    return strategy(attention, mx.array(hidden_states), rotary, "attn1", None)


@pytest.mark.parity
class TestMfswaRoutingIsReal:
    """Establish that the windowed path is distinguishable, so the parity cases mean something."""

    def test_windowing_is_not_global_attention(self, require_reference):
        """The reference's windowed output must differ materially from its global output.

        If it did not, every parity case below would pass just as well against a port that
        never implemented windowing at all.
        """
        import torch

        torch_reference()
        from swiftvr.models.transformer import WanAttnProcessor

        token_grid = (3, 10, 7)
        window = (4, 4)
        state = _seeded_attention_weights(SMALL_DIM, SMALL_HEADS, SMALL_HEAD_DIM, seed=0)
        attention = _build_torch_attention(
            state, dim=SMALL_DIM, heads=SMALL_HEADS, head_dim=SMALL_HEAD_DIM, window_hw=window
        )
        hidden, cos, sin = _synthetic_inputs(token_grid, SMALL_DIM, SMALL_HEAD_DIM, seed=100)

        unshifted = _run_reference(attention, hidden, cos, sin, token_grid, do_shift=False)
        shifted = _run_reference(attention, hidden, cos, sin, token_grid, do_shift=True)

        attention.set_processor(WanAttnProcessor())
        with torch.no_grad():
            global_output = attention(
                torch.from_numpy(hidden.copy()), None, None, (torch.from_numpy(cos), torch.from_numpy(sin))
            )

        scale = float(global_output.abs().max())
        windowed_gap = float((unshifted - global_output).abs().max())
        shift_gap = float((unshifted - shifted).abs().max())
        print(f"\nwindowed-vs-global {windowed_gap:.3e}, shift-vs-noshift {shift_gap:.3e}, absmax {scale:.3e}")

        assert windowed_gap > 0.1 * scale, "windowed attention is indistinguishable from global attention"
        assert shift_gap > 0.1 * scale, "the half-window shift has no effect"

    def test_window_covering_whole_grid_equals_global_attention(self, require_reference):
        """A window at least as large as the grid is one window, which is global attention.

        A structural identity rather than a tolerance: it pins the meaning of the window
        parameter at the degenerate end.
        """
        import torch

        torch_reference()
        from swiftvr.models.transformer import WanAttnProcessor

        token_grid = (2, 6, 5)
        state = _seeded_attention_weights(SMALL_DIM, SMALL_HEADS, SMALL_HEAD_DIM, seed=1)
        attention = _build_torch_attention(
            state, dim=SMALL_DIM, heads=SMALL_HEADS, head_dim=SMALL_HEAD_DIM, window_hw=(64, 64)
        )
        hidden, cos, sin = _synthetic_inputs(token_grid, SMALL_DIM, SMALL_HEAD_DIM, seed=200)

        windowed = _run_reference(attention, hidden, cos, sin, token_grid, do_shift=False)
        attention.set_processor(WanAttnProcessor())
        with torch.no_grad():
            global_output = attention(
                torch.from_numpy(hidden.copy()), None, None, (torch.from_numpy(cos), torch.from_numpy(sin))
            )

        result = compare(windowed, global_output)
        print(f"\nwindow>=grid vs global: {result}")
        assert result.relative_max < 1e-6, f"a single full-grid window must equal global attention: {result}"

    def test_real_weight_geometry_is_actually_shift_sensitive(self):
        """The grid used for the published-weight cases must respond to the shift.

        Boundary clamping silences the half-window shift on any axis of ``window +
        window // 2`` tokens or fewer. On a 16-wide window that is everything up to 24, so
        an innocuous-looking grid such as (20, 18) partitions identically whether the layer
        is shifted or not - and a parity case built on it would pass against a port that
        never implemented the shift. This pins the chosen grid as one where it matters.
        """
        from mflux.models.swiftvr.model.swiftvr_transformer.window_meta import window_axis_starts

        _, height, width = REAL_TOKEN_GRID
        window_h, window_w = REAL_WINDOW
        for size, window, axis in ((height, window_h, "height"), (width, window_w, "width")):
            aligned = window_axis_starts(size, window, do_shift=False)
            shifted = window_axis_starts(size, window, do_shift=True)
            print(f"\n{axis} {size} with window {window}: aligned={aligned} shifted={shifted}")
            assert aligned != shifted, (
                f"{axis} {size} partitions identically with and without the shift "
                f"(clamping absorbs it for any axis <= {window + window // 2}); "
                "the published-weight parity cases would not detect a shift regression"
            )


@pytest.mark.parity
class TestMfswaSyntheticParity:
    """(c) Synthetic weights and inputs, small (T, H, W)."""

    @pytest.mark.parametrize("do_shift", [False, True], ids=["aligned", "shifted"])
    @pytest.mark.parametrize(
        ("token_grid", "window_hw"),
        [
            ((3, 10, 7), (4, 4)),  # both axes exceed the window and do not divide it
            ((1, 8, 8), (4, 4)),  # exact tiling, single frame
            ((4, 5, 3), (4, 4)),  # width below the window: clamped to one column of windows
            ((2, 17, 13), (8, 8)),  # prime-ish extents force clamped trailing windows
        ],
        ids=["ragged", "exact", "narrow", "prime"],
    )
    def test_matches_reference(self, require_reference, token_grid, window_hw, do_shift):
        """Every combination of grid, window and shift parity must agree with the reference.

        The ragged and prime cases are the ones that exercise boundary clamping and
        therefore overlapping windows, where ``owner_pos`` decides which window's output
        survives. A port that picked the other window would fail here and nowhere else.
        """
        state = _seeded_attention_weights(SMALL_DIM, SMALL_HEADS, SMALL_HEAD_DIM, seed=5)
        torch_attention = _build_torch_attention(
            state, dim=SMALL_DIM, heads=SMALL_HEADS, head_dim=SMALL_HEAD_DIM, window_hw=window_hw
        )
        mlx_attention = _build_mlx_attention(state, dim=SMALL_DIM, heads=SMALL_HEADS, head_dim=SMALL_HEAD_DIM)
        hidden, cos, sin = _synthetic_inputs(token_grid, SMALL_DIM, SMALL_HEAD_DIM, seed=300)

        reference = _run_reference(torch_attention, hidden, cos, sin, token_grid, do_shift=do_shift)
        candidate = _run_mlx(mlx_attention, hidden, cos, sin, token_grid, window_hw, do_shift=do_shift)

        result = assert_parity(
            candidate,
            reference,
            label=f"MFSWA grid={token_grid} window={window_hw} shift={do_shift}",
            max_relative=MAX_RELATIVE,
            min_cosine=MIN_COSINE,
        )
        print(f"\nMFSWA {token_grid} win{window_hw} shift={do_shift}: {result}")


@pytest.mark.parity
class TestMfswaRealWeightParity:
    """(c, extended) The published ``attn1`` weights at production width."""

    @pytest.mark.parametrize(
        ("block_index", "do_shift"),
        [(0, False), (1, True)],
        ids=["block0-aligned", "block1-shifted"],
    )
    def test_matches_reference_with_published_weights(self, require_transformer_weights, block_index, do_shift):
        """Real 3072-wide weights, the production 16x16 window, both shift parities.

        Block 0 is aligned and block 1 is shifted under
        ``enable_shifted_window_self_attention``, so these are the two configurations the
        model actually runs.
        """
        token_grid = REAL_TOKEN_GRID
        state = _real_attention_weights(block_index)
        torch_attention = _build_torch_attention(
            state, dim=REAL_DIM, heads=REAL_HEADS, head_dim=REAL_HEAD_DIM, window_hw=REAL_WINDOW
        )
        mlx_attention = _build_mlx_attention(state, dim=REAL_DIM, heads=REAL_HEADS, head_dim=REAL_HEAD_DIM)
        hidden, cos, sin = _synthetic_inputs(token_grid, REAL_DIM, REAL_HEAD_DIM, seed=400 + block_index)

        reference = _run_reference(torch_attention, hidden, cos, sin, token_grid, do_shift=do_shift)
        candidate = _run_mlx(mlx_attention, hidden, cos, sin, token_grid, REAL_WINDOW, do_shift=do_shift)

        result = assert_parity(
            candidate,
            reference,
            label=f"MFSWA block {block_index} (published weights, shift={do_shift})",
            max_relative=MAX_RELATIVE,
            min_cosine=MIN_COSINE,
        )
        print(f"\nMFSWA block{block_index} real weights shift={do_shift}: {result}")

    def test_fusing_projections_is_numerically_free(self, require_transformer_weights):
        """Upstream fuses q/k/v into one linear; mflux keeps them separate by design.

        The divergence is deliberate and documented (fusing would break the 825-tensor
        identity that lets SwiftVR reuse Wan's weight mapping verbatim). This records what
        that choice costs numerically, so the claim in ``mfswa_attention`` that it is "for
        no numerical gain" is measured rather than asserted.
        """
        token_grid = REAL_TOKEN_GRID
        state = _real_attention_weights(0)
        attention = _build_torch_attention(
            state, dim=REAL_DIM, heads=REAL_HEADS, head_dim=REAL_HEAD_DIM, window_hw=REAL_WINDOW
        )
        hidden, cos, sin = _synthetic_inputs(token_grid, REAL_DIM, REAL_HEAD_DIM, seed=500)

        attention.unfuse_projections()
        unfused = _run_reference(attention, hidden, cos, sin, token_grid, do_shift=False)
        attention.fuse_projections()
        fused = _run_reference(attention, hidden, cos, sin, token_grid, do_shift=False)

        result = compare(fused, unfused)
        print(f"\nfused vs unfused (reference, both torch): {result}")
        assert result.relative_max < 1e-6, f"fusing changed the reference output materially: {result}"
