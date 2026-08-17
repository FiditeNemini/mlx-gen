"""DiT parity: one transformer block, then the full one-step denoise.

SwiftVR's DiT is the stock Wan 2.2 TI2V-5B transformer with MFSWA token routing, so this
suite closes the loop: the block tests confirm the surrounding block wiring (modulation,
cross-attention against the frozen prompt embedding, feed-forward) is faithful, and the
full-model test confirms 30 of them plus the patch embedding, condition embedder and
output projection compose into the same velocity the reference produces.

Precision is the axis that matters here
---------------------------------------
Both implementations *ship* bfloat16 - ``ModelConfig.precision`` is ``mx.bfloat16`` and
upstream's own pipeline docstring loads with ``dtype="bfloat16"``. A bf16-vs-fp32
comparison therefore measures bf16, not the port. Every comparison is run twice:

float32
    The algorithmic check. Any real porting bug survives a precision change, so this is
    where the tolerance is tight and where a defect would be caught.

bfloat16
    The shipped configuration. Judged against the reference's *own* bf16 deviation rather
    than an absolute bound, because "as accurate as upstream at the same precision" is the
    only claim worth making and the only one that stays true as MLX kernels change.

Measured on the published checkpoint (M4 Max, mlx 0.30.3, torch 2.8.0):

    one block, float32       rel_max 1.6e-06   cosine 1.00000000
    one block, bfloat16      rel_max 1.2e-02   cosine 0.99998928
    full denoise, float32    rel_max 2.1e-05   cosine 1.00000000
    full denoise, bfloat16   rel_max 8.8e-02   cosine 0.99969235
    (reference's own bf16    rel_max 1.0e-01   cosine 0.99979600)

The full-model cases are marked ``high_memory_requirement``: the reference is a 20 GB
float32 model and the mflux float32 build is another 20 GB. They are deselected by the
default ``addopts`` and run with ``-m high_memory_requirement``.
"""

import gc
import json

import numpy as np
import pytest

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.swiftvr.model.swiftvr_transformer.mfswa_attention import (
    ShiftedWindowRuntime,
    ShiftedWindowSelfAttention,
)
from mflux.models.swiftvr.model.swiftvr_transformer.rope_offset import RopeTemporalOffset
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed as MlxRotaryPosEmbed
from mflux.models.wan.model.wan_transformer.wan_transformer_block import WanTransformerBlock as MlxBlock
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping
from tests.swiftvr.parity.parity_support import (
    PROMPT_EMBEDDING_CHECKPOINT,
    SWIFTVR_SNAPSHOT,
    TRANSFORMER_CHECKPOINT,
    assert_parity,
    compare,
    seeded_normal,
    torch_reference,
)

DIM, FFN_DIM, HEADS, HEAD_DIM, EPS = 3072, 14336, 24, 128, 1e-6
PATCH_SIZE = (1, 2, 2)
ROPE_MAX_SEQ_LEN = 1024
WINDOW_HW = (16, 16)

# 34 is the real 1080p post-patch height (1088 / 16 / 2). It matters that both axes exceed
# 24: with a 16-wide window, boundary clamping absorbs the half-window shift entirely below
# that, so a smaller grid would make the shifted-block case indistinguishable from the
# aligned one. See TestMfswaRoutingIsReal.test_real_weight_geometry_is_actually_shift_sensitive.
BLOCK_TOKEN_GRID = (2, 34, 34)
BLOCK_T_OFFSET = 4

# A latent chunk small enough to run in seconds but large enough that the 16x16 window
# tiles the 20x20 post-patch grid into four overlapping windows rather than one.
DENOISE_LATENT_SHAPE = (1, 48, 3, 40, 40)
DENOISE_T_OFFSET = 5

FLOAT32_MAX_RELATIVE = 1e-3
FLOAT32_MIN_COSINE = 0.999999
# Upstream's bf16 may not be beaten, but it must not be materially exceeded either.
BFLOAT16_TOLERANCE_FACTOR = 1.5


def _transformer_config() -> dict:
    """The published ``transformer/config.json``, minus diffusers bookkeeping keys."""
    with open(SWIFTVR_SNAPSHOT / "transformer" / "config.json") as handle:
        return {key: value for key, value in json.load(handle).items() if not key.startswith("_")}


def _block_state_dict(block_index: int) -> dict:
    """The published tensors of one transformer block, keyed without the block prefix."""
    from safetensors import safe_open

    prefix = f"blocks.{block_index}."
    with safe_open(str(TRANSFORMER_CHECKPOINT), framework="pt") as handle:
        return {key[len(prefix) :]: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}


def _build_torch_block(state_dict: dict, *, do_shift: bool, token_grid: tuple[int, int, int]):
    """A reference block with the shifted-window processor installed on ``attn1`` only."""
    torch_reference()
    from swiftvr.models.transformer import WanShiftWindow2DInferProcessor, WanTransformerBlock

    block = WanTransformerBlock(dim=DIM, ffn_dim=FFN_DIM, num_heads=HEADS, cross_attn_norm=True, eps=EPS)
    block.eval()
    block.load_state_dict(state_dict, strict=True)
    block.attn1.set_processor(WanShiftWindow2DInferProcessor(window_hw=WINDOW_HW))
    block.attn1._thw = token_grid
    block.attn1._do_shift = do_shift
    return block


def _build_mlx_block(state_dict: dict, *, do_shift: bool, token_grid, rotary, dtype):
    """The mflux block carrying the same tensors, mapped through the shipped Wan mapping.

    Routing the copy through :class:`WanWeightMapping` rather than assigning tensors by
    hand means the test also covers the rename the mapping performs (``ffn.net.0.proj`` ->
    ``ffn.net.0``, ``ffn.net.2`` -> ``ffn.net.1``); a hand copy would paper over it.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map

    prefixed = {f"blocks.0.{key}": mx.array(value.numpy()) for key, value in state_dict.items()}
    mapped = WeightMapper.apply_mapping(prefixed, WanWeightMapping.get_transformer_mapping(num_layers=1), num_layers=1)
    block_weights = mapped["blocks"][0]

    block = MlxBlock(dim=DIM, ffn_dim=FFN_DIM, num_heads=HEADS, cross_attn_norm=True, eps=EPS)
    expected = {key for key, _ in tree_flatten(block.parameters())}
    provided = {key for key, _ in tree_flatten(block_weights)}
    if expected != provided:
        raise ValueError(
            f"block weight mismatch: missing {sorted(expected - provided)}, extra {sorted(provided - expected)}"
        )

    block.update(tree_map(lambda array: array.astype(dtype), block_weights))
    mx.eval(block.parameters())
    runtime = ShiftedWindowRuntime(
        token_grid=token_grid,
        rotary_emb=(rotary[0].astype(dtype), rotary[1].astype(dtype)),
        window_hw=WINDOW_HW,
    )
    block.attn1.self_attention_strategy = ShiftedWindowSelfAttention(runtime, do_shift=do_shift)
    return block


def _rotary_tables(token_grid, t_offset):
    """The offset rotary tables from each side, for the block comparisons."""
    torch_reference()
    from swiftvr.models.transformer import WanRotaryPosEmbed
    from swiftvr.streaming.dit import _rope_with_offset

    reference = _rope_with_offset(
        WanRotaryPosEmbed(HEAD_DIM, PATCH_SIZE, ROPE_MAX_SEQ_LEN), *token_grid, t_off=t_offset
    )
    candidate = RopeTemporalOffset(MlxRotaryPosEmbed(HEAD_DIM, PATCH_SIZE, ROPE_MAX_SEQ_LEN))(
        token_grid=token_grid, t_offset=t_offset
    )
    return reference, candidate


def _block_inputs(token_grid, *, seed: int):
    tokens = token_grid[0] * token_grid[1] * token_grid[2]
    return (
        seeded_normal((1, tokens, DIM), seed=seed, scale=0.3),
        seeded_normal((1, 512, DIM), seed=seed + 1, scale=0.3),
        seeded_normal((1, 6, DIM), seed=seed + 2, scale=0.3),
    )


def _bfloat16_rounding_cost(reference) -> float:
    """Largest absolute change from merely storing ``reference`` in bfloat16.

    The yardstick for every bf16 comparison: a deviation at or below this is inside the
    representation itself, before any arithmetic has happened.
    """
    import mlx.core as mx

    values = reference.numpy() if hasattr(reference, "numpy") else np.asarray(reference)
    rounded = np.asarray(mx.array(values).astype(mx.bfloat16).astype(mx.float32))
    return float(np.abs(rounded - values).max())


@pytest.mark.parity
class TestTransformerBlockParity:
    """(e) One transformer block against the reference, with the published weights."""

    @pytest.mark.parametrize(
        ("block_index", "do_shift"),
        [(0, False), (1, True)],
        ids=["block0-aligned", "block1-shifted"],
    )
    def test_block_matches_reference_in_float32(self, require_transformer_weights, block_index, do_shift):
        """float32 on both sides: the algorithmic check on the whole block."""
        import mlx.core as mx
        import torch

        state = _block_state_dict(block_index)
        reference_rotary, candidate_rotary = _rotary_tables(BLOCK_TOKEN_GRID, BLOCK_T_OFFSET)
        hidden, encoder_hidden, temb = _block_inputs(BLOCK_TOKEN_GRID, seed=42 + block_index)

        torch_block = _build_torch_block(state, do_shift=do_shift, token_grid=BLOCK_TOKEN_GRID)
        with torch.no_grad():
            reference = torch_block(
                torch.from_numpy(hidden.copy()),
                torch.from_numpy(encoder_hidden.copy()),
                torch.from_numpy(temb.copy()),
                reference_rotary,
            )

        mlx_block = _build_mlx_block(
            state, do_shift=do_shift, token_grid=BLOCK_TOKEN_GRID, rotary=candidate_rotary, dtype=mx.float32
        )
        candidate = mlx_block(mx.array(hidden), mx.array(encoder_hidden), mx.array(temb), candidate_rotary)

        result = assert_parity(
            candidate,
            reference,
            label=f"transformer block {block_index} (float32, shift={do_shift})",
            max_relative=FLOAT32_MAX_RELATIVE,
            min_cosine=FLOAT32_MIN_COSINE,
        )
        print(f"\nblock {block_index} float32 shift={do_shift}: {result}")

    def test_block_in_bfloat16_stays_within_the_representation_floor(self, require_transformer_weights):
        """bfloat16 - the shipped precision - judged against what bf16 costs by itself.

        Establishes the scale to read the full-model bf16 number at: a whole block of bf16
        arithmetic should land within a small multiple of a single bf16 rounding, not
        orders of magnitude beyond it.
        """
        import mlx.core as mx
        import torch

        state = _block_state_dict(0)
        reference_rotary, candidate_rotary = _rotary_tables(BLOCK_TOKEN_GRID, BLOCK_T_OFFSET)
        hidden, encoder_hidden, temb = _block_inputs(BLOCK_TOKEN_GRID, seed=42)

        torch_block = _build_torch_block(state, do_shift=False, token_grid=BLOCK_TOKEN_GRID)
        with torch.no_grad():
            reference = torch_block(
                torch.from_numpy(hidden.copy()),
                torch.from_numpy(encoder_hidden.copy()),
                torch.from_numpy(temb.copy()),
                reference_rotary,
            )

        mlx_block = _build_mlx_block(
            state, do_shift=False, token_grid=BLOCK_TOKEN_GRID, rotary=candidate_rotary, dtype=mx.bfloat16
        )
        candidate = mlx_block(
            mx.array(hidden).astype(mx.bfloat16),
            mx.array(encoder_hidden).astype(mx.bfloat16),
            mx.array(temb).astype(mx.bfloat16),
            candidate_rotary,
        )

        result = compare(candidate, reference)
        rounding_cost = _bfloat16_rounding_cost(reference)
        print(
            f"\nblock 0 bfloat16: {result}\n"
            f"  bf16 rounding of the reference alone: {rounding_cost:.3e} "
            f"(ratio {result.max_abs_diff / rounding_cost:.2f}x)"
        )

        assert result.cosine > 0.9999, f"bfloat16 block output diverged in direction: {result}"
        assert result.max_abs_diff < 10.0 * rounding_cost, (
            f"bfloat16 block error {result.max_abs_diff:.3e} is more than 10x the bf16 rounding "
            f"floor {rounding_cost:.3e}; that is accumulation beyond what the precision explains"
        )


@pytest.mark.parity
class TestRopeSensitivityOfTheBlock:
    """What the block comparisons can and cannot detect about rotary embeddings.

    Attention sees *relative* position: rotating every query and key by a phase that is an
    affine function of absolute position leaves every pairwise product depending only on
    the difference. SwiftVR keeps no KV cache and every chunk attends only within itself,
    so adding a uniform ``t_offset`` to a chunk cannot change that chunk's output.

    These two tests bound the block comparisons honestly. The first shows they are not
    vacuous - a genuine change in relative structure moves the output by more than half
    its dynamic range. The second records that a uniform offset does not, in *either*
    implementation, so a ``t_offset`` regression must be caught by
    ``test_rope_offset_parity.py`` at the table level, not here.
    """

    def test_block_output_responds_to_relative_rope_structure(self, require_transformer_weights):
        """Reversing the tables along the token axis must change the reference materially.

        Without this, "the block matched" would be consistent with a block that ignored
        its rotary tables entirely.
        """
        import torch

        state = _block_state_dict(0)
        reference_rotary, _ = _rotary_tables(BLOCK_TOKEN_GRID, BLOCK_T_OFFSET)
        hidden, encoder_hidden, temb = _block_inputs(BLOCK_TOKEN_GRID, seed=42)
        block = _build_torch_block(state, do_shift=False, token_grid=BLOCK_TOKEN_GRID)

        def run(rotary):
            with torch.no_grad():
                return block(
                    torch.from_numpy(hidden.copy()),
                    torch.from_numpy(encoder_hidden.copy()),
                    torch.from_numpy(temb.copy()),
                    rotary,
                ).numpy()

        baseline = run(reference_rotary)
        reversed_rotary = (reference_rotary[0].flip(1).contiguous(), reference_rotary[1].flip(1).contiguous())
        scrambled = run(reversed_rotary)

        scale = float(np.abs(baseline).max())
        gap = float(np.abs(baseline - scrambled).max())
        print(f"\nblock response to reversed rope tables: {gap:.3e} against absmax {scale:.3e}")
        assert gap > 0.1 * scale, "the block barely responds to its rotary tables; the parity cases are vacuous"

    def test_uniform_temporal_offset_is_a_no_op_in_both_implementations(self, require_transformer_weights):
        """A uniform ``t_offset`` leaves a single chunk's output unchanged, on both sides.

        Documents the invariance and, more usefully, checks that mflux is invariant to the
        *same* degree as the reference. A port that broke RoPE into something that was not
        a pure rotation would lose the invariance and fail here.
        """
        import mlx.core as mx
        import torch

        state = _block_state_dict(0)
        hidden, encoder_hidden, temb = _block_inputs(BLOCK_TOKEN_GRID, seed=42)
        outputs = {"reference": {}, "mflux": {}}

        for offset in (0, 97):
            reference_rotary, candidate_rotary = _rotary_tables(BLOCK_TOKEN_GRID, offset)
            torch_block = _build_torch_block(state, do_shift=False, token_grid=BLOCK_TOKEN_GRID)
            with torch.no_grad():
                outputs["reference"][offset] = torch_block(
                    torch.from_numpy(hidden.copy()),
                    torch.from_numpy(encoder_hidden.copy()),
                    torch.from_numpy(temb.copy()),
                    reference_rotary,
                ).numpy()
            mlx_block = _build_mlx_block(
                state, do_shift=False, token_grid=BLOCK_TOKEN_GRID, rotary=candidate_rotary, dtype=mx.float32
            )
            outputs["mflux"][offset] = np.asarray(
                mlx_block(mx.array(hidden), mx.array(encoder_hidden), mx.array(temb), candidate_rotary)
            )

        table_gap = float(
            np.abs(
                np.asarray(_rotary_tables(BLOCK_TOKEN_GRID, 0)[1][0])
                - np.asarray(_rotary_tables(BLOCK_TOKEN_GRID, 97)[1][0])
            ).max()
        )
        reference_shift = float(np.abs(outputs["reference"][0] - outputs["reference"][97]).max())
        candidate_shift = float(np.abs(outputs["mflux"][0] - outputs["mflux"][97]).max())
        scale = float(np.abs(outputs["reference"][0]).max())
        print(
            f"\nuniform t_offset 0 -> 97 moves the rope table by {table_gap:.3e} but the block output by "
            f"{reference_shift:.3e} (reference) and {candidate_shift:.3e} (mflux), against absmax {scale:.3e}"
        )

        assert table_gap > 0.5, "the two offsets produced near-identical tables; the invariance is untested"
        assert reference_shift < 1e-3 * scale, f"the reference is not offset-invariant: {reference_shift:.3e}"
        assert candidate_shift < 1e-3 * scale, f"mflux is not offset-invariant: {candidate_shift:.3e}"
        assert candidate_shift < 10.0 * reference_shift, (
            f"mflux is {candidate_shift / reference_shift:.1f}x less offset-invariant than the reference; "
            "its rotary application may not be a pure rotation"
        )


@pytest.mark.high_memory_requirement
@pytest.mark.parity
class TestFullOneStepDenoise:
    """(e) The complete one-step restoration, ``v = DiT(z_lq, t=1000)``, on a tiny latent."""

    @pytest.fixture(scope="class")
    def latents(self) -> np.ndarray:
        return seeded_normal(DENOISE_LATENT_SHAPE, seed=2024, scale=0.8)

    @pytest.fixture(scope="class")
    def reference_velocities(self, require_transformer_weights, latents) -> dict:
        """Run the reference DiT once in float32 and once in bfloat16, then free it.

        Both references come from one load. torch is released before any test builds an
        mflux model, so peak residency is one 20 GB model at a time rather than two.
        """
        import torch
        from safetensors.torch import load_file

        torch_reference()
        from swiftvr.models.transformer import WanTransformer3DModel, enable_shifted_window_self_attention
        from swiftvr.streaming.dit import INFERENCE_TIMESTEP, _dit_forward_chunk, _precompute_cond

        assert INFERENCE_TIMESTEP == 1000.0, "the reference's one-step endpoint moved"

        model = WanTransformer3DModel(**_transformer_config())
        missing, unexpected = model.load_state_dict(load_file(str(TRANSFORMER_CHECKPOINT)), strict=False)
        assert not missing, f"reference DiT is missing {len(missing)} tensors: {missing[:5]}"
        assert not unexpected, f"reference DiT got {len(unexpected)} unexpected tensors: {unexpected[:5]}"
        model.eval()
        enable_shifted_window_self_attention(model, window_hw=WINDOW_HW)
        prompt = load_file(str(PROMPT_EMBEDDING_CHECKPOINT))["prompt_emb"]

        velocities = {}
        for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
            model.to(dtype)
            with torch.inference_mode():
                timestep = torch.full((1,), INFERENCE_TIMESTEP, dtype=torch.float32)
                temb, time_proj, encoder_hidden = _precompute_cond(model, 1, prompt.to(dtype), timestep)
                velocity = _dit_forward_chunk(
                    model,
                    torch.from_numpy(latents).to(dtype),
                    temb,
                    time_proj,
                    encoder_hidden,
                    t_off=DENOISE_T_OFFSET,
                )
            velocities[name] = velocity.float().numpy().copy()
            del temb, time_proj, encoder_hidden, velocity

        del model, prompt
        gc.collect()
        return velocities

    def _mflux_velocity(self, precision):
        """Build the shipped SwiftVR at ``precision`` and return one velocity prediction.

        Goes through :class:`SwiftVR`, so the initializer, the weight-coverage assertions
        and ``install_mfswa`` all run exactly as they do in production.
        """
        import mlx.core as mx

        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.swiftvr.streaming.streaming_dit import INFERENCE_TIMESTEP
        from mflux.models.swiftvr.variants.upscale.swiftvr import SwiftVR

        original = ModelConfig.precision
        try:
            ModelConfig.precision = precision
            model = SwiftVR()
        finally:
            ModelConfig.precision = original

        transformer = model.transformer
        assert transformer.mfswa_installed, "MFSWA was not installed by the initializer"
        assert transformer.window_hw == WINDOW_HW, f"unexpected window {transformer.window_hw}"
        dtype = transformer.transformer.blocks[0].attn1.to_q.weight.dtype
        assert dtype == precision, f"SwiftVR loaded at {dtype}, expected {precision}"

        latents = seeded_normal(DENOISE_LATENT_SHAPE, seed=2024, scale=0.8)
        velocity = transformer.predict_velocity(
            mx.array(latents).astype(dtype),
            timestep=mx.array([INFERENCE_TIMESTEP], dtype=mx.float32),
            prompt_embeds=model.prompt_embeds.astype(dtype),
            t_offset=DENOISE_T_OFFSET,
        )
        mx.eval(velocity)
        result = np.asarray(velocity.astype(mx.float32))
        del model, transformer, velocity
        gc.collect()
        return result

    def test_velocity_matches_reference_in_float32(self, reference_velocities, latents):
        """The algorithmic check on the whole 30-block model.

        A tight bound here is what licenses reading the bfloat16 case as precision rather
        than as a defect.
        """
        import mlx.core as mx

        candidate = self._mflux_velocity(mx.float32)
        reference = reference_velocities["float32"]

        assert np.abs(reference).mean() > 0.01, "the reference DiT returned a near-zero velocity; nothing is under test"

        result = assert_parity(
            candidate,
            reference,
            label="full one-step denoise (float32)",
            max_relative=FLOAT32_MAX_RELATIVE,
            min_cosine=FLOAT32_MIN_COSINE,
        )
        print(f"\nfull denoise float32: {result}")

        restored = compare(latents - candidate, latents - reference)
        print(f"full denoise float32, z_hq = z_lq - v: {restored}")
        assert restored.cosine > FLOAT32_MIN_COSINE, f"restored latents diverged: {restored}"

    def test_velocity_in_bfloat16_is_as_accurate_as_the_reference_at_bfloat16(self, reference_velocities):
        """The shipped precision, judged against the reference's own bfloat16 deviation.

        An absolute bound would be arbitrary: bf16 through 30 blocks moves the velocity by
        ~0.1 relative *in the reference itself*. The claim worth testing is that mflux's
        bf16 is no further from the float32 truth than upstream's bf16 is.
        """
        import mlx.core as mx

        candidate = self._mflux_velocity(mx.bfloat16)
        float32_reference = reference_velocities["float32"]
        bfloat16_reference = reference_velocities["bfloat16"]

        mflux_deviation = compare(candidate, float32_reference)
        reference_deviation = compare(bfloat16_reference, float32_reference)
        like_for_like = compare(candidate, bfloat16_reference)

        print(
            f"\nfull denoise bfloat16:"
            f"\n  mflux bf16   vs torch fp32: {mflux_deviation}"
            f"\n  torch bf16   vs torch fp32: {reference_deviation}"
            f"\n  mflux bf16   vs torch bf16: {like_for_like}"
        )

        assert mflux_deviation.max_abs_diff <= BFLOAT16_TOLERANCE_FACTOR * reference_deviation.max_abs_diff, (
            f"mflux bfloat16 deviates from the float32 reference by {mflux_deviation.max_abs_diff:.3e}, "
            f"more than {BFLOAT16_TOLERANCE_FACTOR}x the reference's own bfloat16 deviation "
            f"{reference_deviation.max_abs_diff:.3e}"
        )
        assert mflux_deviation.cosine > 0.999, f"bfloat16 velocity diverged in direction: {mflux_deviation}"
        assert like_for_like.cosine > 0.999, f"the two bfloat16 runs disagree in direction: {like_for_like}"
