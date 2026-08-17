"""Temporally-offset RoPE parity, including the table extension past ``rope_max_seq_len``.

SwiftVR restores a clip chunk by chunk on one continuous temporal coordinate, so chunk
``n`` slices its rotary table starting at the global latent-frame offset. Two regimes:

Within the table (offset + frames <= 1024)
    Both sides slice a table each built in float32 - torch selects float32 when
    ``torch.backends.mps.is_available()``, which is the case on Apple silicon. Agreement
    is limited only by how each computes ``theta ** (2i/d)`` and the position product.
    The residual grows with absolute position because the angle does: 4.6e-07 at offset 0,
    5.9e-05 at offset 1000.

Past the table (offset + frames > 1024)
    Here mflux *deliberately diverges*. The reference extends the existing table by
    recovering each column's angular step with ``atan2`` and marching it forward in
    float64; mflux has no float64 and rebuilds the table analytically in float32 instead.
    ``rope_offset`` documents this as an intentional choice.

    :class:`TestRopeTableExtension` measures the consequence rather than trusting the
    docstring, and :class:`TestRopeDivergenceStaysBelowRuntimePrecision` establishes the
    only thing that decides whether it matters: the gap is far smaller than the bf16
    rounding the DiT applies to the very same tensors.

Tolerances
----------
``WITHIN_TABLE_MAX_ABS = 3e-4`` - five times the worst measured 5.9e-05 at offset 1000.
Table entries are cosines bounded in [-1, 1], so an absolute bound is already relative.

``EXTENSION_MAX_ABS = 3e-3`` - below bf16 unit roundoff (2**-8 = 3.9e-3). Set at the
point past which the divergence could start to be visible at the precision the model
runs, not at the point where the two happen to agree today.
"""

import numpy as np
import pytest

from mflux.models.swiftvr.model.swiftvr_transformer.rope_offset import RopeTemporalOffset
from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed as MlxRotaryPosEmbed
from tests.swiftvr.parity.parity_support import compare, seeded_normal, torch_reference

ATTENTION_HEAD_DIM = 128
PATCH_SIZE = (1, 2, 2)
ROPE_MAX_SEQ_LEN = 1024

WITHIN_TABLE_MAX_ABS = 3e-4
EXTENSION_MAX_ABS = 3e-3
BFLOAT16_UNIT_ROUNDOFF = 2.0**-8


def _reference_rope():
    torch_reference()
    from swiftvr.models.transformer import WanRotaryPosEmbed

    return WanRotaryPosEmbed(ATTENTION_HEAD_DIM, PATCH_SIZE, ROPE_MAX_SEQ_LEN)


def _mflux_rope() -> RopeTemporalOffset:
    return RopeTemporalOffset(MlxRotaryPosEmbed(ATTENTION_HEAD_DIM, PATCH_SIZE, ROPE_MAX_SEQ_LEN))


def _reference_tables(rope, token_grid, t_offset):
    torch_reference()
    from swiftvr.streaming.dit import _rope_with_offset

    return _rope_with_offset(rope, *token_grid, t_off=t_offset)


def _float64_axis_tables(dim: int, positions: np.ndarray, theta: float = 10000.0):
    """Exact cos/sin for one RoPE axis, in float64, with ``repeat_interleave_real`` layout.

    This is the arbiter for the extension comparison. Neither implementation is the
    reference for the other past position 1024: they use different methods, so the only
    way to say which is right is to compute the answer at a precision that dwarfs both.
    """
    inverse_frequencies = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = np.outer(np.asarray(positions, dtype=np.float64), inverse_frequencies)
    return np.repeat(np.cos(angles), 2, axis=1), np.repeat(np.sin(angles), 2, axis=1)


@pytest.mark.parity
class TestRopeWithinTable:
    """(d) Offsets that stay inside ``rope_max_seq_len``: no extension on either side."""

    @pytest.mark.parametrize(
        ("token_grid", "t_offset"),
        [
            ((3, 8, 8), 0),
            ((3, 8, 8), 7),
            ((2, 12, 10), 101),
            ((2, 6, 6), 1000),
        ],
        ids=["offset0", "offset7", "offset101", "offset1000"],
    )
    def test_offset_tables_match_reference(self, require_reference, token_grid, t_offset):
        """The offset slice must match ``_rope_with_offset`` element for element."""
        reference_rope = _reference_rope()
        mflux_rope = _mflux_rope()

        reference_cos, reference_sin = _reference_tables(reference_rope, token_grid, t_offset)
        candidate_cos, candidate_sin = mflux_rope(token_grid=token_grid, t_offset=t_offset)

        assert mflux_rope._extended_tables is None, "offsets inside the table must not trigger an extension"

        cos_result = compare(np.asarray(candidate_cos), reference_cos)
        sin_result = compare(np.asarray(candidate_sin), reference_sin)
        print(f"\nRoPE offset={t_offset} grid={token_grid} cos: {cos_result}")
        print(f"RoPE offset={t_offset} grid={token_grid} sin: {sin_result}")

        assert cos_result.max_abs_diff < WITHIN_TABLE_MAX_ABS, f"cos table diverged: {cos_result}"
        assert sin_result.max_abs_diff < WITHIN_TABLE_MAX_ABS, f"sin table diverged: {sin_result}"

    def test_temporal_offset_actually_shifts_the_table(self, require_reference):
        """An offset table must equal the unoffset table read further along.

        Without this, a port that ignored ``t_offset`` entirely would still pass the
        comparisons above whenever the reference also ignored it. The reference is used
        as its own control: rows ``[k, k+n)`` of the offset-0 table are rows ``[0, n)`` of
        the offset-k table.
        """
        mflux_rope = _mflux_rope()
        frames, offset = 3, 12
        grid = (frames, 4, 4)
        wide = mflux_rope(token_grid=(frames + offset, 4, 4), t_offset=0)[0]
        shifted = mflux_rope(token_grid=grid, t_offset=offset)[0]

        tokens_per_frame = 4 * 4
        head_dim = wide.shape[-1]
        wide_tail = np.asarray(wide).reshape(1, frames + offset, tokens_per_frame, 1, head_dim)[:, offset:]
        shifted_view = np.asarray(shifted).reshape(1, frames, tokens_per_frame, 1, head_dim)

        result = compare(shifted_view, wide_tail)
        print(f"\nRoPE offset self-consistency: {result}")
        assert result.max_abs_diff == 0.0, f"offset slicing is not a pure shift: {result}"


@pytest.mark.parity
class TestRopeTableExtension:
    """(d) Offsets past ``rope_max_seq_len``, where the two implementations differ by design."""

    @pytest.mark.parametrize("t_offset", [1200, 4000], ids=["just-past", "far-past"])
    def test_extension_diverges_within_the_documented_envelope(self, require_reference, t_offset):
        """Both sides must extend, and the resulting tables must stay within the envelope.

        This is not an exactness claim. mflux rebuilds the table in float32 where the
        reference marches it forward in float64, so the two genuinely differ; the test
        pins how much.
        """
        token_grid = (2, 6, 6)
        reference_rope = _reference_rope()
        mflux_rope = _mflux_rope()
        original_length = reference_rope.freqs_cos.shape[0]

        reference_cos, reference_sin = _reference_tables(reference_rope, token_grid, t_offset)
        candidate_cos, candidate_sin = mflux_rope(token_grid=token_grid, t_offset=t_offset)

        assert reference_rope.freqs_cos.shape[0] > original_length, "the reference table did not extend"
        assert mflux_rope._extended_tables is not None, "mflux did not build an extended table"

        cos_result = compare(np.asarray(candidate_cos), reference_cos)
        sin_result = compare(np.asarray(candidate_sin), reference_sin)
        print(f"\nRoPE extension offset={t_offset} cos: {cos_result}")
        print(f"RoPE extension offset={t_offset} sin: {sin_result}")

        assert cos_result.max_abs_diff < EXTENSION_MAX_ABS, f"cos extension outside envelope: {cos_result}"
        assert sin_result.max_abs_diff < EXTENSION_MAX_ABS, f"sin extension outside envelope: {sin_result}"

    @pytest.mark.parametrize("length", [1456, 2048, 4096], ids=["1456", "2048", "4096"])
    def test_both_extensions_are_accurate_against_float64_truth(self, require_reference, length):
        """Adjudicate the two extension methods against an exact float64 build.

        Records which is closer to the truth. As of this measurement the reference's
        float64 ``atan2`` march is consistently the more accurate of the two by roughly
        1.7x - the opposite of the ordering asserted in ``rope_offset``'s module
        docstring, whose 2.83e-04 figure for the rebuild reproduces exactly while its
        2.96e-04 figure for ``atan2`` does not. The likely cause is that the docstring
        compared a *float32* atan2 extension, whereas ``_ensure_rope_cache_len`` does that
        arithmetic in float64.

        The test asserts only the absolute accuracy of each method, not their ordering, so
        it documents the discrepancy without pinning an implementation detail of the
        reference. Both stay far inside the bf16 envelope, which is what matters at
        runtime.
        """
        torch_reference()
        from swiftvr.streaming.dit import _ensure_rope_cache_len

        reference_rope = _reference_rope()
        _ensure_rope_cache_len(reference_rope, length)
        reference_cos = reference_rope.freqs_cos.numpy()

        mflux_rope = MlxRotaryPosEmbed(ATTENTION_HEAD_DIM, PATCH_SIZE, ROPE_MAX_SEQ_LEN)
        rebuilt, truth = [], []
        for dim in (mflux_rope.t_dim, mflux_rope.h_dim, mflux_rope.w_dim):
            cos, _ = MlxRotaryPosEmbed._get_1d_rotary_pos_embed(dim=dim, max_seq_len=length, theta=mflux_rope.theta)
            rebuilt.append(np.asarray(cos))
            truth.append(_float64_axis_tables(dim, np.arange(length))[0])
        candidate_cos = np.concatenate(rebuilt, axis=1)
        exact_cos = np.concatenate(truth, axis=1)

        tail = slice(ROPE_MAX_SEQ_LEN, None)
        reference_error = float(np.abs(reference_cos[tail] - exact_cos[tail]).max())
        candidate_error = float(np.abs(candidate_cos[tail] - exact_cos[tail]).max())
        print(
            f"\nRoPE extension accuracy at L={length} (rows >= {ROPE_MAX_SEQ_LEN}): "
            f"reference atan2/float64 = {reference_error:.3e}, mflux rebuild/float32 = {candidate_error:.3e}, "
            f"closer = {'reference' if reference_error < candidate_error else 'mflux'}"
        )

        assert reference_error < EXTENSION_MAX_ABS, f"reference extension error {reference_error:.3e}"
        assert candidate_error < EXTENSION_MAX_ABS, f"mflux extension error {candidate_error:.3e}"


@pytest.mark.parity
class TestRopeDivergenceStaysBelowRuntimePrecision:
    """Whether the deliberate extension divergence can affect a real run."""

    @pytest.mark.parametrize("t_offset", [1200, 4000], ids=["just-past", "far-past"])
    def test_rope_gap_is_smaller_than_bfloat16_rounding(self, require_reference, t_offset):
        """Apply both tables to the same query and weigh the gap against bf16 rounding.

        The DiT runs in bfloat16. If rotating a query with the mflux table instead of the
        reference table moves it by less than simply storing that query in bfloat16 does,
        the divergence cannot change a generated frame, whatever the table comparison says.
        """
        import mlx.core as mx

        token_grid = (2, 6, 6)
        tokens = token_grid[0] * token_grid[1] * token_grid[2]
        heads = 24

        reference_cos, reference_sin = _reference_tables(_reference_rope(), token_grid, t_offset)
        candidate_cos, candidate_sin = _mflux_rope()(token_grid=token_grid, t_offset=t_offset)

        query = seeded_normal((1, tokens, heads, ATTENTION_HEAD_DIM), seed=9)
        rotated_reference = self._reference_rotate(query, reference_cos, reference_sin)
        rotated_candidate = WanAttention._apply_rotary_emb(
            mx.array(query), mx.array(candidate_cos), mx.array(candidate_sin)
        )

        gap = compare(np.asarray(rotated_candidate), rotated_reference)
        bfloat16_cost = float(
            np.abs(
                np.asarray(mx.array(rotated_reference.numpy()).astype(mx.bfloat16).astype(mx.float32))
                - rotated_reference.numpy()
            ).max()
        )
        print(
            f"\nRoPE offset={t_offset} applied to q: gap max_abs={gap.max_abs_diff:.3e} "
            f"vs bf16 rounding of the same tensor {bfloat16_cost:.3e} "
            f"(ratio {bfloat16_cost / gap.max_abs_diff:.1f}x)"
        )

        assert gap.max_abs_diff < bfloat16_cost, (
            f"the RoPE extension divergence ({gap.max_abs_diff:.3e}) is no longer dominated by bf16 "
            f"rounding ({bfloat16_cost:.3e}); it could now affect a real run"
        )
        assert gap.cosine > 0.999999, f"rotated query direction diverged: {gap}"

    @staticmethod
    def _reference_rotate(query: np.ndarray, freqs_cos, freqs_sin):
        import torch

        torch_reference()
        from swiftvr.models.transformer import _apply_rotary_emb

        return _apply_rotary_emb(torch.from_numpy(query.copy()), freqs_cos, freqs_sin)
