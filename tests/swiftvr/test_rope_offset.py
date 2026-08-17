"""Temporally-offset rotary tables, without torch.

This is where a ``t_offset`` regression is actually catchable. Inside one chunk the offset
is a mathematical no-op - RoPE encodes relative position, SwiftVR keeps no KV cache, and a
uniform shift of every position cancels - so no DiT output test can see it (see
``test_streaming_dit.TestTemporalOffset``). The tables themselves are the observable, so
they are what these tests read.

``tests/swiftvr/parity/test_rope_offset_parity.py`` compares the same tables against the
reference implementation and skips when torch or the reference tree is absent. This file
holds the properties that need neither.
"""

import mlx.core as mx
import pytest

from mflux.models.swiftvr.model.swiftvr_transformer.rope_offset import ROPE_EXTEND_MARGIN, RopeTemporalOffset
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed

ATTENTION_HEAD_DIM = 128
PATCH_SIZE = (1, 2, 2)
MAX_SEQ_LEN = 64


@pytest.fixture
def rope_offset() -> RopeTemporalOffset:
    return RopeTemporalOffset(WanRotaryPosEmbed(ATTENTION_HEAD_DIM, PATCH_SIZE, MAX_SEQ_LEN))


def _max_difference(first: mx.array, second: mx.array) -> float:
    mx.eval(first, second)
    return float(mx.max(mx.abs(first - second)))


class TestTableShape:
    @pytest.mark.parametrize("token_grid", [(1, 1, 1), (3, 4, 4), (7, 34, 60), (2, 5, 3)])
    def test_the_tables_cover_one_row_per_token(self, rope_offset, token_grid):
        cos, sin = rope_offset(token_grid=token_grid, t_offset=0)
        num_tokens = token_grid[0] * token_grid[1] * token_grid[2]
        assert cos.shape == sin.shape == (1, num_tokens, 1, ATTENTION_HEAD_DIM)

    def test_the_layout_matches_wan_s_own_unoffset_tables(self, rope_offset):
        """At offset zero this must be Wan's table exactly, or the plain and chunked
        routes are running different position encodings."""
        latents = mx.zeros((1, 8, 3, 8, 8))
        wan_cos, wan_sin = rope_offset.rope(latents)
        cos, sin = rope_offset(token_grid=(3, 4, 4), t_offset=0)
        assert _max_difference(cos, wan_cos) == 0.0
        assert _max_difference(sin, wan_sin) == 0.0


class TestTemporalSlice:
    @pytest.mark.parametrize("t_offset", [1, 3, 7, 13])
    def test_an_offset_chunk_reuses_the_rows_of_a_longer_unoffset_chunk(self, rope_offset, t_offset):
        """The offset is a slice start, nothing else: rows [t_offset, t_offset + T) of the
        table a longer chunk would have used."""
        frames, height, width = 3, 4, 4
        offset_cos, offset_sin = rope_offset(token_grid=(frames, height, width), t_offset=t_offset)
        long_cos, long_sin = rope_offset(token_grid=(t_offset + frames, height, width), t_offset=0)
        tokens_per_frame = height * width
        start = t_offset * tokens_per_frame
        end = start + frames * tokens_per_frame
        assert _max_difference(offset_cos, long_cos[:, start:end]) == 0.0
        assert _max_difference(offset_sin, long_sin[:, start:end]) == 0.0

    def test_the_spatial_axes_are_never_offset(self, rope_offset):
        """Upstream's _rope_with_offset takes h_off and w_off, and every caller passes 0."""
        base = rope_offset(token_grid=(1, 4, 4), t_offset=0)[0]
        shifted = rope_offset(token_grid=(1, 4, 4), t_offset=5)[0]
        rope = rope_offset.rope
        temporal_dim = rope.t_dim
        # The temporal block of the encoding moves; the height and width blocks do not.
        assert _max_difference(base[..., :temporal_dim], shifted[..., :temporal_dim]) > 1e-2
        assert _max_difference(base[..., temporal_dim:], shifted[..., temporal_dim:]) == 0.0

    @pytest.mark.parametrize("t_offset", [0, 1, 9])
    def test_frames_within_a_chunk_keep_consecutive_positions(self, rope_offset, t_offset):
        frames, height, width = 4, 2, 2
        cos, _ = rope_offset(token_grid=(frames, height, width), t_offset=t_offset)
        single = [rope_offset(token_grid=(1, height, width), t_offset=t_offset + index)[0] for index in range(frames)]
        stacked = mx.concatenate(single, axis=1)
        assert _max_difference(cos, stacked) == 0.0


class TestTableExtension:
    def test_a_chunk_inside_the_registered_table_uses_it_untouched(self, rope_offset):
        """ModelSaver persists rope.freqs_cos/sin, so growing them would change the shape
        of two keys in every saved checkpoint."""
        registered_shape = rope_offset.rope.freqs_cos.shape
        rope_offset(token_grid=(3, 4, 4), t_offset=MAX_SEQ_LEN - 4)
        assert rope_offset.rope.freqs_cos.shape == registered_shape
        assert rope_offset._extended_tables is None

    def test_a_chunk_past_the_table_extends_a_private_copy(self, rope_offset):
        rope_offset(token_grid=(3, 4, 4), t_offset=MAX_SEQ_LEN)
        assert rope_offset.rope.freqs_cos.shape == (MAX_SEQ_LEN, ATTENTION_HEAD_DIM)
        assert rope_offset._extended_tables is not None
        assert rope_offset._extended_tables[0].shape[0] >= MAX_SEQ_LEN + 3 + ROPE_EXTEND_MARGIN

    def test_the_extension_agrees_with_the_registered_table_on_their_shared_prefix(self, rope_offset):
        """The extension reuses Wan's own generator, so there is no second frequency
        schedule that could drift."""
        rope_offset(token_grid=(3, 4, 4), t_offset=MAX_SEQ_LEN)
        extended_cos, extended_sin = rope_offset._extended_tables
        registered_cos, registered_sin = rope_offset.rope.freqs_cos, rope_offset.rope.freqs_sin
        assert _max_difference(extended_cos[:MAX_SEQ_LEN], registered_cos) == 0.0
        assert _max_difference(extended_sin[:MAX_SEQ_LEN], registered_sin) == 0.0

    def test_the_extension_is_reused_for_the_next_chunk(self, rope_offset):
        rope_offset(token_grid=(3, 4, 4), t_offset=MAX_SEQ_LEN)
        tables = rope_offset._extended_tables
        rope_offset(token_grid=(3, 4, 4), t_offset=MAX_SEQ_LEN + 3)
        assert rope_offset._extended_tables is tables

    def test_clear_cache_drops_the_extension(self, rope_offset):
        rope_offset(token_grid=(3, 4, 4), t_offset=MAX_SEQ_LEN)
        rope_offset.clear_cache()
        assert rope_offset._extended_tables is None


class TestFailClosed:
    @pytest.mark.parametrize("token_grid", [(0, 4, 4), (3, 0, 4), (3, 4, 0), (-1, 4, 4)])
    def test_a_degenerate_grid_raises(self, rope_offset, token_grid):
        with pytest.raises(ValueError, match="positive in every axis"):
            rope_offset(token_grid=token_grid, t_offset=0)

    @pytest.mark.parametrize("t_offset", [-1, -100])
    def test_a_negative_offset_raises(self, rope_offset, t_offset):
        with pytest.raises(ValueError, match="non-negative"):
            rope_offset(token_grid=(3, 4, 4), t_offset=t_offset)
