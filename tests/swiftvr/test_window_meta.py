"""MFSWA index bookkeeping: the gather/scatter tables that replace global attention.

``window_meta`` is the largest piece of genuinely new numerics in the SwiftVR port, and
every way it can be wrong is silent. A wrong owner rule, a wrong window order or an
over-eager interior prune all produce tables of exactly the right shape, so attention
still runs, the output still has the right dimensions, and only the pixels are quietly
wrong. These tests pin the two properties that make the partition correct:

* ``lin_flat[owner_pos] == arange(K)`` - every token's chosen window slot really holds
  that token, so scattering back cannot mix tokens up.
* coverage - every token is owned by exactly one window, in both parities.

They also pin the identity that makes the whole scheme a drop-in for global attention:
when the window covers the grid, gather-then-scatter is the identity map, so a single
window must reproduce Wan's global self-attention exactly. That last one runs a real
``WanAttention`` with real (random) weights both ways and compares the numbers.
"""

import itertools

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.swiftvr.model.swiftvr_transformer.mfswa_attention import (
    ShiftedWindowRuntime,
    ShiftedWindowSelfAttention,
    install_shifted_window_self_attention,
    uninstall_shifted_window_self_attention,
)
from mflux.models.swiftvr.model.swiftvr_transformer.swiftvr_transformer import SwiftVRTransformer
from mflux.models.swiftvr.model.swiftvr_transformer.window_meta import (
    DEFAULT_WINDOW_HW,
    WindowGridCache,
    build_owner_positions,
    build_window_grid,
    build_window_token_indices,
    window_axis_starts,
)
from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention

# Grids chosen to hit every regime: window larger than the axis, window equal to the
# axis, exact multiples, non-multiples, and the real post-patch shapes a 1080p and a
# 720p chunk produce (1088/16/2 = 34 rows, 1920/16/2 = 60 columns; 736/16/2 = 23).
TOKEN_GRIDS = [
    (1, 1, 1),
    (1, 4, 4),
    (2, 16, 16),
    (3, 17, 5),
    (2, 20, 18),
    (1, 33, 33),
    (2, 34, 60),
    (7, 34, 60),
    (4, 23, 40),
    (5, 6, 100),
]
WINDOW_SIZES = [(1, 1), (2, 3), (4, 4), (8, 8), DEFAULT_WINDOW_HW, (32, 32)]


AXIS_SIZES = list(range(1, 70))


def _window_cases():
    """Every (window, parity) pair, swept inside a test rather than as node ids."""
    return list(itertools.product(WINDOW_SIZES, [False, True]))


class TestWindowAxisStarts:
    @pytest.mark.parametrize("do_shift", [False, True])
    def test_starts_are_sorted_unique_and_inside_the_axis(self, do_shift):
        for size in AXIS_SIZES:
            for window in (1, 2, 3, 4, 8, 16):
                window = min(window, size)
                starts = window_axis_starts(size, window, do_shift=do_shift)
                assert starts == sorted(set(starts)), (size, window)
                assert all(0 <= start <= size - window for start in starts), (size, window)

    @pytest.mark.parametrize("do_shift", [False, True])
    def test_the_starts_cover_the_whole_axis(self, do_shift):
        """A coverage hole here means tokens no window ever sees."""
        for size in AXIS_SIZES:
            for window in (1, 2, 3, 4, 8, 16):
                window = min(window, size)
                covered = set()
                for start in window_axis_starts(size, window, do_shift=do_shift):
                    covered.update(range(start, start + window))
                assert covered == set(range(size)), (size, window)

    def test_an_axis_that_fits_in_one_window_yields_a_single_start(self):
        for size in range(1, 17):
            assert window_axis_starts(size, 16, do_shift=False) == [0]
            assert window_axis_starts(size, 16, do_shift=True) == [0]

    def test_the_shift_moves_the_starts_when_the_axis_is_wide_enough(self):
        """With <= 2 windows the clamp absorbs the shift; the two parities must differ
        somewhere, or consecutive layers would partition identically and no information
        would ever cross a window seam."""
        assert window_axis_starts(34, 16, do_shift=False) != window_axis_starts(34, 16, do_shift=True)
        assert window_axis_starts(60, 16, do_shift=False) != window_axis_starts(60, 16, do_shift=True)

    @pytest.mark.parametrize(("size", "window"), [(0, 4), (4, 0), (-1, 4), (4, -2)])
    def test_a_non_positive_extent_raises(self, size, window):
        with pytest.raises(ValueError, match="positive size and window"):
            window_axis_starts(size, window, do_shift=False)


class TestWindowGrid:
    @pytest.mark.parametrize("token_grid", TOKEN_GRIDS)
    def test_scatter_of_gather_is_the_identity_on_token_positions(self, token_grid):
        """``lin_flat[owner_pos] == arange(K)``: the slot a token owns holds that token."""
        for window_hw, do_shift in _window_cases():
            grid = build_window_grid(token_grid, window_hw, do_shift=do_shift)
            gathered = np.array(grid.lin_flat)[np.array(grid.owner_pos)]
            assert np.array_equal(gathered, np.arange(grid.num_tokens)), (window_hw, do_shift)

    @pytest.mark.parametrize("token_grid", TOKEN_GRIDS)
    def test_every_token_is_covered_by_at_least_one_window(self, token_grid):
        for window_hw, do_shift in _window_cases():
            grid = build_window_grid(token_grid, window_hw, do_shift=do_shift)
            assert set(np.array(grid.lin_flat).tolist()) == set(range(grid.num_tokens)), (window_hw, do_shift)

    @pytest.mark.parametrize("token_grid", TOKEN_GRIDS)
    def test_the_table_shape_is_num_windows_times_window_tokens(self, token_grid):
        num_frames, height, width = token_grid
        for window_hw, do_shift in _window_cases():
            grid = build_window_grid(token_grid, window_hw, do_shift=do_shift)
            clamped = (min(window_hw[0], height), min(window_hw[1], width))
            expected_windows = len(window_axis_starts(height, clamped[0], do_shift=do_shift)) * len(
                window_axis_starts(width, clamped[1], do_shift=do_shift)
            )
            assert grid.window_hw == clamped, (window_hw, do_shift)
            assert grid.num_windows == expected_windows, (window_hw, do_shift)
            assert grid.window_tokens == num_frames * clamped[0] * clamped[1], (window_hw, do_shift)
            assert grid.lin_flat.shape == (grid.num_windows * grid.window_tokens,), (window_hw, do_shift)
            assert grid.owner_pos.shape == (grid.num_tokens,), (window_hw, do_shift)
            assert grid.num_tokens == num_frames * height * width, (window_hw, do_shift)

    @pytest.mark.parametrize("token_grid", TOKEN_GRIDS)
    def test_every_window_spans_the_full_temporal_axis(self, token_grid):
        """MFSWA is 2D-spatial with a FULL temporal view; a windowed time axis is a
        different model that would raise nothing."""
        num_frames, height, width = token_grid
        for window_hw, do_shift in _window_cases():
            grid = build_window_grid(token_grid, window_hw, do_shift=do_shift)
            table = np.array(grid.lin_flat).reshape(grid.num_windows, grid.window_tokens)
            frames_per_window = table // (height * width)
            for window in frames_per_window:
                assert set(window.tolist()) == set(range(num_frames)), (window_hw, do_shift)

    @pytest.mark.parametrize("token_grid", TOKEN_GRIDS)
    def test_each_window_is_a_dense_rectangle_inside_the_grid(self, token_grid):
        """Mask-free depends on this: every window is exactly wh x ww and fully interior,
        so there is nothing to mask and no padding token to exclude."""
        _, height, width = token_grid
        for window_hw, do_shift in _window_cases():
            grid = build_window_grid(token_grid, window_hw, do_shift=do_shift)
            table = np.array(grid.lin_flat).reshape(grid.num_windows, grid.window_tokens)
            spatial = table % (height * width)
            rows, columns = spatial // width, spatial % width
            for window_rows, window_columns in zip(rows, columns):
                assert len(set(window_rows.tolist())) == grid.window_hw[0], (window_hw, do_shift)
                assert len(set(window_columns.tolist())) == grid.window_hw[1], (window_hw, do_shift)
                assert max(window_rows) - min(window_rows) == grid.window_hw[0] - 1, (window_hw, do_shift)
                assert max(window_columns) - min(window_columns) == grid.window_hw[1] - 1, (window_hw, do_shift)

    @pytest.mark.parametrize("token_grid", TOKEN_GRIDS)
    def test_ownership_priority_flips_with_the_parity(self, token_grid):
        """Unshifted layers hand an overlap to the lowest-index covering window, shifted
        layers to the highest-index one. That disagreement is the whole mechanism."""
        unshifted = build_window_grid(token_grid, (4, 4), do_shift=False)
        shifted = build_window_grid(token_grid, (4, 4), do_shift=True)
        assert unshifted.prefer_front is True
        assert shifted.prefer_front is False
        for grid in (unshifted, shifted):
            table = np.array(grid.lin_flat).reshape(grid.num_windows, grid.window_tokens)
            owners = np.array(grid.owner_pos) // grid.window_tokens
            for token, owner in enumerate(owners):
                covering = [index for index in range(grid.num_windows) if token in set(table[index].tolist())]
                assert owner == (min(covering) if grid.prefer_front else max(covering))

    def test_a_window_larger_than_the_grid_degenerates_to_one_window(self):
        grid = build_window_grid((3, 8, 8), (16, 16), do_shift=True)
        assert grid.num_windows == 1
        assert grid.window_hw == (8, 8)
        assert grid.window_tokens == grid.num_tokens

    @pytest.mark.parametrize(
        ("token_grid", "window_hw"),
        [((0, 4, 4), (2, 2)), ((1, 0, 4), (2, 2)), ((1, 4, 0), (2, 2)), ((1, 4, 4), (0, 2)), ((1, 4, 4), (2, -1))],
    )
    def test_a_degenerate_grid_or_window_raises(self, token_grid, window_hw):
        with pytest.raises(ValueError, match="positive"):
            build_window_grid(token_grid, window_hw, do_shift=False)

    def test_an_uncovered_token_is_reported_rather_than_left_at_minus_one(self):
        """build_owner_positions is the only guard against a prune that opens a hole."""
        indices = build_window_token_indices((1, 4, 4), [0], [0], (2, 2))
        with pytest.raises(ValueError, match="uncovered"):
            build_owner_positions(indices, prefer_front=True, num_tokens=16)


class TestWindowGridCache:
    def test_a_repeated_request_returns_the_same_tables(self):
        cache = WindowGridCache()
        first = cache.get((2, 34, 60), (16, 16), do_shift=False)
        second = cache.get((2, 34, 60), (16, 16), do_shift=False)
        assert first is second

    def test_parity_and_grid_are_both_part_of_the_key(self):
        cache = WindowGridCache()
        base = cache.get((2, 34, 60), (16, 16), do_shift=False)
        assert cache.get((2, 34, 60), (16, 16), do_shift=True) is not base
        assert cache.get((3, 34, 60), (16, 16), do_shift=False) is not base
        assert cache.get((2, 34, 60), (8, 8), do_shift=False) is not base

    def test_the_cache_is_bounded_and_evicts_the_oldest_entry(self):
        cache = WindowGridCache(max_entries=2)
        first = cache.get((1, 8, 8), (4, 4), do_shift=False)
        cache.get((1, 9, 9), (4, 4), do_shift=False)
        cache.get((1, 10, 10), (4, 4), do_shift=False)
        assert len(cache._store) == 2
        assert cache.get((1, 8, 8), (4, 4), do_shift=False) is not first

    def test_clear_drops_everything(self):
        cache = WindowGridCache()
        first = cache.get((1, 8, 8), (4, 4), do_shift=False)
        cache.clear()
        assert cache.get((1, 8, 8), (4, 4), do_shift=False) is not first

    def test_a_cache_with_no_room_raises(self):
        with pytest.raises(ValueError, match="at least one entry"):
            WindowGridCache(max_entries=0)


def _tiny_transformer(window_hw=(2, 2), **overrides):
    """A real SwiftVRTransformer, small enough to run in a unit test.

    Random weights, real modules, real arithmetic: this is a small model, not a stand-in
    for one. Nothing here compares against the published checkpoint - that is what
    tests/swiftvr/parity is for.
    """
    kwargs = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 8,
        "out_channels": 8,
        "text_dim": 16,
        "freq_dim": 16,
        "ffn_dim": 32,
        "num_layers": 2,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 64,
    }
    kwargs.update(overrides)
    return SwiftVRTransformer(window_hw=window_hw, shift_alternate_layers=True, **kwargs)


class TestWindowedAttentionAgainstGlobalAttention:
    """The identity that makes MFSWA a drop-in: one window == Wan's global attention."""

    @staticmethod
    def _setup(token_grid, window_hw):
        mx.random.seed(11)
        transformer = _tiny_transformer(window_hw=window_hw)
        attention = transformer.transformer.blocks[0].attn1
        num_tokens = token_grid[0] * token_grid[1] * token_grid[2]
        hidden = mx.random.normal((1, num_tokens, attention.inner_dim))
        rotary = transformer.rope_offset(token_grid=token_grid, t_offset=0)
        runtime = ShiftedWindowRuntime(window_hw=window_hw)
        runtime.token_grid = token_grid
        runtime.rotary_emb = rotary
        return attention, hidden, rotary, runtime

    @pytest.mark.parametrize("token_grid", [(1, 4, 4), (3, 5, 6), (2, 8, 8)])
    def test_a_single_window_reproduces_global_self_attention(self, token_grid):
        window_hw = (token_grid[1], token_grid[2])
        attention, hidden, rotary, runtime = self._setup(token_grid, window_hw)

        global_output = attention(hidden, rotary_emb=rotary, attention_name="attn1")
        strategy = ShiftedWindowSelfAttention(runtime, do_shift=False)
        windowed = strategy(attention, hidden, rotary, "attn1", None)

        mx.eval(global_output, windowed)
        assert windowed.shape == global_output.shape
        assert float(mx.max(mx.abs(windowed - global_output))) < 2e-6

    def test_a_partitioned_grid_does_not_reproduce_global_attention(self):
        """The negative control: if it did, the windows would not be restricting anything."""
        token_grid = (2, 8, 8)
        attention, hidden, rotary, runtime = self._setup(token_grid, (2, 2))
        global_output = attention(hidden, rotary_emb=rotary, attention_name="attn1")
        windowed = ShiftedWindowSelfAttention(runtime, do_shift=False)(attention, hidden, rotary, "attn1", None)
        mx.eval(global_output, windowed)
        assert float(mx.max(mx.abs(windowed - global_output))) > 1e-3

    def test_the_two_parities_partition_differently_on_a_real_grid(self):
        """A shift that the boundary clamp absorbs would make consecutive layers identical."""
        token_grid = (2, 34, 60)
        attention, hidden, rotary, runtime = self._setup(token_grid, (16, 16))
        unshifted = ShiftedWindowSelfAttention(runtime, do_shift=False)(attention, hidden, rotary, "attn1", None)
        shifted = ShiftedWindowSelfAttention(runtime, do_shift=True)(attention, hidden, rotary, "attn1", None)
        mx.eval(unshifted, shifted)
        assert float(mx.max(mx.abs(unshifted - shifted))) > 1e-3

    def test_the_runtime_must_be_primed_before_a_chunk(self):
        attention, hidden, rotary, _ = self._setup((1, 4, 4), (4, 4))
        strategy = ShiftedWindowSelfAttention(ShiftedWindowRuntime(window_hw=(4, 4)), do_shift=False)
        with pytest.raises(ValueError, match="token_grid must be set"):
            strategy(attention, hidden, rotary, "attn1", None)

    def test_a_token_count_that_disagrees_with_the_grid_raises(self):
        attention, hidden, rotary, runtime = self._setup((1, 4, 4), (4, 4))
        runtime.token_grid = (1, 4, 5)
        with pytest.raises(ValueError, match="token count mismatch"):
            ShiftedWindowSelfAttention(runtime, do_shift=False)(attention, hidden, rotary, "attn1", None)

    def test_rotary_tables_covering_a_different_chunk_raise(self):
        attention, hidden, rotary, runtime = self._setup((1, 4, 4), (4, 4))
        runtime.rotary_emb = (rotary[0][:, :4], rotary[1][:, :4])
        with pytest.raises(ValueError, match="rotary table mismatch"):
            ShiftedWindowSelfAttention(runtime, do_shift=False)(attention, hidden, rotary, "attn1", None)


class TestInstallation:
    def test_mfswa_is_installed_on_attn1_only_and_alternates_the_shift(self):
        transformer = _tiny_transformer(num_layers=4)
        runtime = ShiftedWindowRuntime()
        strategies = install_shifted_window_self_attention(transformer.transformer, runtime=runtime)
        assert [strategy.do_shift for strategy in strategies] == [False, True, False, True]
        for index, block in enumerate(transformer.transformer.blocks):
            assert block.attn1.self_attention_strategy is strategies[index]
            assert block.attn2.self_attention_strategy is None

    def test_the_shift_can_be_turned_off_for_every_block(self):
        transformer = _tiny_transformer(num_layers=4)
        strategies = install_shifted_window_self_attention(
            transformer.transformer,
            runtime=ShiftedWindowRuntime(),
            shift_alternate_layers=False,
        )
        assert [strategy.do_shift for strategy in strategies] == [False] * 4

    def test_uninstall_restores_global_attention(self):
        transformer = _tiny_transformer(num_layers=2)
        transformer.install_mfswa()
        assert transformer.mfswa_installed is True
        transformer.uninstall_mfswa()
        assert transformer.mfswa_installed is False
        assert all(block.attn1.self_attention_strategy is None for block in transformer.transformer.blocks)

    def test_installing_on_a_build_without_the_hook_fails_closed(self):
        """Without the WanAttention seam the assignment would be inert and SwiftVR would
        silently run Wan's global attention - the exact failure ADR 0002 forbids."""
        transformer = _tiny_transformer(num_layers=1)
        attention = transformer.transformer.blocks[0].attn1
        assert hasattr(attention, "self_attention_strategy")
        del attention.self_attention_strategy
        with pytest.raises(ValueError, match="does not expose the"):
            install_shifted_window_self_attention(transformer.transformer, runtime=ShiftedWindowRuntime())
        attention.self_attention_strategy = None

    def test_installing_on_a_transformer_with_no_blocks_raises(self):
        class Blockless:
            blocks: list = []

        with pytest.raises(ValueError, match="exposes no blocks"):
            install_shifted_window_self_attention(Blockless(), runtime=ShiftedWindowRuntime())

    def test_uninstall_on_a_blockless_object_is_a_no_op(self):
        class Blockless:
            blocks: list = []

        uninstall_shifted_window_self_attention(Blockless())

    def test_a_strategy_on_cross_attention_raises_rather_than_routing_it(self):
        transformer = _tiny_transformer(num_layers=1)
        cross_attention = transformer.transformer.blocks[0].attn2
        runtime = ShiftedWindowRuntime(window_hw=(2, 2))
        runtime.token_grid = (1, 4, 4)
        runtime.rotary_emb = transformer.rope_offset(token_grid=(1, 4, 4), t_offset=0)
        hidden = mx.random.normal((1, 16, cross_attention.inner_dim))
        with pytest.raises(ValueError, match="install it on attn1 only"):
            ShiftedWindowSelfAttention(runtime, do_shift=False)(cross_attention, hidden, None, "attn2", None)

    def test_the_wan_attention_hook_dispatches_to_the_strategy(self):
        """The seam itself: WanAttention.__call__ must consult the strategy before
        projecting, or installation would change nothing."""
        transformer = _tiny_transformer(num_layers=1)
        attention: WanAttention = transformer.transformer.blocks[0].attn1
        token_grid = (1, 8, 8)
        runtime = ShiftedWindowRuntime(window_hw=(2, 2))
        runtime.token_grid = token_grid
        runtime.rotary_emb = transformer.rope_offset(token_grid=token_grid, t_offset=0)
        hidden = mx.random.normal((1, 64, attention.inner_dim))

        without = attention(hidden, rotary_emb=runtime.rotary_emb, attention_name="attn1")
        attention.self_attention_strategy = ShiftedWindowSelfAttention(runtime, do_shift=False)
        with_strategy = attention(hidden, rotary_emb=runtime.rotary_emb, attention_name="attn1")
        mx.eval(without, with_strategy)
        assert float(mx.max(mx.abs(without - with_strategy))) > 1e-3
