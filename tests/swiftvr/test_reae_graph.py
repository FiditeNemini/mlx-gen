"""The Restoration-aware Autoencoder graph and its causal streaming state.

Two things are load-bearing and silent when wrong.

*Topology.* The published checkpoint stores the encoder and decoder as ``nn.Sequential``
with positional keys, so every non-parametric entry (``ReLU``, ``Upsample``, ``Clamp``)
holds an index that later tensors depend on. Insert or drop one and the weight mapping
still applies, just to the wrong layers. The parameter and tensor totals are the cheap
signal that the layer lists still match the checkpoint, which is why they are asserted
against the published numbers here rather than against whatever the code currently
builds.

*Causality.* Chunked encoding is only equivalent to whole-clip encoding because the
MemBlock and TPool boundary buffers cross the chunk seam. Dropping that carry raises
nothing - it just puts a seam in the video. The tests below prove the carry is actually
wired by showing that a reset state produces a different result, and that a chunked run
reproduces the whole-clip run to numerical noise.

Weights are random. These are topology, shape and state-wiring tests on real modules;
numerical agreement with the reference is tests/swiftvr/parity's job.
"""

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.swiftvr.model.swiftvr_reae.reae import (
    REAE_PARAMETER_COUNT,
    REAE_TENSOR_COUNT,
    ReAE,
    summarize_reae_parameters,
)
from mflux.models.swiftvr.model.swiftvr_reae.reae_blocks import (
    Clamp,
    MemBlock,
    TGrow,
    TPool,
    pixel_shuffle_nhwc,
    pixel_unshuffle_nhwc,
)
from mflux.models.swiftvr.streaming.chunk import ChunkSpec, ChunkType, build_chunk_specs
from mflux.models.swiftvr.streaming.streaming_reae import ReAEStackState, ReAEStreamingCodec, run_reae_stack

SPATIAL = 64
LATENT_SPATIAL = SPATIAL // 16


def _clip(frames: int, seed: int = 0, height: int = SPATIAL, width: int = SPATIAL) -> mx.array:
    mx.random.seed(seed)
    return mx.clip(mx.random.uniform(shape=(1, frames, height, width, 3)), 0.0, 1.0)


def _relative_difference(first: mx.array, second: mx.array) -> float:
    delta = float(mx.max(mx.abs(first - second)))
    scale = float(mx.max(mx.abs(first))) or 1.0
    return delta / scale


class TestTopology:
    def test_the_parameter_count_matches_the_published_checkpoint(self):
        summary = summarize_reae_parameters()
        assert summary.parameter_count == REAE_PARAMETER_COUNT == 40_946_364
        assert summary.tensor_count == REAE_TENSOR_COUNT == 128
        assert summary.matches_published_checkpoint is True

    def test_the_summary_reports_a_consistent_split_and_size(self):
        summary = summarize_reae_parameters()
        assert summary.encoder_parameter_count + summary.decoder_parameter_count == summary.parameter_count
        assert summary.float32_bytes == summary.parameter_count * 4
        # 40.95M parameters against the Wan 3D VAE's 704.69M is the point of ReAE.
        assert 40e6 < summary.parameter_count < 41e6

    def test_the_summary_inspects_a_supplied_model_rather_than_a_fresh_one(self):
        model = ReAE()
        assert summarize_reae_parameters(model).parameter_count == summarize_reae_parameters().parameter_count

    def test_the_compression_contract_matches_the_wan_latent_it_replaces(self):
        model = ReAE()
        assert model.latent_channels == 48
        assert model.spatial_scale == 16
        assert model.temporal_scale == 4
        assert model.frames_to_trim == 3

    def test_the_stateful_layers_sit_at_the_checkpoint_indices(self):
        """The mapping generates targets from these positions; a shift silently misaligns."""
        model = ReAE()
        encoder = model.encoder.layers
        decoder = model.decoder.layers
        assert [i for i, layer in enumerate(encoder) if isinstance(layer, MemBlock)] == [4, 5, 6, 9, 10, 11, 14, 15, 16]
        assert [i for i, layer in enumerate(encoder) if isinstance(layer, TPool)] == [2, 7, 12]
        assert [encoder[i].stride for i in (2, 7, 12)] == [2, 2, 1]
        assert [i for i, layer in enumerate(decoder) if isinstance(layer, MemBlock)] == [3, 4, 5, 9, 10, 11, 15, 16, 17]
        assert [i for i, layer in enumerate(decoder) if isinstance(layer, TGrow)] == [7, 13, 19]
        assert [decoder[i].stride for i in (7, 13, 19)] == [1, 2, 2]
        assert isinstance(decoder[0], Clamp)

    @pytest.mark.parametrize(
        ("time_upscale", "space_upscale"),
        [
            ((True, False), (True, True, True)),
            ((True, True), (True, True, False)),
            ((False, False), (True, True, True)),
        ],
    )
    def test_decoder_geometry_that_cannot_invert_the_encoder_raises(self, time_upscale, space_upscale):
        """The reference carries these flags unchecked, so a disabled stage silently
        yields an autoencoder whose decode does not undo its encode."""
        with pytest.raises(ValueError, match="would not invert the encoder"):
            ReAE(decoder_time_upscale=time_upscale, decoder_space_upscale=space_upscale)


class TestBlocks:
    def test_clamp_is_the_reference_soft_limiter(self):
        values = mx.array([-30.0, -3.0, 0.0, 3.0, 30.0])
        clamped = Clamp()(values)
        assert np.allclose(np.array(clamped), np.tanh(np.array(values) / 3.0) * 3.0, atol=1e-6)
        # tanh saturates, so the bound is closed in float32: tanh(10) * 3 rounds to 3.0.
        assert float(mx.max(mx.abs(clamped))) <= 3.0

    def test_a_memblock_rejects_a_past_of_the_wrong_shape(self):
        """A mis-shaped carry is the failure mode of the streaming state."""
        block = MemBlock(8, 8)
        x = mx.zeros((2, 4, 4, 8))
        with pytest.raises(ValueError, match="needs past to match x"):
            block(x, mx.zeros((1, 4, 4, 8)))

    def test_a_memblock_reads_its_past(self):
        block = MemBlock(8, 8)
        mx.random.seed(3)
        x = mx.random.normal((2, 4, 4, 8))
        zero_past = block(x, mx.zeros_like(x))
        real_past = block(x, mx.random.normal((2, 4, 4, 8)))
        assert _relative_difference(zero_past, real_past) > 1e-3

    @pytest.mark.parametrize("stride", [1, 2])
    def test_tpool_folds_frames_into_channels_and_shortens_the_clip(self, stride):
        pool = TPool(6, stride)
        out = pool(mx.random.normal((4, 5, 5, 6)))
        assert out.shape == (4 // stride, 5, 5, 6)

    def test_tpool_rejects_a_partial_temporal_group(self):
        with pytest.raises(ValueError, match="requires a multiple of 2 frames"):
            TPool(6, 2)(mx.random.normal((3, 5, 5, 6)))

    def test_tpool_rejects_a_non_positive_stride(self):
        with pytest.raises(ValueError, match="stride must be positive"):
            TPool(6, 0)

    @pytest.mark.parametrize("stride", [1, 2])
    def test_tgrow_multiplies_the_frame_axis(self, stride):
        grow = TGrow(6, stride)
        out = grow(mx.random.normal((3, 5, 5, 6)))
        assert out.shape == (3 * stride, 5, 5, 6)

    def test_tgrow_is_frame_local(self):
        """No cross-frame mixing is why the streaming runner carries no TGrow state."""
        grow = TGrow(6, 2)
        mx.random.seed(5)
        frames = mx.random.normal((3, 5, 5, 6))
        whole = grow(frames)
        piecewise = mx.concatenate([grow(frames[index : index + 1]) for index in range(3)], axis=0)
        mx.eval(whole, piecewise)
        assert _relative_difference(whole, piecewise) < 1e-6

    def test_tgrow_rejects_an_unsupported_stride(self):
        with pytest.raises(ValueError, match="supports stride 1 or 2"):
            TGrow(6, 3)

    @pytest.mark.parametrize("ratio", [1, 2, 4])
    def test_pixel_shuffle_inverts_pixel_unshuffle(self, ratio):
        mx.random.seed(7)
        frames = mx.random.normal((2, 8, 8, 3))
        round_trip = pixel_shuffle_nhwc(pixel_unshuffle_nhwc(frames, ratio), ratio)
        assert round_trip.shape == frames.shape
        assert float(mx.max(mx.abs(round_trip - frames))) == 0.0

    def test_pixel_unshuffle_rejects_an_indivisible_canvas(self):
        with pytest.raises(ValueError, match="divisible spatial dims"):
            pixel_unshuffle_nhwc(mx.zeros((1, 7, 8, 3)), 2)

    def test_pixel_shuffle_rejects_an_indivisible_channel_count(self):
        with pytest.raises(ValueError, match="channels divisible by"):
            pixel_shuffle_nhwc(mx.zeros((1, 4, 4, 5)), 2)


class TestRoundTripShapes:
    @pytest.mark.parametrize(("total_frames", "clip_len"), [(9, 8), (29, 24), (13, 4), (33, 8)])
    def test_a_whole_clip_decodes_back_to_its_source_frame_count(self, total_frames, clip_len):
        codec = ReAEStreamingCodec(ReAE())
        written = 0
        for spec in build_chunk_specs(total_frames, clip_len):
            latents = codec.encode_chunk(_clip(spec.frame_count, seed=spec.clip_idx), spec)
            assert latents.shape == (1, spec.latent_count, LATENT_SPATIAL, LATENT_SPATIAL, 48)
            for decoded in codec.iter_decode_chunk(latents, spec):
                assert decoded.shape[2:] == (SPATIAL, SPATIAL, 3)
                written += int(decoded.shape[1])
        assert written == total_frames

    def test_decoded_pixels_stay_in_the_unit_range(self):
        codec = ReAEStreamingCodec(ReAE())
        spec = build_chunk_specs(9, 8)[0]
        decoded = codec.decode_chunk(codec.encode_chunk(_clip(spec.frame_count), spec), spec)
        assert float(mx.min(decoded)) >= 0.0
        assert float(mx.max(decoded)) <= 1.0

    def test_a_non_square_canvas_round_trips(self):
        codec = ReAEStreamingCodec(ReAE())
        spec = build_chunk_specs(9, 8)[0]
        frames = _clip(9, height=32, width=96)
        latents = codec.encode_chunk(frames, spec)
        assert latents.shape == (1, spec.latent_count, 2, 6, 48)
        decoded = codec.decode_chunk(latents, spec)
        assert decoded.shape == (1, 9, 32, 96, 3)

    def test_the_decoder_head_trim_happens_once_per_clip(self):
        """frames_to_trim causal head frames are dropped on the first decode only."""
        codec = ReAEStreamingCodec(ReAE())
        specs = build_chunk_specs(29, 24)
        assert codec.is_first_decode is True
        counts = []
        for spec in specs:
            latents = codec.encode_chunk(_clip(spec.frame_count, seed=spec.clip_idx), spec)
            counts.append(int(codec.decode_chunk(latents, spec).shape[1]))
        assert codec.is_first_decode is False
        assert counts[0] == specs[0].latent_count * 4 - ReAE().frames_to_trim
        assert counts[1] == specs[1].latent_count * 4
        assert sum(counts) == 29

    def test_reset_re_arms_the_head_trim_for_the_next_clip(self):
        codec = ReAEStreamingCodec(ReAE())
        spec = build_chunk_specs(9, 8)[0]
        first = codec.decode_chunk(codec.encode_chunk(_clip(9), spec), spec)
        codec.reset()
        assert codec.is_first_decode is True
        second = codec.decode_chunk(codec.encode_chunk(_clip(9), spec), spec)
        assert first.shape == second.shape


class TestCausalState:
    def test_the_memblock_carry_crosses_the_chunk_boundary(self):
        """The proof that causality is wired: a codec that has just seen chunk one must
        encode chunk two differently from a codec that starts fresh."""
        specs = build_chunk_specs(29, 24)
        first_frames = _clip(specs[0].frame_count, seed=1)
        second_frames = _clip(specs[1].frame_count, seed=2)

        model = ReAE()
        carried = ReAEStreamingCodec(model)
        carried.encode_chunk(first_frames, specs[0])
        with_history = carried.encode_chunk(second_frames, specs[1])

        fresh = ReAEStreamingCodec(model)
        # Same weights, same input, same spec - only the boundary state differs. The spec
        # is rebuilt as a FIRST-position LAST so the codec is asked for identical work.
        standalone_spec = ChunkSpec(
            ctype=ChunkType.LAST,
            frame_start=0,
            frame_count=specs[1].frame_count,
            b=specs[1].b,
            clip_idx=0,
            is_first_decode=True,
        )
        without_history = fresh.encode_chunk(second_frames, standalone_spec)

        assert with_history.shape == without_history.shape
        assert _relative_difference(with_history, without_history) > 1e-3

    def test_the_decoder_carry_crosses_the_chunk_boundary(self):
        model = ReAE()
        specs = build_chunk_specs(29, 24)
        codec = ReAEStreamingCodec(model)
        latents_one = codec.encode_chunk(_clip(specs[0].frame_count, seed=1), specs[0])
        latents_two = codec.encode_chunk(_clip(specs[1].frame_count, seed=2), specs[1])
        codec.decode_chunk(latents_one, specs[0])
        with_history = codec.decode_chunk(latents_two, specs[1])

        fresh = ReAEStreamingCodec(model)
        fresh._decode_started = True
        fresh._pending_head_trim = 0
        without_history = fresh.decode_chunk(latents_two)

        assert with_history.shape == without_history.shape
        assert _relative_difference(with_history, without_history) > 1e-3

    def test_chunked_encoding_reproduces_whole_clip_encoding(self):
        """The state exists so that streaming equals the non-streaming result."""
        model = ReAE()
        frames = _clip(16, seed=4)
        whole = ReAEStreamingCodec(model).encode_chunk_incremental(frames)

        streamed = ReAEStreamingCodec(model)
        pieces = [streamed.encode_chunk_incremental(frames[:, start : start + 4]) for start in range(0, 16, 4)]
        chunked = mx.concatenate([piece for piece in pieces if piece is not None], axis=1)
        mx.eval(whole, chunked)
        assert whole.shape == chunked.shape
        assert _relative_difference(whole, chunked) < 1e-5

    def test_chunked_decoding_reproduces_whole_chunk_decoding(self):
        model = ReAE()
        spec = build_chunk_specs(29, 24)[0]
        latents = ReAEStreamingCodec(model).encode_chunk(_clip(spec.frame_count, seed=6), spec)

        whole = ReAEStreamingCodec(model, decode_slice_latents=64).decode_chunk(latents, spec)
        sliced = ReAEStreamingCodec(model, decode_slice_latents=1).decode_chunk(latents, spec)
        mx.eval(whole, sliced)
        assert whole.shape == sliced.shape
        assert _relative_difference(whole, sliced) < 1e-5

    def test_reset_clears_every_carry(self):
        model = ReAE()
        codec = ReAEStreamingCodec(model)
        spec = build_chunk_specs(9, 8)[0]
        first = codec.encode_chunk(_clip(9, seed=8), spec)
        codec.reset()
        again = codec.encode_chunk(_clip(9, seed=8), spec)
        mx.eval(first, again)
        assert _relative_difference(first, again) == 0.0

    def test_the_incremental_encoder_buffers_a_partial_temporal_group(self):
        codec = ReAEStreamingCodec(ReAE())
        assert codec.encode_chunk_incremental(_clip(3, seed=9)) is None
        assert codec.flush_encoder() is not None

    def test_flushing_an_empty_encoder_returns_nothing(self):
        assert ReAEStreamingCodec(ReAE()).flush_encoder() is None

    def test_stack_state_is_allocated_per_layer_slot(self):
        model = ReAE()
        state = ReAEStackState.for_stack(model.encoder.layers)
        assert len(state.previous_frame) == len(model.encoder.layers)
        assert state.live_arrays() == []
        with pytest.raises(ValueError, match="allocated for"):
            run_reae_stack(model.decoder.layers, mx.zeros((1, 1, 4, 4, 48)), state, batch=1)


class TestCodecContract:
    @pytest.mark.parametrize("encode_slice_frames", [0, 1, 2, 3, 5, 6])
    def test_an_encode_slice_that_starves_a_tpool_is_rejected(self, encode_slice_frames):
        with pytest.raises(ValueError, match="must be a multiple of 4"):
            ReAEStreamingCodec(ReAE(), encode_slice_frames=encode_slice_frames)

    def test_a_non_positive_decode_slice_is_rejected(self):
        with pytest.raises(ValueError, match="decode_slice_latents must be positive"):
            ReAEStreamingCodec(ReAE(), decode_slice_latents=0)

    def test_a_frame_count_that_disagrees_with_the_spec_is_rejected(self):
        codec = ReAEStreamingCodec(ReAE())
        spec = build_chunk_specs(29, 24)[0]
        with pytest.raises(ValueError, match="expects 28 frames"):
            codec.encode_chunk(_clip(24), spec)

    def test_channels_first_input_is_rejected_rather_than_reinterpreted(self):
        codec = ReAEStreamingCodec(ReAE())
        spec = build_chunk_specs(9, 8)[0]
        with pytest.raises(ValueError, match="channels last"):
            codec.encode_chunk(mx.zeros((1, 9, 3, SPATIAL, SPATIAL)), spec)

    def test_a_canvas_that_is_not_a_multiple_of_the_spatial_scale_is_rejected(self):
        codec = ReAEStreamingCodec(ReAE())
        spec = build_chunk_specs(9, 8)[0]
        with pytest.raises(ValueError, match="divisible by 16"):
            codec.encode_chunk(_clip(9, height=40, width=64), spec)

    def test_a_rank_four_clip_is_rejected(self):
        with pytest.raises(ValueError, match=r"\[B, T, H, W, C\] pixel frames"):
            ReAEStreamingCodec(ReAE()).encode_chunk_incremental(mx.zeros((9, SPATIAL, SPATIAL, 3)))

    def test_latents_with_the_wrong_channel_count_are_rejected(self):
        with pytest.raises(ValueError, match="48 latent channels"):
            ReAEStreamingCodec(ReAE()).decode_chunk(mx.zeros((1, 2, 4, 4, 16)))

    def test_a_spec_disagreeing_about_the_head_trim_is_rejected(self):
        """Two sources for the trim is exactly how a clip loses or gains three frames."""
        codec = ReAEStreamingCodec(ReAE())
        specs = build_chunk_specs(29, 24)
        latents = codec.encode_chunk(_clip(specs[0].frame_count), specs[0])
        with pytest.raises(ValueError, match="disagreeing sources"):
            codec.iter_decode_chunk(latents, specs[1])
