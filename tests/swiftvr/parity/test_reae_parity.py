"""ReAE encoder and decoder parity: mflux MLX against the upstream torch reference.

Both sides load the same ``reae.safetensors`` and see the same frames. The MLX side goes
through :class:`ReAEStreamingCodec` and the torch side through :class:`StreamingTAE`, so
what is compared is the whole shipped path - pixel unshuffle, the layer stack, the causal
MemBlock/TPool boundary state, the decoder head trim and the pixel shuffle - not a
hand-assembled subset of it.

Tolerance
---------
ReAE weights are float32 on disk and both implementations keep them there, so the only
expected divergence is accumulation order: MLX and torch tile, vectorise and reduce
convolutions differently. float32 unit roundoff is 1.19e-07 and the stacks are 18 and 23
layers deep, so a few parts in 1e-06 is the floor.

``max_relative = 1e-4`` sits ~30x above the measured 3e-06 - loose enough not to be
flaky across MLX releases, tight enough that every plausible porting bug is caught. Such
bugs are not subtle at this scale: a transposed convolution kernel, an off-by-one layer
index, a dropped ``TPool(1)`` or a mis-threaded causal carry all move the output by
O(1e-1) relative or change the shape outright. There is no failure mode that lands
between 1e-4 and 1e-6.

``min_cosine = 0.999999`` is the directional companion. A cosine of 0.98 on a block
output means a real bug, as the task brief notes; the bound is set four orders of
magnitude tighter than that.
"""

import numpy as np
import pytest

from mflux.models.swiftvr.streaming.streaming_reae import ReAEStreamingCodec
from tests.swiftvr.parity.parity_support import (
    assert_parity,
    compare,
    nchw_to_nhwc,
    nhwc_to_nchw,
    paired_chunk_specs,
    seeded_clip,
    torch_reference,
)

MAX_RELATIVE = 1e-4
MIN_COSINE = 0.999999

# 25 frames at clip_len 8 is the shortest schedule exercising all three chunk types:
# FIRST(12) -> MIDDLE(8) -> LAST(5), i.e. every causal boundary the protocol can produce.
FULL_PROTOCOL_FRAMES = 25
FULL_PROTOCOL_CLIP_LEN = 8


def _bfloat16_rounding_cost(reference) -> float:
    """Largest absolute change from merely storing ``reference`` in bfloat16."""
    import mlx.core as mx

    values = reference.numpy() if hasattr(reference, "numpy") else np.asarray(reference)
    rounded = np.asarray(mx.array(values).astype(mx.bfloat16).astype(mx.float32))
    return float(np.abs(rounded - values).max())


def _torch_streaming_tae(model):
    torch_reference()
    from swiftvr.streaming.tae import StreamingTAE

    return StreamingTAE(model)


@pytest.mark.parity
class TestReAEEncoderParity:
    """(a) Encoder: real frames -> latents."""

    @pytest.mark.parametrize(
        ("frames", "height", "width", "seed"),
        [
            (9, 64, 64, 7),
            (5, 32, 48, 21),
        ],
    )
    def test_single_chunk_encoder_matches_reference(self, torch_reae, mlx_reae, frames, height, width, seed):
        """A single LAST chunk: the frame-count-preserving path used for short clips."""
        import mlx.core as mx
        import torch

        clip = seeded_clip(frames=frames, height=height, width=width, seed=seed)
        specs = paired_chunk_specs(frames, FULL_PROTOCOL_CLIP_LEN)
        assert len(specs) == 1, f"{frames} frames at clip_len {FULL_PROTOCOL_CLIP_LEN} is not a single chunk"
        reference_spec, mflux_spec = specs[0]

        with torch.no_grad():
            reference = _torch_streaming_tae(torch_reae).encode_chunk_fixed(torch.from_numpy(clip), reference_spec)
        candidate = ReAEStreamingCodec(mlx_reae).encode_chunk(mx.array(nchw_to_nhwc(clip)), mflux_spec)

        result = assert_parity(
            nhwc_to_nchw(candidate),
            reference,
            label=f"ReAE encoder {frames}f {height}x{width}",
            max_relative=MAX_RELATIVE,
            min_cosine=MIN_COSINE,
        )
        print(f"\nReAE encoder {frames}f {height}x{width}: {result}")

    def test_streaming_encoder_matches_reference_across_all_chunk_types(self, torch_reae, mlx_reae):
        """FIRST -> MIDDLE -> LAST on one codec instance, so the causal carry is under test.

        Chunk 2 and 3 can only match if the MemBlock previous-frame carry and the TPool
        remainder buffer both survived the boundary identically. The MLX codec additionally
        slices each chunk into 4-frame encoder calls internally while the reference runs
        the chunk in one pass, so agreement also shows the carry is correct at a finer
        granularity than the reference ever exercises.
        """
        import mlx.core as mx
        import torch

        clip = seeded_clip(frames=FULL_PROTOCOL_FRAMES, height=64, width=64, seed=11)
        specs = paired_chunk_specs(FULL_PROTOCOL_FRAMES, FULL_PROTOCOL_CLIP_LEN)
        assert [mflux_spec.ctype.value for _, mflux_spec in specs] == ["first", "middle", "last"]

        reference_tae = _torch_streaming_tae(torch_reae)
        codec = ReAEStreamingCodec(mlx_reae)
        clip_nhwc = mx.array(nchw_to_nhwc(clip))

        for reference_spec, mflux_spec in specs:
            window = slice(mflux_spec.frame_start, mflux_spec.frame_start + mflux_spec.frame_count)
            with torch.no_grad():
                reference = reference_tae.encode_chunk_fixed(torch.from_numpy(clip[:, window]), reference_spec)
            candidate = codec.encode_chunk(clip_nhwc[:, window], mflux_spec)
            result = assert_parity(
                nhwc_to_nchw(candidate),
                reference,
                label=f"ReAE encoder chunk {mflux_spec.clip_idx} ({mflux_spec.ctype.value})",
                max_relative=MAX_RELATIVE,
                min_cosine=MIN_COSINE,
            )
            print(f"\nReAE encoder {mflux_spec.ctype.value}: {result}")


@pytest.mark.parity
class TestReAEDecoderParity:
    """(b) Decoder: latents -> frames."""

    def test_decoder_matches_reference_on_encoder_latents(self, torch_reae, mlx_reae):
        """Decode the reference's own latents on both sides.

        Feeding both decoders the *torch* latents isolates the decoder: any divergence is
        the decoder's, not inherited from a 1e-06 encoder disagreement.
        """
        import mlx.core as mx
        import torch

        clip = seeded_clip(frames=9, height=64, width=64, seed=13)
        reference_spec, mflux_spec = paired_chunk_specs(9, FULL_PROTOCOL_CLIP_LEN)[0]

        reference_tae = _torch_streaming_tae(torch_reae)
        with torch.no_grad():
            latents = reference_tae.encode_chunk_fixed(torch.from_numpy(clip), reference_spec)
            reference = reference_tae.decode_chunk_fixed(latents, reference_spec)

        shared = mx.array(nchw_to_nhwc(latents.numpy()))
        candidate = ReAEStreamingCodec(mlx_reae).decode_chunk(shared, mflux_spec)

        assert reference.shape[1] == clip.shape[1], "reference decode must preserve the frame count"
        result = assert_parity(
            nhwc_to_nchw(candidate),
            reference,
            label="ReAE decoder (shared torch latents)",
            max_relative=MAX_RELATIVE,
            min_cosine=MIN_COSINE,
        )
        print(f"\nReAE decoder shared-latents: {result}")

    def test_single_frame_clip_matches_reference(self, torch_reae, mlx_reae):
        """A one-frame clip round-trips identically on both sides.

        ``t = 1`` is a legal ``4a + 1`` length and the CLI can be handed a one-frame video,
        so the codec must be correct there, but every other case in this file is 9, 16 or
        25 frames. The single frame is the degenerate end of the protocol: the encoder
        replicates it to fill a 4-frame slice, one latent comes out, and the decoder's
        3-frame head trim has to leave exactly one frame rather than consuming the only
        real one. That makes it the shape most likely to hide an off-by-one, and the shape
        whose output quality is used to justify withholding a SwiftVR image route - a
        justification that only holds if the port is faithful here.
        """
        import mlx.core as mx
        import torch

        clip = seeded_clip(frames=1, height=64, width=64, seed=17)
        reference_spec, mflux_spec = paired_chunk_specs(1, FULL_PROTOCOL_CLIP_LEN)[0]

        reference_tae = _torch_streaming_tae(torch_reae)
        with torch.no_grad():
            reference_latents = reference_tae.encode_chunk_fixed(torch.from_numpy(clip), reference_spec)
            reference_frames = reference_tae.decode_chunk_fixed(reference_latents, reference_spec)

        codec = ReAEStreamingCodec(mlx_reae)
        candidate_latents = codec.encode_chunk(mx.array(nchw_to_nhwc(clip)), mflux_spec)
        candidate_frames = codec.decode_chunk(mx.array(nchw_to_nhwc(reference_latents.numpy())), mflux_spec)

        assert reference_frames.shape[1] == 1, "a one-frame clip must decode back to one frame"

        encode_result = assert_parity(
            nhwc_to_nchw(candidate_latents),
            reference_latents,
            label="ReAE encoder (single-frame clip)",
            max_relative=MAX_RELATIVE,
            min_cosine=MIN_COSINE,
        )
        decode_result = assert_parity(
            nhwc_to_nchw(candidate_frames),
            reference_frames,
            label="ReAE decoder (single-frame clip)",
            max_relative=MAX_RELATIVE,
            min_cosine=MIN_COSINE,
        )
        print(f"\nReAE single-frame encode: {encode_result}")
        print(f"ReAE single-frame decode: {decode_result}")

    def test_streaming_decoder_matches_reference_across_all_chunk_types(self, torch_reae, mlx_reae):
        """FIRST -> MIDDLE -> LAST decode, including the one-shot head trim on FIRST.

        The decoder carries causal state too, and ``frames_to_trim`` head frames are
        dropped exactly once per clip. A codec that re-trimmed on every chunk, or never
        trimmed, would change the frame count and fail on shape before the numbers matter.
        """
        import mlx.core as mx
        import torch

        clip = seeded_clip(frames=FULL_PROTOCOL_FRAMES, height=64, width=64, seed=17)
        specs = paired_chunk_specs(FULL_PROTOCOL_FRAMES, FULL_PROTOCOL_CLIP_LEN)

        reference_tae = _torch_streaming_tae(torch_reae)
        codec = ReAEStreamingCodec(mlx_reae)
        clip_nhwc = mx.array(nchw_to_nhwc(clip))
        emitted = 0

        for reference_spec, mflux_spec in specs:
            window = slice(mflux_spec.frame_start, mflux_spec.frame_start + mflux_spec.frame_count)
            with torch.no_grad():
                latents = reference_tae.encode_chunk_fixed(torch.from_numpy(clip[:, window]), reference_spec)
                reference = reference_tae.decode_chunk_fixed(latents, reference_spec)
            candidate = codec.decode_chunk(codec.encode_chunk(clip_nhwc[:, window], mflux_spec), mflux_spec)
            result = assert_parity(
                nhwc_to_nchw(candidate),
                reference,
                label=f"ReAE decoder chunk {mflux_spec.clip_idx} ({mflux_spec.ctype.value})",
                max_relative=MAX_RELATIVE,
                min_cosine=MIN_COSINE,
            )
            print(f"\nReAE decoder {mflux_spec.ctype.value}: {result}")
            emitted += int(candidate.shape[1])

        assert emitted == FULL_PROTOCOL_FRAMES, (
            f"the chunk protocol must preserve the frame count: {emitted} out for {FULL_PROTOCOL_FRAMES} in"
        )


@pytest.mark.parity
class TestReAEProductionPrecisionParity:
    """(a, b at shipped precision) ReAE in bfloat16, which is what ``ModelConfig.precision`` loads.

    The float32 cases above isolate the graph; this one measures the configuration a real
    ``mlxgen upscale`` run uses. It is judged against the bfloat16 rounding floor rather
    than an absolute bound, because at this precision the representation, not the port,
    sets the error.
    """

    def test_encode_decode_stays_within_the_bfloat16_floor(self, torch_reae, mlx_reae_bfloat16):
        import mlx.core as mx
        import torch

        clip = seeded_clip(frames=9, height=64, width=64, seed=7)
        reference_spec, mflux_spec = paired_chunk_specs(9, FULL_PROTOCOL_CLIP_LEN)[0]

        reference_tae = _torch_streaming_tae(torch_reae)
        with torch.no_grad():
            reference_latents = reference_tae.encode_chunk_fixed(torch.from_numpy(clip), reference_spec)
            reference_frames = reference_tae.decode_chunk_fixed(reference_latents, reference_spec)

        codec = ReAEStreamingCodec(mlx_reae_bfloat16)
        candidate_latents = codec.encode_chunk(mx.array(nchw_to_nhwc(clip)).astype(mx.bfloat16), mflux_spec)
        candidate_frames = codec.decode_chunk(candidate_latents, mflux_spec)

        encode_result = compare(nhwc_to_nchw(candidate_latents), reference_latents)
        decode_result = compare(nhwc_to_nchw(candidate_frames), reference_frames)
        encode_floor = _bfloat16_rounding_cost(reference_latents)
        decode_floor = _bfloat16_rounding_cost(reference_frames)

        print(f"\nReAE bfloat16 encoder: {encode_result}  (bf16 floor {encode_floor:.3e})")
        print(f"ReAE bfloat16 decoder: {decode_result}  (bf16 floor {decode_floor:.3e})")

        assert encode_result.cosine > 0.999, f"bfloat16 latents diverged in direction: {encode_result}"
        assert decode_result.cosine > 0.999, f"bfloat16 frames diverged in direction: {decode_result}"
        assert encode_result.max_abs_diff < 20.0 * encode_floor, (
            f"bfloat16 encoder error {encode_result.max_abs_diff:.3e} exceeds 20x the bf16 rounding floor "
            f"{encode_floor:.3e}; that is more than the precision explains"
        )
        assert decode_result.max_abs_diff < 20.0 * decode_floor, (
            f"bfloat16 decoder error {decode_result.max_abs_diff:.3e} exceeds 20x the bf16 rounding floor "
            f"{decode_floor:.3e}; that is more than the precision explains"
        )
