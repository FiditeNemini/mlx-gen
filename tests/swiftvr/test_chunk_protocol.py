"""The fixed FIRST/MIDDLE/LAST chunk protocol.

The chunk plan is the contract between the reader, the causal codec and the DiT: it
decides how many source frames each call consumes and how many output frames come back.
Frame accounting is the load-bearing property, and it is silent when wrong - a plan that
loses or duplicates a frame still runs, still writes a video, and only shows up as a
different duration or a stutter at a boundary. Every test below therefore checks the
arithmetic against the source count rather than against a transcribed expectation.

The relationship being pinned, with ``n_lat = clip_len // 4``:

* FIRST  - ``clip_len + 4`` frames  -> ``n_lat + 1`` latents
* MIDDLE - ``clip_len`` frames      -> ``n_lat`` latents
* LAST   - ``4b + 1`` frames        -> ``b + 1`` latents (the codec repeats the final
  frame three times before encoding)

and, once per clip, the decoder drops ``frames_to_trim = 3`` causal head frames, which is
exactly what makes ``4 * total_latents - 3 == total_frames``.
"""

import pytest

from mflux.models.swiftvr.model.swiftvr_reae.reae import ReAE
from mflux.models.swiftvr.streaming.chunk import (
    LATENT_TEMPORAL_DOWNSCALE,
    ChunkSpec,
    ChunkType,
    aligned_frame_count,
    build_chunk_specs,
)

# Every clip length of the form 4a + 1 up to 201 frames, against clip lengths that are
# multiples of 4 from the smallest legal one to well past a single-chunk clip.
ALIGNED_FRAME_COUNTS = list(range(1, 202, LATENT_TEMPORAL_DOWNSCALE))
CLIP_LENGTHS = [4, 8, 12, 16, 20, 24, 32, 48]


class TestFrameAccounting:
    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_the_plan_consumes_every_source_frame_exactly_once(self, clip_len):
        for total_frames in ALIGNED_FRAME_COUNTS:
            specs = build_chunk_specs(total_frames, clip_len)
            assert sum(spec.frame_count for spec in specs) == total_frames, (total_frames, clip_len)
            position = 0
            for spec in specs:
                assert spec.frame_start == position, (total_frames, clip_len)
                position += spec.frame_count
            assert position == total_frames, (total_frames, clip_len)

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_decoded_frames_equal_source_frames_after_the_one_head_trim(self, clip_len):
        """The whole protocol exists to make this identity hold for every (t, clip_len)."""
        frames_to_trim = ReAE().frames_to_trim
        for total_frames in ALIGNED_FRAME_COUNTS:
            specs = build_chunk_specs(total_frames, clip_len)
            decoded = sum(spec.latent_count for spec in specs) * LATENT_TEMPORAL_DOWNSCALE
            assert decoded - frames_to_trim == total_frames, (total_frames, clip_len)

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_exactly_one_chunk_trims_the_decoder_head(self, clip_len):
        for total_frames in ALIGNED_FRAME_COUNTS:
            specs = build_chunk_specs(total_frames, clip_len)
            assert [spec.is_first_decode for spec in specs].count(True) == 1, (total_frames, clip_len)
            assert specs[0].is_first_decode is True, (total_frames, clip_len)

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_chunk_indices_are_dense_and_in_playback_order(self, clip_len):
        for total_frames in ALIGNED_FRAME_COUNTS:
            specs = build_chunk_specs(total_frames, clip_len)
            assert [spec.clip_idx for spec in specs] == list(range(len(specs))), (total_frames, clip_len)


class TestChunkShape:
    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_chunk_types_follow_first_then_middles_then_last(self, clip_len):
        for total_frames in ALIGNED_FRAME_COUNTS:
            types = [spec.ctype for spec in build_chunk_specs(total_frames, clip_len)]
            assert types[-1] is ChunkType.LAST, (total_frames, clip_len)
            assert types.count(ChunkType.LAST) == 1, (total_frames, clip_len)
            if len(types) == 1:
                continue
            assert types[0] is ChunkType.FIRST, (total_frames, clip_len)
            assert all(ctype is ChunkType.MIDDLE for ctype in types[1:-1]), (total_frames, clip_len)

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_each_chunk_length_matches_its_type(self, clip_len):
        for spec in [s for t in ALIGNED_FRAME_COUNTS for s in build_chunk_specs(t, clip_len)]:
            if spec.ctype is ChunkType.FIRST:
                assert spec.frame_count == clip_len + LATENT_TEMPORAL_DOWNSCALE
            elif spec.ctype is ChunkType.MIDDLE:
                assert spec.frame_count == clip_len
            else:
                assert spec.frame_count % LATENT_TEMPORAL_DOWNSCALE == 1
                assert spec.frame_count == LATENT_TEMPORAL_DOWNSCALE * spec.b + 1
                # A trailing LAST is capped by clip_len; a clip small enough to be its own
                # single chunk can reach clip_len + 1, the largest 4a + 1 under clip_len + 4.
                cap = clip_len + 1 if spec.clip_idx == 0 else clip_len
                assert 1 <= spec.frame_count <= cap

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_a_last_chunk_never_needs_more_padding_than_a_middle_chunk_can_supply(self, clip_len):
        """StreamingDiT front-pads LAST up to n_lat + 1 latents; a longer LAST would raise."""
        chunk_latent_frames = clip_len // LATENT_TEMPORAL_DOWNSCALE
        for total_frames in ALIGNED_FRAME_COUNTS:
            last = build_chunk_specs(total_frames, clip_len)[-1]
            assert last.latent_count <= chunk_latent_frames + 1, (total_frames, clip_len)

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_a_clip_that_fits_in_first_is_emitted_as_a_single_last_chunk(self, clip_len):
        """t <= clip_len + 4 has no FIRST: one chunk carries the whole clip and trims the head."""
        for total_frames in range(1, clip_len + 5, LATENT_TEMPORAL_DOWNSCALE):
            specs = build_chunk_specs(total_frames, clip_len)
            assert len(specs) == 1
            assert specs[0].ctype is ChunkType.LAST
            assert specs[0].frame_count == total_frames
            assert specs[0].is_first_decode is True

    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_the_split_happens_at_the_first_length_that_outgrows_clip_len_plus_four(self, clip_len):
        """clip_len is a multiple of 4, so the largest single-chunk clip is clip_len + 1."""
        assert aligned_frame_count(clip_len + 4) == clip_len + 1
        single = build_chunk_specs(clip_len + 1, clip_len)
        assert [spec.ctype for spec in single] == [ChunkType.LAST]
        assert single[0].frame_count == clip_len + 1

        split = build_chunk_specs(clip_len + 5, clip_len)
        assert [spec.ctype for spec in split] == [ChunkType.FIRST, ChunkType.LAST]
        assert split[0].frame_count == clip_len + 4
        assert split[1].frame_count == 1

    def test_middle_chunks_appear_once_the_clip_outgrows_first_plus_last(self):
        specs = build_chunk_specs(101, 24)
        assert [spec.ctype.value for spec in specs] == ["first", "middle", "middle", "middle", "last"]
        assert [spec.frame_count for spec in specs] == [28, 24, 24, 24, 1]
        assert [spec.latent_count for spec in specs] == [7, 6, 6, 6, 1]
        # 26 latents x 4 decoded frames, minus the 3 causal head frames, is 101.
        assert sum(spec.latent_count for spec in specs) * 4 - 3 == 101


class TestLatentCount:
    @pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
    def test_only_last_chunks_gain_a_latent_from_the_repeated_tail_frame(self, clip_len):
        for spec in [s for t in ALIGNED_FRAME_COUNTS for s in build_chunk_specs(t, clip_len)]:
            if spec.ctype is ChunkType.LAST:
                assert spec.latent_count == spec.b + 1
                assert spec.latent_count * LATENT_TEMPORAL_DOWNSCALE == spec.frame_count + 3
            else:
                assert spec.latent_count == spec.frame_count // LATENT_TEMPORAL_DOWNSCALE

    def test_b_is_meaningless_outside_last_chunks(self):
        for spec in build_chunk_specs(101, 24):
            if spec.ctype is not ChunkType.LAST:
                assert spec.b == 0


class TestFailClosed:
    @pytest.mark.parametrize("clip_len", [1, 2, 3, 5, 6, 7, 9, 22, 26])
    def test_a_clip_len_that_is_not_a_multiple_of_four_raises(self, clip_len):
        """ReAE folds 4 pixel frames into 1 latent; a MIDDLE chunk must form whole groups."""
        with pytest.raises(ValueError, match="positive multiple of 4"):
            build_chunk_specs(101, clip_len)

    @pytest.mark.parametrize("clip_len", [0, -4, -24])
    def test_a_non_positive_clip_len_raises(self, clip_len):
        with pytest.raises(ValueError, match="positive multiple of 4"):
            build_chunk_specs(101, clip_len)

    @pytest.mark.parametrize("total_frames", [2, 3, 4, 24, 100, 102])
    def test_a_clip_length_that_is_not_4a_plus_1_raises_and_names_the_fix(self, total_frames):
        with pytest.raises(ValueError, match=r"4a \+ 1") as exc:
            build_chunk_specs(total_frames, 24)
        assert str(aligned_frame_count(total_frames)) in str(exc.value)

    @pytest.mark.parametrize("total_frames", [0, -3, -100])
    def test_a_non_positive_clip_length_raises_instead_of_planning_negative_frames(self, total_frames):
        """Python's modulo makes -3 % 4 == 1, so the 4a + 1 test alone lets a negative through."""
        with pytest.raises(ValueError, match="at least one source frame"):
            build_chunk_specs(total_frames, 24)

    def test_the_clip_len_check_survives_python_dash_o(self):
        """Upstream asserts; an assert is removed by -O and the plan would silently skew."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-O",
                "-c",
                "from mflux.models.swiftvr.streaming.chunk import build_chunk_specs\n"
                "try:\n"
                "    build_chunk_specs(101, 25)\n"
                "except ValueError:\n"
                "    print('raised')\n",
            ],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "raised", result.stderr


class TestAlignedFrameCount:
    def test_alignment_returns_the_largest_4a_plus_1_that_fits(self):
        for raw_total in range(1, 200):
            aligned = aligned_frame_count(raw_total)
            assert aligned % LATENT_TEMPORAL_DOWNSCALE == 1, raw_total
            assert aligned <= raw_total, raw_total
            assert raw_total - aligned < LATENT_TEMPORAL_DOWNSCALE, raw_total
            # The result is always a length build_chunk_specs accepts.
            assert sum(spec.frame_count for spec in build_chunk_specs(aligned, 24)) == aligned, raw_total

    @pytest.mark.parametrize("raw_total", [0, -1, -100])
    def test_a_clip_with_no_frames_raises(self, raw_total):
        with pytest.raises(ValueError, match="at least one source frame"):
            aligned_frame_count(raw_total)


class TestSpecIsImmutable:
    def test_a_spec_cannot_be_edited_after_the_plan_is_built(self):
        """The plan is read by the reader, the codec and the DiT; a mutable spec could drift."""
        spec = build_chunk_specs(101, 24)[0]
        assert isinstance(spec, ChunkSpec)
        with pytest.raises(AttributeError):
            spec.frame_count = 999
