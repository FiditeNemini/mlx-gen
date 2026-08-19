"""Geometry, bounds and the options the SwiftVR route refuses.

These exercise the real functions on real integers and real argv. Nothing here needs the
20 GB checkpoint, so the contract stays covered on machines that have not downloaded it.
"""

import subprocess
import sys

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.swiftvr.streaming.chunk import build_chunk_specs
from mflux.models.swiftvr.variants.upscale.swiftvr import SwiftVR
from mflux.models.swiftvr.variants.upscale.swiftvr_util import SwiftVRUtil
from mflux.utils.scale_factor import ScaleFactor


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mflux.cli.mlx_gen", "upscale", *args],
        capture_output=True,
        text=True,
    )


class TestCanvasGeometry:
    @pytest.mark.parametrize(
        ("height", "width", "expected"),
        [
            (1080, 1920, (1088, 1920)),
            (192, 320, (192, 320)),
            (720, 1280, (736, 1280)),
            (1, 1, (32, 32)),
        ],
    )
    def test_canvas_is_padded_up_to_a_multiple_of_32(self, height, width, expected):
        assert SwiftVRUtil.padded_canvas(height, width) == expected

    @pytest.mark.parametrize(
        ("width", "height"),
        # 1080 is not a multiple of 16, so ScaleFactor.get_scaled_value would report 1072
        # for a 1x request; the source size must survive that unchanged.
        [(1920, 1080), (1280, 720), (320, 192), (854, 481)],
    )
    def test_one_times_resolution_keeps_the_source_size(self, width, height):
        assert SwiftVRUtil.output_canvas(source_width=width, source_height=height, resolution=ScaleFactor(1)) == (
            height,
            width,
        )

    def test_an_explicit_resolution_matching_the_short_side_is_accepted(self):
        assert SwiftVRUtil.output_canvas(source_width=1920, source_height=1080, resolution=1080) == (1080, 1920)

    @pytest.mark.parametrize("resolution", [ScaleFactor(2), 768, 256])
    def test_any_other_resolution_fails_closed(self, resolution):
        with pytest.raises(ValueError, match="restores at the source resolution"):
            SwiftVRUtil.output_canvas(source_width=1280, source_height=720, resolution=resolution)

    def test_canvas_beyond_the_analysed_envelope_is_reported(self):
        assert SwiftVRUtil.canvas_bound_error(1088, 1920) is None
        message = SwiftVRUtil.canvas_bound_error(2176, 3840)
        assert message is not None
        assert "--force-unsafe-video-memory" in message

    def test_frame_limit_follows_the_rotary_table(self):
        # 1024 latent positions x 4 source frames per latent, minus the 4a+1 remainder.
        assert SwiftVRUtil.max_supported_source_frames(1024) == 4093


class TestChunkPlanFrameAccounting:
    @pytest.mark.parametrize(
        ("total_frames", "clip_len"),
        [(9, 8), (25, 8), (29, 24), (101, 24), (1, 4), (121, 24), (241, 20)],
    )
    def test_chunk_plan_consumes_every_source_frame_exactly_once(self, total_frames, clip_len):
        specs = build_chunk_specs(total_frames, clip_len)
        assert sum(spec.frame_count for spec in specs) == total_frames
        position = 0
        for spec in specs:
            assert spec.frame_start == position
            position += spec.frame_count

    @pytest.mark.parametrize(
        ("total_frames", "clip_len"),
        [(9, 8), (25, 8), (29, 24), (101, 24), (121, 24)],
    )
    def test_decoded_frame_count_equals_the_source_frame_count(self, total_frames, clip_len):
        """4 pixel frames per latent, minus the 3 causal head frames dropped once."""
        specs = build_chunk_specs(total_frames, clip_len)
        decoded = sum(spec.latent_count * 4 for spec in specs)
        assert decoded - 3 == total_frames

    def test_only_the_first_chunk_trims_the_decoder_head(self):
        specs = build_chunk_specs(101, 24)
        assert [spec.is_first_decode for spec in specs] == [True] + [False] * (len(specs) - 1)


class TestQuantizationIsRefused:
    """SwiftVR runs bf16 only, and both quantization levels are wrong in different ways.

    At 8 bits Wan's q8 sensitivity list spares every quantizable module this architecture
    has, so the run would be labelled quantized while staying bf16. At 4 bits the quantized
    condition embedder is read as a packed buffer by Wan's low-precision linear helper and
    the first chunk dies inside the timestep projection - after loading 20 GB.
    """

    @pytest.mark.parametrize("bits", [3, 4, 6, 8])
    def test_a_quantization_request_is_refused_before_any_weights_are_read(self, bits):
        with pytest.raises(ValueError, match="does not support --quantize") as exc:
            SwiftVR._assert_quantization_supported(bits)
        assert "ADR 0001" in str(exc.value)

    def test_no_quantization_is_accepted(self):
        assert SwiftVR._assert_quantization_supported(None) is None

    @pytest.mark.parametrize("flag", ["--quantize", "-q"])
    @pytest.mark.parametrize("bits", ["4", "8"])
    def test_the_cli_refuses_it_at_parse_time(self, flag, bits):
        result = _run_cli("--model", "swiftvr", "--video-path", "a.mp4", flag, bits)
        assert result.returncode != 0
        assert f"--quantize {bits} does not apply to SwiftVR" in result.stderr
        # The record's message carries a {value} token that the CLI must interpolate;
        # an unformatted token reaching the user is a wiring bug this line pins.
        assert "{value}" not in result.stderr

    @pytest.mark.parametrize("mode", ["wavelet", "lab"])
    def test_an_unsupported_color_correction_is_refused_at_parse_time(self, mode):
        """The route guard would raise the same incapability only after the 5B weight
        load; the CLI must say it before any weights are read (ADR 0002)."""
        result = _run_cli("--model", "swiftvr", "--video-path", "a.mp4", "--color-correction", mode)
        assert result.returncode != 0
        assert "SwiftVR does not apply color correction" in result.stderr
        assert "seedvr2-3b" in result.stderr

    def test_the_reason_given_for_refusing_q8_is_true(self):
        """The message claims Wan's q8 policy spares every quantizable module here. If it
        ever stops being true, q8 becomes a real option and the message becomes a lie."""
        from mflux.models.swiftvr.model.swiftvr_transformer.swiftvr_transformer import SwiftVRTransformer
        from mflux.models.swiftvr.weights.swiftvr_weight_definition import SwiftVRWeightDefinition

        # Same module tree as the published checkpoint (30 blocks), built narrow so the
        # test costs nothing: the predicate reads paths, not shapes.
        transformer = SwiftVRTransformer(
            window_hw=(2, 2),
            shift_alternate_layers=True,
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=8,
            out_channels=8,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=30,
            cross_attn_norm=True,
            eps=1e-6,
            added_kv_proj_dim=None,
            rope_max_seq_len=64,
        ).transformer
        quantizable = [
            (path, module) for path, module in transformer.named_modules() if hasattr(module, "to_quantized")
        ]
        assert len(quantizable) > 300, "the module tree lost its quantizable linears"

        at_eight = [p for p, m in quantizable if SwiftVRWeightDefinition.quantization_predicate(p, m, 8)]
        assert at_eight == [], "q8 now quantizes something, so refusing it needs a new reason"

        at_four = [p for p, m in quantizable if SwiftVRWeightDefinition.quantization_predicate(p, m, 4)]
        assert any(path.startswith("condition_embedder.") for path in at_four), (
            "q4 no longer quantizes the condition embedder, so the stated failure mode has moved"
        )

    def test_the_refusal_agrees_with_the_save_and_prepare_gates(self):
        """Three surfaces refuse a quantized SwiftVR; an ungated one would contradict them."""
        from mflux.models.swiftvr.weights.swiftvr_weight_definition import SwiftVRWeightDefinition

        with pytest.raises(NotImplementedError, match="ADR 0001"):
            SwiftVRWeightDefinition.for_saving(ModelConfig.swiftvr())
        with pytest.raises(NotImplementedError, match="not supported yet"):
            SwiftVR.save_model(object(), "/tmp/does-not-matter")


class TestUnsupportedOptionsFailClosed:
    def test_dit_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="dit_overlap"):
            SwiftVR._assert_supported_options(dit_overlap=1, color_correction_mode="off")

    @pytest.mark.parametrize("mode", ["wavelet", "lab"])
    def test_color_correction_is_rejected(self, mode):
        with pytest.raises(ValueError, match="does not apply color correction"):
            SwiftVR._assert_supported_options(dit_overlap=0, color_correction_mode=mode)

    def test_supported_combination_is_accepted(self):
        assert SwiftVR._assert_supported_options(dit_overlap=0, color_correction_mode="off") is None


class TestCliRejections:
    """The CLI must refuse options it cannot honour rather than ignoring them."""

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            (["--image-path", "a.png"], "SwiftVR restores video only"),
            (["--video-path", "a.mp4", "--steps", "4"], "--steps does not apply to SwiftVR"),
            (["--video-path", "a.mp4", "--temporal-chunk-size", "49"], "does not apply to SwiftVR"),
            (["--video-path", "a.mp4", "--temporal-chunk-overlap", "8"], "does not apply to SwiftVR"),
            (["--video-path", "a.mp4", "--softness", "0.5"], "--softness does not apply to SwiftVR"),
            (["--video-path", "a.mp4", "--vae-tiling"], "--vae-tiling does not apply to SwiftVR"),
            (["--video-path", "a.mp4", "--seed", "1", "2"], "deterministic"),
        ],
    )
    def test_option_is_rejected_with_an_actionable_message(self, args, expected):
        result = _run_cli("--model", "swiftvr", *args)
        assert result.returncode != 0
        assert expected in result.stderr

    def test_unknown_swiftvr_repo_id_fails_closed(self):
        result = _run_cli("--model", "H-oliday/NotSwiftVR", "--video-path", "a.mp4")
        assert result.returncode != 0
        # An unrecognised repo id belongs to SeedVR2's resolver, which names both families.
        assert "Unsupported SeedVR2 model handle" in result.stderr
        assert "swiftvr" in result.stderr.lower()

    def test_prepare_is_refused_with_a_pointer_to_download(self):
        result = subprocess.run(
            [sys.executable, "-m", "mflux.cli.mlx_gen", "prepare", "--model", "swiftvr", "--path", "/tmp/swiftvr-out"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "mlxgen download --model H-oliday/SwiftVR" in result.stderr


class TestMeasuredMemoryReporting:
    """The peak counter is process-wide and shared with --low-ram's summary line."""

    def test_the_peak_is_read_without_resetting_the_shared_counter(self):
        """SwiftVR forces --low-ram, so MemorySaver prints mx.get_peak_memory() at the end
        of every run. Resetting it per chunk left that line reporting the last (shortest)
        chunk while claiming to report the run."""
        assert not hasattr(SwiftVR, "_reset_peak_memory")
        peak = SwiftVR._peak_memory()
        assert isinstance(peak, int)
        assert peak >= 0

    def test_a_broken_peak_reading_is_not_swallowed(self):
        """The route's only memory guard must not turn itself off in silence (ADR 0002)."""
        import inspect

        source = inspect.getsource(SwiftVR._peak_memory)
        assert "except" not in source


class TestCatalogEntry:
    def test_swiftvr_declares_no_base_model(self):
        """A base_model naming Wan would reroute prepare and download to the Wan family."""
        assert ModelConfig.swiftvr().base_model is None

    def test_swiftvr_does_not_advertise_guidance_or_a_text_encoder(self):
        model_config = ModelConfig.swiftvr()
        assert model_config.supports_guidance is False
        assert model_config.text_encoder_overrides == {}

    def test_swiftvr_shares_the_wan_transformer_shape(self):
        from mflux.models.common.config.model_config import WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE

        overrides = ModelConfig.swiftvr().transformer_overrides
        for key, value in WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE.items():
            assert overrides[key] == value

    def test_every_override_key_is_classified(self):
        """An unclassified key would be silently dropped instead of reaching the model."""
        from mflux.models.swiftvr.swiftvr_initializer import (
            RUNTIME_OVERRIDE_KEYS,
            TRANSFORMER_CONSTRUCTOR_KEYS,
            SwiftVRInitializer,
        )

        overrides = set(ModelConfig.swiftvr().transformer_overrides)
        assert overrides - TRANSFORMER_CONSTRUCTOR_KEYS - RUNTIME_OVERRIDE_KEYS == set()
        kwargs = SwiftVRInitializer.transformer_kwargs(ModelConfig.swiftvr())
        assert kwargs["num_layers"] == 30
        assert kwargs["patch_size"] == (1, 2, 2)
