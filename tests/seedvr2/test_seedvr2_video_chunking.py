import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.seedvr2.variants.upscale import seedvr2 as seedvr2_module
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import DecodedVideoClip, VideoUtil


def _solid_frame(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (16, 16), color)


def _iter_fake_chunk_clips(frames: list[Image.Image], windows: list[tuple[int, int]], audio_present: bool):
    for start_frame, end_frame in windows:
        yield DecodedVideoClip(
            frames=frames[start_frame:end_frame],
            fps=12.0,
            source_width=16,
            source_height=16,
            source_frame_count=8,
            source_duration_seconds=8 / 12.0,
            audio_present=audio_present,
            clip_start_frame=start_frame,
            clip_frame_count=end_frame - start_frame,
        )


@pytest.mark.fast
def test_seedvr2_video_chunk_plan_uses_requested_overlap():
    chunks = SeedVR2Util.plan_video_chunks(frame_count=17, chunk_size=9, overlap=4)

    assert chunks == [(0, 9), (5, 14), (10, 17)]
    assert chunks[0][1] - chunks[1][0] == 4
    assert chunks[1][1] - chunks[2][0] == 4


@pytest.mark.fast
def test_seedvr2_blend_overlapping_frames_returns_crossfade():
    blended = SeedVR2Util.blend_overlapping_frames(
        existing_tail=[_solid_frame((0, 0, 0)), _solid_frame((0, 0, 0))],
        incoming_head=[_solid_frame((255, 255, 255)), _solid_frame((255, 255, 255))],
    )

    first = blended[0].getpixel((0, 0))[0]
    second = blended[1].getpixel((0, 0))[0]
    assert 0 < first < second < 255


@pytest.mark.fast
def test_seedvr2_wavelet_color_reconstruction_supports_video_tensors():
    content = mx.zeros((1, 3, 2, 8, 8), dtype=mx.float32)
    style = mx.ones((1, 3, 2, 8, 8), dtype=mx.float32)

    corrected = SeedVR2Util.apply_color_correction(content, style, mode="wavelet")

    assert corrected.shape == content.shape


@pytest.mark.fast
def test_seedvr2_resize_and_soften_keeps_native_1x_frames_unchanged():
    frame = Image.new("RGB", (320, 240), (10, 20, 30))

    resized, true_h, true_w = SeedVR2Util._resize_and_soften(
        image=frame,
        resolution=ScaleFactor(1),
        softness=0.0,
    )

    assert resized.size == frame.size
    assert (true_h, true_w) == (240, 320)
    assert np.array_equal(np.asarray(resized), np.asarray(frame))


@pytest.mark.fast
def test_vae_util_preserves_temporal_axis_when_requested():
    class FakeVAE:
        @staticmethod
        def encode(image):
            return mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)

        @staticmethod
        def decode(latent):
            return mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32)

    encoded = VAEUtil.encode(FakeVAE(), mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32), preserve_temporal_axis=True)
    decoded = VAEUtil.decode(FakeVAE(), mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32), preserve_temporal_axis=True)

    assert encoded.shape == (1, 16, 1, 4, 4)
    assert decoded.shape == (1, 3, 1, 32, 32)


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_records_chunk_metadata(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [
        _solid_frame((index * 20, index * 20, index * 20))
        for index in range(8)
    ]
    chunk_calls: list[int] = []

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=8,
                source_duration_seconds=8 / 12.0,
                audio_present=True,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], True)),
    )

    def fake_restore_video_frames(self, *, seed, frames, resolution, softness, color_correction_mode):
        chunk_calls.append(len(frames))
        return list(frames), 16, 16, len(frames)

    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", fake_restore_video_frames)
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    file_path = model.restore_video_to_path(
        seed=42,
        video_path=source,
        resolution=256,
        softness=0.0,
        output_path=output,
        export_json_metadata=True,
        temporal_chunk_size=5,
        temporal_chunk_overlap=2,
        color_correction_mode="wavelet",
    )

    assert file_path.exists()
    metadata = json.loads(file_path.with_suffix(".metadata.json").read_text())
    assert metadata["frames"] == 8
    assert metadata["audio_present"] is True
    assert metadata["audio_copied"] is False
    assert metadata["temporal_chunk_size"] == 5
    assert metadata["temporal_chunk_overlap"] == 2
    assert metadata["temporal_chunk_count"] == 2
    assert metadata["temporal_chunk_plan"] == [
        {"start_frame": 0, "end_frame": 5, "frame_count": 5},
        {"start_frame": 3, "end_frame": 8, "frame_count": 5},
    ]
    assert metadata["color_correction_mode"] == "wavelet"
    assert chunk_calls == [5, 5]


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_saved_mp4_keeps_expected_frame_count(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame((index * 25, index * 25, index * 25)) for index in range(8)]
    original_read_video_clip = VideoUtil.read_video_clip

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=8,
                source_duration_seconds=8 / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], False)),
    )
    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", lambda self, **kwargs: (list(kwargs["frames"]), 16, 16, len(kwargs["frames"])))
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    file_path = model.restore_video_to_path(
        seed=42,
        video_path=source,
        resolution=256,
        softness=0.0,
        output_path=output,
        export_json_metadata=False,
        temporal_chunk_size=5,
        temporal_chunk_overlap=2,
    )

    clip = original_read_video_clip(file_path)
    means = [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in clip.frames]

    assert clip.clip_frame_count == 8
    assert len(means) == 8
    assert all(a <= b for a, b in zip(means, means[1:]))


@pytest.mark.fast
def test_seedvr2_restore_video_to_path_cleans_temp_file_on_failure(monkeypatch, tmp_path):
    source = tmp_path / "source.mp4"
    source.touch()
    output = tmp_path / "restored.mp4"

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = object()

    frames = [_solid_frame((index * 25, index * 25, index * 25)) for index in range(8)]
    call_count = {"count": 0}

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "read_video_clip",
        staticmethod(
            lambda path, start_seconds=0.0, max_frames=None: DecodedVideoClip(
                frames=frames[:1],
                fps=12.0,
                source_width=16,
                source_height=16,
                source_frame_count=8,
                source_duration_seconds=8 / 12.0,
                audio_present=False,
                clip_start_frame=0,
                clip_frame_count=1,
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VideoUtil,
        "iter_video_frame_windows",
        staticmethod(lambda path, start_frame=0, windows=None: _iter_fake_chunk_clips(frames, windows or [], False)),
    )

    def fake_restore(self, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise RuntimeError("boom")
        return list(kwargs["frames"]), 16, 16, len(kwargs["frames"])

    monkeypatch.setattr(seedvr2_module.SeedVR2, "_restore_video_frames", fake_restore)
    monkeypatch.setattr(seedvr2_module.VideoUtil, "_latents_to_frames", staticmethod(lambda decoded: decoded))

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    with pytest.raises(RuntimeError, match="boom"):
        model.restore_video_to_path(
            seed=42,
            video_path=source,
            resolution=256,
            softness=0.0,
            output_path=output,
            export_json_metadata=False,
            temporal_chunk_size=5,
            temporal_chunk_overlap=2,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".restored-*")) == []


@pytest.mark.fast
def test_seedvr2_restore_video_frames_trims_temporal_padding(monkeypatch):
    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 6, 32, 32), dtype=mx.float32), 32, 32)),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "encode",
        staticmethod(
            lambda vae, image, tiling_config, preserve_temporal_axis=False: mx.zeros(
                (1, 16, image.shape[2], 4, 4), dtype=mx.float32
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(
            lambda vae, latent, tiling_config, preserve_temporal_axis=False: mx.zeros(
                (1, 3, latent.shape[2], 32, 32), dtype=mx.float32
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    decoded, _, _, padded_frames = model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0)) for _ in range(6)],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert decoded.shape[2] == 6
    assert padded_frames == 9


@pytest.mark.fast
def test_seedvr2_restore_video_frames_keeps_single_frame_temporal_axis(monkeypatch):
    captured: dict[str, tuple[int, ...]] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32), 32, 32)),
    )

    def fake_encode(vae, image, tiling_config, preserve_temporal_axis=False):
        captured["encode_input_shape"] = tuple(image.shape)
        captured["encode_preserve_temporal_axis"] = preserve_temporal_axis
        return mx.zeros((1, 16, 1, 4, 4), dtype=mx.float32)

    def fake_decode(vae, latent, tiling_config, preserve_temporal_axis=False):
        captured["decode_input_shape"] = tuple(latent.shape)
        captured["decode_preserve_temporal_axis"] = preserve_temporal_axis
        return mx.zeros((1, 3, 1, 32, 32), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.VAEUtil, "encode", staticmethod(fake_encode))
    monkeypatch.setattr(seedvr2_module.VAEUtil, "decode", staticmethod(fake_decode))
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    decoded, _, _, padded_frames = model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0))],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert captured["encode_input_shape"] == (1, 3, 1, 32, 32)
    assert captured["encode_preserve_temporal_axis"] is True
    assert captured["decode_input_shape"] == (1, 16, 1, 4, 4)
    assert captured["decode_preserve_temporal_axis"] is True
    assert decoded.shape == (1, 3, 1, 32, 32)
    assert padded_frames == 1


@pytest.mark.fast
def test_seedvr2_restore_video_frames_disables_tiled_encode_for_video(monkeypatch):
    captured: dict[str, object] = {}

    def fake_init(model, model_config, quantize=None, model_path=None):
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = TilingConfig()
        model.bits = quantize
        model.vae = object()
        model.transformer = lambda txt, vid, timestep: mx.zeros_like(vid[:, :16])

    monkeypatch.setattr(seedvr2_module.SeedVR2Initializer, "init", fake_init)
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "preprocess_video_frames",
        staticmethod(lambda frames, resolution, softness: (mx.zeros((1, 3, 6, 32, 32), dtype=mx.float32), 32, 32)),
    )

    def fake_encode(vae, image, tiling_config, preserve_temporal_axis=False):
        captured["vae_encode_tiled"] = tiling_config.vae_encode_tiled if tiling_config is not None else None
        return mx.zeros((1, 16, image.shape[2], 4, 4), dtype=mx.float32)

    monkeypatch.setattr(seedvr2_module.VAEUtil, "encode", staticmethod(fake_encode))
    monkeypatch.setattr(
        seedvr2_module.VAEUtil,
        "decode",
        staticmethod(
            lambda vae, latent, tiling_config, preserve_temporal_axis=False: mx.zeros(
                (1, 3, latent.shape[2], 32, 32), dtype=mx.float32
            )
        ),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2TextEmbeddings,
        "load_positive",
        staticmethod(lambda: mx.zeros((1, 1, 5120), dtype=mx.float32)),
    )
    monkeypatch.setattr(
        seedvr2_module.SeedVR2Util,
        "apply_color_correction",
        staticmethod(lambda content, style, mode="lab": content),
    )

    model = SeedVR2(quantize=8, model_config=ModelConfig.seedvr2_3b())
    model._restore_video_frames(
        seed=42,
        frames=[_solid_frame((0, 0, 0)) for _ in range(6)],
        resolution=256,
        softness=0.0,
        color_correction_mode="wavelet",
    )

    assert captured["vae_encode_tiled"] is False


@pytest.mark.fast
def test_video_util_iter_video_frame_windows_streams_overlapping_windows(tmp_path):
    source = tmp_path / "source.mp4"
    frames = [_solid_frame((index * 25, index * 25, index * 25)) for index in range(8)]

    VideoUtil.save_video(
        frames=frames,
        path=source,
        fps=12.0,
        metadata=None,
        export_json_metadata=False,
        overwrite=True,
        validate_health=False,
    )

    windows = list(VideoUtil.iter_video_frame_windows(source, windows=[(0, 5), (3, 8)]))
    means = [
        [float(np.asarray(frame.convert("L"), dtype=np.float32).mean()) for frame in clip.frames]
        for clip in windows
    ]

    assert [clip.clip_start_frame for clip in windows] == [0, 3]
    assert [clip.clip_frame_count for clip in windows] == [5, 5]
    assert means[0][-2:] == means[1][:2]
