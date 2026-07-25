from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.variants import Wan2_2_TI2V
from tests.wan.test_wan_a14b_config import _fake_t2v_a14b_model, _patch_fake_wan_generation


def test_wan_bracket_condition_video_layout_matches_diffusers_reference():
    # diffusers WanImageToVideoPipeline.prepare_latents with last_image:
    # video_condition = cat([image, zeros(num_frames - 2), last_image]).
    mx.random.seed(3)
    first = mx.random.uniform(shape=(1, 3, 8, 8), dtype=mx.float32) * 2 - 1
    last = mx.random.uniform(shape=(1, 3, 8, 8), dtype=mx.float32) * 2 - 1

    condition = Wan2_2_TI2V._build_video_condition(
        normalized_first_frame=first,
        normalized_last_frame=last,
        num_frames=9,
        batch_size=1,
        precision=mx.float32,
    )
    mx.eval(condition)

    assert condition.shape == (1, 3, 9, 8, 8)
    np.testing.assert_array_equal(np.array(condition[:, :, 0]), np.array(first))
    np.testing.assert_array_equal(np.array(condition[:, :, 1:8]), np.zeros((1, 3, 7, 8, 8), dtype=np.float32))
    np.testing.assert_array_equal(np.array(condition[:, :, 8]), np.array(last))


def test_wan_single_frame_condition_video_unchanged_without_last_frame():
    # No-flag path pin: omitting the last frame must reproduce the historical
    # [first, zeros(num_frames - 1)] layout bitwise.
    mx.random.seed(5)
    first = mx.random.uniform(shape=(1, 3, 8, 8), dtype=mx.float32) * 2 - 1

    condition = Wan2_2_TI2V._build_video_condition(
        normalized_first_frame=first,
        num_frames=9,
        batch_size=1,
        precision=mx.float32,
    )
    legacy = mx.concatenate(
        [first[:, :, None, :, :], mx.zeros((1, 3, 8, 8, 8), dtype=mx.float32)],
        axis=2,
    )
    mx.eval(condition, legacy)

    np.testing.assert_array_equal(np.array(condition), np.array(legacy))


def test_wan_bracket_bf16_condition_build_is_bitwise_identical_to_f32_concat_cast():
    # F2 discipline on the bracket path (cycle-2 review pin): building
    # [first, zeros, last] directly in bf16 must be bitwise identical to the
    # reference-style f32 concat followed by one cast (elementwise cast
    # commutes with concat; zeros cast exactly).
    mx.random.seed(11)
    first = mx.random.uniform(shape=(1, 3, 16, 16), dtype=mx.float32) * 2 - 1
    last = mx.random.uniform(shape=(1, 3, 16, 16), dtype=mx.float32) * 2 - 1

    old_condition = mx.concatenate(
        [
            first[:, :, None, :, :],
            mx.zeros((1, 3, 7, 16, 16), dtype=mx.float32),
            last[:, :, None, :, :],
        ],
        axis=2,
    ).astype(mx.bfloat16)
    new_condition = Wan2_2_TI2V._build_video_condition(
        normalized_first_frame=first,
        normalized_last_frame=last,
        num_frames=9,
        batch_size=1,
        precision=mx.bfloat16,
    )
    mx.eval(old_condition, new_condition)

    assert new_condition.dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.array(old_condition.astype(mx.float32)),
        np.array(new_condition.astype(mx.float32)),
    )


def _diffusers_reference_mask_channels(*, num_frames, latent_height, latent_width, temporal_scale, with_last):
    # Line-for-line numpy port of the diffusers WanImageToVideoPipeline mask
    # packing (prepare_latents), used as the layout oracle.
    mask = np.ones((1, 1, num_frames, latent_height, latent_width), dtype=np.float32)
    if with_last:
        mask[:, :, list(range(1, num_frames - 1))] = 0
    else:
        mask[:, :, list(range(1, num_frames))] = 0
    first_frame_mask = np.repeat(mask[:, :, 0:1], temporal_scale, axis=2)
    mask = np.concatenate([first_frame_mask, mask[:, :, 1:]], axis=2)
    mask = mask.reshape(1, -1, temporal_scale, latent_height, latent_width)
    return mask.transpose(0, 2, 1, 3, 4)


@pytest.mark.parametrize("with_last", [False, True])
def test_wan_i2v_condition_mask_rows_match_diffusers_reference(tmp_path, with_last):
    first_path = tmp_path / "first.png"
    last_path = tmp_path / "last.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    Image.new("RGB", (80, 64), "black").save(last_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])

    condition = model._load_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=9,
        batch_size=1,
        last_image_path=last_path if with_last else None,
    )
    mx.eval(condition)

    assert condition.shape == (1, 20, 3, 8, 10)
    expected_mask = _diffusers_reference_mask_channels(
        num_frames=9,
        latent_height=8,
        latent_width=10,
        temporal_scale=model.vae.temporal_scale,
        with_last=with_last,
    )
    # Channels 0:4 carry the packed temporal mask; 4:20 the VAE latents.
    latent_frames = condition.shape[2]
    np.testing.assert_array_equal(np.array(condition[:, :4]), expected_mask[:, :, :latent_frames])
    if with_last:
        # Endpoint truth: latent frame 0 fully masked-in, middle free, and the
        # final latent frame carries exactly the last pixel frame's mask slot.
        packed = np.array(condition[:, :4])
        assert packed[0, :, 0].min() == 1.0
        assert packed[0, :, 1].max() == 0.0
        np.testing.assert_array_equal(
            packed[0, :, -1, 0, 0],
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )


def test_wan_bracket_condition_cache_keys_include_last_image_identity(tmp_path):
    first_path = tmp_path / "first.png"
    last_path = tmp_path / "last.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    Image.new("RGB", (80, 64), "black").save(last_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])

    plain = model._encode_video_condition(image_path=first_path, height=64, width=80, num_frames=9, batch_size=1)
    bracketed = model._encode_video_condition(
        image_path=first_path, height=64, width=80, num_frames=9, batch_size=1, last_image_path=last_path
    )
    mx.eval(plain, bracketed)

    assert len(model.image_condition_cache) == 2
    assert not np.array_equal(np.array(plain), np.array(bracketed))


def test_wan_bracket_condition_cache_evicts_oldest_key_first(monkeypatch, tmp_path):
    # Cycle-2 review pin: the 2-entry condition cache evicts in insertion
    # order (next(iter(cache)) relies on dict ordering), so after a third
    # distinct key lands, the two NEWEST keys must still serve hits and the
    # oldest must re-encode.
    first_path = tmp_path / "first.png"
    last_a = tmp_path / "last_a.png"
    last_b = tmp_path / "last_b.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    Image.new("RGB", (80, 64), "black").save(last_a)
    Image.new("RGB", (80, 64), "gray").save(last_b)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])
    loads = {"count": 0}
    original_load = Wan2_2_TI2V._load_video_condition

    def counting_load(self, **kwargs):
        loads["count"] += 1
        return original_load(self, **kwargs)

    monkeypatch.setattr(Wan2_2_TI2V, "_load_video_condition", counting_load)

    def encode(last):
        return model._encode_video_condition(
            image_path=first_path, height=64, width=80, num_frames=9, batch_size=1, last_image_path=last
        )

    encode(None)  # plain(first): oldest
    encode(last_a)  # bracket(first, A)
    encode(last_b)  # bracket(first, B): evicts plain(first)
    assert loads["count"] == 3
    assert len(model.image_condition_cache) == 2

    encode(last_a)
    encode(last_b)
    assert loads["count"] == 3  # both newest keys are cache hits
    encode(None)
    assert loads["count"] == 4  # the evicted oldest key re-encodes


def test_wan_bracket_last_image_honors_resize_mode_like_first_frame(tmp_path):
    # Cycle-2 review pin: resize_mode must reach the LAST anchor through the
    # same _normalized_condition_frame geometry as the first frame. The first
    # image matches the canvas exactly (identical under every mode), so any
    # pad-vs-resize difference in the encoded condition can only come from
    # the last anchor's mapping.
    first_path = tmp_path / "first.png"
    last_path = tmp_path / "last_wide.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    Image.new("RGB", (160, 16), "white").save(last_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])

    # Deterministic geometry: pad letterboxes the mismatched-aspect white
    # last image with black bars (normalized exactly -1); resize stretches
    # white across the full canvas (normalized exactly +1).
    padded = np.array(Wan2_2_TI2V._normalized_condition_frame(last_path, height=64, width=80, resize_mode="pad"))
    stretched = np.array(Wan2_2_TI2V._normalized_condition_frame(last_path, height=64, width=80, resize_mode="resize"))
    np.testing.assert_array_equal(padded[0, :, 0, :], np.full((3, 80), -1.0, dtype=np.float32))
    np.testing.assert_array_equal(stretched[0, :, 0, :], np.full((3, 80), 1.0, dtype=np.float32))

    condition_pad = model._load_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=9,
        batch_size=1,
        last_image_path=last_path,
        resize_mode="pad",
    )
    condition_resize = model._load_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=9,
        batch_size=1,
        last_image_path=last_path,
        resize_mode="resize",
    )
    mx.eval(condition_pad, condition_resize)
    assert not np.array_equal(np.array(condition_pad), np.array(condition_resize))


def test_wan_generate_passes_last_image_to_condition_and_metadata(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    last_path = tmp_path / "last.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    Image.new("RGB", (80, 64), "black").save(last_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}

    def encode_video_condition(**kwargs):
        observed["condition"] = kwargs
        return mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32)

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(model, "_encode_video_condition", encode_video_condition)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=5,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        image_path=str(first_path),
        last_image_path=str(last_path),
    )

    assert observed["condition"]["last_image_path"] == str(last_path)
    assert observed["to_video"]["extra_metadata"]["last_image_path"] == str(last_path)


def test_wan_generate_metadata_omits_last_image_when_absent(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}
    monkeypatch.setattr(model, "_encode_video_condition", lambda **kwargs: mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32))

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)

    model.generate_video(
        seed=1,
        prompt="a slow wave",
        width=64,
        height=64,
        num_frames=5,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        image_path=str(first_path),
    )

    assert "last_image_path" not in observed["to_video"]["extra_metadata"]


def test_wan_last_image_requires_image_path(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="last_image_path requires image_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=5,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            last_image_path="last.png",
        )


def test_wan_last_image_rejected_on_video_to_video(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="last_image_path requires image_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=5,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            video_path="input.mp4",
            last_image_path="last.png",
        )


def test_wan_last_image_rejected_on_expand_timesteps_path(monkeypatch, tmp_path):
    # TI2V-5B first-frame conditioning (expand_timesteps) has no last-frame slot.
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()
    model.transformer = SimpleNamespace(patch_size=(1, 2, 2), in_channels=48, out_channels=48)
    model.transformer_2 = None
    model.vae = SimpleNamespace(z_dim=48, temporal_scale=4, spatial_scale=8)

    with pytest.raises(ValueError, match="expand_timesteps"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=5,
            num_inference_steps=2,
            guidance=1,
            image_path=str(first_path),
            last_image_path=str(first_path),
        )


def test_wan_last_image_requires_at_least_two_frames(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="at least 2 frames"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=1,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
            last_image_path=str(first_path),
        )


def test_wan_vace_rejects_last_image():
    from mflux.models.wan.variants import WanVace

    model = WanVace.__new__(WanVace)

    with pytest.raises(ValueError, match="VACE does not support last_image_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            last_image_path="last.png",
        )
