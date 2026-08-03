from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.variants import Wan2_2_TI2V
from tests.wan.test_wan_a14b_config import _fake_t2v_a14b_model, _patch_fake_wan_generation


def _gradient_frames(count, *, height=8, width=8, seed=7):
    mx.random.seed(seed)
    return [mx.random.uniform(shape=(1, 3, height, width), dtype=mx.float32) * 2 - 1 for _ in range(count)]


def _write_frames(tmp_path, count, *, size=(80, 64)):
    paths = []
    for index in range(count):
        path = tmp_path / f"ctx_{index}.png"
        Image.new("RGB", size, (index * 17 % 256, 128, 255 - index * 13 % 256)).save(path)
        paths.append(path)
    return paths


def _a14b_i2v_model_with_vae():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])
    return model


def test_wan_context_condition_video_layout_places_head_zeros_and_last():
    first, *context = _gradient_frames(5)
    last = _gradient_frames(1, seed=11)[0]

    condition = Wan2_2_TI2V._build_video_condition(
        normalized_first_frame=first,
        normalized_context_frames=context,
        normalized_last_frame=last,
        num_frames=13,
        batch_size=1,
        precision=mx.float32,
    )
    mx.eval(condition)

    assert condition.shape == (1, 3, 13, 8, 8)
    np.testing.assert_array_equal(np.array(condition[:, :, 0]), np.array(first))
    for index, frame in enumerate(context, start=1):
        np.testing.assert_array_equal(np.array(condition[:, :, index]), np.array(frame))
    np.testing.assert_array_equal(np.array(condition[:, :, 5:12]), np.zeros((1, 3, 7, 8, 8), dtype=np.float32))
    np.testing.assert_array_equal(np.array(condition[:, :, 12]), np.array(last))


def test_wan_context_condition_without_last_fills_zeros_to_the_end():
    first, *context = _gradient_frames(5)

    condition = Wan2_2_TI2V._build_video_condition(
        normalized_first_frame=first,
        normalized_context_frames=context,
        num_frames=13,
        batch_size=1,
        precision=mx.float32,
    )
    mx.eval(condition)

    np.testing.assert_array_equal(np.array(condition[:, :, 5:]), np.zeros((1, 3, 8, 8, 8), dtype=np.float32))


def test_wan_no_context_condition_video_is_bitwise_identical_to_legacy_layout():
    # The context parameter must be a strict generalization: the None path
    # reproduces the historical [first, zeros x (n - 1)] build bitwise.
    first = _gradient_frames(1, seed=5)[0]

    condition = Wan2_2_TI2V._build_video_condition(
        normalized_first_frame=first,
        normalized_context_frames=None,
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


def _reference_context_mask_channels(*, num_frames, latent_height, latent_width, temporal_scale, head, with_last):
    # Generalized numpy port of the diffusers WanImageToVideoPipeline mask
    # packing: the kept head is [0, head) instead of frame 0 alone.
    mask = np.ones((1, 1, num_frames, latent_height, latent_width), dtype=np.float32)
    end = num_frames - 1 if with_last else num_frames
    mask[:, :, list(range(head, end))] = 0
    first_frame_mask = np.repeat(mask[:, :, 0:1], temporal_scale, axis=2)
    mask = np.concatenate([first_frame_mask, mask[:, :, 1:]], axis=2)
    mask = mask.reshape(1, -1, temporal_scale, latent_height, latent_width)
    return mask.transpose(0, 2, 1, 3, 4)


@pytest.mark.parametrize("with_last", [False, True])
@pytest.mark.parametrize("context_count", [4, 8])
def test_wan_context_mask_keeps_whole_head_latent_groups(tmp_path, with_last, context_count):
    paths = _write_frames(tmp_path, context_count + 2)
    first_path, *context_paths, last_path = paths
    model = _a14b_i2v_model_with_vae()

    condition = model._load_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=17,
        batch_size=1,
        context_image_paths=context_paths,
        last_image_path=last_path if with_last else None,
    )
    mx.eval(condition)

    assert condition.shape == (1, 20, 5, 8, 10)
    expected_mask = _reference_context_mask_channels(
        num_frames=17,
        latent_height=8,
        latent_width=10,
        temporal_scale=model.vae.temporal_scale,
        head=1 + context_count,
        with_last=with_last,
    )
    latent_frames = condition.shape[2]
    np.testing.assert_array_equal(np.array(condition[:, :4]), expected_mask[:, :, :latent_frames])
    packed = np.array(condition[:, :4])
    head_latent_frames = 1 + context_count // model.vae.temporal_scale
    # Head latent groups fully kept; the first free group fully zero (except
    # the bracketed tail slot when a last image is present).
    assert packed[0, :, :head_latent_frames].min() == 1.0
    if with_last:
        np.testing.assert_array_equal(
            packed[0, :, -1, 0, 0],
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        assert packed[0, :, head_latent_frames:-1].max() == 0.0
    else:
        assert packed[0, :, head_latent_frames:].max() == 0.0


def test_wan_context_condition_cache_key_includes_context_identity(tmp_path):
    paths = _write_frames(tmp_path, 5)
    first_path, *context_paths = paths
    model = _a14b_i2v_model_with_vae()

    plain = model._encode_video_condition(image_path=first_path, height=64, width=80, num_frames=9, batch_size=1)
    contexted = model._encode_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=9,
        batch_size=1,
        context_image_paths=context_paths,
    )
    mx.eval(plain, contexted)

    assert len(model.image_condition_cache) == 2
    assert not np.array_equal(np.array(plain), np.array(contexted))


def test_wan_context_noise_is_deterministic_and_touches_only_head_latents():
    temporal_scale = 4
    mx.random.seed(23)
    condition = mx.random.uniform(shape=(1, 20, 5, 8, 10), dtype=mx.float32)
    mx.eval(condition)

    noised_a = Wan2_2_TI2V._apply_context_noise(
        condition=condition,
        context_noise=20.0,
        head_frame_count=5,
        temporal_scale=temporal_scale,
        seed=42,
    )
    noised_b = Wan2_2_TI2V._apply_context_noise(
        condition=condition,
        context_noise=20.0,
        head_frame_count=5,
        temporal_scale=temporal_scale,
        seed=42,
    )
    noised_other_seed = Wan2_2_TI2V._apply_context_noise(
        condition=condition,
        context_noise=20.0,
        head_frame_count=5,
        temporal_scale=temporal_scale,
        seed=43,
    )
    mx.eval(noised_a, noised_b, noised_other_seed)

    head_latent_frames = 1 + (5 - 1) // temporal_scale
    # Deterministic per seed; different across seeds.
    np.testing.assert_array_equal(np.array(noised_a), np.array(noised_b))
    assert not np.array_equal(np.array(noised_a), np.array(noised_other_seed))
    # Mask channels stay binary truth and free frames stay untouched.
    np.testing.assert_array_equal(np.array(noised_a[:, :4]), np.array(condition[:, :4]))
    np.testing.assert_array_equal(
        np.array(noised_a[:, 4:, head_latent_frames:]),
        np.array(condition[:, 4:, head_latent_frames:]),
    )
    # The head is exactly the SkyReels-style blend (1 - t/1000) * latent +
    # (t/1000) * noise with the seed-derived key.
    head_before = condition[:, 4:, :head_latent_frames]
    expected_noise = mx.random.normal(head_before.shape, dtype=head_before.dtype, key=mx.random.key(42 + 0x9E3779B9))
    expected_head = (1.0 - 0.02) * head_before + 0.02 * expected_noise
    mx.eval(expected_head)
    np.testing.assert_allclose(
        np.array(noised_a[:, 4:, :head_latent_frames]),
        np.array(expected_head),
        rtol=1e-6,
        atol=1e-7,
    )


def test_wan_generate_passes_context_to_condition_and_metadata(monkeypatch, tmp_path):
    paths = _write_frames(tmp_path, 5)
    first_path, *context_paths = paths
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}

    def encode_video_condition(**kwargs):
        observed["condition"] = kwargs
        return mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32)

    def apply_context_noise(**kwargs):
        observed["context_noise"] = {key: value for key, value in kwargs.items() if key != "condition"}
        return kwargs["condition"]

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(model, "_encode_video_condition", encode_video_condition)
    monkeypatch.setattr(Wan2_2_TI2V, "_apply_context_noise", staticmethod(apply_context_noise))
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)

    model.generate_video(
        seed=1,
        prompt="the starship keeps rising",
        width=64,
        height=64,
        num_frames=13,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        image_path=str(first_path),
        context_image_paths=[str(path) for path in context_paths],
        context_noise=20.0,
    )

    assert observed["condition"]["context_image_paths"] == [str(path) for path in context_paths]
    assert observed["context_noise"]["head_frame_count"] == 5
    assert observed["context_noise"]["context_noise"] == 20.0
    assert observed["context_noise"]["seed"] == 1
    extras = observed["to_video"]["extra_metadata"]
    assert extras["context_image_paths"] == [str(path) for path in context_paths]
    assert extras["context_noise"] == 20.0


def test_wan_generate_metadata_omits_context_when_absent(monkeypatch, tmp_path):
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

    assert "context_image_paths" not in observed["to_video"]["extra_metadata"]
    assert "context_noise" not in observed["to_video"]["extra_metadata"]


def test_wan_context_frames_require_image_path(monkeypatch):
    # Covers the t2v rejection: no image_path means no conditioned head.
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="context_image_paths requires image_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            context_image_paths=["c1.png", "c2.png", "c3.png", "c4.png"],
        )


def test_wan_context_frames_rejected_on_video_to_video(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="context_image_paths requires image_path"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            video_path="input.mp4",
            context_image_paths=["c1.png", "c2.png", "c3.png", "c4.png"],
        )


def test_wan_context_frames_rejected_on_expand_timesteps_path(tmp_path):
    # TI2V-5B first-frame conditioning (expand_timesteps) has no multi-frame head.
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
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            image_path=str(first_path),
            context_image_paths=[str(first_path)] * 4,
        )


def test_wan_context_frames_count_must_fill_whole_latent_groups(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="multiple of 4"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
            context_image_paths=[str(first_path)] * 3,
        )


def test_wan_context_frames_head_bound_is_thirteen(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="maximum is 13"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=33,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
            context_image_paths=[str(first_path)] * 16,
        )


def test_wan_context_frames_require_free_frames_beyond_head(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="requires at least 13 frames"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=9,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
            context_image_paths=[str(first_path)] * 8,
        )


def test_wan_context_noise_requires_context_frames(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="context_noise requires context_image_paths"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
            context_noise=20.0,
        )


def test_wan_context_noise_range_is_validated(monkeypatch, tmp_path):
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="within \\[0, 1000\\]"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
            context_image_paths=[str(first_path)] * 4,
            context_noise=1200.0,
        )


def test_wan_vace_rejects_context_frames():
    from mflux.models.wan.variants import WanVace

    model = WanVace.__new__(WanVace)

    with pytest.raises(ValueError, match="VACE does not support context_image_paths"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            context_image_paths=["c1.png"],
        )
    with pytest.raises(ValueError, match="VACE does not support context_noise"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            context_noise=20.0,
        )


def test_wan_context_frames_honor_resize_mode_like_first_frame(tmp_path):
    # resize_mode must reach every context frame through the same
    # _normalized_condition_frame geometry as the first frame (0097 rule).
    first_path = tmp_path / "first.png"
    Image.new("RGB", (80, 64), "white").save(first_path)
    wide_paths = []
    for index in range(4):
        path = tmp_path / f"wide_{index}.png"
        Image.new("RGB", (160, 16), "white").save(path)
        wide_paths.append(path)
    model = _a14b_i2v_model_with_vae()

    condition_pad = model._load_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=13,
        batch_size=1,
        context_image_paths=wide_paths,
        resize_mode="pad",
    )
    condition_resize = model._load_video_condition(
        image_path=first_path,
        height=64,
        width=80,
        num_frames=13,
        batch_size=1,
        context_image_paths=wide_paths,
        resize_mode="resize",
    )
    mx.eval(condition_pad, condition_resize)
    assert not np.array_equal(np.array(condition_pad), np.array(condition_resize))
