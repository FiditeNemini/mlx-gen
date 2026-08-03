from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import (
    LoRAApplicationError,
    LoRAFileReport,
    LoRALoader,
)
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.variants import Wan2_2_TI2V
from mflux.models.wan.variants.wan_svi import MOTION_LATENT_KEY, WanSvi
from mflux.models.wan.wan_initializer import WanInitializer
from tests.wan.test_wan_a14b_config import _fake_t2v_a14b_model, _patch_fake_wan_generation
from tests.wan.test_wan_context_frames import _reference_context_mask_channels


def _a14b_i2v_model_with_vae():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])
    return model


def _write_anchor(tmp_path, *, name="anchor.png", size=(80, 64)):
    path = tmp_path / name
    Image.new("RGB", size, (200, 40, 90)).save(path)
    return path


def _write_motion_latents(tmp_path, *, latent_frames=4, latent_height=8, latent_width=10, name="prev.svi_latent"):
    mx.random.seed(3)
    latents = mx.random.normal((1, 16, latent_frames, latent_height, latent_width), dtype=mx.float32)
    mx.eval(latents)
    path = WanSvi.export_motion_latents(
        tmp_path / f"{name}.safetensors",
        latents=latents,
        width=latent_width * 8,
        height=latent_height * 8,
        num_frames=(latent_frames - 1) * 4 + 1,
        model_name="test-model",
    )
    return path, latents


def _svi_report(role, *, unmatched=0):
    return LoRAFileReport(
        requested_path=f"{role}.safetensors",
        resolved_path=f"/resolved/{role}.safetensors",
        scale=1.0,
        role=role,
        total_key_count=800,
        matched_key_count=800 - unmatched,
        unmatched_key_count=unmatched,
        applied_target_count=400,
    )


def _mark_svi_pack_loaded(model):
    model.svi_lora_paths = ["/resolved/high.safetensors", "/resolved/low.safetensors"]
    model.svi_lora_reports = (
        _svi_report("high_noise_transformer"),
        _svi_report("low_noise_transformer"),
    )


# --- conditioning layout ---------------------------------------------------


def test_svi_first_clip_condition_is_anchor_plus_true_zero_latents(tmp_path):
    model = _a14b_i2v_model_with_vae()
    anchor_path = _write_anchor(tmp_path)

    condition = WanSvi.build_condition(
        model,
        anchor_image_path=anchor_path,
        motion_latent_path=None,
        motion_latent_count=1,
        height=64,
        width=80,
        num_frames=17,
        batch_size=1,
        resize_mode="resize",
    )
    mx.eval(condition)

    # 4 mask channels + 16 latent channels, (17 - 1) // 4 + 1 = 5 latent frames.
    assert condition.shape == (1, 20, 5, 8, 10)
    anchor_latent = model._load_first_frame_condition(image_path=anchor_path, height=64, width=80)
    mx.eval(anchor_latent)
    np.testing.assert_array_equal(np.array(condition[:, 4:, 0:1]), np.array(anchor_latent))
    # THE SVI convention: padding is zero LATENTS, not the VAE encode of zero
    # frames (which is non-zero and belongs to the stock i2v convention).
    np.testing.assert_array_equal(
        np.array(condition[:, 4:, 1:]),
        np.zeros((1, 16, 4, 8, 10), dtype=np.float32),
    )
    stock = model._load_video_condition(image_path=anchor_path, height=64, width=80, num_frames=17, batch_size=1)
    mx.eval(stock)
    assert not np.array_equal(np.array(stock[:, 4:, 1:]), np.array(condition[:, 4:, 1:]))


def test_svi_condition_mask_marks_only_the_first_frame_group(tmp_path):
    model = _a14b_i2v_model_with_vae()
    anchor_path = _write_anchor(tmp_path)
    motion_path, _ = _write_motion_latents(tmp_path)

    condition = WanSvi.build_condition(
        model,
        anchor_image_path=anchor_path,
        motion_latent_path=motion_path,
        motion_latent_count=1,
        height=64,
        width=80,
        num_frames=17,
        batch_size=1,
        resize_mode="resize",
    )
    mx.eval(condition)

    # SVI keeps the STANDARD first-frame mask: the motion latent at temporal
    # position 1 stays mask=0 (the fine-tuned model reads it positionally).
    expected_mask = _reference_context_mask_channels(
        num_frames=17,
        latent_height=8,
        latent_width=10,
        temporal_scale=model.vae.temporal_scale,
        head=1,
        with_last=False,
    )
    np.testing.assert_array_equal(np.array(condition[:, :4]), expected_mask[:, :, :5])


def test_svi_continuation_condition_places_motion_latent_after_anchor(tmp_path):
    model = _a14b_i2v_model_with_vae()
    anchor_path = _write_anchor(tmp_path)
    motion_path, motion_latents = _write_motion_latents(tmp_path, latent_frames=4)

    for count in (1, 2):
        condition = WanSvi.build_condition(
            model,
            anchor_image_path=anchor_path,
            motion_latent_path=motion_path,
            motion_latent_count=count,
            height=64,
            width=80,
            num_frames=17,
            batch_size=1,
            resize_mode="resize",
        )
        mx.eval(condition)

        # The LAST `count` entries of the exported tensor land at positions
        # 1..count (reference: prev_last_latent[:, -num_motion_latent:]).
        np.testing.assert_array_equal(
            np.array(condition[:, 4:, 1 : 1 + count]),
            np.array(motion_latents[:, :, -count:]),
        )
        np.testing.assert_array_equal(
            np.array(condition[:, 4:, 1 + count :]),
            np.zeros((1, 16, 5 - 1 - count, 8, 10), dtype=np.float32),
        )


# --- motion latent file IO -------------------------------------------------


def test_svi_export_import_round_trip_is_exact(tmp_path):
    mx.random.seed(11)
    latents = mx.random.normal((1, 16, 4, 8, 10), dtype=mx.float32)
    mx.eval(latents)

    exported = WanSvi.export_motion_latents(
        tmp_path / "clip.svi_latent.safetensors",
        latents=latents,
        width=80,
        height=64,
        num_frames=13,
        model_name="test-model",
    )
    loaded = WanSvi.load_motion_latents(exported, count=4, z_dim=16, latent_height=8, latent_width=10)
    mx.eval(loaded)

    np.testing.assert_array_equal(np.array(loaded), np.array(latents))


def test_svi_export_requires_safetensors_suffix(tmp_path):
    with pytest.raises(ValueError, match="safetensors"):
        WanSvi.export_motion_latents(
            tmp_path / "clip.latent.bin",
            latents=mx.zeros((1, 16, 1, 8, 10), dtype=mx.float32),
            width=80,
            height=64,
            num_frames=5,
            model_name="test-model",
        )


def test_svi_motion_latent_load_rejections(tmp_path):
    motion_path, _ = _write_motion_latents(tmp_path, latent_frames=2)

    with pytest.raises(ValueError, match="does not exist"):
        WanSvi.load_motion_latents(
            tmp_path / "missing.safetensors", count=1, z_dim=16, latent_height=8, latent_width=10
        )
    with pytest.raises(ValueError, match="canvas"):
        WanSvi.load_motion_latents(motion_path, count=1, z_dim=16, latent_height=16, latent_width=20)
    with pytest.raises(ValueError, match="exceeds the 2 temporal entries"):
        WanSvi.load_motion_latents(motion_path, count=3, z_dim=16, latent_height=8, latent_width=10)

    bogus = tmp_path / "bogus.safetensors"
    mx.save_safetensors(str(bogus), {"other": mx.zeros((2, 2))})
    with pytest.raises(ValueError, match=f"no '{MOTION_LATENT_KEY}' tensor"):
        WanSvi.load_motion_latents(bogus, count=1, z_dim=16, latent_height=8, latent_width=10)

    wrong_rank = tmp_path / "wrong_rank.safetensors"
    mx.save_safetensors(str(wrong_rank), {MOTION_LATENT_KEY: mx.zeros((8, 2, 8, 10), dtype=mx.float32)})
    with pytest.raises(ValueError, match="expected"):
        WanSvi.load_motion_latents(wrong_rank, count=1, z_dim=16, latent_height=8, latent_width=10)


def test_svi_assembly_trim_frames_matches_reference_overlap():
    # The authors stitch continuation clips with the first FIVE frames removed
    # for one motion latent: 1 anchor-restoration frame + 4 re-rendered frames.
    assert WanSvi.assembly_trim_frames(temporal_scale=4, motion_latent_count=1, is_continuation=False) == 0
    assert WanSvi.assembly_trim_frames(temporal_scale=4, motion_latent_count=1, is_continuation=True) == 5
    assert WanSvi.assembly_trim_frames(temporal_scale=4, motion_latent_count=2, is_continuation=True) == 9


# --- generate_video bind contract -------------------------------------------


def _svi_capable_fake_model(monkeypatch):
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _mark_svi_pack_loaded(model)
    return model


def test_wan_generate_svi_binds_condition_export_and_metadata(monkeypatch, tmp_path):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    anchor_path = _write_anchor(tmp_path)
    motion_path, _ = _write_motion_latents(tmp_path, latent_frames=1, latent_height=8, latent_width=8)
    export_path = tmp_path / "next.svi_latent.safetensors"
    observed = {}

    def encode_svi_condition(**kwargs):
        observed["condition"] = kwargs
        return mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32)

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(model, "_encode_svi_condition", encode_svi_condition)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)

    model.generate_video(
        seed=7,
        prompt="the drone rises out of the steam",
        width=64,
        height=64,
        num_frames=13,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        svi_anchor_image_path=str(anchor_path),
        svi_motion_latent_path=str(motion_path),
        svi_motion_latent_count=1,
        svi_motion_latent_export_path=export_path,
    )

    assert observed["condition"]["anchor_image_path"] == str(anchor_path)
    assert observed["condition"]["motion_latent_path"] == str(motion_path)
    assert observed["condition"]["motion_latent_count"] == 1
    # The final latent was exported for the next clip (real file, loadable).
    assert export_path.exists()
    reloaded = WanSvi.load_motion_latents(export_path, count=1, z_dim=16, latent_height=8, latent_width=8)
    assert reloaded.shape == (1, 16, 1, 8, 8)
    extras = observed["to_video"]["extra_metadata"]
    assert extras["svi_anchor_image_path"] == str(anchor_path)
    assert extras["svi_motion_latent_path"] == str(motion_path)
    assert extras["svi_motion_latent_count"] == 1
    assert extras["svi_motion_latent_export"] == str(export_path)
    assert extras["svi_assembly_trim_frames"] == 5
    assert extras["svi_lora_high"]["unmatched_key_count"] == 0
    assert extras["svi_lora_low"]["role"] == "low_noise_transformer"


def test_wan_generate_svi_first_clip_metadata_has_no_trim(monkeypatch, tmp_path):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    anchor_path = _write_anchor(tmp_path)
    observed = {}

    monkeypatch.setattr(model, "_encode_svi_condition", lambda **kwargs: mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32))

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)

    model.generate_video(
        seed=7,
        prompt="a silver survey drone hovering over a geyser field",
        width=64,
        height=64,
        num_frames=13,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        svi_anchor_image_path=str(anchor_path),
    )

    extras = observed["to_video"]["extra_metadata"]
    assert extras["svi_assembly_trim_frames"] == 0
    assert "svi_motion_latent_path" not in extras
    assert "svi_motion_latent_export" not in extras


def test_wan_generate_non_svi_metadata_omits_svi_keys(monkeypatch, tmp_path):
    first_path = _write_anchor(tmp_path, name="first.png")
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

    assert not any(key.startswith("svi_") for key in observed["to_video"]["extra_metadata"])


# --- rejection matrix --------------------------------------------------------


def test_svi_rejected_without_the_lora_pack(monkeypatch, tmp_path):
    model = _fake_t2v_a14b_model()
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer.in_channels = 36
    model.transformer_2.in_channels = 36
    _patch_fake_wan_generation(monkeypatch, model)
    anchor_path = _write_anchor(tmp_path)

    with pytest.raises(ValueError, match="requires the SVI LoRA pair"):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            svi_anchor_image_path=str(anchor_path),
        )


def test_non_svi_run_rejected_when_the_pack_is_loaded(monkeypatch, tmp_path):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model)
    first_path = _write_anchor(tmp_path, name="first.png")

    with pytest.raises(ValueError, match="constructed with the SVI LoRA pair"):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            image_path=str(first_path),
        )


@pytest.mark.parametrize(
    ("conflict_kwargs", "match"),
    [
        ({"image_path": "first.png"}, "conflicts with image_path"),
        ({"last_image_path": "last.png"}, "conflicts with last_image_path"),
        ({"video_path": "input.mp4"}, "conflicts with video_path"),
        (
            {"context_image_paths": ["c1.png", "c2.png", "c3.png", "c4.png"]},
            "conflicts with context_image_paths",
        ),
    ],
)
def test_svi_conflicts_fail_loudly(monkeypatch, tmp_path, conflict_kwargs, match):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model)
    anchor_path = _write_anchor(tmp_path)

    with pytest.raises(ValueError, match=match):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            svi_anchor_image_path=str(anchor_path),
            **conflict_kwargs,
        )


def test_svi_motion_latent_requires_anchor(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model)

    with pytest.raises(ValueError, match="svi_motion_latent_path requires svi_anchor_image_path"):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            svi_motion_latent_path="prev.safetensors",
        )


def test_svi_motion_latent_count_requires_motion_latent(monkeypatch, tmp_path):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model)
    anchor_path = _write_anchor(tmp_path)

    with pytest.raises(ValueError, match="svi_motion_latent_count requires svi_motion_latent_path"):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            svi_anchor_image_path=str(anchor_path),
            svi_motion_latent_count=2,
        )


def test_svi_rejected_on_expand_timesteps_path(tmp_path):
    # TI2V-5B first-frame conditioning has no 20-channel conditioning stream.
    anchor_path = _write_anchor(tmp_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()
    model.transformer = SimpleNamespace(patch_size=(1, 2, 2), in_channels=48, out_channels=48)
    model.transformer_2 = None
    model.vae = SimpleNamespace(z_dim=48, temporal_scale=4, spatial_scale=8)
    _mark_svi_pack_loaded(model)

    with pytest.raises(ValueError, match="does not support SVI conditioning"):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=13,
            num_inference_steps=2,
            guidance=1,
            svi_anchor_image_path=str(anchor_path),
        )


def test_svi_requires_a_free_latent_group(monkeypatch, tmp_path):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model)
    anchor_path = _write_anchor(tmp_path)
    motion_path, _ = _write_motion_latents(tmp_path)

    with pytest.raises(ValueError, match="free latent group"):
        model.generate_video(
            seed=1,
            prompt="p",
            width=64,
            height=64,
            num_frames=5,
            num_inference_steps=2,
            guidance=1,
            guidance_2=1,
            svi_anchor_image_path=str(anchor_path),
            svi_motion_latent_path=str(motion_path),
        )


def test_svi_continuation_beyond_65_frames_warns(monkeypatch, tmp_path, capsys):
    model = _svi_capable_fake_model(monkeypatch)
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    anchor_path = _write_anchor(tmp_path)
    motion_path, _ = _write_motion_latents(tmp_path, latent_frames=1, latent_height=8, latent_width=8)

    monkeypatch.setattr(model, "_encode_svi_condition", lambda **kwargs: mx.zeros((1, 20, 1, 8, 8), dtype=mx.float32))
    monkeypatch.setattr(
        "mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches",
        lambda **kwargs: SimpleNamespace(),
    )

    model.generate_video(
        seed=1,
        prompt="p",
        width=64,
        height=64,
        num_frames=69,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
        svi_anchor_image_path=str(anchor_path),
        svi_motion_latent_path=str(motion_path),
    )

    assert "65 frames" in capsys.readouterr().out


def test_wan_vace_rejects_svi():
    from mflux.models.wan.variants import WanVace

    model = WanVace.__new__(WanVace)

    with pytest.raises(ValueError, match="VACE does not support SVI"):
        model.generate_video(seed=1, prompt="p", svi_anchor_image_path="anchor.png")
    with pytest.raises(ValueError, match="VACE does not support SVI"):
        model.generate_video(seed=1, prompt="p", svi_motion_latent_path="prev.safetensors")


# --- strict LoRA pack loading ------------------------------------------------


def test_svi_lora_strict_match_rejects_unmatched_keys():
    with pytest.raises(LoRAApplicationError, match="left 3 of 800 keys unmatched"):
        WanInitializer._require_strict_svi_key_match(_svi_report("high_noise_transformer", unmatched=3))
    WanInitializer._require_strict_svi_key_match(_svi_report("high_noise_transformer", unmatched=0))


def test_svi_lora_pair_is_indivisible():
    model = SimpleNamespace(transformer_2=SimpleNamespace(), model_config=SimpleNamespace(model_name="m"))

    with pytest.raises(LoRAApplicationError, match="indivisible"):
        WanInitializer._apply_svi_loras(model, svi_lora_high_path="high.safetensors", svi_lora_low_path=None)
    with pytest.raises(LoRAApplicationError, match="indivisible"):
        WanInitializer._apply_svi_loras(model, svi_lora_high_path=None, svi_lora_low_path="low.safetensors")


def test_svi_lora_pair_requires_dual_expert_model():
    model = SimpleNamespace(transformer_2=None, model_config=SimpleNamespace(model_name="Wan-VACE"))

    with pytest.raises(LoRAApplicationError, match="dual-expert"):
        WanInitializer._apply_svi_loras(
            model, svi_lora_high_path="high.safetensors", svi_lora_low_path="low.safetensors"
        )


def test_svi_lora_absent_pair_initializes_empty_state():
    model = SimpleNamespace(transformer_2=SimpleNamespace(), model_config=SimpleNamespace(model_name="m"))

    WanInitializer._apply_svi_loras(model, svi_lora_high_path=None, svi_lora_low_path=None)

    assert model.svi_lora_paths == []
    assert model.svi_lora_reports == ()


# --- PEFT adapter-infix normalization ----------------------------------------


def test_peft_adapter_infix_is_stripped_for_single_adapter_files():
    weights = {
        "blocks.0.self_attn.q.lora_A.default.weight": mx.zeros((2, 2)),
        "blocks.0.self_attn.q.lora_B.default.weight": mx.zeros((2, 2)),
    }

    normalized = LoRALoader._normalize_peft_adapter_infix(weights)

    assert set(normalized) == {
        "blocks.0.self_attn.q.lora_A.weight",
        "blocks.0.self_attn.q.lora_B.weight",
    }


def test_peft_adapter_infix_left_untouched_for_multi_adapter_files():
    weights = {
        "blocks.0.self_attn.q.lora_A.style.weight": mx.zeros((2, 2)),
        "blocks.0.self_attn.q.lora_A.motion.weight": mx.zeros((2, 2)),
    }

    normalized = LoRALoader._normalize_peft_adapter_infix(weights)

    assert set(normalized) == set(weights)


def test_peft_adapter_infix_noop_for_plain_keys():
    weights = {
        "blocks.0.self_attn.q.lora_A.weight": mx.zeros((2, 2)),
        "blocks.0.self_attn.q.alpha": mx.zeros((1,)),
    }

    normalized = LoRALoader._normalize_peft_adapter_infix(weights)

    assert set(normalized) == set(weights)
