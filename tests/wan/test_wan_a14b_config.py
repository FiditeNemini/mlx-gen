import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.variants import Wan2_2_TI2V
from mflux.models.wan.variants.wan2_2_ti2v import _GUIDANCE_2_UNSET
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping


def test_wan_a14b_t2v_config_resolves_exact_repo():
    config = ModelConfig.from_name("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert config is ModelConfig.wan2_2_t2v_a14b()
    assert config.transformer_overrides["has_transformer_2"] is True
    assert config.transformer_overrides["expand_timesteps"] is False
    assert config.transformer_overrides["boundary_ratio"] == 0.875
    assert config.transformer_overrides["flow_shift"] == 3.0
    assert config.transformer_overrides["vae_variant"] == "wan21"
    assert config.transformer_overrides["default_guidance"] == 4.0
    assert config.transformer_overrides["default_guidance_2"] == 3.0
    assert "低质量" in config.transformer_overrides["default_negative_prompt"]


def test_wan_ti2v_5b_config_matches_official_generation_defaults():
    config = ModelConfig.from_name("Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    assert config is ModelConfig.wan2_2_ti2v_5b()
    assert config.transformer_overrides["flow_shift"] == 5.0
    assert config.transformer_overrides["default_guidance"] == 5.0
    assert config.transformer_overrides["default_steps"] == 50
    assert config.transformer_overrides["default_fps"] == 24
    assert config.transformer_overrides["default_frames"] == 121
    assert "低质量" in config.transformer_overrides["default_negative_prompt"]
    assert config.transformer_overrides.get("default_guidance_2") is None


def test_wan_a14b_i2v_config_resolves_local_path_by_alias():
    config = ModelConfig.from_name("models/wan2.2-i2v-a14b-8bit")

    assert config.base_model == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    assert config.transformer_overrides["in_channels"] == 36
    assert config.transformer_overrides["out_channels"] == 16
    assert config.transformer_overrides["supports_image_to_video"] is True
    assert config.transformer_overrides["default_guidance"] == 3.5
    assert config.transformer_overrides["default_guidance_2"] == 3.5


def test_wan_a14b_weight_definition_includes_second_transformer():
    definition = WanWeightDefinition.for_config(ModelConfig.wan2_2_t2v_a14b())
    components = definition.get_components()

    assert [component.name for component in components] == ["transformer", "transformer_2", "vae"]
    assert components[0].num_layers == 40
    assert components[1].num_layers == 40
    assert "transformer_2/*.safetensors" in definition.get_download_patterns()


def test_wan_a14b_vae_mapping_uses_wan21_flat_encoder_and_plural_upsamplers():
    targets = {target.to_pattern for target in WanWeightMapping.get_vae_mapping(variant="wan21")}

    assert "encoder.down_blocks.0.conv1.conv3d.weight" in targets
    assert "encoder.down_blocks.2.resample_conv.weight" in targets
    assert "decoder.up_blocks.0.upsamplers.0.resample_conv.weight" in targets


def test_wan_a14b_vae_config_uses_16_channel_latents_and_scale_8():
    vae = Wan2_2_VAE(**ModelConfig.wan2_2_t2v_a14b().transformer_overrides["vae_config"])

    assert vae.z_dim == 16
    assert vae.spatial_scale == 8
    assert vae.temporal_scale == 4
    assert vae.patch_size == 1
    assert vae.latents_mean.shape == (16,)


def test_wan_a14b_boundary_selects_low_noise_transformer_below_boundary():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(name="high")
    model.transformer_2 = SimpleNamespace(name="low")

    high, high_guidance = model._select_transformer_and_guidance(
        timestep=900,
        boundary_timestep=875,
        guidance=4.0,
        guidance_2=3.0,
    )
    low, low_guidance = model._select_transformer_and_guidance(
        timestep=800,
        boundary_timestep=875,
        guidance=4.0,
        guidance_2=3.0,
    )

    assert high.name == "high"
    assert high_guidance == 4.0
    assert low.name == "low"
    assert low_guidance == 3.0


def test_wan_a14b_default_guidance_pair_uses_model_defaults():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()

    guidance, guidance_2 = model._resolve_guidance_pair(guidance=None, guidance_2=_GUIDANCE_2_UNSET)

    assert guidance == 4.0
    assert guidance_2 == 3.0


def test_wan_a14b_explicit_guidance_without_guidance_2_follows_guidance():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()

    guidance, guidance_2 = model._resolve_guidance_pair(guidance=4.5, guidance_2=_GUIDANCE_2_UNSET)

    assert guidance == 4.5
    assert guidance_2 == 4.5


def test_wan_ti2v_5b_default_guidance_has_no_low_noise_stage():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    guidance, guidance_2 = model._resolve_guidance_pair(guidance=None, guidance_2=_GUIDANCE_2_UNSET)

    assert guidance == 5.0
    assert guidance_2 is None


def test_wan_guidance_2_requires_two_transformer_boundary():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_ti2v_5b()

    with pytest.raises(ValueError, match="guidance_2 is only supported"):
        model.generate_video(
            seed=1,
            prompt="a slow wave",
            width=128,
            height=128,
            num_frames=5,
            num_inference_steps=1,
            guidance_2=2.0,
        )


def test_wan_python_api_infers_a14b_config_from_model_path(monkeypatch):
    observed = {}

    def fake_init(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(WanInitializer, "init", fake_init)

    Wan2_2_TI2V(model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    assert observed["model_config"] is ModelConfig.wan2_2_t2v_a14b()
    assert observed["model_path"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def test_wan_runtime_contract_rejects_mismatched_t2v_channels():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(in_channels=16, out_channels=16)
    model.transformer_2 = SimpleNamespace(in_channels=16, out_channels=16)
    model.vae = SimpleNamespace(z_dim=48)

    with pytest.raises(ValueError, match="Wan runtime config mismatch"):
        model._validate_runtime_contract(is_image_to_video=False)


def test_wan_runtime_contract_rejects_consistent_ti2v_modules_under_a14b_config():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(in_channels=48, out_channels=48)
    model.transformer_2 = None
    model.vae = SimpleNamespace(z_dim=48)

    with pytest.raises(ValueError, match="transformer.in_channels"):
        model._validate_runtime_contract(is_image_to_video=False)


def test_wan_source_config_rejects_a14b_checkpoint_with_ti2v_runtime(tmp_path):
    _write_wan_source_configs(
        tmp_path,
        has_transformer_2=True,
        boundary_ratio=0.875,
        transformer_in_channels=16,
        transformer_out_channels=16,
        transformer_layers=40,
        transformer_heads=40,
        transformer_ffn_dim=13824,
        vae_z_dim=16,
        vae_base_dim=96,
    )

    with pytest.raises(ValueError, match="Wan source/config mismatch"):
        WanInitializer._validate_source_config(tmp_path, ModelConfig.wan2_2_ti2v_5b())


def test_wan_source_config_accepts_matching_a14b_checkpoint(tmp_path):
    _write_wan_source_configs(
        tmp_path,
        has_transformer_2=True,
        boundary_ratio=0.875,
        transformer_in_channels=16,
        transformer_out_channels=16,
        transformer_layers=40,
        transformer_heads=40,
        transformer_ffn_dim=13824,
        vae_z_dim=16,
        vae_base_dim=96,
    )

    WanInitializer._validate_source_config(tmp_path, ModelConfig.wan2_2_t2v_a14b())


def test_wan_transformer_rejects_wrong_input_channels_before_conv():
    transformer = WanTransformer(
        in_channels=16,
        out_channels=16,
        num_layers=0,
        num_attention_heads=1,
        attention_head_dim=8,
        text_dim=16,
        ffn_dim=16,
    )

    with pytest.raises(ValueError, match="input channel mismatch"):
        transformer(
            hidden_states=mx.zeros((1, 48, 1, 4, 4)),
            timestep=mx.array([1], dtype=mx.float32),
            encoder_hidden_states=mx.zeros((1, 1, 16)),
        )


def test_wan_runtime_contract_accepts_a14b_t2v_channels():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_t2v_a14b()
    model.transformer = SimpleNamespace(in_channels=16, out_channels=16)
    model.transformer_2 = SimpleNamespace(in_channels=16, out_channels=16)
    model.vae = SimpleNamespace(z_dim=16)

    model._validate_runtime_contract(is_image_to_video=False)


def test_wan_runtime_contract_requires_image_for_i2v_model():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.model_config = ModelConfig.wan2_2_i2v_a14b()
    model.transformer = SimpleNamespace(in_channels=36)
    model.vae = SimpleNamespace(z_dim=16)

    with pytest.raises(ValueError, match="requires image-to-video input"):
        model._validate_runtime_contract(is_image_to_video=False)


def test_wan_a14b_prepare_latents_uses_vae_channels_not_i2v_input_channels():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.transformer = SimpleNamespace(in_channels=36)
    model.vae = SimpleNamespace(z_dim=16, temporal_scale=4, spatial_scale=8)

    latents = model.prepare_latents(seed=1, batch_size=1, height=64, width=80, num_frames=9)
    mx.eval(latents)

    assert latents.shape == (1, 16, 3, 8, 10)


def test_wan_a14b_i2v_condition_uses_20_condition_channels(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (80, 64), "white").save(image_path)
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.vae = Wan2_2_VAE(**ModelConfig.wan2_2_i2v_a14b().transformer_overrides["vae_config"])

    condition = model._encode_video_condition(
        image_path=image_path,
        height=64,
        width=80,
        num_frames=9,
        batch_size=1,
    )
    mx.eval(condition)

    assert condition.shape == (1, 20, 3, 8, 10)


def _write_wan_source_configs(
    path,
    *,
    has_transformer_2: bool,
    boundary_ratio: float | None,
    transformer_in_channels: int,
    transformer_out_channels: int,
    transformer_layers: int,
    transformer_heads: int,
    transformer_ffn_dim: int,
    vae_z_dim: int,
    vae_base_dim: int,
) -> None:
    (path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "WanPipeline",
                "boundary_ratio": boundary_ratio,
                "transformer_2": ["diffusers", "WanTransformer3DModel"] if has_transformer_2 else [None, None],
            }
        )
    )
    transformer_config = {
        "_class_name": "WanTransformer3DModel",
        "in_channels": transformer_in_channels,
        "out_channels": transformer_out_channels,
        "num_layers": transformer_layers,
        "num_attention_heads": transformer_heads,
        "ffn_dim": transformer_ffn_dim,
        "patch_size": [1, 2, 2],
    }
    for component in ("transformer", "transformer_2"):
        if component == "transformer_2" and not has_transformer_2:
            continue
        component_path = path / component
        component_path.mkdir()
        (component_path / "config.json").write_text(json.dumps(transformer_config))
    vae_path = path / "vae"
    vae_path.mkdir()
    (vae_path / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLWan", "z_dim": vae_z_dim, "base_dim": vae_base_dim}))
