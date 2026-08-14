import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.preview.preview_decoder import (
    TINY_DECODERS,
    PreviewDecoder,
    PreviewDecoderUnavailable,
)
from mflux.models.common.preview.tiny_autoencoder import TinyAutoencoder
from mflux.models.common.preview.tiny_autoencoder_loader import TinyAutoencoderLoader


def _write_checkpoint(tmp_path, latent_channels=16, *, drop_key=None):
    """Write a synthetic diffusers-format AutoencoderTiny checkpoint."""
    config = {
        "_class_name": "AutoencoderTiny",
        "act_fn": "relu",
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": latent_channels,
        "decoder_block_out_channels": [64, 64, 64, 64],
        "encoder_block_out_channels": [64, 64, 64, 64],
        "num_decoder_blocks": [3, 3, 3, 1],
        "num_encoder_blocks": [1, 3, 3, 3],
        "upsampling_scaling_factor": 2,
        "scaling_factor": 1.0,
        "shift_factor": 0.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    reference = TinyAutoencoder(latent_channels=latent_channels)
    weights = {}
    for key, value in tree_flatten(reference.parameters()):
        torch_key = key.replace(".layers.layers.", ".layers.").replace(".conv.layers.", ".conv.")
        # Published checkpoints store convolutions as (out, in, kH, kW).
        weights[torch_key] = mx.transpose(value, (0, 3, 1, 2)) if value.ndim == 4 else value
    if drop_key is not None:
        weights.pop(drop_key)
    mx.save_safetensors(str(tmp_path / "diffusion_pytorch_model.safetensors"), weights)
    return reference


@pytest.mark.fast
def test_tiny_autoencoder_matches_reference_graph_shape():
    model = TinyAutoencoder(latent_channels=16)
    keys = {key for key, _ in tree_flatten(model.parameters())}

    # Reference TAESD decoder: conv_in, 3+3+3+1 blocks, 3 upsample convs, conv_out.
    assert model.spatial_scale == 8
    assert sum(value.size for _, value in tree_flatten(model.parameters())) == 1_229_443
    assert "decoder.layers.layers.0.weight" in keys
    assert "decoder.layers.layers.18.bias" in keys
    # The three inter-stage convolutions carry no bias; only the output conv does.
    for index in (6, 11, 16):
        assert f"decoder.layers.layers.{index}.weight" in keys
        assert f"decoder.layers.layers.{index}.bias" not in keys


@pytest.mark.fast
def test_tiny_autoencoder_decode_shape_and_range():
    model = TinyAutoencoder(latent_channels=16)
    decoded = model.decode(mx.zeros((1, 16, 12, 20)))
    assert decoded.shape == (1, 3, 96, 160)

    # The clamp keeps extreme latents finite, and output follows the [-1, 1] convention.
    extreme = model.decode(mx.full((1, 16, 8, 8), 500.0))
    assert bool(mx.all(mx.isfinite(extreme)))


@pytest.mark.fast
def test_tiny_autoencoder_preserves_temporal_axis():
    model = TinyAutoencoder(latent_channels=16)
    assert model.decode(mx.zeros((1, 16, 1, 8, 8))).shape == (1, 3, 1, 64, 64)
    assert model.decode(mx.zeros((1, 16, 8, 8))).shape == (1, 3, 64, 64)


@pytest.mark.fast
def test_tiny_autoencoder_rejects_multi_frame_latents():
    """Silently previewing frame 0 of a clip would misrepresent the generation."""
    model = TinyAutoencoder(latent_channels=16)
    with pytest.raises(ValueError, match="single frames"):
        model.decode(mx.zeros((1, 16, 4, 8, 8)))


@pytest.mark.fast
def test_midblock_gn_variant_adds_pool_branch_to_deepest_group_only():
    model = TinyAutoencoder(latent_channels=32, use_midblock_gn=True)
    keys = {key for key, _ in tree_flatten(model.parameters())}

    # Upstream applies the branch to the first decoder group (blocks 2, 3, 4) only.
    for index in (2, 3, 4):
        assert f"decoder.layers.layers.{index}.pool.layers.0.weight" in keys
        assert f"decoder.layers.layers.{index}.pool.layers.1.weight" in keys
    for index in (7, 12, 17):
        assert f"decoder.layers.layers.{index}.pool.layers.0.weight" not in keys

    assert model.decode(mx.zeros((1, 32, 8, 8))).shape == (1, 3, 64, 64)


@pytest.mark.fast
def test_loader_accepts_config_overrides_for_configless_repos(tmp_path):
    reference = TinyAutoencoder(latent_channels=32, use_midblock_gn=True)
    weights = {}
    for key, value in tree_flatten(reference.parameters()):
        torch_key = (
            key.replace(".layers.layers.", ".layers.")
            .replace(".conv.layers.", ".conv.")
            .replace(".pool.layers.", ".pool.")  # fmt: skip
        )
        weights[torch_key] = mx.transpose(value, (0, 3, 1, 2)) if value.ndim == 4 else value
    mx.save_safetensors(str(tmp_path / "taef2.safetensors"), weights)

    loaded = TinyAutoencoderLoader.load(
        str(tmp_path),
        config_overrides={"latent_channels": 32, "use_midblock_gn": True},
        weight_file="taef2.safetensors",
    )
    reference_params = dict(tree_flatten(reference.parameters()))
    loaded_params = dict(tree_flatten(loaded.parameters()))
    assert set(reference_params) == set(loaded_params)
    for key, value in reference_params.items():
        assert mx.array_equal(value, loaded_params[key]), key


@pytest.mark.fast
def test_loader_requires_architecture_when_config_absent(tmp_path):
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), {"decoder.layers.0.weight": mx.zeros((64, 16, 3, 3))})
    with pytest.raises(FileNotFoundError, match="explicit architecture"):
        TinyAutoencoderLoader.load(str(tmp_path))


@pytest.mark.fast
def test_loader_keeps_zero_scaling_factor(tmp_path):
    _write_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["shift_factor"] = None  # published configs use null or omit the key
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert TinyAutoencoderLoader.load(str(tmp_path)).shift_factor == 0.0


@pytest.mark.fast
def test_loader_round_trips_published_key_layout(tmp_path):
    reference = _write_checkpoint(tmp_path)
    loaded = TinyAutoencoderLoader.load(str(tmp_path))

    reference_params = dict(tree_flatten(reference.parameters()))
    loaded_params = dict(tree_flatten(loaded.parameters()))
    assert set(reference_params) == set(loaded_params)
    for key, value in reference_params.items():
        assert mx.array_equal(value, loaded_params[key]), key


@pytest.mark.fast
def test_loader_rejects_incomplete_checkpoint(tmp_path):
    _write_checkpoint(tmp_path, drop_key="decoder.layers.0.weight")
    with pytest.raises(ValueError, match="weight mismatch"):
        TinyAutoencoderLoader.load(str(tmp_path))


@pytest.mark.fast
def test_loader_keeps_directional_structure(tmp_path):
    """A kH/kW transpose is invisible on symmetric input; use a directional kernel."""
    _write_checkpoint(tmp_path)
    model = TinyAutoencoderLoader.load(str(tmp_path))

    latents = np.zeros((1, 16, 16, 16), dtype=np.float32)
    latents[:, :, 4, :] = 3.0  # a horizontal bar
    decoded = np.array(model.decode(mx.array(latents)).astype(mx.float32))[0].mean(axis=0)
    row_energy = np.abs(decoded - decoded.mean()).mean(axis=1)
    column_energy = np.abs(decoded - decoded.mean()).mean(axis=0)
    # The response must stay banded across rows, not columns.
    assert row_energy.std() > column_energy.std() * 5


@pytest.mark.fast
def test_preview_decoder_mapping_is_explicit_not_channel_count():
    # Every registered entry names a concrete latent space and checkpoint.
    for latent_space, spec in TINY_DECODERS.items():
        assert spec.repo_id and spec.latent_channels > 0 and latent_space

    assert PreviewDecoder.available_for("flux.1") is True
    assert PreviewDecoder.available_for("qwen-image") is False
    assert PreviewDecoder.available_for(None) is False


@pytest.mark.fast
def test_preview_decoder_resolution_modes():
    class _Vae:
        latent_space = "unmapped-space"

    class _Model:
        vae = _Vae()

    model = _Model()
    assert PreviewDecoder.resolve(model, mode="full") is None
    # auto degrades to the full VAE rather than failing a generation...
    assert PreviewDecoder.resolve(model, mode="auto") is None
    # ...while an explicit request surfaces the problem.
    with pytest.raises(PreviewDecoderUnavailable, match="No tiny preview decoder"):
        PreviewDecoder.resolve(model, mode="tiny")


@pytest.mark.fast
def test_stepwise_handler_prefers_preview_decoder(tmp_path):
    from mflux.callbacks.instances.stepwise_handler import StepwiseHandler

    calls = {"tiny": 0, "vae": 0}

    class _Vae:
        def decode(self, latents):
            calls["vae"] += 1
            return mx.zeros((1, 3, 64, 64))

    class _Tiny:
        def decode(self, latents, vae=None):
            calls["tiny"] += 1
            return mx.zeros((1, 3, 64, 64))

    class _Model:
        vae = _Vae()
        bits = None
        lora_paths = None
        lora_scales = None

    class _LatentCreator:
        @staticmethod
        def unpack_latents(latents, height, width):
            return latents

    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    config = Config(width=64, height=64, num_inference_steps=2, model_config=ModelConfig.z_image_turbo())
    handler = StepwiseHandler(
        model=_Model(),
        output_dir=str(tmp_path),
        latent_creator=_LatentCreator,
        preview_decoder=_Tiny(),
    )
    handler.call_in_loop(t=0, seed=1, prompt="", latents=mx.zeros((1, 16, 8, 8)), config=config, time_steps=None)

    assert calls == {"tiny": 1, "vae": 0}
