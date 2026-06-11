from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.ernie_image.weights.ernie_image_lora_mapping import ErnieImageLoRAMapping


class _TinyErnieAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 3, bias=False)
        self.to_out = [nn.Linear(4, 3, bias=False)]


class _TinyErnieMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.linear_fc2 = nn.Linear(6, 4, bias=False)


class _TinyErnieBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _TinyErnieAttention()
        self.mlp = _TinyErnieMLP()


class _TinyErnieTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_TinyErnieBlock()]
        self.final_linear = nn.Linear(4, 3, bias=False)


@pytest.mark.fast
def test_ernie_diffusion_model_lora_keys_apply_to_attention_and_mlp(tmp_path: Path):
    lora_path = tmp_path / "ernie-anime.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.layers.0.self_attention.to_q.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.layers.0.self_attention.to_q.lora_B.weight": mx.zeros((3, 2)),
            "diffusion_model.layers.0.mlp.gate_proj.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.layers.0.mlp.gate_proj.lora_B.weight": mx.zeros((6, 2)),
        },
    )
    transformer = _TinyErnieTransformer()

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=ErnieImageLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[0.8],
    )

    assert isinstance(transformer.layers[0].self_attention.to_q, LoRALinear)
    assert isinstance(transformer.layers[0].mlp.gate_proj, LoRALinear)
    assert result.reports[0].matched_key_count == 4
    assert result.reports[0].unmatched_key_count == 0
    assert result.reports[0].applied_target_count == 2


@pytest.mark.fast
def test_ernie_default_weight_aliases_apply_to_final_linear(tmp_path: Path):
    lora_path = tmp_path / "ernie-default.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "base_model.model.final_linear.lora_A.default.weight": mx.zeros((2, 4)),
            "base_model.model.final_linear.lora_B.default.weight": mx.zeros((3, 2)),
        },
    )
    transformer = _TinyErnieTransformer()

    LoRALoader.load_and_apply_lora(
        lora_mapping=ErnieImageLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
    )

    assert isinstance(transformer.final_linear, LoRALinear)
