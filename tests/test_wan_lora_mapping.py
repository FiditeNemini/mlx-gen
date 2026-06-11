from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights.wan_lora_mapping import WanLoRAMapping


class _TinyWanAttention(nn.Module):
    def __init__(self, with_image_proj: bool = False):
        super().__init__()
        self.to_q = nn.Linear(4, 3, bias=False)
        self.to_k = nn.Linear(4, 3, bias=False)
        self.to_v = nn.Linear(4, 3, bias=False)
        self.to_out = [nn.Linear(3, 4, bias=False)]
        self.add_k_proj = nn.Linear(4, 3, bias=False) if with_image_proj else None
        self.add_v_proj = nn.Linear(4, 3, bias=False) if with_image_proj else None


class _TinyWanFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [nn.Linear(4, 6, bias=False), nn.Linear(6, 4, bias=False)]


class _TinyWanBlock(nn.Module):
    def __init__(self, with_image_proj: bool = False):
        super().__init__()
        self.attn1 = _TinyWanAttention()
        self.attn2 = _TinyWanAttention(with_image_proj=with_image_proj)
        self.ffn = _TinyWanFFN()


class _TinyWanTransformer(nn.Module):
    def __init__(self, with_image_proj: bool = False):
        super().__init__()
        self.blocks = [_TinyWanBlock(with_image_proj=with_image_proj)]


@pytest.mark.fast
def test_wan_non_diffusers_lora_keys_apply_to_attention_and_ffn(tmp_path: Path):
    lora_path = tmp_path / "wan-ti2v-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": mx.zeros((3, 2)),
            "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.cross_attn.k.lora_B.weight": mx.zeros((3, 2)),
            "diffusion_model.blocks.0.ffn.0.lora_A.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.ffn.0.lora_B.weight": mx.zeros((6, 2)),
            "diffusion_model.blocks.0.ffn.2.lora_A.weight": mx.zeros((2, 6)),
            "diffusion_model.blocks.0.ffn.2.lora_B.weight": mx.zeros((4, 2)),
        },
    )
    transformer = _TinyWanTransformer()

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
        role="transformer",
    )

    assert isinstance(transformer.blocks[0].attn1.to_q, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.to_k, LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[0], LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[1], LoRALinear)
    assert result.reports[0].matched_key_count == 8
    assert result.reports[0].unmatched_key_count == 0


@pytest.mark.fast
def test_wan_musubi_lora_keys_apply_to_attention_and_ffn(tmp_path: Path):
    lora_path = tmp_path / "wan-musubi-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "lora_unet_blocks_0_self_attn_q.lora_down.weight": mx.zeros((2, 4)),
            "lora_unet_blocks_0_self_attn_q.lora_up.weight": mx.zeros((3, 2)),
            "lora_unet_blocks_0_cross_attn_o.lora_down.weight": mx.zeros((2, 3)),
            "lora_unet_blocks_0_cross_attn_o.lora_up.weight": mx.zeros((4, 2)),
            "lora_unet_blocks_0_ffn_0.lora_down.weight": mx.zeros((2, 4)),
            "lora_unet_blocks_0_ffn_0.lora_up.weight": mx.zeros((6, 2)),
            "lora_unet_blocks_0_ffn_2.lora_down.weight": mx.zeros((2, 6)),
            "lora_unet_blocks_0_ffn_2.lora_up.weight": mx.zeros((4, 2)),
        },
    )
    transformer = _TinyWanTransformer()

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[0.8],
        role="transformer",
    )

    assert isinstance(transformer.blocks[0].attn1.to_q, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.to_out[0], LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[0], LoRALinear)
    assert isinstance(transformer.blocks[0].ffn.net[1], LoRALinear)
    assert result.reports[0].matched_key_count == 8
    assert result.reports[0].unmatched_key_count == 0


@pytest.mark.fast
def test_wan_i2v_expands_t2v_lora_to_image_projection_layers(tmp_path: Path):
    lora_path = tmp_path / "wan-t2v-only-lora.safetensors"
    mx.save_safetensors(
        str(lora_path),
        {
            "diffusion_model.blocks.0.cross_attn.k.lora_down.weight": mx.zeros((2, 4)),
            "diffusion_model.blocks.0.cross_attn.k.lora_up.weight": mx.zeros((3, 2)),
        },
    )
    transformer = _TinyWanTransformer(with_image_proj=True)

    result = LoRALoader.load_and_apply_lora_detailed(
        lora_mapping=WanLoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_path)],
        lora_scales=[1.0],
        role="transformer",
        state_dict_transform=WanInitializer._transform_wan_lora_state_dict,
    )

    assert isinstance(transformer.blocks[0].attn2.to_k, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.add_k_proj, LoRALinear)
    assert isinstance(transformer.blocks[0].attn2.add_v_proj, LoRALinear)
    assert result.reports[0].matched_key_count == 6
    assert result.reports[0].unmatched_key_count == 0
