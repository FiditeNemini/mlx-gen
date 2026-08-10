import mlx.core as mx

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.weights import WanWeightDefinition


def test_bernini_precision_override_matches_official_keep_set():
    definition = WanWeightDefinition.for_config(ModelConfig.bernini_r_1_3b())
    transformer = next(component for component in definition.get_components() if component.name == "transformer")

    assert transformer.precision_override is not None
    assert transformer.precision_override("scale_shift_table") == mx.float32
    assert transformer.precision_override("blocks.0.scale_shift_table") == mx.float32
    assert transformer.precision_override("patch_embedding.weight") is None
    assert transformer.precision_override("patch_embedding.bias") is None
    assert transformer.precision_override("condition_embedder.time_embedder.linear_1.weight") == mx.float32
    assert transformer.precision_override("condition_embedder.time_embedder.linear_2.bias") == mx.float32
    assert transformer.precision_override("condition_embedder.time_proj.weight") is None
    assert transformer.precision_override("condition_embedder.time_proj.bias") is None
    assert transformer.precision_override("condition_embedder.text_embedder.linear_1.weight") is None
    assert transformer.precision_override("condition_embedder.text_embedder.linear_2.bias") is None
    assert transformer.precision_override("blocks.0.attn1.norm_q.weight") is None
    assert transformer.precision_override("blocks.0.attn1.norm_k.weight") is None
    assert transformer.precision_override("blocks.0.norm2.weight") == mx.float32
    assert transformer.precision_override("blocks.0.norm1.weight") == mx.float32
    assert transformer.precision_override("blocks.0.norm3.weight") == mx.float32
    assert transformer.precision_override("norm_out.weight") is None
    assert transformer.precision_override("proj_out.weight") is None
