from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.fibo.fibo_initializer import FIBOInitializer
from mflux.models.fibo.model.fibo_text_encoder import smol_lm3_3b_attention as smol_attention_module
from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_attention import SmolLM3_3B_SelfAttention
from mflux.models.fibo.model.fibo_text_encoder.smol_lm3_3b_text_encoder import SmolLM3_3B_TextEncoder
from mflux.models.fibo.model.fibo_transformer import fibo_single_attention as fibo_single_attention_module
from mflux.models.fibo.model.fibo_transformer.fibo_joint_attention import FiboJointAttention
from mflux.models.fibo.model.fibo_transformer.fibo_single_attention import FiboSingleAttention
from mflux.models.fibo.model.fibo_transformer.transformer import FiboTransformer
from mflux.models.fibo.model.fibo_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.fibo.weights.fibo_weight_definition import FIBOWeightDefinition
from mflux.utils.tensor_health import TensorHealthError


class QuantizableModule:
    def to_quantized(self):
        return self


def test_fibo_attention_mask_preserves_text_padding():
    config = SimpleNamespace(height=16, width=16)
    prompt_attention_mask = mx.array(
        [
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=mx.float32,
    )
    encoder_hidden_states = mx.zeros((2, 3, 4), dtype=mx.float32)

    attention_mask = FiboTransformer._compute_attention_mask(
        batch_size=2,
        config=config,
        encoder_hidden_states=encoder_hidden_states,
        max_tokens=3,
        prompt_attention_mask=prompt_attention_mask,
    )

    mask = np.array(attention_mask)
    assert mask.shape == (2, 1, 4, 4)
    assert np.isneginf(mask[0, 0, 1, 0])
    assert np.isneginf(mask[0, 0, 0, 1])
    assert mask[0, 0, 0, 2] == 0
    assert mask[1, 0, 1, 0] == 0


def test_fibo_transformer_attention_mask_uses_negative_infinity():
    attention_mask = FiboTransformer._prepare_attention_mask(
        mx.array([[1, 0, 1]], dtype=mx.float32),
    )

    mask = np.array(attention_mask)
    assert mask.shape == (1, 1, 3, 3)
    assert np.isneginf(mask[0, 0, 0, 1])
    assert np.isneginf(mask[0, 0, 1, 0])
    assert mask[0, 0, 0, 2] == 0


def test_fibo_text_encoder_attention_mask_uses_negative_infinity():
    attention_mask = SmolLM3_3B_TextEncoder._build_attention_mask(
        mx.array([[1, 0, 1]], dtype=mx.int32),
    )

    mask = np.array(attention_mask)
    assert mask.shape == (1, 1, 3, 3)
    assert np.isneginf(mask[0, 0, 0, 1])
    assert np.isneginf(mask[0, 0, 0, 2])
    assert np.isneginf(mask[0, 0, 2, 1])
    assert mask[0, 0, 2, 0] == 0


def test_fibo_attention_mask_rejects_batch_mismatch():
    config = SimpleNamespace(height=16, width=16)
    prompt_attention_mask = mx.ones((1, 3), dtype=mx.float32)
    encoder_hidden_states = mx.zeros((2, 3, 4), dtype=mx.float32)

    with pytest.raises(ValueError, match="batch"):
        FiboTransformer._compute_attention_mask(
            batch_size=2,
            config=config,
            encoder_hidden_states=encoder_hidden_states,
            max_tokens=3,
            prompt_attention_mask=prompt_attention_mask,
        )


def test_fibo_block_health_fails_on_non_finite_tensor(monkeypatch):
    monkeypatch.setenv("MFLUX_FIBO_BLOCK_HEALTH", "1")
    config = SimpleNamespace(
        scheduler=SimpleNamespace(timesteps=mx.array([900.0])),
        num_inference_steps=1,
        guidance=5.0,
    )

    with pytest.raises(TensorHealthError, match="fibo\\.transformer\\.joint\\.7\\.hidden_states"):
        FiboTransformer._check_block_outputs(
            enabled=FiboTransformer._block_health_enabled(),
            block_name="joint.7",
            t=0,
            config=config,
            hidden_states=mx.array([[float("nan")]], dtype=mx.float32),
            encoder_hidden_states=mx.zeros((1, 1), dtype=mx.float32),
        )


def test_fibo_single_attention_broadcasts_mask_to_heads(monkeypatch):
    captured = {}

    def fake_sdpa(query, key, value, *, scale, mask):
        captured["mask_shape"] = mask.shape
        return mx.zeros_like(query)

    monkeypatch.setattr(fibo_single_attention_module, "scaled_dot_product_attention", fake_sdpa)
    attention = FiboSingleAttention(dim=8, num_attention_heads=2, attention_head_dim=4)
    hidden_states = mx.zeros((1, 3, 8), dtype=mx.float32)
    rotary = (mx.ones((3, 4), dtype=mx.float32), mx.zeros((3, 4), dtype=mx.float32))
    mask = mx.zeros((1, 1, 3, 3), dtype=mx.float32)

    attention(hidden_states=hidden_states, image_rotary_emb=rotary, attention_mask=mask)

    assert captured["mask_shape"] == (1, 2, 3, 3)


def test_fibo_single_attention_zeroes_fully_masked_query_rows():
    attention = FiboSingleAttention(dim=8, num_attention_heads=2, attention_head_dim=4)
    hidden_states = mx.ones((1, 3, 8), dtype=mx.bfloat16)
    rotary = (mx.ones((3, 4), dtype=mx.float32), mx.zeros((3, 4), dtype=mx.float32))
    mask = np.zeros((1, 1, 3, 3), dtype=np.float32)
    mask[:, :, 1, :] = -np.inf

    output = attention(
        hidden_states=hidden_states,
        image_rotary_emb=rotary,
        attention_mask=mx.array(mask),
    )

    output_np = np.array(output.astype(mx.float32))
    assert np.isfinite(output_np).all()
    assert np.allclose(output_np[:, 1, :], 0.0)


def test_fibo_joint_attention_zeroes_fully_masked_query_rows():
    attention = FiboJointAttention(dim=8, num_attention_heads=2, attention_head_dim=4)
    hidden_states = mx.ones((1, 2, 8), dtype=mx.bfloat16)
    encoder_hidden_states = mx.ones((1, 2, 8), dtype=mx.bfloat16)
    rotary = (mx.ones((4, 4), dtype=mx.float32), mx.zeros((4, 4), dtype=mx.float32))
    mask = np.zeros((1, 1, 4, 4), dtype=np.float32)
    mask[:, :, 0, :] = -np.inf

    hidden_output, encoder_output = attention(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        image_rotary_emb=rotary,
        attention_mask=mx.array(mask),
    )

    assert np.isfinite(np.array(hidden_output.astype(mx.float32))).all()
    encoder_np = np.array(encoder_output.astype(mx.float32))
    assert np.isfinite(encoder_np).all()


def test_fibo_transformer_norm_out_matches_diffusers_bias_contract():
    transformer = FiboTransformer()

    assert "bias" in transformer.norm_out.linear.parameters()


def test_fibo_weight_validation_rejects_old_prepared_transformer_without_norm_out_bias():
    weights = LoadedWeights(
        components={"transformer": {"norm_out": {"linear": {"weight": mx.zeros((2, 2))}}}},
        meta_data=MetaData(),
    )

    with pytest.raises(ValueError, match="norm_out\\.linear\\.bias"):
        FIBOInitializer._validate_weights(weights)


def test_fibo_weight_validation_accepts_nested_norm_out_bias():
    weights = LoadedWeights(
        components={"transformer": {"norm_out": {"linear": {"bias": mx.zeros((2,))}}}},
        meta_data=MetaData(),
    )

    FIBOInitializer._validate_weights(weights)


def test_fibo_weight_validation_accepts_flat_norm_out_bias():
    weights = LoadedWeights(
        components={"transformer": {"norm_out.linear.bias": mx.zeros((2,))}},
        meta_data=MetaData(),
    )

    FIBOInitializer._validate_weights(weights)


def test_fibo_weight_validation_rejects_quantized_norm_out_linear_layout():
    weights = LoadedWeights(
        components={
            "transformer": {
                "norm_out": {
                    "linear": {
                        "weight": mx.zeros((2, 2)),
                        "bias": mx.zeros((2,)),
                        "scales": mx.zeros((2, 1)),
                        "biases": mx.zeros((2, 1)),
                    }
                }
            }
        },
        meta_data=MetaData(quantization_level=8),
    )

    with pytest.raises(ValueError, match="norm_out\\.linear"):
        FIBOInitializer._validate_weights(weights)


def test_fibo_weight_validation_rejects_quantized_q8_sensitive_transformer_layout():
    weights = LoadedWeights(
        components={
            "transformer": {
                "norm_out": {"linear": {"weight": mx.zeros((2, 2)), "bias": mx.zeros((2,))}},
                "context_embedder": {
                    "weight": mx.zeros((2, 2)),
                    "scales": mx.zeros((2, 1)),
                    "biases": mx.zeros((2, 1)),
                },
            }
        },
        meta_data=MetaData(quantization_level=8),
    )

    with pytest.raises(ValueError, match="context_embedder"):
        FIBOInitializer._validate_weights(weights)


def test_fibo_weight_validation_rejects_quantized_vae_layout():
    weights = LoadedWeights(
        components={
            "transformer": {"norm_out": {"linear": {"bias": mx.zeros((2,))}}},
            "vae": {"decoder": {"mid": {"attn": {"weight": mx.zeros((2, 2)), "scales": mx.zeros((2, 1))}}}},
        },
        meta_data=MetaData(quantization_level=8),
    )

    with pytest.raises(ValueError, match="VAE"):
        FIBOInitializer._validate_weights(weights)


def test_fibo_q4_and_q8_keep_sensitive_paths_bf16():
    components = {component.name: component for component in FIBOWeightDefinition.get_components()}
    module = QuantizableModule()

    assert components["vae"].skip_quantization
    for bits in (4, 8):
        assert not FIBOWeightDefinition.quantization_predicate("x_embedder", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("context_embedder", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("time_embed.timestep_embedder.linear_1", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("caption_projection.0.linear", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("transformer_blocks.0.norm1.linear", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("transformer_blocks.0.norm1_context.linear", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("single_transformer_blocks.0.norm.linear", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("norm_out.linear", module, bits)
        assert not FIBOWeightDefinition.quantization_predicate("proj_out", module, bits)
        assert FIBOWeightDefinition.quantization_predicate("transformer_blocks.0.attn.to_q", module, bits)
        assert FIBOWeightDefinition.quantization_predicate("transformer_blocks.0.ff.net.2", module, bits)


def test_fibo_source_components_load_with_bf16_preserving_paths():
    components = {component.name: component for component in FIBOWeightDefinition.get_components()}

    assert components["vae"].loading_mode == "single"
    assert components["vae"].precision == mx.bfloat16
    assert components["transformer"].loading_mode == "multi_glob"
    assert components["transformer"].precision == mx.bfloat16
    assert components["text_encoder"].loading_mode == "multi_glob"
    assert components["text_encoder"].precision == mx.bfloat16


def test_fibo_download_patterns_include_scheduler_contract():
    patterns = set(FIBOWeightDefinition.get_download_patterns())

    assert "model_index.json" in patterns
    assert "scheduler/*.json" in patterns


def test_fibo_vae_decoder_uses_mlx_fork_temporal_contract():
    vae = Wan2_2_VAE()

    assert vae.decoder.temporal_upsample == []
    decoded = vae.decode(mx.zeros((1, vae.Z_DIM, 1, 4, 4), dtype=mx.bfloat16))
    assert decoded.shape == (1, 3, 1, 64, 64)


def test_smol_lm3_respects_transformers_nope_pattern():
    encoder = SmolLM3_3B_TextEncoder(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    assert [layer.self_attn.use_rope for layer in encoder.layers] == [True, True, True, False]


def test_smol_lm3_attention_skips_rope_when_nope_layer(monkeypatch):
    called = False

    def fake_apply_rope(q, k, cos, sin):
        nonlocal called
        called = True
        return q, k

    monkeypatch.setattr(smol_attention_module.SmolLM3_3B_SelfAttention, "_apply_rope", fake_apply_rope)
    attention = SmolLM3_3B_SelfAttention(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        use_rope=False,
    )
    hidden_states = mx.zeros((1, 3, 8), dtype=mx.float32)
    mask = mx.zeros((1, 1, 3, 3), dtype=mx.float32)
    rotary = (mx.ones((1, 1, 3, 4), dtype=mx.float32), mx.zeros((1, 1, 3, 4), dtype=mx.float32))

    attention(hidden_states=hidden_states, attention_mask=mask, cos_sin=rotary)

    assert called is False
