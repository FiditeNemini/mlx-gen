import mlx.core as mx
import numpy as np
import pytest
import torch

from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_transformer import wan_attention as wan_attention_module
from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from mflux.models.wan.variants.wan_bernini import BerniniRenderer


def _tiny_transformer(num_layers: int = 1) -> WanTransformer:
    return WanTransformer(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=12,
        freq_dim=8,
        ffn_dim=24,
        num_layers=num_layers,
        cross_attn_norm=True,
        rope_max_seq_len=16,
    )


def _official_source_rope(
    *,
    latent_shape: tuple[int, ...],
    patch_size: tuple[int, int, int],
    attention_head_dim: int,
    source_id: float,
    theta: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, frames, height, width = latent_shape
    p_t, p_h, p_w = patch_size
    grid = (frames // p_t, height // p_h, width // p_w)
    t_complex = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
    h_complex = attention_head_dim // 6
    w_complex = attention_head_dim // 6

    def phases(dim: int, positions: np.ndarray) -> np.ndarray:
        frequencies = 1.0 / (theta ** (np.arange(0, dim * 2, 2, dtype=np.float64) / (dim * 2)))
        return np.exp(1j * positions[:, None] * frequencies[None, :])

    temporal = phases(t_complex, np.arange(grid[0], dtype=np.float64))[:, None, None, :]
    vertical = phases(h_complex, np.arange(grid[1], dtype=np.float64))[None, :, None, :]
    horizontal = phases(w_complex, np.arange(grid[2], dtype=np.float64))[None, None, :, :]
    temporal = np.broadcast_to(temporal, (*grid, t_complex))
    vertical = np.broadcast_to(vertical, (*grid, h_complex))
    horizontal = np.broadcast_to(horizontal, (*grid, w_complex))
    spatial = np.concatenate([temporal, vertical, horizontal], axis=-1).reshape(1, -1, 1, attention_head_dim // 2)

    source_frequencies = 1.0 / (theta ** (np.arange(0, attention_head_dim, 2, dtype=np.float64) / attention_head_dim))
    source_phase = np.exp(1j * float(source_id) * source_frequencies).reshape(1, 1, 1, -1)
    combined = spatial * source_phase
    return np.repeat(combined.real, 2, axis=-1), np.repeat(combined.imag, 2, axis=-1)


@pytest.mark.parametrize(
    ("source_id", "attention_head_dim"),
    [(0.0, 12), (1.0, 12), (5.0, 12), (2.25, 12), (2.25, 128)],
)
def test_bernini_source_rotary_matches_official_complex_phase(source_id, attention_head_dim):
    patch_size = (1, 2, 2)
    latent = mx.zeros((1, 4, 3, 4, 6), dtype=mx.float32)
    rope = WanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=16,
    )

    actual_cos, actual_sin = rope(latent, source_id=source_id)
    expected_cos, expected_sin = _official_source_rope(
        latent_shape=latent.shape,
        patch_size=patch_size,
        attention_head_dim=attention_head_dim,
        source_id=source_id,
    )

    np.testing.assert_allclose(np.asarray(actual_cos), expected_cos, rtol=1e-6, atol=2e-6)
    np.testing.assert_allclose(np.asarray(actual_sin), expected_sin, rtol=1e-6, atol=2e-6)


def test_bernini_source_rotary_supports_ids_interpolated_for_eight_sources():
    latent = mx.zeros((1, 4, 1, 4, 4), dtype=mx.float32)
    rope = WanRotaryPosEmbed(attention_head_dim=12, patch_size=(1, 2, 2), max_seq_len=8)

    for source_id in BerniniRenderer._source_ids(8):
        actual_cos, actual_sin = rope(latent, source_id=source_id)
        expected_cos, expected_sin = _official_source_rope(
            latent_shape=latent.shape,
            patch_size=(1, 2, 2),
            attention_head_dim=12,
            source_id=source_id,
        )
        np.testing.assert_allclose(np.asarray(actual_cos), expected_cos, rtol=1e-6, atol=2e-6)
        np.testing.assert_allclose(np.asarray(actual_sin), expected_sin, rtol=1e-6, atol=2e-6)


def test_bernini_rotary_application_preserves_query_dtype_after_source_phase_mixing():
    rng = np.random.default_rng(12)
    hidden_states = mx.array(rng.normal(size=(1, 7, 3, 12)).astype(np.float32), dtype=mx.bfloat16)
    latent = mx.zeros((1, 4, 1, 2, 14), dtype=mx.float32)
    rope = WanRotaryPosEmbed(attention_head_dim=12, patch_size=(1, 2, 2), max_seq_len=16)
    freqs_cos, freqs_sin = rope(latent, source_id=2.25)

    actual = WanAttention._apply_rotary_emb(hidden_states, freqs_cos, freqs_sin)

    assert actual.dtype == hidden_states.dtype


def test_bernini_attention_qk_norm_and_sdpa_preserve_runtime_dtype(monkeypatch):
    class _Identity:
        def __call__(self, x):
            return x

    class _Recorder:
        def __init__(self):
            self.last_dtype = None

        def __call__(self, x):
            self.last_dtype = x.dtype
            return x

    attention = WanAttention(dim=8, heads=2, dim_head=4)
    attention.to_q = _Identity()
    attention.to_k = _Identity()
    attention.to_v = _Identity()
    attention.to_out = [_Identity()]
    attention.norm_q = _Recorder()
    attention.norm_k = _Recorder()
    recorded = {}

    def _fake_sdpa(query, key, value, *, scale):
        recorded["query_dtype"] = query.dtype
        recorded["key_dtype"] = key.dtype
        recorded["value_dtype"] = value.dtype
        recorded["scale"] = scale
        return mx.zeros_like(query)

    monkeypatch.setattr(wan_attention_module, "scaled_dot_product_attention", _fake_sdpa)

    hidden_states = mx.array(np.arange(24, dtype=np.float32).reshape(1, 3, 8), dtype=mx.bfloat16)

    output = attention(hidden_states)
    mx.eval(output)

    assert attention.norm_q.last_dtype == mx.float32
    assert attention.norm_k.last_dtype == mx.float32
    assert recorded == {
        "query_dtype": mx.float32,
        "key_dtype": mx.float32,
        "value_dtype": mx.bfloat16,
        "scale": attention.scale,
    }
    assert output.dtype == hidden_states.dtype


def test_bernini_attention_diffusers_rmsnorm_matches_bf16_weight_contract():
    attention = WanAttention(dim=8, heads=2, dim_head=4)
    hidden_states = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8)

    actual = WanAttention._apply_diffusers_rms_norm(hidden_states, attention.norm_q)
    variance = mx.mean(mx.square(hidden_states.astype(mx.float32)), axis=-1, keepdims=True)
    expected = hidden_states.astype(mx.float32) * mx.rsqrt(variance + attention.norm_q.eps)
    expected = expected.astype(attention.norm_q.weight.dtype) * attention.norm_q.weight
    mx.eval(actual, expected)

    assert actual.dtype == attention.norm_q.weight.dtype
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
        rtol=0,
        atol=0,
    )


def test_bernini_attention_real_norms_keep_runtime_value_dtype_and_float32_qk_into_sdpa(monkeypatch):
    class _Identity:
        def __call__(self, x):
            return x

    attention = WanAttention(dim=8, heads=2, dim_head=4)
    attention.to_q = _Identity()
    attention.to_k = _Identity()
    attention.to_v = _Identity()
    attention.to_out = [_Identity()]
    recorded = {}

    def _fake_sdpa(query, key, value, *, scale):
        recorded["query_dtype"] = query.dtype
        recorded["key_dtype"] = key.dtype
        recorded["value_dtype"] = value.dtype
        return mx.zeros_like(query)

    monkeypatch.setattr(wan_attention_module, "scaled_dot_product_attention", _fake_sdpa)

    hidden_states = mx.array(np.arange(24, dtype=np.float32).reshape(1, 3, 8), dtype=mx.bfloat16)
    output = attention(hidden_states)
    mx.eval(output)

    assert recorded == {
        "query_dtype": mx.float32,
        "key_dtype": mx.float32,
        "value_dtype": mx.bfloat16,
    }
    assert output.dtype == hidden_states.dtype


def test_bernini_condition_embedder_linear_matches_torch_bf16_accumulation_contract():
    embedding = WanTimeTextImageEmbedding(dim=8, time_freq_dim=4, time_proj_dim=12, text_embed_dim=6)
    weight = np.zeros((12, 8), dtype=np.float32)
    weight[0, :4] = [0.5, -0.25, 0.125, -0.0625]
    weight[1, :4] = [-0.5, 0.25, -0.125, 0.0625]
    weight[2, 4:] = [0.75, -0.375, 0.1875, -0.09375]
    weight[3, 4:] = [-0.75, 0.375, -0.1875, 0.09375]
    bias = np.array(
        [0.25, -0.25, 0.125, -0.125, 0.0, 0.5, -0.5, 0.0625, -0.0625, 0.0, 0.03125, -0.03125],
        dtype=np.float32,
    )
    embedding.time_proj.weight = mx.array(weight, dtype=mx.bfloat16)
    embedding.time_proj.bias = mx.array(bias, dtype=mx.bfloat16)
    hidden_states = mx.array(
        [
            [0.5, -1.0, 1.5, -2.0, 0.25, -0.5, 0.75, -1.0],
            [-0.5, 1.0, -1.5, 2.0, -0.25, 0.5, -0.75, 1.0],
        ],
        dtype=mx.bfloat16,
    )

    actual = WanTimeTextImageEmbedding._apply_torch_low_precision_linear(embedding.time_proj, hidden_states)

    expected = np.asarray(hidden_states.astype(mx.float32)) @ weight.T + bias
    expected = torch.from_numpy(expected).to(torch.bfloat16).float().numpy()

    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("input_dtype", [mx.bfloat16, mx.float32])
def test_bernini_patch_embedding_matches_torch_bf16_accumulation_contract(input_dtype):
    model = _tiny_transformer(num_layers=1)
    weight_shape = model.patch_embedding.weight.shape
    bias_shape = model.patch_embedding.bias.shape
    weight = (((np.arange(np.prod(weight_shape), dtype=np.float32) % 9) - 4) / 16).reshape(weight_shape)
    bias = (((np.arange(np.prod(bias_shape), dtype=np.float32) % 7) - 3) / 16).reshape(bias_shape)
    model.patch_embedding.weight = mx.array(weight, dtype=mx.bfloat16)
    model.patch_embedding.bias = mx.array(bias, dtype=mx.bfloat16)
    hidden_states = mx.array(
        (
            ((np.arange(1 * 4 * 2 * 4 * 4, dtype=np.float32) % 11) - 5) / 8
        ).reshape(1, 4, 2, 4, 4),
        dtype=input_dtype,
    )

    actual = model._patch_embed(hidden_states)

    conv = torch.nn.Conv3d(4, model.inner_dim, kernel_size=model.patch_size, stride=model.patch_size, bias=True).to(
        dtype=torch.bfloat16
    )
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(np.transpose(weight, (0, 4, 1, 2, 3))).to(dtype=torch.bfloat16))
        conv.bias.copy_(torch.from_numpy(bias).to(dtype=torch.bfloat16))
        expected = conv(torch.from_numpy(np.asarray(hidden_states.astype(mx.float32))).to(dtype=torch.bfloat16))
        expected = expected.flatten(2).transpose(1, 2).float().numpy()

    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        expected,
        rtol=0,
        atol=0,
    )


def test_bernini_packed_single_target_matches_ordinary_wan_forward():
    mx.random.seed(31)
    model = _tiny_transformer(num_layers=2)
    latent = mx.random.normal((1, 4, 2, 4, 6), dtype=mx.float32)
    timestep = mx.array([700.0], dtype=mx.float32)
    encoder_hidden_states = mx.random.normal((1, 5, 12), dtype=mx.float32)

    ordinary = model(latent, timestep, encoder_hidden_states)
    packed = model.forward_packed(
        latent_segments=[latent],
        source_ids=[0.0],
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    mx.eval(ordinary, packed)

    np.testing.assert_array_equal(np.asarray(packed), np.asarray(ordinary))


def test_bernini_packed_heterogeneous_segments_extract_and_unpatch_only_target():
    mx.random.seed(47)
    model = _tiny_transformer(num_layers=0)
    first_reference = mx.random.normal((1, 4, 1, 4, 4), dtype=mx.float32)
    second_reference = mx.random.normal((1, 4, 2, 6, 4), dtype=mx.float32)
    target = mx.random.normal((1, 4, 3, 4, 6), dtype=mx.float32)
    timestep = mx.array([500.0], dtype=mx.float32)
    encoder_hidden_states = mx.random.normal((1, 5, 12), dtype=mx.float32)

    packed = model.forward_packed(
        latent_segments=[first_reference, second_reference, target],
        source_ids=[1.0, 2.5, 0.0],
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    target_only = model(target, timestep, encoder_hidden_states)
    mx.eval(packed, target_only)

    assert packed.shape == target.shape
    np.testing.assert_array_equal(np.asarray(packed), np.asarray(target_only))


def test_bernini_packed_heterogeneous_segments_run_through_attention():
    mx.random.seed(53)
    model = _tiny_transformer(num_layers=1)
    reference = mx.random.normal((1, 4, 1, 4, 4), dtype=mx.float32)
    target = mx.random.normal((1, 4, 2, 4, 6), dtype=mx.float32)
    timestep = mx.array([500.0], dtype=mx.float32)
    encoder_hidden_states = mx.random.normal((1, 5, 12), dtype=mx.float32)

    with_reference = model.forward_packed(
        latent_segments=[reference, target],
        source_ids=[1.0, 0.0],
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    with_zero_reference = model.forward_packed(
        latent_segments=[mx.zeros_like(reference), target],
        source_ids=[1.0, 0.0],
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    mx.eval(with_reference, with_zero_reference)

    assert with_reference.shape == target.shape
    assert bool(mx.all(mx.isfinite(with_reference)).item())
    assert float(mx.max(mx.abs(with_reference - with_zero_reference)).item()) > 1e-6


@pytest.mark.parametrize(
    ("source_ids", "target_segment_index", "message"),
    [
        ([1.0], 0, "target segment must use source ID 0"),
        ([0.0, 0.0], 1, "Conditioning segments must use positive source IDs"),
        ([float("nan")], 0, "must be finite"),
    ],
)
def test_bernini_packed_rejects_invalid_source_roles(source_ids, target_segment_index, message):
    model = _tiny_transformer(num_layers=0)
    latent = mx.zeros((1, 4, 1, 4, 4), dtype=mx.float32)
    segments = [latent] * len(source_ids)

    with pytest.raises(ValueError, match=message):
        model.forward_packed(
            latent_segments=segments,
            source_ids=source_ids,
            target_segment_index=target_segment_index,
            timestep=mx.array([500.0], dtype=mx.float32),
            encoder_hidden_states=mx.zeros((1, 5, 12), dtype=mx.float32),
        )


def test_bernini_packed_rejects_non_patch_aligned_segment():
    model = _tiny_transformer(num_layers=0)

    with pytest.raises(ValueError, match="must be divisible by patch size"):
        model.forward_packed(
            latent_segments=[mx.zeros((1, 4, 1, 5, 4), dtype=mx.float32)],
            source_ids=[0.0],
            timestep=mx.array([500.0], dtype=mx.float32),
            encoder_hidden_states=mx.zeros((1, 5, 12), dtype=mx.float32),
        )
