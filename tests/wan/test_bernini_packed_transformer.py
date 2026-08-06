import mlx.core as mx
import numpy as np
import pytest

from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_transformer.wan_embedding import WanRotaryPosEmbed
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
