from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mflux.models.wan.model.wan_transformer.wan_activation import WanActivation
from mflux.models.wan.model.wan_transformer.wan_embedding import WanTimeTextImageEmbedding


class _FakeTimestepsProjection:
    def __call__(self, timestep: mx.array) -> mx.array:
        return timestep


class _FakeTimeEmbedder:
    def __init__(self, output: mx.array):
        self.output = output
        self.linear_1 = SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32))

    def __call__(self, timestep: mx.array) -> mx.array:
        return self.output


class _CapturingProjector:
    def __init__(self):
        self.last_input = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        self.last_input = hidden_states
        return hidden_states


class _IdentityTextEmbedder:
    def __call__(self, encoder_hidden_states: mx.array) -> mx.array:
        return encoder_hidden_states


def test_wan_time_text_image_embedding_projects_timesteps_from_casted_time_embedding():
    model = WanTimeTextImageEmbedding.__new__(WanTimeTextImageEmbedding)
    raw_temb = mx.array([[0.1001, 10.1234, -7.7777]], dtype=mx.float32)
    projector = _CapturingProjector()
    model.timesteps_proj = _FakeTimestepsProjection()
    model.time_embedder = _FakeTimeEmbedder(raw_temb)
    model.time_proj = projector
    model.text_embedder = _IdentityTextEmbedder()
    encoder_hidden_states = mx.zeros((1, 3), dtype=mx.bfloat16)

    temb, timestep_proj, returned_text = model(
        timestep=mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32),
        encoder_hidden_states=encoder_hidden_states,
    )

    casted_temb = raw_temb.astype(mx.bfloat16)
    expected_fp32_input = np.asarray(WanActivation.silu(casted_temb).astype(mx.float32))

    np.testing.assert_allclose(np.asarray(projector.last_input.astype(mx.float32)), expected_fp32_input)
    np.testing.assert_allclose(np.asarray(temb.astype(mx.float32)), np.asarray(casted_temb.astype(mx.float32)))
    expected_bf16_output = np.asarray(mx.array(expected_fp32_input, dtype=mx.float32).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_allclose(
        np.asarray(timestep_proj.astype(mx.float32)),
        expected_bf16_output,
    )
    assert temb.dtype == mx.bfloat16
    assert timestep_proj.dtype == mx.bfloat16
    assert returned_text.dtype == mx.bfloat16
