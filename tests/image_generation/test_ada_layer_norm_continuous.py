import mlx.core as mx

from mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous import AdaLayerNormContinuous


def test_ada_layer_norm_continuous_projects_conditioning_in_hidden_state_dtype(monkeypatch):
    module = AdaLayerNormContinuous(embedding_dim=4, conditioning_embedding_dim=4)
    captured = {}

    class _Linear:
        def __call__(self, x):
            captured["dtype"] = x.dtype
            return mx.zeros((x.shape[0], 8), dtype=x.dtype)

    monkeypatch.setattr(module, "linear", _Linear())

    hidden_states = mx.ones((1, 2, 4), dtype=mx.float16)
    conditioning = mx.ones((1, 4), dtype=mx.float32)

    output = module(hidden_states, conditioning)

    assert captured["dtype"] == mx.float16
    assert output.dtype == mx.float16
