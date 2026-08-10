import mlx.core as mx
import numpy as np
import torch

from mflux.models.wan.model.wan_transformer.wan_fp32_layer_norm import FP32LayerNorm


def _torch_fp32_layer_norm(x: np.ndarray, weight: np.ndarray | None, bias: np.ndarray | None, eps: float) -> np.ndarray:
    x_t = torch.from_numpy(x)
    weight_t = None if weight is None else torch.from_numpy(weight)
    bias_t = None if bias is None else torch.from_numpy(bias)
    y = torch.nn.functional.layer_norm(
        x_t.float(),
        (x.shape[-1],),
        None if weight_t is None else weight_t.float(),
        None if bias_t is None else bias_t.float(),
        eps,
    )
    return y.to(x_t.dtype).numpy()


def test_fp32_layer_norm_matches_torch_without_affine():
    x = np.random.default_rng(0).standard_normal((2, 5, 7)).astype(np.float32)
    layer = FP32LayerNorm(7, eps=1e-6, affine=False)

    actual = np.asarray(layer(mx.array(x, dtype=mx.bfloat16)).astype(mx.float32))
    expected = _torch_fp32_layer_norm(x.astype(np.float32), None, None, 1e-6)

    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)


def test_fp32_layer_norm_matches_torch_with_affine():
    x = np.random.default_rng(1).standard_normal((2, 3, 11)).astype(np.float32)
    weight = np.random.default_rng(2).standard_normal((11,)).astype(np.float32)
    bias = np.random.default_rng(3).standard_normal((11,)).astype(np.float32)
    layer = FP32LayerNorm(11, eps=1e-6, affine=True)
    layer.weight = mx.array(weight)
    layer.bias = mx.array(bias)

    actual = np.asarray(layer(mx.array(x, dtype=mx.bfloat16)).astype(mx.float32))
    expected = _torch_fp32_layer_norm(x.astype(np.float32), weight, bias, 1e-6)

    np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3)
