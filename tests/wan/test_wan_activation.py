import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from mflux.models.wan.model.wan_transformer.wan_activation import WanActivation


def test_wan_gelu_tanh_matches_torch_reference():
    values = np.linspace(-6.0, 6.0, 49, dtype=np.float32)
    x = mx.array(values, dtype=mx.float32)

    actual = np.array(WanActivation.gelu_tanh(x))
    expected = F.gelu(torch.tensor(values), approximate="tanh").numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_wan_gelu_tanh_keeps_large_bfloat16_inputs_finite():
    x = mx.array([128.0, -128.0, 256.0, -256.0], dtype=mx.bfloat16)

    actual = WanActivation.gelu_tanh(x)

    assert actual.dtype == mx.bfloat16
    assert bool(mx.all(mx.isfinite(actual)).item())


def test_wan_gelu_tanh_matches_torch_bfloat16_castback_contract():
    values = np.linspace(-6.0, 6.0, 49, dtype=np.float32)
    x = mx.array(values, dtype=mx.bfloat16)

    actual = np.asarray(WanActivation.gelu_tanh(x).astype(mx.float32))
    expected = F.gelu(torch.tensor(values, dtype=torch.float32), approximate="tanh").to(torch.bfloat16).float().numpy()

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / 1024)


def test_wan_silu_matches_torch_bfloat16_castback_contract():
    values = np.linspace(-6.0, 6.0, 49, dtype=np.float32)
    x = mx.array(values, dtype=mx.bfloat16)

    actual = np.asarray(WanActivation.silu(x).astype(mx.float32))
    expected = F.silu(torch.tensor(values, dtype=torch.float32)).to(torch.bfloat16).float().numpy()

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
