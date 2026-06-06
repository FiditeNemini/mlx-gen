import mlx.core as mx
import numpy as np

from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import _step


def test_flow_match_step_uses_float32_accumulation_before_output_cast():
    latents = mx.array([1.0], dtype=mx.float16)
    noise = mx.array([4.0], dtype=mx.float16)
    current_sigma = mx.array(1.0, dtype=mx.float32)
    next_sigma = mx.array(0.93, dtype=mx.float32)

    result = _step(noise=noise, latents=latents, s1=next_sigma, s2=current_sigma)
    mx.eval(result)

    expected = (latents.astype(mx.float32) + (next_sigma - current_sigma) * noise.astype(mx.float32)).astype(
        mx.float16
    )
    naive_half = latents + (next_sigma - current_sigma).astype(mx.float16) * noise

    np.testing.assert_array_equal(np.array(result), np.array(expected))
    assert not np.array_equal(np.array(result), np.array(naive_half))
