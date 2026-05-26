import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.variants import Wan2_2_TI2V

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources" / "wan" / "parity"
WAN_MODEL_ENV = "MFLUX_WAN_PARITY_MODEL"
RUN_PARITY_ENV = "MFLUX_RUN_LOCAL_WAN_PARITY"


pytestmark = pytest.mark.skipif(
    os.getenv(RUN_PARITY_ENV) != "1",
    reason=f"set {RUN_PARITY_ENV}=1 to run full local Wan parity tests",
)


@pytest.fixture(scope="module")
def wan_model() -> Wan2_2_TI2V:
    return Wan2_2_TI2V(model_path=os.getenv(WAN_MODEL_ENV, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))


def test_wan_full_transformer_matches_diffusers_fixture(wan_model):
    hidden = _load_mx("wan_full_parity_hidden.npy").astype(ModelConfig.precision)
    text = _load_mx("wan_full_parity_text.npy").astype(ModelConfig.precision)
    timestep = _load_mx("wan_full_parity_timestep.npy")
    expected = _load_np("wan_full_parity_torch_output.npy")

    output = wan_model.transformer(hidden_states=hidden, timestep=timestep, encoder_hidden_states=text)
    mx.eval(output)

    _assert_close(_to_np(output), expected, mean_abs_limit=0.02, max_abs_limit=0.08)


def test_wan_vae_encoder_matches_diffusers_fixture(wan_model):
    image = _load_mx("wan_vae_gradient_input.npy").astype(ModelConfig.precision)
    expected = _load_np("wan_vae_gradient_diffusers_latent.npy")

    output = wan_model.vae.encode(image)
    mx.eval(output)

    _assert_close(_to_np(output), expected, mean_abs_limit=0.005, max_abs_limit=0.03)


def _load_np(name: str) -> np.ndarray:
    return np.load(RESOURCE_DIR / name)


def _load_mx(name: str) -> mx.array:
    return mx.array(_load_np(name))


def _to_np(value: mx.array) -> np.ndarray:
    return np.array(value.astype(mx.float32))


def _assert_close(actual: np.ndarray, expected: np.ndarray, mean_abs_limit: float, max_abs_limit: float) -> None:
    assert actual.shape == expected.shape
    delta = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    mean_abs = float(delta.mean())
    max_abs = float(delta.max())
    assert mean_abs <= mean_abs_limit, f"mean_abs={mean_abs:.6f} limit={mean_abs_limit:.6f}"
    assert max_abs <= max_abs_limit, f"max_abs={max_abs:.6f} limit={max_abs_limit:.6f}"
