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
WAN_PARITY_PROMPT = "A short cinematic video of a glowing orange glass sphere floating above calm teal water"
WAN_PARITY_NEGATIVE_PROMPT = "blur, low quality, distorted, text, watermark, noisy"
WAN_PARITY_PROMPT_LENGTH = 64
WAN_PARITY_PROMPT_MEAN_ABS_LIMIT = 0.001
WAN_PARITY_PROMPT_MAX_ABS_LIMIT = 0.01


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


def test_wan_vae_decode_matches_diffusers_fixture(wan_model):
    latents = _load_mx("wan_vae_decode_normalized_input.npy").astype(ModelConfig.precision)
    expected = _load_np("wan_vae_decode_diffusers_output.npy")

    output = wan_model.vae.decode_normalized_latents(latents)
    mx.eval(output)

    _assert_close(_to_np(output), expected, mean_abs_limit=0.006, max_abs_limit=0.04)


def test_wan_prompt_embeds_match_diffusers_fixture(wan_model):
    expected_prompt = _load_np("wan_prompt_embeds_diffusers.npy")
    expected_negative = _load_np("wan_negative_prompt_embeds_diffusers.npy")

    prompt, negative = wan_model.encode_prompt(
        prompt=WAN_PARITY_PROMPT,
        negative_prompt=WAN_PARITY_NEGATIVE_PROMPT,
        do_classifier_free_guidance=True,
        max_sequence_length=WAN_PARITY_PROMPT_LENGTH,
    )
    mx.eval(prompt, negative)

    _assert_close(
        _to_np(prompt),
        expected_prompt,
        mean_abs_limit=WAN_PARITY_PROMPT_MEAN_ABS_LIMIT,
        max_abs_limit=WAN_PARITY_PROMPT_MAX_ABS_LIMIT,
    )
    _assert_close(
        _to_np(negative),
        expected_negative,
        mean_abs_limit=WAN_PARITY_PROMPT_MEAN_ABS_LIMIT,
        max_abs_limit=WAN_PARITY_PROMPT_MAX_ABS_LIMIT,
    )


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
