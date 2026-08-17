"""Availability guards for the SwiftVR parity harness.

Every fixture here is a *skip* condition, never a substitute. If torch, the reference
source tree or a checkpoint is absent the affected tests are skipped with the reason
printed; none of them fall back to a synthetic stand-in, because a parity test that
passes without the reference present would be worse than no test at all (ADR 0002).
"""

import pytest

from tests.swiftvr.parity.parity_support import (
    SWIFTVR_REFERENCE_ROOT,
    load_mlx_reae,
    load_torch_reae,
    reae_weights_available,
    reference_available,
    torch_available,
    transformer_checkpoint_status,
    transformer_weights_available,
)


def pytest_configure(config):
    config.addinivalue_line("markers", "parity: MLX-vs-torch numerical parity against real weights")


@pytest.fixture(scope="session")
def require_torch():
    if not torch_available():
        pytest.skip("torch is not importable")


@pytest.fixture(scope="session")
def require_reference(require_torch):
    if not reference_available():
        pytest.skip(f"SwiftVR reference source tree not found at {SWIFTVR_REFERENCE_ROOT}")


@pytest.fixture(scope="session")
def require_reae_weights(require_reference):
    if not reae_weights_available():
        pytest.skip("reae.safetensors is not downloaded")


@pytest.fixture(scope="session")
def require_transformer_weights(require_reference):
    if not transformer_weights_available():
        pytest.skip(f"SwiftVR DiT checkpoint unusable ({transformer_checkpoint_status()})")


@pytest.fixture(scope="session")
def torch_reae(require_reae_weights):
    """The reference ReAE with the published weights, built once per session."""
    return load_torch_reae()


@pytest.fixture(scope="session")
def mlx_reae(require_reae_weights):
    """The mflux ReAE with the published weights, built once per session."""
    return load_mlx_reae()


@pytest.fixture(scope="session")
def mlx_reae_bfloat16(require_reae_weights):
    """The mflux ReAE at ``ModelConfig.precision``, i.e. what a real run loads."""
    import mlx.core as mx

    return load_mlx_reae(dtype=mx.bfloat16)
