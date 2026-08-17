"""Shared plumbing for the SwiftVR MLX-vs-torch numerical parity harness.

Every test in this package runs the *same weights* through the *same inputs* on both
implementations and compares the outputs. Nothing here fabricates a model, a weight or a
tensor of expected results: the torch side is the upstream SwiftVR source tree, the MLX
side is ``mflux.models.swiftvr``, and both read the published ``H-oliday/SwiftVR``
checkpoint off disk.

Importing the reference
-----------------------
``swiftvr/__init__.py`` pulls in ``decord``, a video reader with no arm64 macOS wheel and
no bearing on numerics. Rather than stub it - a stub is a lie the harness would have to
keep telling - :func:`torch_reference` binds a package object with the right ``__path__``
and never executes that ``__init__``. Submodules import normally, relative imports
resolve, and every line of reference *math* is the real one.

Layout
------
The reference is channels-first ``[N, T, C, H, W]``; mflux is channels-last
``[B, T, H, W, C]``. :func:`nchw_to_nhwc` and :func:`nhwc_to_nchw` convert whole rank-5
clips, and comparisons always happen in the reference's layout so a transpose bug cannot
cancel itself out.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

# Both locations can be redirected, so the harness runs on a machine that keeps the
# reference or the snapshot somewhere else - and so the skip path itself is testable by
# pointing them at an empty directory (see tests/swiftvr/test_parity_gating.py).
SWIFTVR_REFERENCE_ROOT = Path(os.environ.get("SWIFTVR_PARITY_REFERENCE", "/tmp/svr/SwiftVR-main"))
SWIFTVR_SNAPSHOT = Path(
    os.environ.get(
        "SWIFTVR_PARITY_SNAPSHOT",
        "/Users/albou/.cache/huggingface/hub/models--H-oliday--SwiftVR/snapshots/743ed2530c550764905400f38eb6cc41af5abc80",
    )
)
REAE_CHECKPOINT = SWIFTVR_SNAPSHOT / "reae.safetensors"
TRANSFORMER_CHECKPOINT = SWIFTVR_SNAPSHOT / "transformer" / "diffusion_pytorch_model.safetensors"
PROMPT_EMBEDDING_CHECKPOINT = SWIFTVR_SNAPSHOT / "prompt_embedding.safetensors"

# Byte size of the fully downloaded DiT. A shorter file is a partial download, which
# safetensors will happily open and then fail on deep inside a tensor read.
TRANSFORMER_CHECKPOINT_BYTES = 19_999_235_584


def torch_available() -> bool:
    """Whether torch imports. The repo depends on torch>=2.8 but tests must not assume it."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def reference_available() -> bool:
    """Whether the upstream SwiftVR source tree is present and importable."""
    return (SWIFTVR_REFERENCE_ROOT / "swiftvr" / "models" / "reae.py").is_file()


def reae_weights_available() -> bool:
    """Whether ``reae.safetensors`` is on disk."""
    return REAE_CHECKPOINT.is_file()


def transformer_weights_available() -> bool:
    """Whether the DiT checkpoint is on disk *and* complete.

    A partial download is the failure mode worth guarding: the file exists, the header
    parses, and the error only surfaces once a tensor near the end is read.
    """
    if not TRANSFORMER_CHECKPOINT.is_file():
        return False
    return TRANSFORMER_CHECKPOINT.stat().st_size == TRANSFORMER_CHECKPOINT_BYTES


def transformer_checkpoint_status() -> str:
    """Human-readable reason the DiT checkpoint is or is not usable."""
    if not TRANSFORMER_CHECKPOINT.is_file():
        return f"missing: {TRANSFORMER_CHECKPOINT}"
    size = TRANSFORMER_CHECKPOINT.stat().st_size
    if size != TRANSFORMER_CHECKPOINT_BYTES:
        pct = 100.0 * size / TRANSFORMER_CHECKPOINT_BYTES
        return f"incomplete: {size} of {TRANSFORMER_CHECKPOINT_BYTES} bytes ({pct:.1f}%)"
    return "complete"


@lru_cache(maxsize=1)
def torch_reference() -> types.ModuleType:
    """Bind and return the upstream ``swiftvr`` package without running its ``__init__``.

    The top-level ``__init__`` imports ``decord``; the submodules that hold the math do
    not. Registering a package object carrying only ``__path__`` lets ``import
    swiftvr.models.reae`` resolve through the normal machinery while that ``__init__``
    never executes.

    Raises:
        FileNotFoundError: If the reference source tree is absent.
    """
    if not reference_available():
        raise FileNotFoundError(f"SwiftVR reference source not found at {SWIFTVR_REFERENCE_ROOT}")
    existing = sys.modules.get("swiftvr")
    if existing is not None and getattr(existing, "_mflux_parity_shim", False):
        return existing
    package = types.ModuleType("swiftvr")
    package.__path__ = [str(SWIFTVR_REFERENCE_ROOT / "swiftvr")]
    package.__package__ = "swiftvr"
    package._mflux_parity_shim = True
    sys.modules["swiftvr"] = package
    return package


# --------------------------------------------------------------------------- #
# Layout conversion                                                           #
# --------------------------------------------------------------------------- #


def nchw_to_nhwc(clip: Any) -> np.ndarray:
    """``[N, T, C, H, W]`` -> ``[N, T, H, W, C]``. Accepts numpy, mx or torch input."""
    array = to_numpy(clip)
    if array.ndim != 5:
        raise ValueError(f"Expected a rank-5 clip, got shape {array.shape}.")
    return np.ascontiguousarray(np.transpose(array, (0, 1, 3, 4, 2)))


def nhwc_to_nchw(clip: Any) -> np.ndarray:
    """``[N, T, H, W, C]`` -> ``[N, T, C, H, W]``. Accepts numpy, mx or torch input.

    Taking any of the three matters: an mflux output is a possibly-bfloat16 ``mx.array``
    that numpy cannot view directly, and silently requiring a pre-converted array would
    make every call site responsible for a conversion it can get subtly wrong.
    """
    array = to_numpy(clip)
    if array.ndim != 5:
        raise ValueError(f"Expected a rank-5 clip, got shape {array.shape}.")
    return np.ascontiguousarray(np.transpose(array, (0, 1, 4, 2, 3)))


def to_numpy(value: Any) -> np.ndarray:
    """Materialize an mx.array or a torch.Tensor as a float64 numpy array.

    float64 is used for the *comparison* only, so the metric itself contributes no
    rounding to a number the test then thresholds.
    """
    if isinstance(value, mx.array):
        return np.asarray(value.astype(mx.float32), dtype=np.float64)
    if hasattr(value, "detach"):
        return value.detach().to("cpu").to(dtype=__import__("torch").float32).numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Comparison metrics                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParityResult:
    """Element-wise agreement between a candidate tensor and a reference tensor.

    Attributes:
        max_abs_diff: Largest absolute deviation, in the units of the tensor.
        mean_abs_diff: Mean absolute deviation.
        cosine: Cosine similarity of the flattened tensors. Sensitive to direction and
            insensitive to scale, which is why it is reported alongside, never instead
            of, the absolute figures.
        relative_max: ``max_abs_diff`` divided by the reference's largest magnitude, the
            scale-free number to compare against a tolerance.
        reference_absmax: Largest magnitude in the reference, for context.
        shape: Shape both tensors share.
    """

    max_abs_diff: float
    mean_abs_diff: float
    cosine: float
    relative_max: float
    reference_absmax: float
    shape: tuple[int, ...]

    def __str__(self) -> str:
        return (
            f"shape={self.shape} max_abs={self.max_abs_diff:.3e} mean_abs={self.mean_abs_diff:.3e} "
            f"rel_max={self.relative_max:.3e} cosine={self.cosine:.8f} ref_absmax={self.reference_absmax:.3e}"
        )


def compare(candidate: Any, reference: Any) -> ParityResult:
    """Compare a candidate against a reference, both converted to float64 numpy.

    Raises:
        ValueError: If the shapes differ, or if either side holds a non-finite value.
            A NaN silently poisons every metric, so it is a hard error rather than an
            interesting number.
    """
    lhs = to_numpy(candidate)
    rhs = to_numpy(reference)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: candidate {lhs.shape} vs reference {rhs.shape}.")
    if not np.isfinite(lhs).all():
        raise ValueError(f"Candidate holds {int((~np.isfinite(lhs)).sum())} non-finite values.")
    if not np.isfinite(rhs).all():
        raise ValueError(f"Reference holds {int((~np.isfinite(rhs)).sum())} non-finite values.")

    diff = np.abs(lhs - rhs)
    flat_lhs = lhs.reshape(-1)
    flat_rhs = rhs.reshape(-1)
    denominator = float(np.linalg.norm(flat_lhs) * np.linalg.norm(flat_rhs))
    cosine = float(np.dot(flat_lhs, flat_rhs) / denominator) if denominator > 0.0 else 1.0
    reference_absmax = float(np.abs(rhs).max())
    return ParityResult(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        cosine=cosine,
        relative_max=float(diff.max() / reference_absmax) if reference_absmax > 0.0 else float(diff.max()),
        reference_absmax=reference_absmax,
        shape=lhs.shape,
    )


def assert_parity(
    candidate: Any,
    reference: Any,
    *,
    label: str,
    max_relative: float,
    min_cosine: float,
) -> ParityResult:
    """Compare and fail with the full metric line when either bound is breached.

    Args:
        candidate: The MLX output.
        reference: The torch output.
        label: What is being compared, echoed in the failure message.
        max_relative: Upper bound on ``max_abs_diff / reference_absmax``.
        min_cosine: Lower bound on cosine similarity.

    Returns:
        The measured :class:`ParityResult`, so a passing test can still report numbers.
    """
    result = compare(candidate, reference)
    failures = []
    if result.relative_max > max_relative:
        failures.append(f"relative_max {result.relative_max:.3e} > {max_relative:.3e}")
    if result.cosine < min_cosine:
        failures.append(f"cosine {result.cosine:.10f} < {min_cosine:.10f}")
    if failures:
        raise AssertionError(f"{label} parity FAILED ({'; '.join(failures)})\n  measured: {result}")
    return result


# --------------------------------------------------------------------------- #
# Model construction                                                          #
# --------------------------------------------------------------------------- #


def load_torch_reae():
    """Build the reference ReAE and load ``reae.safetensors`` with ``strict=True``.

    ``strict=True`` is the reference's own contract: any key mismatch raises here rather
    than leaving a layer at its random initialisation, which would make every downstream
    parity number meaningless.
    """
    import torch

    torch_reference()
    from swiftvr.models.reae import ReAE as TorchReAE

    model = TorchReAE(checkpoint_path=str(REAE_CHECKPOINT))
    model.eval()
    model.to(torch.float32)
    return model


def load_mlx_reae(dtype: "mx.Dtype | None" = None):
    """Build the mflux ReAE and apply ``reae.safetensors`` through the repo's own mapping.

    Uses :class:`SwiftVRWeightMapping` and :class:`WeightMapper` rather than a
    hand-written key translation, so the harness validates the mapping the product
    actually ships. Coverage is asserted both ways: a missing key would leave a random
    tensor in place and an extra key would mean the mapping targets something the module
    tree does not have.

    Args:
        dtype: Cast the weights to this dtype after mapping. ``None`` keeps the
            checkpoint's float32, which isolates the graph from precision; pass
            ``mx.bfloat16`` to reproduce what ``ModelConfig.precision`` ships.

    Raises:
        ValueError: If mapped keys and model parameters do not correspond exactly.
    """
    from mlx.utils import tree_flatten, tree_map

    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from mflux.models.swiftvr.model.swiftvr_reae.reae import ReAE
    from mflux.models.swiftvr.weights.swiftvr_weight_mapping import SwiftVRWeightMapping

    raw = mx.load(str(REAE_CHECKPOINT))
    model = ReAE()
    mapped = WeightMapper.apply_mapping(raw, SwiftVRWeightMapping.get_reae_mapping())
    provided = {key for key, _ in tree_flatten(mapped)}
    expected = {key for key, _ in tree_flatten(model.parameters())}
    if provided != expected:
        raise ValueError(
            f"ReAE weight coverage mismatch: missing {sorted(expected - provided)}, extra {sorted(provided - expected)}"
        )
    if dtype is not None:
        mapped = tree_map(lambda array: array.astype(dtype), mapped)
    model.update(mapped)
    mx.eval(model.parameters())
    return model


def seeded_clip(
    *,
    frames: int,
    height: int,
    width: int,
    channels: int = 3,
    seed: int = 0,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """A deterministic ``[1, T, C, H, W]`` float32 clip in ``[low, high]``.

    Structured rather than pure noise: a moving radial gradient plus a mild texture. Pure
    uniform noise is a pathological input for an autoencoder trained on natural video -
    it lands far outside the data manifold, so agreement there says little about
    agreement on real frames, and disagreement is hard to attribute.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float64).reshape(frames, 1, 1, 1)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64).reshape(1, 1, height, 1)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64).reshape(1, 1, 1, width)
    c = np.arange(channels, dtype=np.float64).reshape(1, channels, 1, 1)

    centre_x = 0.6 * np.sin(0.35 * t + 0.4 * c)
    centre_y = 0.6 * np.cos(0.27 * t + 0.2 * c)
    radius = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2)
    signal = 0.5 + 0.35 * np.cos(4.0 * radius + 0.3 * t) * np.exp(-0.8 * radius)
    signal = signal + 0.04 * rng.standard_normal((frames, channels, height, width))

    clip = np.clip(signal, 0.0, 1.0)
    clip = low + (high - low) * clip
    return np.ascontiguousarray(clip[None].astype(np.float32))


def seeded_normal(shape: tuple[int, ...], *, seed: int, scale: float = 1.0) -> np.ndarray:
    """Deterministic float32 Gaussian noise of the given shape."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray((scale * rng.standard_normal(shape)).astype(np.float32))


def paired_chunk_specs(total_frames: int, clip_len: int) -> list[tuple[Any, Any]]:
    """Build the chunk schedule on both sides and assert the two agree field for field.

    The reference and mflux each define their own ``ChunkType`` enum. They are distinct
    Python classes, so ``mflux_spec.ctype == torch_ChunkType.LAST`` is silently False and
    a reference routine handed an mflux spec would quietly take the wrong branch. Each
    implementation is therefore given its own spec object, and this function is the place
    that proves the two schedules are the same schedule.

    Returns:
        ``(torch_spec, mflux_spec)`` pairs in chunk order.

    Raises:
        AssertionError: If the two schedules disagree in length or in any field.
    """
    torch_reference()
    from swiftvr.streaming.chunk import build_chunk_specs as reference_build

    from mflux.models.swiftvr.streaming.chunk import build_chunk_specs as mflux_build

    reference_specs = reference_build(total_frames, clip_len)
    mflux_specs = mflux_build(total_frames, clip_len)
    assert len(reference_specs) == len(mflux_specs), (
        f"chunk schedules differ in length for t={total_frames}, clip_len={clip_len}: "
        f"{len(reference_specs)} reference vs {len(mflux_specs)} mflux"
    )
    for reference_spec, mflux_spec in zip(reference_specs, mflux_specs):
        reference_fields = (
            reference_spec.ctype.value,
            reference_spec.frame_start,
            reference_spec.frame_count,
            reference_spec.b,
            reference_spec.clip_idx,
            reference_spec.is_first_decode,
        )
        mflux_fields = (
            mflux_spec.ctype.value,
            mflux_spec.frame_start,
            mflux_spec.frame_count,
            mflux_spec.b,
            mflux_spec.clip_idx,
            mflux_spec.is_first_decode,
        )
        assert reference_fields == mflux_fields, (
            f"chunk spec mismatch for t={total_frames}, clip_len={clip_len}: "
            f"reference {reference_fields} vs mflux {mflux_fields}"
        )
    return list(zip(reference_specs, mflux_specs))
