import json
import subprocess
import sys

import pytest

# Dependency-creep gate (0088): `import mflux` must stay free of the heavy
# libraries below. Measured 2026-07-22 on M5 Max: the import dropped from
# ~1.34-1.62 s (cold-ish) / ~240 ms (warm) to ~60 ms warm once huggingface_hub
# (httpx+rich), PIL, and numpy left the module-scope import chain.
# numpy and mlx.core remain acceptable costs for modules that genuinely need
# them, but nothing on the plain `import mflux` chain does today.
FORBIDDEN_TOP_LEVEL_MODULES = (
    "torch",
    "transformers",
    "tokenizers",
    "matplotlib",
    "httpx",
    "huggingface_hub",
    "cv2",
    "av",
    "rich",
    # Achieved by 0088 (output_paths no longer routes through ImageUtil and
    # dimension_resolver defers PIL.Image): keep it locked in.
    "PIL",
)


def _modules_loaded_by(statement: str) -> list[str]:
    # Same interpreter as the test venv; a fresh subprocess gives a clean
    # sys.modules snapshot without pytest's own imports polluting it.
    result = subprocess.run(
        [sys.executable, "-c", f"{statement}; import sys, json; print(json.dumps(sorted(sys.modules)))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _forbidden_offenders(loaded_modules: list[str]) -> list[str]:
    return sorted(
        module
        for module in loaded_modules
        for forbidden in FORBIDDEN_TOP_LEVEL_MODULES
        if module == forbidden or module.startswith(f"{forbidden}.")
    )


@pytest.mark.fast
def test_import_mflux_stays_free_of_heavy_libraries():
    offenders = _forbidden_offenders(_modules_loaded_by("import mflux"))

    assert not offenders, (
        f"`import mflux` pulled forbidden heavy modules: {offenders}. "
        "Keep huggingface_hub/PIL/torch-class imports function-local (see backlog 0088)."
    )


@pytest.mark.fast
def test_import_mlxgen_stays_free_of_heavy_libraries():
    # `mlxgen` copies mflux's public names eagerly, so a name that resolves through a heavy
    # module has to stay lazy on both packages or the gate only holds for one of them.
    offenders = _forbidden_offenders(_modules_loaded_by("import mlxgen"))

    assert not offenders, (
        f"`import mlxgen` pulled forbidden heavy modules: {offenders}. "
        "Lazily-exported mflux names must be skipped by the mlxgen eager copy loop."
    )
