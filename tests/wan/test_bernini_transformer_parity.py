import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "bernini_transformer_parity.py"


def _load_parity_class():
    spec = importlib.util.spec_from_file_location("bernini_transformer_parity", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BerniniTransformerParity


BerniniTransformerParity = _load_parity_class()


def test_bernini_transformer_parity_inputs_are_pinned(tmp_path):
    inputs_path = tmp_path / "inputs.npz"

    BerniniTransformerParity._write_inputs(inputs_path, seed=8103)

    manifest = json.loads((tmp_path / "inputs.json").read_text())
    inputs = np.load(inputs_path)
    assert manifest["input_sha256"] == "953c24f9f34bf8055a8e41022ee5372b8c02fa74195e6e994d837950b37dcdcd"
    assert manifest["official_source_revision"] == "2d2b4591ac053ec25c6371b01a5a6746679e5793"
    assert manifest["checkpoint_revision"] == "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
    assert inputs["rotary_query"].shape == (1, 7, 12, 128)


def test_bernini_transformer_parity_precision_thresholds_lock_runtime_and_float32_profiles():
    assert BerniniTransformerParity.PROFILE_THRESHOLDS == {
        "float32": {"min_cosine_similarity": 0.99999, "max_relative_l2": 0.001},
        "runtime": {"min_cosine_similarity": 0.9995, "max_relative_l2": 0.025},
    }
    reference = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    metrics = BerniniTransformerParity._metrics(
        reference=reference,
        actual=reference + np.array([1e-4, 0.0, -1e-4]),
    )
    assert metrics["cosine_similarity"] > 0.999999
    assert metrics["relative_l2"] < 1e-4
    assert metrics["max_absolute_error"] < 1.01e-4
