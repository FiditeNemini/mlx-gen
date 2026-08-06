import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "bernini_apg_parity.py"


def _load_parity_class():
    spec = importlib.util.spec_from_file_location("bernini_apg_parity", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BerniniAPGParity


BerniniAPGParity = _load_parity_class()


def test_bernini_apg_parity_inputs_are_pinned(tmp_path):
    inputs_path = tmp_path / "inputs.npz"
    manifest_path = tmp_path / "inputs.json"

    BerniniAPGParity._write_inputs(
        inputs_path,
        manifest_path=manifest_path,
        seed=BerniniAPGParity.DEFAULT_SEED,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["input_sha256"] == "69c6fa61f891fbc352a48eed4177c4bc2d9471a36c064cb2a4c6974f18491847"
    assert manifest["official_source_revision"] == "2d2b4591ac053ec25c6371b01a5a6746679e5793"
    assert manifest["cases"]["momentum_step_2"]["updates"] == 2
    assert manifest["cases"]["overflow_clip"]["absolute_tolerance"] == 0.0


def test_bernini_apg_torch_export_and_mlx_artifacts_match_pinned_goldens(monkeypatch, tmp_path):
    inputs_path = tmp_path / "inputs.npz"
    manifest_path = tmp_path / "inputs.json"
    BerniniAPGParity._write_inputs(
        inputs_path,
        manifest_path=manifest_path,
        seed=BerniniAPGParity.DEFAULT_SEED,
    )
    monkeypatch.setattr(
        BerniniAPGParity,
        "_validate_reference_source",
        staticmethod(
            lambda _reference_root: (
                BerniniAPGParity.OFFICIAL_SOURCE_REVISION,
                BerniniAPGParity.OFFICIAL_NORMALIZE_DIFF_SHA256,
            )
        ),
    )

    BerniniAPGParity._run_torch(
        inputs_path=inputs_path,
        manifest_path=manifest_path,
        output_dir=tmp_path,
        reference_root=tmp_path,
    )
    BerniniAPGParity._run_mlx(
        inputs_path=inputs_path,
        manifest_path=manifest_path,
        output_dir=tmp_path,
    )
    BerniniAPGParity._compare(output_dir=tmp_path, manifest_path=manifest_path)

    report = json.loads((tmp_path / "parity_report.json").read_text())
    torch_outputs = np.load(tmp_path / "torch_outputs.npz")
    assert report["passed"] is True
    assert report["comparisons"]["overflow_clip__output_0"]["max_absolute_error"] == 0.0
    assert report["comparisons"]["momentum_step_2__state_1"]["max_absolute_error"] == 0.0
    assert report["comparisons"]["large_base_norm__output_0"]["max_absolute_error"] <= 5e-7
    assert BerniniAPGParity._array_hash(torch_outputs["tiny_base_norm__output_0"]) == (
        "b8608d3435a634cd507ef783e9eea6e463210374ade02243ebcab1ee87732b26"
    )
    assert BerniniAPGParity._array_hash(torch_outputs["large_base_norm__output_0"]) == (
        "bd8d35c45fe81ea3be64a1334125ea9b65a5ec619c9336dcdccffd1e5f38ce5f"
    )
    assert BerniniAPGParity._array_hash(torch_outputs["momentum_step_2__output_1"]) == (
        "70dae408862c31f34c74b5768e08d73672975f8aa1e28d792a91ada1775348b2"
    )
