import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "bernini_vae_parity.py"


def _load_parity_class():
    spec = importlib.util.spec_from_file_location("bernini_vae_parity", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.BerniniVaeParity


BerniniVaeParity = _load_parity_class()


def test_bernini_tiled_vae_parity_rejects_fixture_that_does_not_tile(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bernini_vae_parity.py",
            "--stage",
            "compare",
            "--output-dir",
            str(tmp_path),
            "--checkpoint-root",
            str(tmp_path / BerniniVaeParity.OFFICIAL_COMPONENT_REVISION),
            "--tile-spatial",
        ],
    )

    with pytest.raises(ValueError, match="greater than 256"):
        BerniniVaeParity.main()


def test_bernini_tiled_vae_stage_reports_are_bound_to_matching_policy_and_revision(tmp_path):
    checkpoint = tmp_path / BerniniVaeParity.OFFICIAL_COMPONENT_REVISION
    torch_report = {
        "tile_spatial": True,
        "input_shape": [1, 3, 9, 272, 320],
        "checkpoint_root": str(checkpoint),
        "diffusers_version": BerniniVaeParity.PINNED_DIFFUSERS_VERSION,
    }
    mlx_report = {
        "tile_spatial": True,
        "input_shape": [1, 3, 9, 272, 320],
        "checkpoint_root": str(checkpoint),
    }

    BerniniVaeParity._validate_stage_reports(
        torch_report=torch_report,
        mlx_report=mlx_report,
        tile_spatial=True,
    )

    mlx_report["tile_spatial"] = False
    with pytest.raises(ValueError, match="does not match"):
        BerniniVaeParity._validate_stage_reports(
            torch_report=torch_report,
            mlx_report=mlx_report,
            tile_spatial=True,
        )


def test_bernini_tiled_vae_seam_metrics_expose_boundary_concentration():
    reference = np.zeros((1, 3, 1, 272, 320), dtype=np.float32)
    actual = reference.copy()
    actual[..., 188:196, :] = 1.0
    actual[..., :, 188:196] = 1.0

    metrics = BerniniVaeParity._seam_metrics(reference, actual)

    assert metrics["horizontal_boundaries"] == [192]
    assert metrics["vertical_boundaries"] == [192]
    assert metrics["combined_seam_mae"] == 1.0
    assert metrics["interior_mae"] == 0.0
    assert metrics["seam_to_interior_ratio"] is None
