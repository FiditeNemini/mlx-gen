import importlib.util
import json
import sys
from pathlib import Path


def _load_bundle_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location(
        "bernini_official_parity_bundle",
        tools_dir / "bernini_official_parity_bundle.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "validation_outputs"
        / "bernini_r_1_3b_2026_08_10"
        / "official_parity"
        / "run_official_public_cases.py"
    )
    spec = importlib.util.spec_from_file_location("run_official_public_cases", runner_path)
    assert spec is not None and spec.loader is not None
    runner_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner_module
    spec.loader.exec_module(runner_module)
    return runner_module


def test_official_parity_bundle_copies_generated_output_and_metadata(tmp_path, monkeypatch):
    module = _load_bundle_module()
    bundle = module.BerniniOfficialParityBundle

    workspace_root = tmp_path / "workspace"
    case_dir = workspace_root / "validation_outputs" / "bernini_r_1_3b_2026_08_11" / "t2v_case_run" / "t2v"
    case_dir.mkdir(parents=True)

    for name in ("README.md", "input_sheet.png", "official_sheet.png", "mlx_sheet.png"):
        (case_dir / name).write_bytes(b"stub")

    output_path = workspace_root / "validation_outputs" / "bernini_r_1_3b_2026_08_11" / "t2v_case_run" / "t2v" / "t2v.mp4"
    output_path.write_bytes(b"video")
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps({"frames": 81}))

    proof = {
        "id": "t2v",
        "title": "T2V example",
        "task_type": "t2v",
        "output": str(output_path.relative_to(workspace_root)),
        "metadata": str(metadata_path.relative_to(workspace_root)),
        "official_output": "/tmp/official_t2v.mp4",
        "observed_result": ["Manual review accepted this row as working."],
    }
    (case_dir / "proof.json").write_text(json.dumps(proof))

    output_dir = workspace_root / "bundle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bernini_official_parity_bundle.py",
            "--workspace-root",
            str(workspace_root),
            "--search-root",
            "validation_outputs/bernini_r_1_3b_2026_08_11",
            "--output-dir",
            str(output_dir),
            "--case-id",
            "t2v",
        ],
    )

    bundle.main()

    copied_output = output_dir / "t2v" / "t2v.mp4"
    copied_metadata = output_dir / "t2v" / "t2v.metadata.json"
    assert copied_output.read_bytes() == b"video"
    assert json.loads(copied_metadata.read_text()) == {"frames": 81}

    bundled_proof = json.loads((output_dir / "t2v" / "proof.json").read_text())
    assert bundled_proof["bundled_artifacts"] == {
        "output": "t2v.mp4",
        "metadata": "t2v.metadata.json",
    }

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["cases"] == [
        {
            "id": "t2v",
            "title": "T2V example",
            "task_type": "t2v",
            "source_dir": str(case_dir),
            "bundle_dir": str(output_dir / "t2v"),
            "output": str(output_path.relative_to(workspace_root)),
            "official_output": "/tmp/official_t2v.mp4",
            "observed_result": ["Manual review accepted this row as working."],
            "bundle_output": "t2v.mp4",
            "bundle_metadata": "t2v.metadata.json",
        }
    ]


def test_official_public_case_manifest_matches_documented_1_3b_release_rows():
    module = _load_bundle_module()
    bundle = module.BerniniOfficialParityBundle

    runner_module = _load_runner_module()

    expected_ids = (
        "t2i",
        "i2i",
        "t2v",
        "v2v_case1",
        "mv2v",
        "v2v_case3",
        "r2v",
        "r2v_case2",
        "rv2v_case1",
        "ads2v",
    )

    manifest_ids = tuple(entry["id"] for entry in runner_module.OfficialPublicBerniniCases.MANIFEST)
    assert manifest_ids == expected_ids
    assert bundle.CASE_IDS == expected_ids


def test_official_public_cases_resolve_source_aspect_output_for_exact_noise():
    runner_module = _load_runner_module()
    cases = runner_module.OfficialPublicBerniniCases

    class DummyModel:
        @staticmethod
        def _plan_condition_metadata(
            *,
            video_path,
            requested_height,
            requested_width,
            requested_frames,
            fps,
            canvas_policy,
            max_condition_size,
        ):
            assert video_path == Path("/tmp/source.mp4")
            assert requested_width == 848
            assert requested_height == 480
            assert requested_frames == 81
            assert fps == 16
            assert canvas_policy == "source-aspect"
            assert max_condition_size == 848
            return {
                "output_width": 848,
                "output_height": 448,
                "output_frames": 81,
            }

    media_spec = {
        "kind": "video",
        "width": 848,
        "height": 480,
        "frames": 81,
        "fps": 16,
    }
    resolved = cases._resolved_media_spec(
        model=DummyModel(),
        media_spec=media_spec,
        video_path=Path("/tmp/source.mp4"),
        max_condition_size=848,
    )

    assert resolved == {
        "kind": "video",
        "width": 848,
        "height": 448,
        "frames": 81,
        "fps": 16,
    }
    assert media_spec == {
        "kind": "video",
        "width": 848,
        "height": 480,
        "frames": 81,
        "fps": 16,
    }
