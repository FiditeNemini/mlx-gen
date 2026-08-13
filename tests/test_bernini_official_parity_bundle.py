import importlib.util
import json
import sys
from pathlib import Path

OFFICIAL_PUBLIC_CASE_IDS = (
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

    inventory_path = (
        Path(__file__).resolve().parents[1]
        / "docs/assets/validation/bernini-r-1.3b-2026-08-04/official_example_inventory.json"
    )
    inventory = json.loads(inventory_path.read_text())
    inventory_groups = tuple(case["group"] for case in inventory["cases"])
    expected_groups = (
        "t2i",
        "i2i",
        "t2v",
        "v2v",
        "mv2v",
        "v2v",
        "r2v",
        "r2v",
        "rv2v",
        "ads2v",
    )
    assert inventory_groups == expected_groups
    assert bundle.CASE_IDS == OFFICIAL_PUBLIC_CASE_IDS


def test_bernini_source_aspect_video_resolution_matches_official_parity_harness(monkeypatch):
    from mflux.models.wan.variants.wan_bernini import BerniniRenderer
    from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT

    model = BerniniRenderer.__new__(BerniniRenderer)

    class FakeVideoInfo:
        source_frame_count = 81
        source_width = 1920
        source_height = 1080
        fps = 16.0

    monkeypatch.setattr(
        "mflux.models.wan.variants.wan_bernini.VideoUtil.inspect_video",
        lambda path: FakeVideoInfo(),
    )
    monkeypatch.setattr(
        BerniniRenderer,
        "_smart_video_indices",
        lambda self, **kwargs: list(range(kwargs["max_frames"])),
    )

    plan = model._plan_condition_metadata(
        video_path=Path("/tmp/source.mp4"),
        requested_height=480,
        requested_width=848,
        requested_frames=81,
        fps=16,
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        max_condition_size=848,
    )

    expected_width, expected_height = BerniniRenderer._condition_dimensions(
        width=1920,
        height=1080,
        max_size=848,
    )
    assert plan["output_width"] == expected_width
    assert plan["output_height"] == expected_height
    assert plan["output_frames"] == 81
