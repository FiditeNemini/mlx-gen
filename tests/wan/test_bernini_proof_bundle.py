import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageFilter


def _load_proof_bundle():
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    sys.path.insert(0, str(tools_dir))
    spec = importlib.util.spec_from_file_location("bernini_proof_bundle", tools_dir / "bernini_proof_bundle.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BerniniProofBundle


BerniniProofBundle = _load_proof_bundle()


def _current_portable_runs():
    runs = []
    for case in BerniniProofBundle.cases().values():
        prompt = case.prompt_override or f"current prompt for {case.prompt_json}"
        runs.append(
            {
                "case": asdict(case),
                "prompt": prompt,
                "case_fingerprint": BerniniProofBundle._case_fingerprint(case=case, prompt=prompt),
            }
        )
    return runs


def _use_test_prompt_hashes(monkeypatch):
    monkeypatch.setattr(
        BerniniProofBundle,
        "EXPECTED_UPSTREAM_PROMPT_SHA256",
        {
            prompt_json: hashlib.sha256(f"current prompt for {prompt_json}".encode()).hexdigest()
            for prompt_json in BerniniProofBundle.EXPECTED_UPSTREAM_PROMPT_SHA256
        },
    )


def test_proof_reuse_requires_current_case_fingerprint(tmp_path, monkeypatch):
    case = BerniniProofBundle.cases()["r2v_eight_reference"]
    prompt = "official prompt"
    output = tmp_path / "proof.mp4"
    output.write_bytes(b"video")
    legacy = {
        "output_path": str(output),
        "case": json.loads(json.dumps(asdict(case))),
        "prompt": prompt,
    }

    assert not BerniniProofBundle._can_reuse(existing=legacy, case=case, prompt=prompt)

    fingerprinted = {
        **legacy,
        "case_fingerprint": BerniniProofBundle._case_fingerprint(case=case, prompt=prompt),
    }
    assert BerniniProofBundle._can_reuse(existing=fingerprinted, case=case, prompt=prompt)
    assert not BerniniProofBundle._can_reuse(
        existing=fingerprinted,
        case=replace(case, width=case.width + 16),
        prompt=prompt,
    )
    assert not BerniniProofBundle._can_reuse(existing=fingerprinted, case=case, prompt=f"{prompt} changed")
    monkeypatch.setattr(
        BerniniProofBundle,
        "EXPECTED_RUNTIME_POLICY",
        {**BerniniProofBundle.EXPECTED_RUNTIME_POLICY, "transformer_precision_policy_id": "changed"},
    )
    assert not BerniniProofBundle._can_reuse(existing=fingerprinted, case=case, prompt=prompt)


def test_proof_fingerprint_tracks_generation_implementation(tmp_path, monkeypatch):
    repository_root = tmp_path / "repo"
    implementation_path = repository_root / "src/implementation.py"
    implementation_path.parent.mkdir(parents=True)
    implementation_path.write_text("first")
    module_globals = BerniniProofBundle._implementation_fingerprint.__globals__
    monkeypatch.setitem(module_globals, "__file__", str(repository_root / "tools/bernini_proof_bundle.py"))
    monkeypatch.setattr(BerniniProofBundle, "IMPLEMENTATION_GLOBS", ())
    monkeypatch.setattr(BerniniProofBundle, "IMPLEMENTATION_PATHS", ("src/implementation.py",))

    first = BerniniProofBundle._implementation_fingerprint()
    implementation_path.write_text("second")

    assert BerniniProofBundle._implementation_fingerprint() != first


def test_proof_implementation_fingerprint_covers_transitive_runtime_inputs():
    repository_root = Path(__file__).resolve().parents[2]
    paths = {
        path.relative_to(repository_root).as_posix()
        for path in BerniniProofBundle._implementation_files(repository_root=repository_root)
    }

    assert {
        "src/mflux/models/wan/model/wan_transformer/wan_transformer_block.py",
        "src/mflux/models/wan/scheduler/wan_timestep_grid.py",
        "src/mflux/models/wan/model/wan_vae/wan_2_2_vae.py",
        "src/mflux/models/wan/wan_initializer.py",
        "src/mflux/models/common/config/model_config.py",
        "src/mflux/models/wan/cli/wan_generate.py",
        "pyproject.toml",
        "tools/bernini_proof_bundle.py",
        "tools/generation_memory_benchmark.py",
        "uv.lock",
    } <= paths


def test_reference_root_requires_exact_clean_official_revision(tmp_path, monkeypatch):
    reference_root = tmp_path / "official"
    (reference_root / "assets/testcases").mkdir(parents=True)
    (reference_root / "assets/testcases/README.md").write_text("cases")
    (reference_root / "LICENSE").write_text("license")
    outputs = {
        ("rev-parse", "--show-toplevel"): str(reference_root),
        ("rev-parse", "HEAD"): BerniniProofBundle.OFFICIAL_SOURCE_REVISION,
        ("status", "--porcelain", "--untracked-files=no", "--", "assets/testcases", "LICENSE"): "",
    }
    monkeypatch.setattr(
        BerniniProofBundle,
        "_git_output",
        staticmethod(lambda _root, *args: outputs[args]),
    )

    BerniniProofBundle._validate_reference_root(reference_root)
    outputs[("rev-parse", "HEAD")] = "wrong"
    with pytest.raises(ValueError, match="official revision"):
        BerniniProofBundle._validate_reference_root(reference_root)
    outputs[("rev-parse", "HEAD")] = BerniniProofBundle.OFFICIAL_SOURCE_REVISION
    outputs[("status", "--porcelain", "--untracked-files=no", "--", "assets/testcases", "LICENSE")] = " M input"
    with pytest.raises(ValueError, match="differ from"):
        BerniniProofBundle._validate_reference_root(reference_root)


def test_required_runtime_policy_covers_complete_low_ram_vae_path():
    policy = BerniniProofBundle.EXPECTED_RUNTIME_POLICY

    assert policy["low_ram"] is True
    assert policy["vae_low_memory_policy_active"] is True
    assert policy["clear_cache_each_transformer_block"] is True
    assert policy["release_denoisers_before_decode"] is True
    assert policy["vae_feature_cache_policy_id"] == "wan-compact-feature-cache-v1"
    assert policy["vae_spatial_tiling"] is True
    assert policy["vae_spatial_tiling_policy_id"] == "wan21-diffusers-0.35.2-256x256-stride192-v1"
    assert policy["wan_decode_mode"] == "bounded_tile_major_spatial_vae"


def test_required_runtime_environment_fields_cover_reproducibility_contract():
    assert BerniniProofBundle.REQUIRED_RUNTIME_ENVIRONMENT_FIELDS == (
        "mlx_version",
        "python_version",
        "python_implementation",
        "runtime_platform",
        "numpy_version",
        "python_executable",
    )


def test_visual_review_is_bound_to_output_and_every_case_sheet():
    run = {
        "case": {"frames": 2},
        "artifact_hashes": {
            "output": {"sha256": "output"},
            "sheets": {
                "mlx": {"sha256": "mlx"},
                "references": {"sha256": "references"},
            },
        },
    }
    review = {
        "status": "pass_with_limitations",
        "output_sha256": "output",
        "sheet_sha256": {"mlx": "mlx", "references": "references"},
        "reviewed_frame_indices": [0, 1],
        "notes": "Bounded pass.",
    }

    assert all(BerniniProofBundle._visual_review_check(run=run, review=review).values())
    review["output_sha256"] = "stale"
    assert not BerniniProofBundle._visual_review_check(run=run, review=review)["output_hash"]


def test_required_quality_negative_or_structural_result_blocks_overall_pass():
    cases = {case_id: {"status": "pass"} for case_id in BerniniProofBundle.QUALITY_CASE_IDS}
    cases["rv2v_reference_black_ab"] = {"status": "negative_result"}
    checks = BerniniProofBundle._visual_quality_checks(visual_cases=cases)

    assert not checks["rv2v_reference_black_ab"]
    assert not BerniniProofBundle._visual_quality_passes(
        visual_review_complete=True,
        run_ids=set(BerniniProofBundle.QUALITY_CASE_IDS),
        visual_quality_checks=checks,
    )

    cases["rv2v_reference_black_ab"] = {"status": "structural_only"}
    assert not BerniniProofBundle._visual_quality_checks(visual_cases=cases)["rv2v_reference_black_ab"]


def test_pass_with_limitations_requires_minor_severity():
    assert BerniniProofBundle._quality_review_accepted(
        {"status": "pass_with_limitations", "limitation_severity": "minor"}
    )
    assert not BerniniProofBundle._quality_review_accepted(
        {"status": "pass_with_limitations", "limitation_severity": "major"}
    )
    assert not BerniniProofBundle._quality_review_accepted({"status": "pass_with_limitations"})


def test_quality_contract_rejects_latent_group_cadence(monkeypatch, tmp_path):
    case = asdict(BerniniProofBundle.cases()["v2v_snowman"])
    output = tmp_path / "output.mp4"
    output.write_bytes(b"video")
    result = {
        "case": case,
        "output_path": str(output),
        "metadata": {},
        "video_health": {"status": "ok"},
        "sheet_details": {
            "worst_transitions": {
                "temporal_diagnostics": {"boundary_to_non_boundary_ratio": 1.50},
            }
        },
    }
    monkeypatch.setattr(BerniniProofBundle, "_expected_timeline_groups", staticmethod(lambda _: {}))
    monkeypatch.setattr(BerniniProofBundle, "_sheet_contract_passes", staticmethod(lambda **_: True))

    checks = BerniniProofBundle._contract_checks(result)

    assert not checks["latent_group_continuity"]
    result["sheet_details"]["worst_transitions"]["temporal_diagnostics"]["boundary_to_non_boundary_ratio"] = 1.18
    assert BerniniProofBundle._contract_checks(result)["latent_group_continuity"]


def test_proof_cli_exits_nonzero_when_required_quality_gate_fails(tmp_path, monkeypatch):
    reference_root = tmp_path / "official"
    output_dir = tmp_path / "proof"
    args = SimpleNamespace(
        reference_root=reference_root,
        output_dir=output_dir,
        durable_dir=None,
        cases=(),
        rerun=False,
        sample_interval_ms=250,
    )
    monkeypatch.setattr(BerniniProofBundle, "_parse_args", staticmethod(lambda: args))
    monkeypatch.setattr(BerniniProofBundle, "_validate_reference_root", staticmethod(lambda _: None))
    monkeypatch.setattr(BerniniProofBundle, "_existing_runs", staticmethod(lambda _: {}))
    monkeypatch.setattr(BerniniProofBundle, "_ordered_runs", staticmethod(lambda _: []))
    monkeypatch.setattr(BerniniProofBundle, "_refresh_case_sheets", staticmethod(lambda **_: None))
    monkeypatch.setattr(BerniniProofBundle, "_save_summary_sheet", staticmethod(lambda **_: None))
    monkeypatch.setattr(BerniniProofBundle, "_save_role_control_sheet", staticmethod(lambda **_: None))
    monkeypatch.setattr(BerniniProofBundle, "_write_sheet_manifest", staticmethod(lambda **_: None))
    monkeypatch.setattr(
        BerniniProofBundle,
        "_write_report",
        staticmethod(lambda **_: {"passed": False, "visual_quality_passed": False}),
    )
    monkeypatch.setattr(BerniniProofBundle, "_compact_report", staticmethod(lambda report: report))

    with pytest.raises(SystemExit) as error:
        BerniniProofBundle.main()

    assert error.value.code == 1


def test_timeline_sheet_is_high_resolution_and_contains_every_frame(tmp_path):
    images = [Image.new("RGB", (32, 24), (index, 0, 0)) for index in range(17)]
    output_path = tmp_path / "timeline.png"

    details = BerniniProofBundle._save_image_sheet(
        images=images,
        output_path=output_path,
        title="all frames",
        indices=list(range(17)),
        label_prefix="frame",
        resampling=Image.Resampling.NEAREST,
        fixed_columns=BerniniProofBundle.TIMELINE_COLUMNS,
        cell_size=BerniniProofBundle.TIMELINE_CELL_SIZE,
        fps=16,
    )

    assert details["width"] >= 5120
    assert details["columns"] == BerniniProofBundle.TIMELINE_COLUMNS
    assert details["sample_indices"] == list(range(17))
    assert details["includes_all_frames"] is True
    assert details["resampling"] == "nearest"
    assert details["integer_magnification"] is True
    assert details["downsampled"] is False
    assert details["input_sizes"] == [[32, 24]] * 17
    assert len(details["rendered_sizes"]) == 17
    assert len(details["decoded_frame_sha256"]) == 17
    assert all(len(value) == 64 for value in details["decoded_frame_sha256"])
    assert all(width >= 32 and height >= 24 for width, height in details["rendered_sizes"])
    assert details["label_font_size"] >= 64
    with Image.open(output_path) as sheet:
        assert sheet.size == (details["width"], details["height"])


def test_required_quality_cases_use_the_official_release_profile():
    cases = BerniniProofBundle.cases()

    for case_id in BerniniProofBundle.QUALITY_CASE_IDS:
        case = cases[case_id]
        assert case.frames == 81
        assert case.steps == 40
        assert case.max_condition_size == 848
        assert (case.width, case.height) in {(848, 480), (480, 848)}

    r2v = cases["r2v_eight_reference"]
    assert r2v.prompt_json == "assets/testcases/r2v/r2v_case2.json"
    assert r2v.prompt_override is None


def test_long_timeline_is_split_into_ordered_high_resolution_pages(tmp_path, monkeypatch):
    images = [Image.new("RGB", (32, 24), (index, 0, 0)) for index in range(25)]
    monkeypatch.setattr(BerniniProofBundle, "TIMELINE_CELL_SIZE", 64)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_PADDING", 4)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_HEADER_HEIGHT", 32)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_LABEL_HEIGHT", 16)
    monkeypatch.setattr(BerniniProofBundle, "TITLE_FONT_SIZE", 12)
    monkeypatch.setattr(BerniniProofBundle, "LABEL_FONT_SIZE", 10)

    sheets, details = BerniniProofBundle._save_timeline_sheet_group(
        images=images,
        run_dir=tmp_path,
        label="mlx",
        title="all frames",
        indices=list(range(25)),
        label_prefix="frame",
        fps=16,
        resampling=Image.Resampling.NEAREST,
        cell_size=64,
    )

    assert list(sheets) == ["mlx_page_01", "mlx_page_02", "mlx_page_03"]
    assert all(Path(path).is_file() for path in sheets.values())
    assert details["mlx_page_01"]["sample_indices"] == list(range(9))
    assert details["mlx_page_02"]["sample_indices"] == list(range(9, 18))
    assert details["mlx_page_03"]["sample_indices"] == list(range(18, 25))
    assert details["mlx_page_01"]["page_count"] == 3
    assert details["mlx_page_03"]["page_number"] == 3
    assert not (tmp_path / "mlx_contact_sheet.png").exists()


def test_overview_sheet_contract_requires_5k_lossless_render(tmp_path):
    output_path = tmp_path / "overview.png"
    rows = [
        (
            "case",
            [(index, Image.new("RGB", (320, 192), (index, 0, 0))) for index in range(5)],
        )
    ]

    details = BerniniProofBundle._save_row_sheet(rows=rows, output_path=output_path, title="overview")

    assert details["width"] >= 5120
    assert details["downsampled"] is False
    assert BerniniProofBundle._overview_sheet_contract_passes(path=output_path, details=details)


def test_case_sheet_pixel_binding_rejects_blank_misordered_and_degraded_content(tmp_path, monkeypatch):
    case = replace(
        BerniniProofBundle.cases()["r2v_848_condition_smoke"],
        frames=3,
        reference_images=("reference-a.png", "reference-b.png"),
    )
    run_dir = tmp_path / case.case_id
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True)
    reference_paths = [input_dir / "reference_00.png", input_dir / "reference_01.png"]
    Image.new("RGB", (24, 32), (220, 20, 20)).save(reference_paths[0])
    Image.new("RGB", (32, 24), (20, 20, 220)).save(reference_paths[1])
    output_path = run_dir / "output.mp4"
    output_path.write_bytes(b"output")
    frames = [Image.new("RGB", (32, 24), (20 + index * 70, 40, 80)) for index in range(3)]
    video_util = BerniniProofBundle._save_case_sheets.__globals__["VideoUtil"]
    monkeypatch.setattr(
        video_util,
        "read_video_clip",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(frames=frames, fps=8)),
    )
    monkeypatch.setattr(BerniniProofBundle, "TIMELINE_CELL_SIZE", 64)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_PADDING", 4)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_HEADER_HEIGHT", 32)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_LABEL_HEIGHT", 16)
    monkeypatch.setattr(BerniniProofBundle, "TITLE_FONT_SIZE", 12)
    monkeypatch.setattr(BerniniProofBundle, "LABEL_FONT_SIZE", 10)

    def regenerate():
        return BerniniProofBundle._save_case_sheets(
            case=case,
            reference_paths=reference_paths,
            source_path=None,
            official_path=None,
            output_path=output_path,
            run_dir=run_dir,
            output_metadata={},
        )

    sheets, details = regenerate()
    run = {
        "case": asdict(case),
        "input_reference_paths": [str(path) for path in reference_paths],
        "input_source_path": None,
        "official_output_path": None,
        "output_path": str(output_path),
        "metadata": {},
        "sheets": sheets,
        "sheet_details": details,
    }
    BerniniProofBundle._verify_case_sheet_pixels(run=run)

    mlx_path = Path(sheets["mlx"])
    with Image.open(mlx_path) as sheet:
        Image.new("RGB", sheet.size, (127, 127, 127)).save(mlx_path)
    with pytest.raises(ValueError, match="pixel binding"):
        BerniniProofBundle._verify_case_sheet_pixels(run=run)

    sheets, details = regenerate()
    run.update(sheets=sheets, sheet_details=details)
    reference_images = [Image.open(path).convert("RGB") for path in reversed(reference_paths)]
    BerniniProofBundle._save_image_sheet(
        images=reference_images,
        output_path=Path(sheets["references"]),
        title=f"{case.case_id}: ordered references",
        indices=[0, 1],
        label_prefix="image",
        resampling=Image.Resampling.LANCZOS,
        fixed_columns=BerniniProofBundle.TIMELINE_COLUMNS,
        cell_size=BerniniProofBundle.TIMELINE_CELL_SIZE,
    )
    with pytest.raises(ValueError, match="pixel binding"):
        BerniniProofBundle._verify_case_sheet_pixels(run=run)

    sheets, details = regenerate()
    run.update(sheets=sheets, sheet_details=details)
    mlx_path = Path(sheets["mlx"])
    with Image.open(mlx_path) as sheet:
        sheet.convert("RGB").filter(ImageFilter.GaussianBlur(radius=1)).save(mlx_path)
    with pytest.raises(ValueError, match="pixel binding"):
        BerniniProofBundle._verify_case_sheet_pixels(run=run)

    reordered = copy.deepcopy(run)
    reordered["input_reference_paths"] = list(reversed(reordered["input_reference_paths"]))
    with pytest.raises(ValueError, match="reference ordering"):
        BerniniProofBundle._verify_case_sheet_pixels(run=reordered)


def test_source_sheet_sampling_is_recomputed_from_retained_video(monkeypatch):
    case = replace(
        BerniniProofBundle.cases()["v2v_snowman"],
        width=32,
        height=32,
        frames=5,
        fps=8,
        max_condition_size=32,
    )
    source_info = SimpleNamespace(
        source_frame_count=5,
        source_width=32,
        source_height=24,
        fps=8.0,
    )
    video_util = BerniniProofBundle._verify_source_sheet_sampling.__globals__["VideoUtil"]
    monkeypatch.setattr(video_util, "inspect_video", staticmethod(lambda _path: source_info))
    renderer = BerniniProofBundle._verify_source_sheet_sampling.__globals__["BerniniRenderer"]
    expected_indices = renderer._smart_video_indices(
        total_frames=5,
        video_fps=8,
        fps=8,
        frame_factor=4,
        max_frames=5,
        add_one=True,
    )
    output_height, output_width = renderer._closest_spatial_size_for_ratio(
        requested_height=case.height,
        requested_width=case.width,
        source_height=24,
        source_width=32,
        multiple_h=16,
        multiple_w=16,
    )
    condition_width, condition_height = renderer._condition_dimensions(
        width=output_width,
        height=output_height,
        max_size=case.max_condition_size,
    )
    metadata = {
        "canvas_policy": "source-aspect",
        "condition_resize_backend": "pillow-bicubic",
        "source_width": 32,
        "source_height": 24,
        "source_fps": 8.0,
        "source_frame_count": 5,
        "source_sample_indices": expected_indices,
        "video_condition_width": condition_width,
        "video_condition_height": condition_height,
        "video_condition_frames": len(expected_indices),
    }

    BerniniProofBundle._verify_source_sheet_sampling(
        case=case,
        source_path=Path("source.mp4"),
        metadata=metadata,
    )

    changed = {**metadata, "source_sample_indices": list(reversed(expected_indices))}
    with pytest.raises(ValueError, match="source-sheet sampling"):
        BerniniProofBundle._verify_source_sheet_sampling(
            case=case,
            source_path=Path("source.mp4"),
            metadata=changed,
        )

    changed = {**metadata, "condition_resize_backend": "unattested-resizer"}
    with pytest.raises(ValueError, match="source-sheet sampling"):
        BerniniProofBundle._verify_source_sheet_sampling(
            case=case,
            source_path=Path("source.mp4"),
            metadata=changed,
        )


def test_overview_sheet_pixel_binding_rejects_blank_and_reordered_rows(tmp_path, monkeypatch):
    case_ids = (
        "rv2v_reference_pinstripe_ab",
        "rv2v_reference_black_ab",
        "rv2v_reference_none_ab",
    )
    runs = []
    decoded = {}
    for run_index, case_id in enumerate(case_ids):
        output_path = tmp_path / f"{case_id}.mp4"
        output_path.write_bytes(case_id.encode())
        runs.append({"case": {"case_id": case_id}, "output_path": str(output_path)})
        decoded[str(output_path)] = [
            Image.new("RGB", (32, 24), (run_index * 70, frame_index * 40, 20)) for frame_index in range(4)
        ]
    video_util = BerniniProofBundle._save_summary_sheet.__globals__["VideoUtil"]
    monkeypatch.setattr(
        video_util,
        "read_video_clip",
        staticmethod(lambda path, **_kwargs: SimpleNamespace(frames=decoded[str(path)], fps=8)),
    )
    monkeypatch.setattr(BerniniProofBundle, "SUMMARY_COLUMNS", 3)
    monkeypatch.setattr(BerniniProofBundle, "SUMMARY_CELL_WIDTH", 64)
    monkeypatch.setattr(BerniniProofBundle, "SUMMARY_CELL_HEIGHT", 48)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_PADDING", 4)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_HEADER_HEIGHT", 32)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_LABEL_HEIGHT", 16)
    monkeypatch.setattr(BerniniProofBundle, "TITLE_FONT_SIZE", 12)
    monkeypatch.setattr(BerniniProofBundle, "LABEL_FONT_SIZE", 10)
    summary_path = tmp_path / "summary.png"
    role_path = tmp_path / "role.png"
    summary_details = BerniniProofBundle._save_summary_sheet(runs=runs, output_path=summary_path)
    role_details = BerniniProofBundle._save_role_control_sheet(runs=runs, output_path=role_path)
    BerniniProofBundle._verify_overview_sheet_pixels(
        runs=runs,
        summary_path=summary_path,
        summary_details=summary_details,
        role_path=role_path,
        role_details=role_details,
    )

    with Image.open(summary_path) as sheet:
        Image.new("RGB", sheet.size, (127, 127, 127)).save(summary_path)
    with pytest.raises(ValueError, match="summary overview sheet pixel binding"):
        BerniniProofBundle._verify_overview_sheet_pixels(
            runs=runs,
            summary_path=summary_path,
            summary_details=summary_details,
            role_path=role_path,
            role_details=role_details,
        )

    BerniniProofBundle._save_summary_sheet(runs=runs, output_path=summary_path)
    BerniniProofBundle._save_role_control_sheet(runs=list(reversed(runs)), output_path=role_path)
    with pytest.raises(ValueError, match="role-control overview sheet pixel binding"):
        BerniniProofBundle._verify_overview_sheet_pixels(
            runs=runs,
            summary_path=summary_path,
            summary_details=summary_details,
            role_path=role_path,
            role_details=role_details,
        )


def test_worst_transition_sheet_selects_and_pairs_largest_frame_changes(tmp_path, monkeypatch):
    case = replace(
        BerniniProofBundle.cases()["r2v_eight_reference"],
        frames=6,
        reference_images=(),
        official_output=None,
    )
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"output")
    values = [0, 1, 2, 200, 201, 202]
    frames = [Image.new("RGB", (16, 16), (value, value, value)) for value in values]
    video_util = BerniniProofBundle._save_case_sheets.__globals__["VideoUtil"]
    monkeypatch.setattr(
        video_util,
        "read_video_clip",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(frames=frames, fps=16)),
    )

    _, details = BerniniProofBundle._save_case_sheets(
        case=case,
        reference_paths=[],
        source_path=None,
        official_path=None,
        output_path=output_path,
        run_dir=tmp_path,
        output_metadata={},
    )

    assert details["worst_transitions"]["transition_start_indices"][0] == 2
    assert details["worst_transitions"]["sample_indices"][:2] == [2, 3]
    assert details["worst_transitions"]["width"] >= 3072


def test_worst_transition_selection_detects_localized_change(tmp_path, monkeypatch):
    case = replace(
        BerniniProofBundle.cases()["r2v_eight_reference"],
        frames=3,
        reference_images=(),
        official_output=None,
    )
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"output")
    frame_0 = Image.new("RGB", (64, 64), (0, 0, 0))
    frame_1 = Image.new("RGB", (64, 64), (20, 20, 20))
    frame_2 = frame_1.copy()
    for y in range(16):
        for x in range(16):
            frame_2.putpixel((x, y), (255, 255, 255))
    video_util = BerniniProofBundle._save_case_sheets.__globals__["VideoUtil"]
    monkeypatch.setattr(
        video_util,
        "read_video_clip",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(frames=[frame_0, frame_1, frame_2], fps=16)),
    )

    _, details = BerniniProofBundle._save_case_sheets(
        case=case,
        reference_paths=[],
        source_path=None,
        official_path=None,
        output_path=output_path,
        run_dir=tmp_path,
        output_metadata={},
    )

    transitions = details["worst_transitions"]
    assert transitions["transition_start_indices"][0] == 1
    assert transitions["transition_max_tile_mae"][0] > transitions["transition_max_tile_mae"][1]
    assert transitions["transition_global_mae"][0] < transitions["transition_global_mae"][1]


def test_source_sheet_uses_exact_conditioned_frame_indices(tmp_path, monkeypatch):
    case = replace(
        BerniniProofBundle.cases()["v2v_snowman"],
        frames=3,
        source_video="source.mp4",
        official_output=None,
    )
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "output.mp4"
    source_path.write_bytes(b"source")
    output_path.write_bytes(b"output")
    source_frames = [Image.new("RGB", (32, 24), (index, 0, 0)) for index in range(5)]
    output_frames = [Image.new("RGB", (32, 24), (0, index, 0)) for index in range(3)]
    video_util = BerniniProofBundle._save_case_sheets.__globals__["VideoUtil"]
    monkeypatch.setattr(
        video_util,
        "read_video_clip",
        staticmethod(
            lambda path, max_frames=None: SimpleNamespace(
                frames=source_frames if Path(path) == source_path else output_frames,
                fps=10,
            )
        ),
    )
    monkeypatch.setattr(BerniniProofBundle, "TIMELINE_COLUMNS", 3)
    monkeypatch.setattr(BerniniProofBundle, "TIMELINE_CELL_SIZE", 64)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_PADDING", 4)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_HEADER_HEIGHT", 32)
    monkeypatch.setattr(BerniniProofBundle, "SHEET_LABEL_HEIGHT", 16)
    monkeypatch.setattr(BerniniProofBundle, "TITLE_FONT_SIZE", 12)
    monkeypatch.setattr(BerniniProofBundle, "LABEL_FONT_SIZE", 10)

    _, details = BerniniProofBundle._save_case_sheets(
        case=case,
        reference_paths=[],
        source_path=source_path,
        official_path=None,
        output_path=output_path,
        run_dir=tmp_path,
        output_metadata={
            "source_sample_indices": [0, 2, 4],
            "video_condition_width": 32,
            "video_condition_height": 24,
        },
    )

    assert details["source"]["sample_indices"] == [0, 2, 4]
    assert details["source"]["input_sizes"] == [[32, 24]] * 3
    assert details["source"]["downsampled"] is False
    assert details["mlx"]["sample_indices"] == [0, 1, 2]


def test_visual_review_metadata_rejects_future_or_unattested_review():
    now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    review = {
        "schema_version": BerniniProofBundle.VISUAL_REVIEW_SCHEMA_VERSION,
        "reviewed_at": (now + timedelta(minutes=10)).isoformat(),
        "reviewer": "adversarial review",
        "scope": "all frames and transition sheets",
    }

    checks = BerniniProofBundle._visual_review_metadata_checks(review, now=now)

    assert checks["schema_version"]
    assert checks["reviewed_at_parseable"]
    assert not checks["reviewed_at_not_future"]
    review["reviewed_at"] = now.isoformat()
    review["reviewer"] = ""
    checks = BerniniProofBundle._visual_review_metadata_checks(review, now=now)
    assert checks["reviewed_at_not_future"]
    assert not checks["reviewer_present"]


def test_refresh_case_sheets_regenerates_derived_evidence_for_reused_run(tmp_path, monkeypatch):
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"video")
    output_path.with_suffix(".metadata.json").write_text("{}")
    run = {
        "case": {"case_id": "r2v_eight_reference"},
        "output_path": str(output_path),
        "input_reference_paths": [],
        "input_source_path": None,
        "official_output_path": None,
        "sheets": {"mlx": "stale"},
        "sheet_details": {"mlx": {"sheet_contract_version": 0}},
    }
    monkeypatch.setattr(
        BerniniProofBundle,
        "_save_case_sheets",
        staticmethod(lambda **_: ({"mlx": "fresh"}, {"mlx": {"sheet_contract_version": 1}})),
    )

    BerniniProofBundle._refresh_case_sheets(runs=[run])

    assert run["sheets"] == {"mlx": "fresh"}
    assert run["sheet_details"]["mlx"]["sheet_contract_version"] == 1


def test_sheet_contract_rejects_stale_low_resolution_sheet(tmp_path):
    path = tmp_path / "mlx.png"
    Image.new("RGB", (320, 240)).save(path)
    result = {
        "case": {"frames": 1},
        "metadata": {},
        "sheets": {"mlx": str(path)},
        "sheet_details": {
            "mlx": {
                "sheet_contract_version": BerniniProofBundle.SHEET_CONTRACT_VERSION,
                "width": 320,
                "height": 240,
                "cell_width": 192,
                "cell_height": 192,
                "title_font_size": 11,
                "label_font_size": 11,
                "source_frame_count": 1,
                "sample_indices": [0],
                "includes_all_frames": True,
                "resampling": "nearest",
            }
        },
    }

    assert not BerniniProofBundle._sheet_contract_passes(result=result, expected_sheets={"mlx"})


def test_durable_export_replaces_stale_files_and_copies_supplemental_evidence(tmp_path, monkeypatch):
    evidence_root = tmp_path / "evidence"
    output_dir = evidence_root / "cycle4_proof"
    (output_dir / "cases" / "run_1" / "case").mkdir(parents=True)
    (output_dir / "cases" / "run_1" / "case" / "output.mp4").write_bytes(b"video")
    for name in (
        "bernini_proof_report.json",
        "bernini_proof_report.md",
        "output_summary_contact_sheet.png",
        "role_control_contact_sheet.png",
        "sheet_manifest.json",
        "visual_review.json",
        "visual_review.md",
    ):
        if name == "bernini_proof_report.json":
            (output_dir / name).write_text(json.dumps({"schema_version": 3, "runs": _current_portable_runs()}))
        else:
            (output_dir / name).write_text("{}" if name.endswith(".json") else "proof")
    (evidence_root / "parity").mkdir()
    (evidence_root / "parity" / "report.json").write_text(
        json.dumps({"cache": str(Path.home() / ".cache" / "model"), "output": str(evidence_root / "parity")})
    )
    (evidence_root / "diagnostics").mkdir()
    (evidence_root / "diagnostics" / "comparison.png").write_bytes(b"image")
    for source_name in BerniniProofBundle.SUPPLEMENTAL_EVIDENCE_DIRS:
        source = evidence_root / source_name
        source.mkdir()
        (source / "evidence.json").write_text(
            json.dumps({"source": source_name, "artifact_path": str(source / "evidence.json")})
        )
    reference_root = tmp_path / "official"
    reference_root.mkdir()
    (reference_root / "LICENSE").write_text("Apache-2.0")
    durable_dir = tmp_path / "docs" / "bundle"
    durable_dir.mkdir(parents=True)
    (durable_dir / "portable_manifest.json").write_text(json.dumps({"kind": "portable_bernini_r_1_3b_proof_bundle"}))
    (durable_dir / "stale.txt").write_text("stale")
    monkeypatch.setattr(BerniniProofBundle, "_refresh_portable_report", staticmethod(lambda _: None))
    monkeypatch.setattr(BerniniProofBundle, "verify_portable_bundle", staticmethod(lambda _: {"verified": True}))

    BerniniProofBundle._export_durable_bundle(
        output_dir=output_dir,
        durable_dir=durable_dir,
        reference_root=reference_root,
    )

    assert not (durable_dir / "stale.txt").exists()
    assert (durable_dir / "cases" / "run_1" / "case" / "output.mp4").is_file()
    assert (durable_dir / "parity" / "report.json").is_file()
    assert (durable_dir / "diagnostics" / "comparison.png").is_file()
    for target_name in BerniniProofBundle.SUPPLEMENTAL_EVIDENCE_DIRS.values():
        copied = durable_dir / target_name / "evidence.json"
        assert copied.is_file()
        assert json.loads(copied.read_text())["artifact_path"] == f"<bundle-root>/{target_name}/evidence.json"
    assert (durable_dir / "UPSTREAM_BERNINI_LICENSE.txt").is_file()
    assert (durable_dir / "portable_manifest.json").is_file()
    portable_report = (durable_dir / "parity" / "report.json").read_text()
    assert str(Path.home()) not in portable_report
    assert str(evidence_root) not in portable_report
    assert "<user-home>" in portable_report
    assert "<bundle-root>/parity" in portable_report
    assert "<validation-root>" not in portable_report


def test_portable_bundle_verifier_rejects_extra_modified_missing_compatibility_and_symlinks(tmp_path, monkeypatch):
    _use_test_prompt_hashes(monkeypatch)
    durable_dir = tmp_path / "bundle"
    durable_dir.mkdir()
    review = {"schema_version": BerniniProofBundle.VISUAL_REVIEW_SCHEMA_VERSION, "cases": {}}
    (durable_dir / "visual_review.json").write_text(json.dumps(review))
    (durable_dir / "bernini_proof_report.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "passed": False,
                "visual_quality_passed": False,
                "visual_review": review,
                "runs": _current_portable_runs(),
            }
        )
    )
    (durable_dir / "sheet_manifest.json").write_text(
        json.dumps({"sheet_contract_version": BerniniProofBundle.SHEET_CONTRACT_VERSION})
    )
    payload = durable_dir / "evidence.bin"
    payload.write_bytes(b"evidence")
    compatibility = durable_dir / "component_compatibility.json"
    compatibility.write_text("{}")
    monkeypatch.setattr(BerniniProofBundle, "_verify_component_compatibility", staticmethod(lambda _: None))
    monkeypatch.setattr(
        BerniniProofBundle,
        "_verify_portable_report_evidence",
        staticmethod(lambda **_: {"passed": False, "visual_quality_passed": False}),
    )
    BerniniProofBundle._write_portable_manifest(durable_dir)

    result = BerniniProofBundle.verify_portable_bundle(durable_dir)

    assert result == {
        "verified": True,
        "integrity_verified": True,
        "quality_certified": False,
        "entry_count": 5,
        "report_passed": False,
        "visual_quality_passed": False,
    }
    compatibility.unlink()
    BerniniProofBundle._write_portable_manifest(durable_dir)
    with pytest.raises(ValueError, match="compatibility record is missing"):
        BerniniProofBundle.verify_portable_bundle(durable_dir)
    compatibility.write_text("{}")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    symlink = durable_dir / "linked.json"
    symlink.symlink_to(outside)
    BerniniProofBundle._write_portable_manifest(durable_dir)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        BerniniProofBundle.verify_portable_bundle(durable_dir)
    symlink.unlink()
    BerniniProofBundle._write_portable_manifest(durable_dir)
    extra = durable_dir / "unmanifested.txt"
    extra.write_text("extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        BerniniProofBundle.verify_portable_bundle(durable_dir)
    extra.unlink()
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity verification"):
        BerniniProofBundle.verify_portable_bundle(durable_dir)


def test_portable_bundle_verifier_rejects_stale_case_profile(tmp_path, monkeypatch):
    _use_test_prompt_hashes(monkeypatch)
    durable_dir = tmp_path / "bundle"
    durable_dir.mkdir()
    review = {"schema_version": BerniniProofBundle.VISUAL_REVIEW_SCHEMA_VERSION, "cases": {}}
    report = {
        "schema_version": 3,
        "passed": False,
        "visual_quality_passed": False,
        "visual_review": review,
        "runs": _current_portable_runs(),
    }
    (durable_dir / "visual_review.json").write_text(json.dumps(review))
    (durable_dir / "bernini_proof_report.json").write_text(json.dumps(report))
    (durable_dir / "sheet_manifest.json").write_text(
        json.dumps({"sheet_contract_version": BerniniProofBundle.SHEET_CONTRACT_VERSION})
    )
    (durable_dir / "component_compatibility.json").write_text("{}")
    monkeypatch.setattr(BerniniProofBundle, "_verify_component_compatibility", staticmethod(lambda _: None))
    monkeypatch.setattr(
        BerniniProofBundle,
        "_verify_portable_report_evidence",
        staticmethod(lambda **_: {"passed": False, "visual_quality_passed": False}),
    )
    BerniniProofBundle._write_portable_manifest(durable_dir)

    assert BerniniProofBundle.verify_portable_bundle(durable_dir)["verified"]

    report["runs"][0]["case"]["width"] += 16
    (durable_dir / "bernini_proof_report.json").write_text(json.dumps(report))
    BerniniProofBundle._write_portable_manifest(durable_dir)

    with pytest.raises(ValueError, match="is stale"):
        BerniniProofBundle.verify_portable_bundle(durable_dir)


def test_portable_profile_rejects_self_consistent_but_unpinned_prompt(monkeypatch):
    _use_test_prompt_hashes(monkeypatch)
    runs = _current_portable_runs()
    forged = runs[0]
    case = BerniniProofBundle.cases()[forged["case"]["case_id"]]
    forged["prompt"] = "self-consistent forged prompt"
    forged["case_fingerprint"] = BerniniProofBundle._case_fingerprint(case=case, prompt=forged["prompt"])
    report = json.loads(json.dumps({"runs": runs}))
    fingerprints = BerniniProofBundle._portable_case_fingerprints(report)
    manifest = {
        "case_fingerprints": fingerprints,
        "proof_profile_sha256": BerniniProofBundle._proof_profile_hash(fingerprints),
    }

    with pytest.raises(ValueError, match="pinned prompt source"):
        BerniniProofBundle._verify_current_portable_profile(report=report, manifest=manifest)


def test_primary_report_recomputation_rejects_aggregate_quality_claim_without_artifacts(tmp_path):
    report = {
        "kind": "bernini_r_1_3b_mlx_model_backed_proof",
        "official_source_revision": BerniniProofBundle.OFFICIAL_SOURCE_REVISION,
        "official_source_root": "<official-source-root>",
        "component_compatibility": BerniniProofBundle.EXPECTED_COMPONENT_PROVENANCE,
        "runs": [{"case": {"case_id": "forged"}, "passed": True}],
        "machine_contract_passed": True,
        "visual_review_complete": True,
        "visual_quality_passed": True,
        "passed": True,
    }

    with pytest.raises(ValueError, match="proof path|malformed input"):
        BerniniProofBundle._verify_portable_report_evidence(
            durable_dir=tmp_path,
            report=report,
            visual_review={"cases": {}},
            sheet_manifest={},
        )


def test_portable_run_recomputes_metadata_artifacts_memory_health_and_pass(tmp_path, monkeypatch):
    durable_dir = tmp_path / "bundle"
    case_dir = durable_dir / "cases" / "run_1" / "case"
    inputs_dir = case_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    paths = {
        "output": case_dir / "output_1f.mp4",
        "metadata": case_dir / "output_1f.metadata.json",
        "stdout": case_dir / "stdout.log",
        "stderr": case_dir / "stderr.log",
        "reference": inputs_dir / "reference.png",
        "source": inputs_dir / "source.mp4",
        "official": inputs_dir / "official.mp4",
        "sheet": case_dir / "mlx.png",
    }
    for label, path in paths.items():
        if label == "metadata":
            continue
        path.write_bytes(label.encode())
    portable = {label: f"<bundle-root>/{path.relative_to(durable_dir).as_posix()}" for label, path in paths.items()}
    metadata = {
        "reference_image_paths": [portable["reference"]],
        "video_path": portable["source"],
    }
    paths["metadata"].write_text(json.dumps(metadata))

    def artifact(label):
        path = paths[label]
        return {
            "path": portable[label],
            "sha256": BerniniProofBundle._sha256(path),
            "size_bytes": path.stat().st_size,
        }

    health = {
        "status": "ok",
        "kind": "video",
        "path": portable["output"],
        "frames": 1,
        "width": 16,
        "height": 16,
        "fps": 1.0,
    }
    samples = [
        {"rss_bytes": 10, "darwin_physical_footprint_bytes": 20},
        {"rss_bytes": 30, "darwin_physical_footprint_bytes": None},
    ]
    run = {
        "case": {"case_id": "case"},
        "output_path": portable["output"],
        "stdout_path": portable["stdout"],
        "stderr_path": portable["stderr"],
        "input_reference_paths": [portable["reference"]],
        "input_source_path": portable["source"],
        "official_output_path": portable["official"],
        "sheets": {"mlx": portable["sheet"]},
        "sheet_details": {"mlx": {}},
        "metadata": metadata,
        "video_health": health,
        "samples": samples,
        "sampler": {
            "sample_count": 2,
            "peak_sampled_rss_bytes": 30,
            "peak_sampled_darwin_physical_footprint_bytes": 20,
        },
        "artifact_hashes": {
            "output": artifact("output"),
            "metadata": artifact("metadata"),
            "stdout": artifact("stdout"),
            "stderr": artifact("stderr"),
            "references": [artifact("reference")],
            "source": artifact("source"),
            "official_output": artifact("official"),
            "sheets": {"mlx": artifact("sheet")},
        },
        "contract_checks": {"synthetic": True},
        "passed": True,
        "output_sha256": BerniniProofBundle._sha256(paths["output"]),
    }

    def video_health(path):
        return {**health, "path": str(path)}

    def contract_checks(resolved):
        assert resolved["metadata"]["reference_image_paths"] == [str(paths["reference"])]
        assert resolved["metadata"]["video_path"] == str(paths["source"])
        return {"synthetic": True}

    monkeypatch.setattr(
        BerniniProofBundle._verify_portable_run.__globals__["GenerationMemoryBenchmark"],
        "_video_health",
        staticmethod(video_health),
    )
    monkeypatch.setattr(BerniniProofBundle, "_verify_case_sheet_pixels", staticmethod(lambda **_: None))
    monkeypatch.setattr(BerniniProofBundle, "_contract_checks", staticmethod(contract_checks))

    assert BerniniProofBundle._verify_portable_run(durable_dir=durable_dir, run=run)["passed"]

    mutations = (
        ("memory", lambda value: value["sampler"].update(sample_count=1), "memory summary"),
        ("metadata", lambda value: value["metadata"].update(video_path=None), "metadata sidecar"),
        (
            "artifact",
            lambda value: value["artifact_hashes"]["output"].update(sha256="0" * 64),
            "artifact hash",
        ),
        ("health", lambda value: value["video_health"].update(frames=2), "video-health"),
        ("pass", lambda value: value.update(passed=False), "run contract"),
    )
    for _, mutate, message in mutations:
        changed = copy.deepcopy(run)
        mutate(changed)
        with pytest.raises(ValueError, match=message):
            BerniniProofBundle._verify_portable_run(durable_dir=durable_dir, run=changed)


def test_bundle_local_json_path_verifier_rejects_missing_and_machine_local_targets(tmp_path):
    durable_dir = tmp_path / "bundle"
    durable_dir.mkdir()
    target = durable_dir / "evidence" / "report.json"
    target.parent.mkdir()
    target.write_text("{}")
    record = durable_dir / "record.json"
    record.write_text(json.dumps({"path": "<bundle-root>/evidence/report.json"}))

    BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)

    record.write_text(json.dumps({"path": "<bundle-root>/evidence/missing.json"}))
    with pytest.raises(ValueError, match="unresolved bundle-local"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    record.write_text(json.dumps({"path": "/tmp/historical-bernini/output.json"}))
    with pytest.raises(ValueError, match="machine-local"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    record.write_text(json.dumps({"path": "<validation-root>/cycle/output.json"}))
    with pytest.raises(ValueError, match="validation-root dependency"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    record.write_text(json.dumps({"<bundle-root>/evidence/missing.json": "key-position"}))
    with pytest.raises(ValueError, match="unresolved bundle-local"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    record.write_text(json.dumps({"/tmp/historical-bernini/output.json": "key-position"}))
    with pytest.raises(ValueError, match="machine-local"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    record.write_text(json.dumps({"<validation-root>/cycle/output.json": "key-position"}))
    with pytest.raises(ValueError, match="validation-root dependency"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    linked = durable_dir / "linked.json"
    linked.symlink_to(outside)
    record.write_text(json.dumps({"path": "<bundle-root>/linked.json"}))
    with pytest.raises(ValueError, match="unresolved bundle-local"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)


def test_bundle_local_json_path_verifier_rejects_embedded_and_cross_platform_local_paths(tmp_path):
    durable_dir = tmp_path / "bundle"
    durable_dir.mkdir()
    target = durable_dir / "evidence" / "report.json"
    target.parent.mkdir()
    target.write_text("{}")
    record = durable_dir / "record.json"
    record.write_text(json.dumps({"note": "bound at <bundle-root>/evidence/report.json exactly"}))

    BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)

    for leaked in (
        "/private/var/folders/xx/run/report.json",
        "/var/folders/xx/run/report.json",
        "/home/user/run/report.json",
        "/Volumes/work/run/report.json",
        r"C:\\Users\\user\\run\\report.json",
        "~/run/report.json",
    ):
        record.write_text(json.dumps({"path": leaked}))
        with pytest.raises(ValueError, match="machine-local"):
            BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
    record.write_text(json.dumps({"note": "prefix <bundle-root>/evidence/missing.json suffix"}))
    with pytest.raises(ValueError, match="unresolved bundle-local"):
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)


def test_durable_export_target_guard_rejects_protected_and_unmarked_paths(tmp_path, monkeypatch):
    user_home = tmp_path / "home"
    repository = user_home / "projects" / "mlx-gen"
    output = repository / "validation_outputs" / "proof"
    reference = tmp_path / "official" / "Bernini"
    for path in (repository, output, reference):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: user_home))

    for unsafe in (
        Path("/"),
        user_home,
        user_home / "projects",
        repository,
        output,
        output / "bundle",
        reference,
        reference / "bundle",
        tmp_path,
    ):
        with pytest.raises(ValueError, match="Refusing"):
            BerniniProofBundle._validate_durable_target(
                durable_dir=unsafe,
                output_dir=output,
                reference_root=reference,
                repository_root=repository,
            )

    unmarked = repository / "docs" / "proof"
    unmarked.mkdir(parents=True)
    (unmarked / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="unmarked"):
        BerniniProofBundle._validate_durable_target(
            durable_dir=unmarked,
            output_dir=output,
            reference_root=reference,
            repository_root=repository,
        )

    safe = repository / "docs" / "new-proof"
    BerniniProofBundle._validate_durable_target(
        durable_dir=safe,
        output_dir=output,
        reference_root=reference,
        repository_root=repository,
    )
    safe.mkdir(parents=True)
    (safe / "portable_manifest.json").write_text(json.dumps({"kind": "portable_bernini_r_1_3b_proof_bundle"}))
    BerniniProofBundle._validate_durable_target(
        durable_dir=safe,
        output_dir=output,
        reference_root=reference,
        repository_root=repository,
    )


def test_component_compatibility_verifies_negative_visual_review_artifact_bindings(tmp_path, monkeypatch):
    durable_dir = tmp_path / "bundle"
    diagnosis_dir = durable_dir / "diagnostics" / "final_latent_three_way_decode"
    diagnosis_dir.mkdir(parents=True)
    backends = ("mlx_tiled_runtime", "mlx_untiled", "torch_diffusers_0_35_2")
    native_hashes = {}
    videos = {}
    sheets = {}
    reviewed = {}
    decoded_videos = {}
    for backend_index, name in enumerate(backends):
        video_path = diagnosis_dir / f"{name}.mp4"
        sheet_path = diagnosis_dir / f"{name}_all_frames_5k.png"
        video_path.write_bytes(f"video-{name}".encode())
        hashes = []
        frames = []
        native_dir = diagnosis_dir / "native_frames" / name
        native_dir.mkdir(parents=True)
        for index in range(17):
            image = Image.new("RGB", (320, 176), (32 + backend_index * 20, 16 + index * 8, 7))
            image.save(native_dir / f"frame_{index:03d}.png")
            hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
            frames.append(image)
        sheet = Image.new("RGB", (5280, 4352), "#17191d")
        for index, image in enumerate(frames):
            row, column = divmod(index, 4)
            x = 32 + column * (1280 + 32)
            y = 160 + 32 + row * (704 + 96 + 32)
            sheet.paste(image.resize((1280, 704), Image.Resampling.NEAREST), (x, y))
        sheet.save(sheet_path)
        videos[name] = BerniniProofBundle._sha256(video_path)
        sheets[name] = BerniniProofBundle._sha256(sheet_path)
        native_hashes[name] = hashes
        reviewed[name] = list(range(17))
        decoded_videos[name] = frames

    def read_video(path, **_):
        frames = decoded_videos[Path(path).stem]
        return SimpleNamespace(
            frames=frames,
            clip_frame_count=17,
            source_width=320,
            source_height=176,
            fps=16.0,
        )

    monkeypatch.setattr(
        BerniniProofBundle,
        "_verify_cycle17_tensor_artifacts",
        staticmethod(lambda **_: None),
    )
    monkeypatch.setattr(
        BerniniProofBundle._verify_component_compatibility.__globals__["VideoUtil"], "read_video_clip", read_video
    )

    mlx_report = {"schema_version": 1, "kind": "bernini_mlx_final_latent_decode_diagnosis"}
    torch_report = {"schema_version": 1, "kind": "bernini_torch_final_latent_decode_diagnosis"}
    (diagnosis_dir / "mlx_report.json").write_text(json.dumps(mlx_report))
    (diagnosis_dir / "torch_report.json").write_text(json.dumps(torch_report))

    review = {
        "schema_version": 1,
        "kind": "bernini_identical_final_latent_three_way_decode_manual_review",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": "test reviewer",
        "execution_mode": "native-frame-and-high-resolution-contact-sheet-review",
        "status": "negative_result",
        "scope": "all native frames and high-resolution sheets",
        "findings": {"decode_implementations_materially_agree": True},
        "reviewed_native_frame_indices": reviewed,
        "artifact_sha256": {
            "videos": videos,
            "contact_sheets": sheets,
            "native_frames": native_hashes,
        },
    }
    review_path = diagnosis_dir / "manual_visual_review.json"
    review_path.write_text(json.dumps(review))
    diagnosis = {
        "schema_version": 2,
        "kind": "bernini_identical_final_latent_three_way_decode",
        "structural_checks_passed": True,
        "visual_disposition": "negative_result",
        "mlx_report": mlx_report,
        "torch_report": torch_report,
        "native_frame_sha256": native_hashes,
        "encoded_video_metrics": {
            "tiled_vs_untiled": BerniniProofBundle._array_comparison_metrics(
                np.stack([np.asarray(frame, dtype=np.float64) for frame in decoded_videos["mlx_untiled"]]),
                np.stack([np.asarray(frame, dtype=np.float64) for frame in decoded_videos["mlx_tiled_runtime"]]),
                data_range=255.0,
            ),
            "torch_vs_mlx_untiled": BerniniProofBundle._array_comparison_metrics(
                np.stack([np.asarray(frame, dtype=np.float64) for frame in decoded_videos["torch_diffusers_0_35_2"]]),
                np.stack([np.asarray(frame, dtype=np.float64) for frame in decoded_videos["mlx_untiled"]]),
                data_range=255.0,
            ),
        },
        "manual_visual_review": {
            "path": "<bundle-root>/diagnostics/final_latent_three_way_decode/manual_visual_review.json",
            "sha256": BerniniProofBundle._sha256(review_path),
        },
    }
    diagnosis_path = diagnosis_dir / "decode_comparison_report.json"
    diagnosis_path.write_text(json.dumps(diagnosis))
    for label, relative_path in BerniniProofBundle.EXPECTED_PARITY_REPORTS.items():
        if label.startswith("final_latent_three_way_decode"):
            continue
        path = durable_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        schema_version, kind = BerniniProofBundle.EXPECTED_PARITY_REPORT_HEADERS[label]
        path.write_text(json.dumps({"schema_version": schema_version, "kind": kind, "passed": True}))
    compatibility = BerniniProofBundle._expected_component_compatibility()
    (durable_dir / "component_compatibility.json").write_text(json.dumps(compatibility))

    BerniniProofBundle._verify_component_compatibility(durable_dir)

    Image.new("RGB", (2, 2), "red").save(diagnosis_dir / "native_frames" / "mlx_tiled_runtime" / "frame_001.png")
    with pytest.raises(ValueError, match="native-frame .* failed"):
        BerniniProofBundle._verify_component_compatibility(durable_dir)
