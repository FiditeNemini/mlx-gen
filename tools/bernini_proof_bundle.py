import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from generation_memory_benchmark import CommandVariant, GenerationMemoryBenchmark
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageFont,
    ImageOps,
    __version__ as PILLOW_VERSION,
)

from mflux.models.wan.model.wan_vae.wan_2_2_vae import Wan2_2_VAE
from mflux.models.wan.variants.wan_bernini import BerniniRenderer
from mflux.models.wan.wan_text_encoder_loader import WanTextEncoderLoader
from mflux.models.wan.weights import WanWeightDefinition
from mflux.utils.video_util import VideoUtil


@dataclass(frozen=True)
class BerniniProofCase:
    case_id: str
    proof_kind: str
    prompt_json: str | None
    prompt_override: str | None
    reference_images: tuple[str, ...]
    source_video: str | None
    official_output: str | None
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    seed: int
    max_condition_size: int


class BerniniProofBundle:
    CASE_CONTRACT_VERSION = 5
    SHEET_CONTRACT_VERSION = 5
    VISUAL_REVIEW_SCHEMA_VERSION = 4
    MAX_REVIEW_CLOCK_SKEW_SECONDS = 300
    MAX_QUALITY_LATENT_BOUNDARY_RATIO = 1.30
    TRANSITION_TILE_SIZE = 32
    OFFICIAL_SOURCE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
    EXPECTED_COMPONENT_PROVENANCE = {
        "text_encoder": {
            "source": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "source_role": "base",
            "revision": "ec4d2cb062b548996b179d493fdd05340de702a1",
        },
        "tokenizer": {
            "source": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "source_role": "base",
            "revision": "ec4d2cb062b548996b179d493fdd05340de702a1",
        },
        "vae": {
            "source": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "source_role": "base",
            "revision": "ec4d2cb062b548996b179d493fdd05340de702a1",
        },
        "transformer": {
            "source": "ByteDance/Bernini-R-1.3B-Diffusers",
            "source_role": "transformer",
            "revision": "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce",
        },
    }
    EXPECTED_RUNTIME_POLICY = {
        "text_encoder_precision_policy_id": WanTextEncoderLoader.BERNINI_PRECISION_POLICY_ID,
        "transformer_precision_policy_id": WanWeightDefinition.BERNINI_TRANSFORMER_PRECISION_POLICY_ID,
        "transformer_default_weight_precision": "bfloat16",
        "transformer_fp32_weight_keys": list(WanWeightDefinition.BERNINI_TRANSFORMER_FP32_KEYS),
        "low_ram": True,
        "vae_low_memory_policy_active": True,
        "clear_cache_each_transformer_block": True,
        "release_denoisers_before_decode": True,
        "vae_feature_cache_policy_id": Wan2_2_VAE.COMPACT_FEATURE_CACHE_POLICY_ID,
        "vae_encode_cache_materialization": "eager-contiguous-per-slice",
        "vae_decode_cache_materialization": "eager-contiguous-per-slice",
        "vae_spatial_tiling": True,
        "vae_spatial_tiling_policy_id": Wan2_2_VAE.SPATIAL_TILING_POLICY_ID,
        "wan_decode_mode": Wan2_2_VAE.TILED_DECODE_MODE,
    }
    REQUIRED_RUNTIME_ENVIRONMENT_FIELDS = (
        "mlx_version",
        "python_version",
        "python_implementation",
        "runtime_platform",
        "numpy_version",
        "python_executable",
    )
    IMPLEMENTATION_GLOBS = ("src/mflux/**/*.py",)
    IMPLEMENTATION_PATHS = (
        "pyproject.toml",
        "uv.lock",
        "tools/bernini_proof_bundle.py",
        "tools/generation_memory_benchmark.py",
    )
    SUPPLEMENTAL_EVIDENCE_DIRS = {
        "cycle15_tiled_vae_parity": "parity/vae_tiled_9f_272x320",
        "cycle16_tiled_vae_memory_848x480x81_corrected": "memory/vae_tiled_81f_848x480",
        "cycle17_latent_decode_diagnosis": "diagnostics/final_latent_three_way_decode",
        "cycle12_umt5_exact": "diagnostics/cycle12_source_generation",
    }
    EXPECTED_PARITY_REPORTS = {
        "scheduler": "parity/scheduler_diffusers_0_35_2_report.json",
        "apg": "parity/apg/parity_report.json",
        "transformer_fp32": "parity/transformer/parity_report.json",
        "transformer_runtime_bf16_five_slice": "parity/transformer_5slice/parity_runtime_report.json",
        "vae_bf16_17_frame": "parity/vae_17f/vae_parity_report.json",
        "vae_tiled_bf16_9_frame": "parity/vae_tiled_9f_272x320/vae_parity_report.json",
        "vae_tiled_memory_81_frame": "memory/vae_tiled_81f_848x480/vae_memory_report.json",
        "final_latent_three_way_decode": ("diagnostics/final_latent_three_way_decode/decode_comparison_report.json"),
        "final_latent_three_way_decode_visual_review": (
            "diagnostics/final_latent_three_way_decode/manual_visual_review.json"
        ),
    }
    EXPECTED_RUNTIME_CONTRACT = {
        "scheduler": "UniPCMultistepScheduler",
        "scheduler_reference": "diffusers-0.35.2",
        "flow_shift": 5.0,
        "source_conditioning": "independent-vae-packed-segments",
        "reference_count": {"minimum": 0, "maximum": 8},
        "max_condition_size": {"minimum": 16, "maximum": 1280, "multiple": 16},
    }
    EXPECTED_PARITY_REPORT_HEADERS = {
        "scheduler": (1, "bernini_diffusers_0_35_2_unipc_parity"),
        "apg": (1, "bernini_official_apg_parity"),
        "transformer_fp32": (1, "bernini_official_full_transformer_parity"),
        "transformer_runtime_bf16_five_slice": (2, "bernini_official_full_transformer_parity"),
        "vae_bf16_17_frame": (2, "bernini_wan21_vae_default_bf16_vs_official_fp32"),
        "vae_tiled_bf16_9_frame": (2, "bernini_wan21_vae_tiled_bf16_vs_official_fp32"),
        "vae_tiled_memory_81_frame": (1, "bernini_wan2_1_tiled_vae_production_geometry_memory"),
    }
    EXPECTED_UPSTREAM_PROMPT_SHA256 = {
        "assets/testcases/r2v/r2v_case2.json": "c391fd8b8f398b7eed42b221e63f7d5afd705bfec99d1cbef7b87b0bc0be7421",
        "assets/testcases/rv2v/rv2v_case1.json": "c75184e9a2266ac6321b8ab6fc97746da3812d592f9c84b16c178bf7de53a86f",
        "assets/testcases/v2v/v2v_case1.json": "bb6a4e91c0a89a33465d939cae70d0ae3de653d05fba08067221c040bc94dc0c",
    }
    REVIEWED_STATUSES = {"pass", "pass_with_limitations", "negative_result", "structural_only", "fail"}
    QUALITY_ACCEPTED_STATUSES = {"pass", "pass_with_limitations"}
    QUALITY_CASE_IDS = {
        "r2v_eight_reference",
        "rv2v_garment",
        "rv2v_reference_pinstripe_ab",
        "rv2v_reference_black_ab",
        "v2v_snowman",
    }
    TIMELINE_COLUMNS = 3
    TIMELINE_CELL_SIZE = 1664
    TIMELINE_PAGE_SIZE = 9
    TRANSITION_COLUMNS = 2
    SUMMARY_COLUMNS = 4
    SUMMARY_CELL_WIDTH = 1280
    SUMMARY_CELL_HEIGHT = 960
    SHEET_PADDING = 48
    SHEET_HEADER_HEIGHT = 240
    SHEET_LABEL_HEIGHT = 128
    TITLE_FONT_SIZE = 88
    LABEL_FONT_SIZE = 72

    @staticmethod
    def cases() -> dict[str, BerniniProofCase]:
        r2v_images = tuple(f"assets/testcases/r2v/source_img{index}.png" for index in range(8))
        return {
            "r2v_eight_reference": BerniniProofCase(
                case_id="r2v_eight_reference",
                proof_kind="recommended-quality-and-eight-reference-bound",
                prompt_json="assets/testcases/r2v/r2v_case2.json",
                prompt_override=None,
                reference_images=r2v_images,
                source_video=None,
                official_output="assets/testcases/r2v/r2v_case2_out.mp4",
                width=848,
                height=480,
                frames=81,
                fps=16,
                steps=40,
                seed=8108,
                max_condition_size=848,
            ),
            "rv2v_garment": BerniniProofCase(
                case_id="rv2v_garment",
                proof_kind="recommended-quality-and-role-control",
                prompt_json="assets/testcases/rv2v/rv2v_case1.json",
                prompt_override=None,
                reference_images=("assets/testcases/rv2v/ref_case1.jpg",),
                source_video="assets/testcases/rv2v/source_case1.mp4",
                official_output="assets/testcases/rv2v/rv2v_case1_out.mp4",
                width=480,
                height=848,
                frames=81,
                fps=16,
                steps=40,
                seed=8106,
                max_condition_size=848,
            ),
            "rv2v_no_reference_control": BerniniProofCase(
                case_id="rv2v_no_reference_control",
                proof_kind="same-seed-source-only-control",
                prompt_json="assets/testcases/rv2v/rv2v_case1.json",
                prompt_override=None,
                reference_images=(),
                source_video="assets/testcases/rv2v/source_case1.mp4",
                official_output=None,
                width=176,
                height=320,
                frames=17,
                fps=16,
                steps=20,
                seed=8106,
                max_condition_size=320,
            ),
            "rv2v_no_source_control": BerniniProofCase(
                case_id="rv2v_no_source_control",
                proof_kind="same-seed-reference-only-control",
                prompt_json="assets/testcases/rv2v/rv2v_case1.json",
                prompt_override=None,
                reference_images=("assets/testcases/rv2v/ref_case1.jpg",),
                source_video=None,
                official_output=None,
                width=176,
                height=320,
                frames=17,
                fps=16,
                steps=20,
                seed=8106,
                max_condition_size=320,
            ),
            "rv2v_reference_pinstripe_ab": BerniniProofCase(
                case_id="rv2v_reference_pinstripe_ab",
                proof_kind="recommended-quality-same-prompt-same-seed-reference-ab-pinstripe",
                prompt_json=None,
                prompt_override=(
                    "Replace the person's outer shirt with the garment from image0 while keeping the inner "
                    "undershirt unchanged. Preserve the original person, body pose, camera framing, lighting, "
                    "background, pants, hair, skin, shadows, and motion."
                ),
                reference_images=("assets/testcases/rv2v/ref_case1.jpg",),
                source_video="assets/testcases/rv2v/source_case1.mp4",
                official_output=None,
                width=480,
                height=848,
                frames=81,
                fps=16,
                steps=40,
                seed=8112,
                max_condition_size=848,
            ),
            "rv2v_reference_black_ab": BerniniProofCase(
                case_id="rv2v_reference_black_ab",
                proof_kind="recommended-quality-same-prompt-same-seed-reference-ab-black",
                prompt_json=None,
                prompt_override=(
                    "Replace the person's outer shirt with the garment from image0 while keeping the inner "
                    "undershirt unchanged. Preserve the original person, body pose, camera framing, lighting, "
                    "background, pants, hair, skin, shadows, and motion."
                ),
                reference_images=("assets/testcases/r2v/source_img2.png",),
                source_video="assets/testcases/rv2v/source_case1.mp4",
                official_output=None,
                width=480,
                height=848,
                frames=81,
                fps=16,
                steps=40,
                seed=8112,
                max_condition_size=848,
            ),
            "rv2v_reference_none_ab": BerniniProofCase(
                case_id="rv2v_reference_none_ab",
                proof_kind="same-prompt-same-seed-no-reference-control",
                prompt_json=None,
                prompt_override=(
                    "Replace the person's outer shirt with the garment from image0 while keeping the inner "
                    "undershirt unchanged. Preserve the original person, body pose, camera framing, lighting, "
                    "background, pants, hair, skin, shadows, and motion."
                ),
                reference_images=(),
                source_video="assets/testcases/rv2v/source_case1.mp4",
                official_output=None,
                width=176,
                height=320,
                frames=17,
                fps=16,
                steps=20,
                seed=8112,
                max_condition_size=320,
            ),
            "v2v_snowman": BerniniProofCase(
                case_id="v2v_snowman",
                proof_kind="recommended-quality",
                prompt_json="assets/testcases/v2v/v2v_case1.json",
                prompt_override=None,
                reference_images=(),
                source_video="assets/testcases/v2v/source_case1.mp4",
                official_output="assets/testcases/v2v/v2v_case1_out.mp4",
                width=848,
                height=480,
                frames=81,
                fps=16,
                steps=40,
                seed=8107,
                max_condition_size=848,
            ),
            "r2v_848_condition_smoke": BerniniProofCase(
                case_id="r2v_848_condition_smoke",
                proof_kind="condition-size-memory-control",
                prompt_json=None,
                prompt_override="The referenced marble statue slowly turns its head, fixed camera.",
                reference_images=("assets/testcases/r2v/source_img0.png",),
                source_video=None,
                official_output=None,
                width=128,
                height=128,
                frames=5,
                fps=8,
                steps=1,
                seed=8110,
                max_condition_size=848,
            ),
            "r2v_1280_condition_smoke": BerniniProofCase(
                case_id="r2v_1280_condition_smoke",
                proof_kind="condition-size-memory-bound",
                prompt_json=None,
                prompt_override="The referenced marble statue slowly turns its head, fixed camera.",
                reference_images=("assets/testcases/r2v/source_img0.png",),
                source_video=None,
                official_output=None,
                width=128,
                height=128,
                frames=5,
                fps=8,
                steps=1,
                seed=8110,
                max_condition_size=1280,
            ),
        }

    @staticmethod
    def main() -> None:
        args = BerniniProofBundle._parse_args()
        reference_root = args.reference_root.resolve()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        BerniniProofBundle._validate_reference_root(reference_root)
        os.environ["MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING"] = "1"
        selected = [BerniniProofBundle.cases()[name] for name in args.cases]
        runs_by_id = BerniniProofBundle._existing_runs(output_dir)
        for index, case in enumerate(selected, start=1):
            existing = runs_by_id.get(case.case_id)
            prompt = BerniniProofBundle._prompt(case=case, reference_root=reference_root)
            if not args.rerun and BerniniProofBundle._can_reuse(existing=existing, case=case, prompt=prompt):
                print(f"[{index}/{len(selected)}] reusing {case.case_id}", flush=True)
                existing["case_fingerprint"] = BerniniProofBundle._case_fingerprint(case=case, prompt=prompt)
                continue
            print(f"[{index}/{len(selected)}] running {case.case_id}", flush=True)
            runs_by_id[case.case_id] = BerniniProofBundle._run_case(
                case=case,
                reference_root=reference_root,
                output_dir=output_dir,
                sample_interval_ms=args.sample_interval_ms,
            )
            runs = BerniniProofBundle._ordered_runs(runs_by_id)
            BerniniProofBundle._write_report(output_dir=output_dir, reference_root=reference_root, runs=runs)
        runs = BerniniProofBundle._ordered_runs(runs_by_id)
        BerniniProofBundle._refresh_case_sheets(runs=runs)
        summary_sheet_details = BerniniProofBundle._save_summary_sheet(
            runs=runs,
            output_path=output_dir / "output_summary_contact_sheet.png",
        )
        role_control_sheet_details = BerniniProofBundle._save_role_control_sheet(
            runs=runs, output_path=output_dir / "role_control_contact_sheet.png"
        )
        BerniniProofBundle._write_sheet_manifest(
            output_dir=output_dir,
            runs=runs,
            summary_sheet_details=summary_sheet_details,
            role_control_sheet_details=role_control_sheet_details,
        )
        report = BerniniProofBundle._write_report(
            output_dir=output_dir,
            reference_root=reference_root,
            runs=runs,
            summary_sheet_details=summary_sheet_details,
            role_control_sheet_details=role_control_sheet_details,
        )
        if args.durable_dir is not None:
            BerniniProofBundle._export_durable_bundle(
                output_dir=output_dir,
                durable_dir=args.durable_dir.resolve(),
                reference_root=reference_root,
            )
        print(json.dumps(BerniniProofBundle._compact_report(report), indent=2, sort_keys=True), flush=True)
        if not report["passed"]:
            raise SystemExit(1)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Generate the model-backed Bernini MP4, contact-sheet, provenance, and memory proof bundle."
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("validation_outputs/bernini_r_1_3b_2026_08_03/cycle4_proof"),
        )
        parser.add_argument("--reference-root", type=Path, required=True)
        parser.add_argument(
            "--durable-dir",
            type=Path,
            help="Copy a sanitized, self-contained proof bundle to this documentation directory.",
        )
        parser.add_argument("--sample-interval-ms", type=int, default=250)
        parser.add_argument(
            "--rerun", action="store_true", help="Regenerate selected cases even when proof outputs exist."
        )
        parser.add_argument(
            "--cases",
            nargs="+",
            choices=tuple(BerniniProofBundle.cases()),
            default=tuple(BerniniProofBundle.cases()),
        )
        args = parser.parse_args()
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        return args

    @staticmethod
    def _existing_runs(output_dir: Path) -> dict[str, dict[str, Any]]:
        report_path = output_dir / "bernini_proof_report.json"
        if not report_path.is_file():
            return {}
        report = json.loads(report_path.read_text())
        return {run["case"]["case_id"]: run for run in report.get("runs", [])}

    @staticmethod
    def _ordered_runs(runs_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        order = {case_id: index for index, case_id in enumerate(BerniniProofBundle.cases())}
        return sorted(runs_by_id.values(), key=lambda run: order.get(run["case"]["case_id"], len(order)))

    @staticmethod
    def _validate_reference_root(reference_root: Path) -> None:
        if (
            not (reference_root / "assets" / "testcases" / "README.md").is_file()
            or not (reference_root / "LICENSE").is_file()
        ):
            raise FileNotFoundError(f"Official Bernini testcases were not found below {reference_root}.")
        repository_root = BerniniProofBundle._git_output(reference_root, "rev-parse", "--show-toplevel")
        if Path(repository_root).resolve() != reference_root.resolve():
            raise ValueError(f"Bernini reference root is not the root of its Git checkout: {reference_root}")
        revision = BerniniProofBundle._git_output(reference_root, "rev-parse", "HEAD")
        if revision != BerniniProofBundle.OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                "Bernini proof inputs require official revision "
                f"{BerniniProofBundle.OFFICIAL_SOURCE_REVISION}, got {revision}."
            )
        changed_inputs = BerniniProofBundle._git_output(
            reference_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "assets/testcases",
            "LICENSE",
        )
        if changed_inputs:
            raise ValueError("Bernini proof inputs differ from the pinned official revision.")

    @staticmethod
    def _git_output(reference_root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(reference_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise ValueError(f"Could not validate the Bernini reference checkout: {detail}")
        return result.stdout.strip()

    @staticmethod
    def _run_case(
        *,
        case: BerniniProofCase,
        reference_root: Path,
        output_dir: Path,
        sample_interval_ms: int,
    ) -> dict[str, Any]:
        run_dir = output_dir / "cases" / "run_1" / case.case_id
        input_dir = run_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        prompt = BerniniProofBundle._prompt(case=case, reference_root=reference_root)
        reference_paths = [
            BerniniProofBundle._copy_input(
                reference_root / relative, input_dir / f"reference_{index:02d}{Path(relative).suffix}"
            )
            for index, relative in enumerate(case.reference_images)
        ]
        source_path = (
            BerniniProofBundle._copy_input(
                reference_root / case.source_video,
                input_dir / f"source{Path(case.source_video).suffix}",
            )
            if case.source_video is not None
            else None
        )
        official_path = (
            BerniniProofBundle._copy_input(
                reference_root / case.official_output,
                input_dir / f"official_output{Path(case.official_output).suffix}",
            )
            if case.official_output is not None
            else None
        )
        argv = BerniniProofBundle._command(
            case=case,
            prompt=prompt,
            reference_paths=reference_paths,
            source_path=source_path,
            output_path=Path(f"{case.case_id}_{case.frames}f.mp4"),
        )
        variant = CommandVariant(
            name=case.case_id,
            argv=argv,
            output_path=Path(f"{case.case_id}_{case.frames}f.mp4"),
            cwd=Path.cwd(),
            env={
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING": "1",
            },
        )
        result = GenerationMemoryBenchmark._run_variant(
            profile="cases",
            variant=variant,
            run_index=1,
            output_dir=output_dir,
            sample_interval_ms=sample_interval_ms,
        )
        output_path = Path(result["output_path"])
        metadata_path = output_path.with_suffix(".metadata.json")
        output_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        sheets, sheet_details = BerniniProofBundle._save_case_sheets(
            case=case,
            reference_paths=reference_paths,
            source_path=source_path,
            official_path=official_path,
            output_path=output_path,
            run_dir=run_dir,
            output_metadata=output_metadata,
        )
        result.update(
            {
                "case": asdict(case),
                "case_fingerprint": BerniniProofBundle._case_fingerprint(case=case, prompt=prompt),
                "prompt": prompt,
                "input_reference_paths": [str(path) for path in reference_paths],
                "input_source_path": str(source_path) if source_path is not None else None,
                "official_output_path": str(official_path) if official_path is not None else None,
                "metadata": output_metadata,
                "sheets": sheets,
                "sheet_details": sheet_details,
                "video_health": GenerationMemoryBenchmark._video_health(output_path),
            }
        )
        result["contract_checks"] = BerniniProofBundle._contract_checks(result)
        result["passed"] = result["video_health"]["status"] == "ok" and all(result["contract_checks"].values())
        return result

    @staticmethod
    def _can_reuse(*, existing: dict[str, Any] | None, case: BerniniProofCase, prompt: str) -> bool:
        if existing is None or not Path(existing.get("output_path", "")).is_file():
            return False
        expected_fingerprint = BerniniProofBundle._case_fingerprint(case=case, prompt=prompt)
        stored_fingerprint = existing.get("case_fingerprint")
        return stored_fingerprint == expected_fingerprint

    @staticmethod
    def _case_fingerprint(*, case: BerniniProofCase, prompt: str) -> str:
        payload = {
            "case_contract_version": BerniniProofBundle.CASE_CONTRACT_VERSION,
            "case": asdict(case),
            "prompt": prompt,
            "model": "bernini-r-1.3b",
            "official_source_revision": BerniniProofBundle.OFFICIAL_SOURCE_REVISION,
            "component_provenance": BerniniProofBundle.EXPECTED_COMPONENT_PROVENANCE,
            "runtime_policy": BerniniProofBundle.EXPECTED_RUNTIME_POLICY,
            "implementation_fingerprint": BerniniProofBundle._implementation_fingerprint(),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _implementation_fingerprint() -> str:
        repository_root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for path in BerniniProofBundle._implementation_files(repository_root=repository_root):
            relative_path = path.relative_to(repository_root).as_posix()
            digest.update(relative_path.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _implementation_files(*, repository_root: Path) -> list[Path]:
        paths = {repository_root / relative_path for relative_path in BerniniProofBundle.IMPLEMENTATION_PATHS}
        for pattern in BerniniProofBundle.IMPLEMENTATION_GLOBS:
            paths.update(path for path in repository_root.glob(pattern) if path.is_file())
        if not paths:
            raise FileNotFoundError("Bernini proof implementation fingerprint has no inputs.")
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Bernini proof implementation input is missing: {missing[0]}")
        return sorted(paths)

    @staticmethod
    def _prompt(*, case: BerniniProofCase, reference_root: Path) -> str:
        if case.prompt_override is not None:
            return case.prompt_override
        if case.prompt_json is None:
            raise ValueError(f"{case.case_id} has no prompt source.")
        return str(json.loads((reference_root / case.prompt_json).read_text())["prompt"])

    @staticmethod
    def _copy_input(source: Path, target: Path) -> Path:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        return target.resolve()

    @staticmethod
    def _command(
        *,
        case: BerniniProofCase,
        prompt: str,
        reference_paths: list[Path],
        source_path: Path | None,
        output_path: Path,
    ) -> list[str]:
        argv = [
            "uv",
            "run",
            "mlxgen",
            "generate",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            prompt,
            "--width",
            str(case.width),
            "--height",
            str(case.height),
            "--frames",
            str(case.frames),
            "--fps",
            str(case.fps),
            "--steps",
            str(case.steps),
            "--seed",
            str(case.seed),
            "--max-condition-size",
            str(case.max_condition_size),
            "--low-ram",
            "--metadata",
            "--failure-diagnostics",
            "--replace",
            "--no-progress",
            "--output",
            str(output_path),
        ]
        if source_path is not None:
            argv.extend(["--video", str(source_path)])
        for path in reference_paths:
            argv.extend(["--reference-image", str(path)])
        return argv

    @staticmethod
    def _save_case_sheets(
        *,
        case: BerniniProofCase,
        reference_paths: list[Path],
        source_path: Path | None,
        official_path: Path | None,
        output_path: Path,
        run_dir: Path,
        output_metadata: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        sheets: dict[str, str] = {}
        details: dict[str, dict[str, Any]] = {}
        if reference_paths:
            path = run_dir / "reference_contact_sheet.png"
            reference_images = [Image.open(reference).convert("RGB") for reference in reference_paths]
            details["references"] = BerniniProofBundle._save_image_sheet(
                images=reference_images,
                output_path=path,
                title=f"{case.case_id}: ordered references",
                indices=list(range(len(reference_images))),
                label_prefix="image",
                resampling=Image.Resampling.LANCZOS,
                fixed_columns=BerniniProofBundle.TIMELINE_COLUMNS,
                cell_size=BerniniProofBundle.TIMELINE_CELL_SIZE,
            )
            sheets["references"] = str(path)
        for label, video_path in (("source", source_path), ("official", official_path), ("mlx", output_path)):
            if video_path is None:
                continue
            clip = VideoUtil.read_video_clip(video_path, max_frames=None)
            sheet_images = clip.frames
            if label == "source":
                indices = [int(index) for index in output_metadata.get("source_sample_indices", [])]
                if not indices:
                    raise ValueError(f"{case.case_id} source proof is missing source_sample_indices metadata.")
                condition_width = int(output_metadata.get("video_condition_width", 0))
                condition_height = int(output_metadata.get("video_condition_height", 0))
                if condition_width <= 0 or condition_height <= 0:
                    raise ValueError(f"{case.case_id} source proof is missing conditioned source dimensions.")
                sheet_images = [
                    frame.convert("RGB").resize(
                        (condition_width, condition_height),
                        resample=Image.Resampling.BICUBIC,
                    )
                    for frame in clip.frames
                ]
            elif label in {"official", "mlx"}:
                indices = list(range(len(sheet_images)))
            title_label = (
                "upstream qualitative target (checkpoint and recipe unattested)"
                if label == "official"
                else "conditioned source timeline"
                if label == "source"
                else "mlx timeline"
            )
            group_sheets, group_details = BerniniProofBundle._save_timeline_sheet_group(
                images=sheet_images,
                run_dir=run_dir,
                label=label,
                title=f"{case.case_id}: {title_label}",
                indices=indices,
                label_prefix="frame",
                fps=float(clip.fps),
                resampling=(Image.Resampling.NEAREST if label == "mlx" else Image.Resampling.LANCZOS),
                cell_size=BerniniProofBundle.TIMELINE_CELL_SIZE,
            )
            sheets.update(group_sheets)
            details.update(group_details)
            if label == "mlx" and len(sheet_images) > 1:
                transition_metrics = BerniniProofBundle._transition_metrics(sheet_images)
                transition_starts = sorted(
                    range(len(transition_metrics)),
                    key=lambda index: (
                        -transition_metrics[index]["max_tile_mae"],
                        -transition_metrics[index]["global_mae"],
                        index,
                    ),
                )[: min(5, len(transition_metrics))]
                transition_indices = [index for start in transition_starts for index in (start, start + 1)]
                transition_path = run_dir / "mlx_worst_transitions_contact_sheet.png"
                transition_details = BerniniProofBundle._save_image_sheet(
                    images=sheet_images,
                    output_path=transition_path,
                    title=f"{case.case_id}: highest localized-change transitions · inspect each before/after pair",
                    indices=transition_indices,
                    label_prefix="frame",
                    fps=float(clip.fps),
                    resampling=Image.Resampling.NEAREST,
                    fixed_columns=BerniniProofBundle.TRANSITION_COLUMNS,
                    cell_size=BerniniProofBundle.TIMELINE_CELL_SIZE,
                )
                transition_details.update(
                    {
                        "transition_start_indices": transition_starts,
                        "transition_global_mae": [
                            round(transition_metrics[index]["global_mae"], 4) for index in transition_starts
                        ],
                        "transition_max_tile_mae": [
                            round(transition_metrics[index]["max_tile_mae"], 4) for index in transition_starts
                        ],
                        "temporal_diagnostics": BerniniProofBundle._temporal_diagnostics(transition_metrics),
                    }
                )
                details["worst_transitions"] = transition_details
                sheets["worst_transitions"] = str(transition_path)
        return sheets, details

    @staticmethod
    def _save_timeline_sheet_group(
        *,
        images: list[Image.Image],
        run_dir: Path,
        label: str,
        title: str,
        indices: list[int],
        label_prefix: str,
        fps: float | None,
        resampling: Image.Resampling,
        cell_size: int,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        for stale_path in run_dir.glob(f"{label}_contact_sheet*.png"):
            stale_path.unlink()
        page_size = BerniniProofBundle.TIMELINE_PAGE_SIZE
        pages = [indices[start : start + page_size] for start in range(0, len(indices), page_size)]
        page_count = len(pages)
        sheets: dict[str, str] = {}
        details: dict[str, dict[str, Any]] = {}
        for page_index, page_indices in enumerate(pages, start=1):
            if page_count == 1:
                key = label
                output_path = run_dir / f"{label}_contact_sheet.png"
                page_title = title
            else:
                key = f"{label}_page_{page_index:02d}"
                output_path = run_dir / f"{label}_contact_sheet_page_{page_index:02d}.png"
                page_title = f"{title} · page {page_index}/{page_count}"
            detail = BerniniProofBundle._save_image_sheet(
                images=images,
                output_path=output_path,
                title=page_title,
                indices=page_indices,
                label_prefix=label_prefix,
                fps=fps,
                resampling=resampling,
                fixed_columns=BerniniProofBundle.TIMELINE_COLUMNS,
                cell_size=cell_size,
            )
            detail.update(
                {
                    "timeline_group": label,
                    "page_number": page_index,
                    "page_count": page_count,
                }
            )
            sheets[key] = str(output_path)
            details[key] = detail
        return sheets, details

    @staticmethod
    def _refresh_case_sheets(*, runs: list[dict[str, Any]]) -> None:
        cases = BerniniProofBundle.cases()
        for run in runs:
            case = cases[run["case"]["case_id"]]
            output_path = Path(run["output_path"])
            metadata_path = output_path.with_suffix(".metadata.json")
            output_metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
            sheets, details = BerniniProofBundle._save_case_sheets(
                case=case,
                reference_paths=[Path(path) for path in run.get("input_reference_paths", [])],
                source_path=Path(run["input_source_path"]) if run.get("input_source_path") else None,
                official_path=Path(run["official_output_path"]) if run.get("official_output_path") else None,
                output_path=output_path,
                run_dir=output_path.parent,
                output_metadata=output_metadata,
            )
            run["sheets"] = sheets
            run["sheet_details"] = details

    @staticmethod
    def _save_image_sheet(
        *,
        images: list[Image.Image],
        output_path: Path,
        title: str,
        indices: list[int],
        label_prefix: str,
        resampling: Image.Resampling,
        fixed_columns: int,
        cell_size: int,
        fps: float | None = None,
    ) -> dict[str, Any]:
        if not images or not indices:
            raise ValueError(f"Cannot build an empty contact sheet: {title}")
        if any(index < 0 or index >= len(images) for index in indices):
            raise ValueError(f"Contact-sheet indices are out of range for {title}: {indices}")
        columns = max(1, min(fixed_columns, len(indices)))
        rows = math.ceil(len(indices) / columns)
        padding = BerniniProofBundle.SHEET_PADDING
        label_height = BerniniProofBundle.SHEET_LABEL_HEIGHT
        cell_width = cell_height = cell_size
        title_font = BerniniProofBundle._font(BerniniProofBundle.TITLE_FONT_SIZE, bold=True)
        label_font = BerniniProofBundle._font(BerniniProofBundle.LABEL_FONT_SIZE)
        sheet_width = padding * (columns + 1) + cell_width * columns
        title_lines = BerniniProofBundle._wrap_text(title, font=title_font, max_width=sheet_width - 2 * padding)
        title_line_height = title_font.getbbox("Ag")[3] - title_font.getbbox("Ag")[1]
        header = max(
            BerniniProofBundle.SHEET_HEADER_HEIGHT,
            padding * 2 + len(title_lines) * title_line_height,
        )
        sheet_height = header + padding + rows * (cell_height + label_height + padding)
        sheet = Image.new(
            "RGB",
            (sheet_width, sheet_height),
            (18, 20, 26),
        )
        draw = ImageDraw.Draw(sheet)
        for line_index, line in enumerate(title_lines):
            draw.text(
                (padding, padding + line_index * title_line_height),
                line,
                fill=(240, 242, 247),
                font=title_font,
            )
        magnifications = []
        input_sizes = []
        rendered_sizes = []
        decoded_frame_sha256 = []
        for sample_position, index in enumerate(indices):
            row, column = divmod(sample_position, columns)
            image, magnification = BerniniProofBundle._proof_image(
                images[index],
                size=(cell_width, cell_height),
                resampling=resampling,
            )
            magnifications.append(magnification)
            input_sizes.append([images[index].width, images[index].height])
            rendered_sizes.append([image.width, image.height])
            decoded_frame_sha256.append(BerniniProofBundle._image_sha256(images[index]))
            cell_x = padding + column * (cell_width + padding)
            cell_y = header + padding + row * (cell_height + label_height + padding)
            x = cell_x + (cell_width - image.width) // 2
            y = cell_y + (cell_height - image.height) // 2
            sheet.paste(image, (x, y))
            timestamp = f" · {index / fps:.3f}s" if fps is not None and fps > 0 else ""
            draw.text(
                (cell_x, cell_y + cell_height + 10), f"{label_prefix}{index}{timestamp}", fill="white", font=label_font
            )
        sheet.save(output_path)
        return {
            "sheet_contract_version": BerniniProofBundle.SHEET_CONTRACT_VERSION,
            "width": sheet.width,
            "height": sheet.height,
            "columns": columns,
            "rows": rows,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "title_font_size": BerniniProofBundle.TITLE_FONT_SIZE,
            "label_font_size": BerniniProofBundle.LABEL_FONT_SIZE,
            "source_frame_count": len(images),
            "sample_count": len(indices),
            "sample_indices": indices,
            "includes_all_frames": indices == list(range(len(images))),
            "resampling": BerniniProofBundle._resampling_name(resampling),
            "magnifications": magnifications,
            "integer_magnification": all(value is None or float(value).is_integer() for value in magnifications),
            "input_sizes": input_sizes,
            "rendered_sizes": rendered_sizes,
            "decoded_frame_sha256": decoded_frame_sha256,
            "downsampled": any(
                rendered_width < input_width or rendered_height < input_height
                for (rendered_width, rendered_height), (input_width, input_height) in zip(
                    rendered_sizes,
                    input_sizes,
                    strict=True,
                )
            ),
        }

    @staticmethod
    def _save_summary_sheet(*, runs: list[dict[str, Any]], output_path: Path) -> dict[str, Any] | None:
        if not runs:
            return None
        rows = []
        for run in runs:
            clip = VideoUtil.read_video_clip(run["output_path"])
            indices = BerniniProofBundle._uniform_indices(len(clip.frames), maximum=BerniniProofBundle.SUMMARY_COLUMNS)
            rows.append((run["case"]["case_id"], [(index, clip.frames[index]) for index in indices]))
        return BerniniProofBundle._save_row_sheet(
            rows=rows,
            output_path=output_path,
            title="Bernini-R MLX overview · inspect each all-frame sheet before disposition",
        )

    @staticmethod
    def _save_role_control_sheet(*, runs: list[dict[str, Any]], output_path: Path) -> dict[str, Any] | None:
        role_ids = {
            "rv2v_reference_pinstripe_ab",
            "rv2v_reference_black_ab",
            "rv2v_reference_none_ab",
        }
        selected = [run for run in runs if run["case"]["case_id"] in role_ids]
        if len(selected) != len(role_ids):
            return None
        rows = []
        for run in selected:
            clip = VideoUtil.read_video_clip(run["output_path"])
            indices = BerniniProofBundle._uniform_indices(len(clip.frames), maximum=BerniniProofBundle.SUMMARY_COLUMNS)
            rows.append((run["case"]["case_id"], [(index, clip.frames[index]) for index in indices]))
        return BerniniProofBundle._save_row_sheet(
            rows=rows,
            output_path=output_path,
            title="Same prompt and seed reference comparison · none row uses the V2V route",
        )

    @staticmethod
    def _save_row_sheet(
        *,
        rows: list[tuple[str, list[tuple[int, Image.Image]]]],
        output_path: Path,
        title: str,
    ) -> dict[str, Any]:
        label_width = 720
        cell_width = BerniniProofBundle.SUMMARY_CELL_WIDTH
        cell_height = BerniniProofBundle.SUMMARY_CELL_HEIGHT
        padding = BerniniProofBundle.SHEET_PADDING
        header = BerniniProofBundle.SHEET_HEADER_HEIGHT
        label_height = BerniniProofBundle.SHEET_LABEL_HEIGHT
        column_count = max(len(images) for _, images in rows)
        title_font = BerniniProofBundle._font(BerniniProofBundle.TITLE_FONT_SIZE, bold=True)
        label_font = BerniniProofBundle._font(BerniniProofBundle.LABEL_FONT_SIZE)
        sheet_width = max(
            label_width + column_count * (cell_width + padding) + padding,
            int(title_font.getlength(title)) + 2 * padding,
        )
        sheet = Image.new(
            "RGB",
            (
                sheet_width,
                header + len(rows) * (cell_height + label_height + padding) + padding,
            ),
            (18, 20, 26),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((padding, 34), title, fill=(240, 242, 247), font=title_font)
        input_sizes = []
        rendered_sizes = []
        for row_index, (label, images) in enumerate(rows):
            y = header + padding + row_index * (cell_height + label_height + padding)
            draw.text((padding, y + 16), label, fill=(240, 242, 247), font=label_font)
            for column, (frame_index, source) in enumerate(images):
                image = ImageOps.contain(source.convert("RGB"), (cell_width, cell_height), Image.Resampling.NEAREST)
                input_sizes.append([source.width, source.height])
                rendered_sizes.append([image.width, image.height])
                x = label_width + column * (cell_width + padding) + (cell_width - image.width) // 2
                sheet.paste(image, (x, y + (cell_height - image.height) // 2))
                draw.text(
                    (label_width + column * (cell_width + padding), y + cell_height + 10),
                    f"frame {frame_index}",
                    fill="white",
                    font=label_font,
                )
        sheet.save(output_path)
        return {
            "sheet_contract_version": BerniniProofBundle.SHEET_CONTRACT_VERSION,
            "width": sheet.width,
            "height": sheet.height,
            "columns": column_count,
            "rows": len(rows),
            "cell_width": cell_width,
            "cell_height": cell_height,
            "title_font_size": BerniniProofBundle.TITLE_FONT_SIZE,
            "label_font_size": BerniniProofBundle.LABEL_FONT_SIZE,
            "resampling": "nearest",
            "input_sizes": input_sizes,
            "rendered_sizes": rendered_sizes,
            "downsampled": any(
                rendered_width < input_width or rendered_height < input_height
                for (rendered_width, rendered_height), (input_width, input_height) in zip(
                    rendered_sizes,
                    input_sizes,
                    strict=True,
                )
            ),
        }

    @staticmethod
    def _uniform_indices(frame_count: int, *, maximum: int) -> list[int]:
        if frame_count < 1:
            raise ValueError("Cannot sample an empty frame sequence.")
        count = min(maximum, frame_count)
        return np.linspace(0, frame_count - 1, count, dtype=int).tolist()

    @staticmethod
    def _transition_mae(images: list[Image.Image]) -> list[float]:
        return [metrics["global_mae"] for metrics in BerniniProofBundle._transition_metrics(images)]

    @staticmethod
    def _transition_metrics(images: list[Image.Image]) -> list[dict[str, float]]:
        if len(images) < 2:
            return []
        arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) for image in images]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("Transition proof frames must have one consistent shape.")
        metrics = []
        tile_size = BerniniProofBundle.TRANSITION_TILE_SIZE
        for before, after in zip(arrays, arrays[1:]):
            delta = np.abs(after - before)
            tile_mae = [
                float(np.mean(delta[y : y + tile_size, x : x + tile_size]))
                for y in range(0, delta.shape[0], tile_size)
                for x in range(0, delta.shape[1], tile_size)
            ]
            metrics.append(
                {
                    "global_mae": float(np.mean(delta)),
                    "max_tile_mae": max(tile_mae),
                }
            )
        return metrics

    @staticmethod
    def _temporal_diagnostics(metrics: list[dict[str, float]]) -> dict[str, Any]:
        global_mae = [metrics_row["global_mae"] for metrics_row in metrics]
        boundary = [value for index, value in enumerate(global_mae) if index % 4 == 0]
        non_boundary = [value for index, value in enumerate(global_mae) if index % 4 != 0]
        boundary_mean = float(np.mean(boundary)) if boundary else 0.0
        non_boundary_mean = float(np.mean(non_boundary)) if non_boundary else 0.0
        return {
            "transition_count": len(metrics),
            "mean_global_mae": round(float(np.mean(global_mae)), 4) if global_mae else 0.0,
            "max_global_mae": round(max(global_mae), 4) if global_mae else 0.0,
            "max_local_tile_mae": round(max((row["max_tile_mae"] for row in metrics), default=0.0), 4),
            "latent_boundary_start_indices": list(range(0, len(metrics), 4)),
            "latent_boundary_mean_global_mae": round(boundary_mean, 4),
            "non_boundary_mean_global_mae": round(non_boundary_mean, 4),
            "boundary_to_non_boundary_ratio": (
                round(boundary_mean / non_boundary_mean, 4) if non_boundary_mean > 0 else None
            ),
            "near_freeze_start_indices": [index for index, value in enumerate(global_mae) if value < 0.1],
        }

    @staticmethod
    def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        del bold
        font = ImageFont.load_default(size=size)
        if not isinstance(font, ImageFont.FreeTypeFont):
            raise RuntimeError("The installed Pillow build lacks its scalable embedded proof-sheet font.")
        return font

    @staticmethod
    def _wrap_text(text: str, *, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.getlength(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    @staticmethod
    def _resampling_name(resampling: Image.Resampling) -> str:
        return "nearest" if resampling == Image.Resampling.NEAREST else "lanczos"

    @staticmethod
    def _write_sheet_manifest(
        *,
        output_dir: Path,
        runs: list[dict[str, Any]],
        summary_sheet_details: dict[str, Any] | None,
        role_control_sheet_details: dict[str, Any] | None,
    ) -> None:
        manifest = {
            "schema_version": 1,
            "sheet_contract_version": BerniniProofBundle.SHEET_CONTRACT_VERSION,
            "pillow_version": PILLOW_VERSION,
            "font": "Pillow embedded scalable default",
            "cases": {
                run["case"]["case_id"]: {
                    label: {
                        "path": Path(path).resolve().relative_to(output_dir.resolve()).as_posix(),
                        "sha256": BerniniProofBundle._sha256(Path(path)),
                        **run["sheet_details"][label],
                    }
                    for label, path in sorted(run["sheets"].items())
                }
                for run in runs
            },
            "summary": {
                "path": "output_summary_contact_sheet.png",
                "sha256": BerniniProofBundle._sha256(output_dir / "output_summary_contact_sheet.png"),
                **(summary_sheet_details or {}),
            },
            "role_control": {
                "path": "role_control_contact_sheet.png",
                "sha256": BerniniProofBundle._sha256(output_dir / "role_control_contact_sheet.png"),
                **(role_control_sheet_details or {}),
            },
        }
        (output_dir / "sheet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    @staticmethod
    def _proof_image(
        source: Image.Image,
        *,
        size: tuple[int, int],
        resampling: Image.Resampling,
    ) -> tuple[Image.Image, float | None]:
        image = source.convert("RGB")
        if resampling == Image.Resampling.NEAREST and image.width <= size[0] and image.height <= size[1]:
            scale = max(1, min(size[0] // image.width, size[1] // image.height))
            return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST), float(scale)
        return ImageOps.contain(image, size, resampling), None

    @staticmethod
    def _sheet_contract_passes(*, result: dict[str, Any], expected_sheets: set[str]) -> bool:
        sheets = result.get("sheets") or {}
        details = result.get("sheet_details") or {}
        if set(sheets) != expected_sheets or set(details) != expected_sheets:
            return False
        for label in expected_sheets:
            path = Path(sheets[label])
            detail = details[label]
            if not path.is_file() or detail.get("sheet_contract_version") != BerniniProofBundle.SHEET_CONTRACT_VERSION:
                return False
            with Image.open(path) as sheet:
                if [sheet.width, sheet.height] != [detail.get("width"), detail.get("height")]:
                    return False
            if detail.get("title_font_size", 0) < 80 or detail.get("label_font_size", 0) < 64:
                return False
            sample_count = int(detail.get("sample_count", 0))
            expected_columns = min(
                BerniniProofBundle.TRANSITION_COLUMNS
                if label == "worst_transitions"
                else BerniniProofBundle.TIMELINE_COLUMNS,
                sample_count,
            )
            if (
                detail.get("columns") != expected_columns
                or detail.get("cell_width", 0) < BerniniProofBundle.TIMELINE_CELL_SIZE
                or detail.get("cell_height", 0) < BerniniProofBundle.TIMELINE_CELL_SIZE
                or detail.get("width", 0)
                < expected_columns * BerniniProofBundle.TIMELINE_CELL_SIZE
                + (expected_columns + 1) * BerniniProofBundle.SHEET_PADDING
            ):
                return False
            if label != "references" and detail.get("downsampled") is not False:
                return False
            input_sizes = detail.get("input_sizes")
            rendered_sizes = detail.get("rendered_sizes")
            decoded_frame_sha256 = detail.get("decoded_frame_sha256")
            if (
                not isinstance(input_sizes, list)
                or not isinstance(rendered_sizes, list)
                or not isinstance(decoded_frame_sha256, list)
                or len(input_sizes) != sample_count
                or len(rendered_sizes) != sample_count
                or len(decoded_frame_sha256) != sample_count
                or any(
                    not isinstance(size, list)
                    or len(size) != 2
                    or not all(isinstance(value, int) and value > 0 for value in size)
                    for size in [*input_sizes, *rendered_sizes]
                )
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in decoded_frame_sha256
                )
            ):
                return False
        try:
            timeline_groups = BerniniProofBundle._expected_timeline_groups(result)
        except (FileNotFoundError, RuntimeError, ValueError):
            return False
        for group, expected_indices in timeline_groups.items():
            keys = BerniniProofBundle._timeline_sheet_keys(group, len(expected_indices))
            combined_indices: list[int] = []
            for page_number, key in enumerate(keys, start=1):
                detail = details[key]
                start = (page_number - 1) * BerniniProofBundle.TIMELINE_PAGE_SIZE
                expected_page_indices = expected_indices[start : start + BerniniProofBundle.TIMELINE_PAGE_SIZE]
                if (
                    detail.get("timeline_group") != group
                    or detail.get("page_number") != page_number
                    or detail.get("page_count") != len(keys)
                    or detail.get("sample_indices") != expected_page_indices
                    or detail.get("includes_all_frames") is not (len(keys) == 1 and group in {"mlx", "official"})
                    or (group == "mlx" and detail.get("resampling") != "nearest")
                ):
                    return False
                if group in {"mlx", "official"} and detail.get("source_frame_count") != len(expected_indices):
                    return False
                if group == "source" and detail.get("source_frame_count", 0) <= max(expected_indices):
                    return False
                combined_indices.extend(detail["sample_indices"])
            if combined_indices != expected_indices:
                return False
        if "source" in timeline_groups:
            metadata = result.get("metadata") or {}
            expected_size = [metadata.get("video_condition_width"), metadata.get("video_condition_height")]
            for key in BerniniProofBundle._timeline_sheet_keys("source", len(timeline_groups["source"])):
                if any(size != expected_size for size in details[key].get("input_sizes", [])):
                    return False
        reference_detail = details.get("references")
        if reference_detail is not None and (
            reference_detail.get("source_frame_count") != len(result["case"]["reference_images"])
            or reference_detail.get("sample_indices") != list(range(len(result["case"]["reference_images"])))
        ):
            return False
        return True

    @staticmethod
    def _timeline_sheet_keys(label: str, sample_count: int) -> list[str]:
        if sample_count < 1:
            raise ValueError(f"Timeline group {label!r} must contain at least one frame.")
        page_count = math.ceil(sample_count / BerniniProofBundle.TIMELINE_PAGE_SIZE)
        if page_count == 1:
            return [label]
        return [f"{label}_page_{page_number:02d}" for page_number in range(1, page_count + 1)]

    @staticmethod
    def _expected_timeline_groups(result: dict[str, Any]) -> dict[str, list[int]]:
        case = result["case"]
        groups = {"mlx": list(range(int(case["frames"])))}
        if case["source_video"]:
            source_indices = [int(index) for index in (result.get("metadata") or {}).get("source_sample_indices", [])]
            if not source_indices:
                raise ValueError("Source proof is missing exact sampled-frame indices.")
            groups["source"] = source_indices
        if case["official_output"]:
            official_path = Path(result.get("official_output_path", ""))
            if not official_path.is_file():
                raise FileNotFoundError(official_path)
            official_clip = VideoUtil.read_video_clip(official_path, max_frames=None)
            groups["official"] = list(range(len(official_clip.frames)))
        return groups

    @staticmethod
    def _contract_checks(result: dict[str, Any]) -> dict[str, bool]:
        case = result["case"]
        metadata = result.get("metadata") or {}
        video_health = result.get("video_health") or {}
        sampler = result.get("sampler") or {}
        sheets = result.get("sheets") or {}
        sheet_topology_resolved = True
        try:
            timeline_groups = BerniniProofBundle._expected_timeline_groups(result)
        except (FileNotFoundError, RuntimeError, ValueError):
            timeline_groups = {}
            sheet_topology_resolved = False
        expected_sheets = {"worst_transitions"}
        for label, indices in timeline_groups.items():
            expected_sheets.update(BerniniProofBundle._timeline_sheet_keys(label, len(indices)))
        if case["reference_images"]:
            expected_sheets.add("references")
        expected_mode = (
            "rv2v"
            if case["source_video"] and case["reference_images"]
            else "v2v_apg"
            if case["source_video"]
            else "r2v_apg"
        )
        checks = {
            "process_exit": result.get("returncode") == 0,
            "output_exists": Path(result.get("output_path", "")).is_file(),
            "video_health": video_health.get("status") == "ok",
            "guidance_mode": metadata.get("bernini_guidance_mode") == expected_mode,
            "reference_count": metadata.get("reference_image_count") == len(case["reference_images"]),
            "reference_order": metadata.get("reference_image_paths", []) == result.get("input_reference_paths", []),
            "source_path": metadata.get("video_path") == result.get("input_source_path"),
            "factored_sources": metadata.get("factored_component_sources") is True,
            "component_provenance": metadata.get("component_source_provenance")
            == BerniniProofBundle.EXPECTED_COMPONENT_PROVENANCE,
            "runtime_policy": all(
                metadata.get(key) == value for key, value in BerniniProofBundle.EXPECTED_RUNTIME_POLICY.items()
            ),
            "runtime_environment": all(
                isinstance(metadata.get(key), str) and bool(metadata[key])
                for key in BerniniProofBundle.REQUIRED_RUNTIME_ENVIRONMENT_FIELDS
            ),
            "runtime_precision": metadata.get("precision") == "mlx.core.bfloat16",
            "bf16_unquantized": metadata.get("quantize") is None,
            "renderer_checkpoint": metadata.get("model") == "ByteDance/Bernini-R-1.3B-Diffusers",
            "renderer_only": metadata.get("bernini_renderer_only") is True,
            "steps": metadata.get("steps") == case["steps"],
            "width": video_health.get("width") == case["width"] == metadata.get("output_width"),
            "height": video_health.get("height") == case["height"] == metadata.get("output_height"),
            "frames": video_health.get("frames") == case["frames"] == metadata.get("output_frames"),
            "fps": abs(float(video_health.get("fps", 0)) - case["fps"]) < 1e-6 and metadata.get("fps") == case["fps"],
            "max_condition_size": metadata.get("max_condition_size") == case["max_condition_size"],
            "prompt_exact": metadata.get("prompt") == result.get("prompt"),
            "prompt_truncation_recorded": (
                isinstance(metadata.get("prompt_tokens"), int) and isinstance(metadata.get("prompt_truncated"), bool)
            ),
            "prompt_truncation_expected": metadata.get("prompt_truncated")
            is (case.get("prompt_json") == "assets/testcases/r2v/r2v_case2.json"),
            "physical_memory_sampled": (
                isinstance(sampler.get("peak_sampled_darwin_physical_footprint_bytes"), int)
                and sampler["peak_sampled_darwin_physical_footprint_bytes"] > 0
                and sampler.get("sample_count", 0) > 0
            ),
            "within_18gb_physical_memory": (
                isinstance(sampler.get("peak_sampled_darwin_physical_footprint_bytes"), int)
                and sampler["peak_sampled_darwin_physical_footprint_bytes"] <= 18 * 1024**3
            ),
            "sheet_topology": sheet_topology_resolved,
            "required_sheets": expected_sheets == set(sheets) and all(Path(path).is_file() for path in sheets.values()),
            "sheet_contract": BerniniProofBundle._sheet_contract_passes(
                result=result,
                expected_sheets=expected_sheets,
            ),
            "stdout_log": Path(result.get("stdout_path", "")).is_file(),
            "stderr_log": Path(result.get("stderr_path", "")).is_file(),
            "nonzero_temporal_change": float(video_health.get("temporal_continuity", {}).get("mean_temporal_mae", 0))
            > 0,
        }
        if case["case_id"] in BerniniProofBundle.QUALITY_CASE_IDS:
            boundary_ratio = (
                (result.get("sheet_details") or {})
                .get("worst_transitions", {})
                .get("temporal_diagnostics", {})
                .get("boundary_to_non_boundary_ratio")
            )
            checks["latent_group_continuity"] = (
                isinstance(boundary_ratio, (int, float))
                and math.isfinite(float(boundary_ratio))
                and float(boundary_ratio) <= BerniniProofBundle.MAX_QUALITY_LATENT_BOUNDARY_RATIO
            )
        if len(case["reference_images"]) == 8:
            checks["eight_source_ids"] = metadata.get("condition_source_ids") == [
                1.0,
                1.5714285373687744,
                2.142857074737549,
                2.7142858505249023,
                3.2857141494750977,
                3.857142925262451,
                4.4285712242126465,
                5.0,
            ]
            checks["eight_latent_shapes"] = len(metadata.get("condition_latent_shapes", [])) == 8
        return checks

    @staticmethod
    def _write_report(
        *,
        output_dir: Path,
        reference_root: Path,
        runs: list[dict[str, Any]],
        summary_sheet_details: dict[str, Any] | None = None,
        role_control_sheet_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for run in runs:
            output_path = Path(run["output_path"])
            metadata_path = output_path.with_suffix(".metadata.json")
            if output_path.is_file():
                run["video_health"] = GenerationMemoryBenchmark._video_health(output_path)
            if metadata_path.is_file():
                run["metadata"] = json.loads(metadata_path.read_text())
            run["artifact_hashes"] = BerniniProofBundle._artifact_hashes(run=run)
            run["output_sha256"] = run["artifact_hashes"]["output"]["sha256"]
            run["contract_checks"] = BerniniProofBundle._contract_checks(run)
            run["passed"] = bool(run.get("video_health", {}).get("status") == "ok") and all(
                run["contract_checks"].values()
            )
        visual_review_path = output_dir / "visual_review.json"
        visual_review = (
            json.loads(visual_review_path.read_text())
            if visual_review_path.is_file()
            else {"schema_version": 1, "cases": {}}
        )
        visual_cases = visual_review.get("cases", {})
        summary_sheet_path = output_dir / "output_summary_contact_sheet.png"
        role_sheet_path = output_dir / "role_control_contact_sheet.png"
        sheet_manifest_path = output_dir / "sheet_manifest.json"
        overview_sheet_checks = {
            "summary": BerniniProofBundle._overview_sheet_contract_passes(
                path=summary_sheet_path,
                details=summary_sheet_details,
            ),
            "role_control": BerniniProofBundle._overview_sheet_contract_passes(
                path=role_sheet_path,
                details=role_control_sheet_details,
            ),
        }
        machine_contract_passed = (
            bool(runs) and all(run["passed"] for run in runs) and all(overview_sheet_checks.values())
        )
        review_bindings = {
            "summary_contact_sheet_sha256": (
                BerniniProofBundle._sha256(summary_sheet_path) if summary_sheet_path.is_file() else None
            ),
            "role_control_contact_sheet_sha256": (
                BerniniProofBundle._sha256(role_sheet_path) if role_sheet_path.is_file() else None
            ),
            "sheet_manifest_sha256": (
                BerniniProofBundle._sha256(sheet_manifest_path) if sheet_manifest_path.is_file() else None
            ),
        }
        visual_review_checks = {
            run["case"]["case_id"]: BerniniProofBundle._visual_review_check(
                run=run,
                review=visual_cases.get(run["case"]["case_id"], {}),
            )
            for run in runs
        }
        visual_review_metadata_checks = BerniniProofBundle._visual_review_metadata_checks(visual_review)
        global_visual_bindings_match = all(visual_review.get(key) == value for key, value in review_bindings.items())
        visual_review_complete = (
            bool(runs)
            and all(all(checks.values()) for checks in visual_review_checks.values())
            and all(visual_review_metadata_checks.values())
            and global_visual_bindings_match
        )
        run_ids = {run["case"]["case_id"] for run in runs}
        visual_quality_checks = BerniniProofBundle._visual_quality_checks(visual_cases=visual_cases)
        visual_quality_passed = BerniniProofBundle._visual_quality_passes(
            visual_review_complete=visual_review_complete,
            run_ids=run_ids,
            visual_quality_checks=visual_quality_checks,
        )
        report = {
            "schema_version": 3,
            "kind": "bernini_r_1_3b_mlx_model_backed_proof",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "official_source_root": str(reference_root),
            "official_source_revision": BerniniProofBundle.OFFICIAL_SOURCE_REVISION,
            "upstream_comparison_scope": (
                "The bundled upstream clips are qualitative targets only. Their producing checkpoint and "
                "inference recipe are not attested, so they are not 1.3B parity baselines."
            ),
            "component_compatibility": BerniniProofBundle.EXPECTED_COMPONENT_PROVENANCE,
            "method": {
                "process_isolation": "Each case runs as a fresh low-RAM BF16 CLI process.",
                "memory": "The parent samples whole process-tree RSS and Darwin physical footprint.",
                "visual_review": "Generated MP4s and contact sheets are retained; explicit visual inspection is recorded separately.",
            },
            "runs": runs,
            "summary_contact_sheet": str(output_dir / "output_summary_contact_sheet.png"),
            "summary_contact_sheet_details": summary_sheet_details,
            "role_control_contact_sheet": str(output_dir / "role_control_contact_sheet.png"),
            "role_control_contact_sheet_details": role_control_sheet_details,
            "overview_sheet_checks": overview_sheet_checks,
            "sheet_manifest": str(sheet_manifest_path),
            "visual_review_path": str(visual_review_path),
            "visual_review": visual_review,
            "visual_review_metadata_checks": visual_review_metadata_checks,
            "visual_review_checks": visual_review_checks,
            "visual_review_global_bindings": {
                **review_bindings,
                "matched": global_visual_bindings_match,
            },
            "machine_contract_passed": machine_contract_passed,
            "visual_review_complete": visual_review_complete,
            "visual_quality_checks": visual_quality_checks,
            "visual_quality_passed": visual_quality_passed,
            "visual_review_passed": visual_quality_passed,
            "passed": machine_contract_passed and visual_quality_passed,
        }
        (output_dir / "bernini_proof_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        (output_dir / "bernini_proof_report.md").write_text(BerniniProofBundle._markdown_report(report))
        return report

    @staticmethod
    def _overview_sheet_contract_passes(*, path: Path, details: dict[str, Any] | None) -> bool:
        if not path.is_file() or not isinstance(details, dict):
            return False
        with Image.open(path) as sheet:
            if [sheet.width, sheet.height] != [details.get("width"), details.get("height")]:
                return False
        return bool(
            details.get("sheet_contract_version") == BerniniProofBundle.SHEET_CONTRACT_VERSION
            and details.get("width", 0) >= 5120
            and details.get("cell_width", 0) >= BerniniProofBundle.SUMMARY_CELL_WIDTH
            and details.get("cell_height", 0) >= BerniniProofBundle.SUMMARY_CELL_HEIGHT
            and details.get("title_font_size", 0) >= 80
            and details.get("label_font_size", 0) >= 64
            and details.get("downsampled") is False
        )

    @staticmethod
    def _visual_review_metadata_checks(
        review: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, bool]:
        reviewed_at = review.get("reviewed_at")
        timestamp: datetime | None = None
        if isinstance(reviewed_at, str):
            try:
                timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            raise ValueError("Review validation requires a timezone-aware current time.")
        return {
            "schema_version": review.get("schema_version") == BerniniProofBundle.VISUAL_REVIEW_SCHEMA_VERSION,
            "reviewed_at_parseable": timestamp is not None and timestamp.tzinfo is not None,
            "reviewed_at_not_future": (
                timestamp is not None
                and timestamp.tzinfo is not None
                and timestamp.astimezone(timezone.utc)
                <= now.astimezone(timezone.utc) + timedelta(seconds=BerniniProofBundle.MAX_REVIEW_CLOCK_SKEW_SECONDS)
            ),
            "reviewer_present": bool(str(review.get("reviewer", "")).strip()),
            "scope_present": bool(str(review.get("scope", "")).strip()),
        }

    @staticmethod
    def _artifact_hashes(*, run: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(run["output_path"])
        metadata_path = output_path.with_suffix(".metadata.json")
        artifacts: dict[str, Any] = {
            "output": BerniniProofBundle._hashed_artifact(output_path),
            "metadata": BerniniProofBundle._hashed_artifact(metadata_path),
            "stdout": BerniniProofBundle._hashed_artifact(Path(run["stdout_path"])),
            "stderr": BerniniProofBundle._hashed_artifact(Path(run["stderr_path"])),
            "references": [
                BerniniProofBundle._hashed_artifact(Path(path)) for path in run.get("input_reference_paths", [])
            ],
            "source": (
                BerniniProofBundle._hashed_artifact(Path(run["input_source_path"]))
                if run.get("input_source_path")
                else None
            ),
            "official_output": (
                BerniniProofBundle._hashed_artifact(Path(run["official_output_path"]))
                if run.get("official_output_path")
                else None
            ),
            "sheets": {
                label: BerniniProofBundle._hashed_artifact(Path(path))
                for label, path in sorted((run.get("sheets") or {}).items())
            },
        }
        return artifacts

    @staticmethod
    def _hashed_artifact(path: Path) -> dict[str, Any]:
        return {
            "path": str(path),
            "sha256": BerniniProofBundle._sha256(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }

    @staticmethod
    def _visual_review_check(*, run: dict[str, Any], review: dict[str, Any]) -> dict[str, bool]:
        artifact_hashes = run["artifact_hashes"]
        expected_sheet_hashes = {label: artifact["sha256"] for label, artifact in artifact_hashes["sheets"].items()}
        expected_frame_indices = list(range(int(run["case"]["frames"])))
        return {
            "reviewed_status": review.get("status") in BerniniProofBundle.REVIEWED_STATUSES,
            "output_hash": review.get("output_sha256") == artifact_hashes["output"]["sha256"],
            "sheet_hashes": review.get("sheet_sha256") == expected_sheet_hashes,
            "every_output_frame_reviewed": review.get("reviewed_frame_indices") == expected_frame_indices,
            "notes_present": bool(str(review.get("notes", "")).strip()),
        }

    @staticmethod
    def _visual_quality_checks(*, visual_cases: dict[str, dict[str, Any]]) -> dict[str, bool]:
        return {
            case_id: BerniniProofBundle._quality_review_accepted(visual_cases.get(case_id, {}))
            for case_id in sorted(BerniniProofBundle.QUALITY_CASE_IDS)
        }

    @staticmethod
    def _quality_review_accepted(review: dict[str, Any]) -> bool:
        status = review.get("status")
        if status == "pass":
            return True
        return status == "pass_with_limitations" and review.get("limitation_severity") == "minor"

    @staticmethod
    def _visual_quality_passes(
        *,
        visual_review_complete: bool,
        run_ids: set[str],
        visual_quality_checks: dict[str, bool],
    ) -> bool:
        return (
            visual_review_complete
            and BerniniProofBundle.QUALITY_CASE_IDS.issubset(run_ids)
            and all(visual_quality_checks.values())
        )

    @staticmethod
    def _export_durable_bundle(*, output_dir: Path, durable_dir: Path, reference_root: Path) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        BerniniProofBundle._validate_durable_target(
            durable_dir=durable_dir,
            output_dir=output_dir,
            reference_root=reference_root,
            repository_root=repository_root,
        )
        if durable_dir.exists():
            shutil.rmtree(durable_dir)
        durable_dir.mkdir(parents=True, exist_ok=True)
        copy_ignore = shutil.ignore_patterns(".DS_Store")
        shutil.copytree(
            output_dir / "cases",
            durable_dir / "cases",
            dirs_exist_ok=True,
            ignore=copy_ignore,
        )
        for name in (
            "bernini_proof_report.json",
            "bernini_proof_report.md",
            "output_summary_contact_sheet.png",
            "role_control_contact_sheet.png",
            "sheet_manifest.json",
            "visual_review.json",
            "visual_review.md",
        ):
            shutil.copy2(output_dir / name, durable_dir / name)
        evidence_root = output_dir.parent
        for name in ("parity", "diagnostics"):
            source = evidence_root / name
            if not source.is_dir():
                raise FileNotFoundError(f"Supplemental Bernini evidence is missing: {source}")
            shutil.copytree(source, durable_dir / name, ignore=copy_ignore)
        for source_name, target_name in BerniniProofBundle.SUPPLEMENTAL_EVIDENCE_DIRS.items():
            source = evidence_root / source_name
            if not source.is_dir():
                raise FileNotFoundError(f"Supplemental Bernini evidence is missing: {source}")
            shutil.copytree(source, durable_dir / target_name, ignore=copy_ignore)
        smoke_source = evidence_root / "cycle4_smoke"
        if smoke_source.is_dir():
            shutil.copytree(smoke_source, durable_dir / "smoke", dirs_exist_ok=True, ignore=copy_ignore)
        shutil.copy2(reference_root / "LICENSE", durable_dir / "UPSTREAM_BERNINI_LICENSE.txt")
        portable_sources = [
            (output_dir, "<bundle-root>"),
            (evidence_root / "parity", "<bundle-root>/parity"),
            (evidence_root / "diagnostics", "<bundle-root>/diagnostics"),
            (evidence_root / "cycle4_smoke", "<bundle-root>/smoke"),
            *(
                (evidence_root / source_name, f"<bundle-root>/{target_name}")
                for source_name, target_name in BerniniProofBundle.SUPPLEMENTAL_EVIDENCE_DIRS.items()
            ),
            (evidence_root, "<validation-root>"),
            (reference_root, "<official-source-root>"),
            (repository_root, "<repo-root>"),
            (Path.home(), "<user-home>"),
        ]
        replacements = []
        for source, target in portable_sources:
            replacements.append((str(source), target))
            try:
                relative_source = source.relative_to(repository_root).as_posix()
            except ValueError:
                continue
            if relative_source not in {"", "."}:
                replacements.append((relative_source, target))
        replacements.sort(key=lambda item: len(item[0]), reverse=True)
        BerniniProofBundle._sanitize_durable_text(
            durable_dir=durable_dir,
            replacements=tuple(replacements),
        )
        BerniniProofBundle._refresh_portable_report(durable_dir)
        BerniniProofBundle._write_component_compatibility(durable_dir)
        BerniniProofBundle._write_upstream_attribution(durable_dir)
        BerniniProofBundle._write_portable_manifest(durable_dir)
        BerniniProofBundle.verify_portable_bundle(durable_dir)

    @staticmethod
    def _validate_durable_target(
        *,
        durable_dir: Path,
        output_dir: Path,
        reference_root: Path,
        repository_root: Path,
    ) -> None:
        target = durable_dir.resolve()
        output = output_dir.resolve()
        reference = reference_root.resolve()
        repository = repository_root.resolve()
        user_home = Path.home().resolve()
        if durable_dir.is_symlink():
            raise ValueError(f"Refusing symlinked durable proof target: {durable_dir}")
        if any(
            BerniniProofBundle._path_is_within(protected, target)
            for protected in (user_home, repository, output, reference)
        ):
            raise ValueError(f"Refusing durable proof target that contains a protected path: {durable_dir}")
        if BerniniProofBundle._path_is_within(target, output) or BerniniProofBundle._path_is_within(target, reference):
            raise ValueError(f"Refusing durable proof target inside a proof source: {durable_dir}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"Durable proof target is not a directory: {durable_dir}")
        if target.is_dir() and any(target.iterdir()):
            manifest_path = target / "portable_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                raise ValueError(
                    f"Refusing to replace an unmarked non-empty durable proof target: {durable_dir}"
                ) from None
            if manifest.get("kind") != "portable_bernini_r_1_3b_proof_bundle":
                raise ValueError(f"Refusing to replace a foreign durable proof target: {durable_dir}")

    @staticmethod
    def _path_is_within(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def verify_portable_bundle(durable_dir: Path) -> dict[str, Any]:
        manifest_path = durable_dir / "portable_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Portable proof manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 2 or manifest.get("kind") != "portable_bernini_r_1_3b_proof_bundle":
            raise ValueError("Portable proof manifest has an unsupported contract.")
        if manifest.get("case_contract_version") != BerniniProofBundle.CASE_CONTRACT_VERSION:
            raise ValueError("Portable proof bundle uses a stale Bernini case contract.")
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Portable proof manifest entries must be a list.")
        expected_paths: set[str] = set()
        for entry in entries:
            relative_path = entry.get("path") if isinstance(entry, dict) else None
            if not isinstance(relative_path, str):
                raise ValueError("Portable proof manifest contains a malformed path entry.")
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts or relative_path in expected_paths:
                raise ValueError(f"Portable proof manifest contains an unsafe or duplicate path: {relative_path}")
            expected_paths.add(relative_path)
        actual_paths = {
            path.relative_to(durable_dir).as_posix()
            for path in durable_dir.rglob("*")
            if path.is_file() and path != manifest_path
        }
        symlinks = [path.relative_to(durable_dir).as_posix() for path in durable_dir.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(f"Portable proof bundle must not contain symlinks: {sorted(symlinks)}")
        if actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)
            unexpected = sorted(actual_paths - expected_paths)
            raise ValueError(f"Portable proof inventory mismatch; missing={missing}, unexpected={unexpected}")
        for entry in entries:
            path = durable_dir / entry["path"]
            if path.stat().st_size != entry.get("size_bytes") or BerniniProofBundle._sha256(path) != entry.get(
                "sha256"
            ):
                raise ValueError(f"Portable proof artifact failed integrity verification: {entry['path']}")
        report = json.loads((durable_dir / "bernini_proof_report.json").read_text())
        review = json.loads((durable_dir / "visual_review.json").read_text())
        sheet_manifest = json.loads((durable_dir / "sheet_manifest.json").read_text())
        if report.get("schema_version") != 3:
            raise ValueError("Portable proof report is not schema v3.")
        if review.get("schema_version") != BerniniProofBundle.VISUAL_REVIEW_SCHEMA_VERSION:
            raise ValueError("Portable visual review has an unsupported schema.")
        if sheet_manifest.get("sheet_contract_version") != BerniniProofBundle.SHEET_CONTRACT_VERSION:
            raise ValueError("Portable sheet manifest has an unsupported rendering contract.")
        if report.get("visual_review") != review:
            raise ValueError("Portable report does not embed the exact visual review record.")
        BerniniProofBundle._verify_current_portable_profile(report=report, manifest=manifest)
        BerniniProofBundle._verify_bundle_local_json_paths(durable_dir)
        compatibility_path = durable_dir / "component_compatibility.json"
        if not compatibility_path.is_file():
            raise ValueError("Portable component compatibility record is missing.")
        BerniniProofBundle._verify_component_compatibility(durable_dir)
        recomputed = BerniniProofBundle._verify_portable_report_evidence(
            durable_dir=durable_dir,
            report=report,
            visual_review=review,
            sheet_manifest=sheet_manifest,
        )
        return {
            "verified": True,
            "integrity_verified": True,
            "quality_certified": recomputed["passed"] and recomputed["visual_quality_passed"],
            "entry_count": len(entries),
            "report_passed": recomputed["passed"],
            "visual_quality_passed": recomputed["visual_quality_passed"],
        }

    @staticmethod
    def _verify_bundle_local_json_paths(durable_dir: Path) -> None:
        def strings(value: Any):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(key, str):
                        yield key
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)

        for json_path in durable_dir.rglob("*.json"):
            payload = json.loads(json_path.read_text())
            for value in strings(payload):
                if "<validation-root>/" in value:
                    raise ValueError(f"Portable JSON retains a validation-root dependency: {json_path}")
                machine_local = re.search(
                    r"(?:^|[^A-Za-z0-9])/(?:Users|home|tmp|private/(?:tmp|var)|var/folders|Volumes)(?:/|$)",
                    value,
                )
                windows_local = re.search(r"(?:^|[^A-Za-z0-9])[A-Za-z]:[\\/]", value)
                home_relative = re.search(r"(?:^|[\s\"'=(])~[\\/]", value)
                if machine_local or windows_local or home_relative:
                    raise ValueError(f"Portable JSON leaks an unmapped machine-local path: {json_path}")
                occurrences = re.findall(r"<bundle-root>/[A-Za-z0-9_./-]+", value)
                if "<bundle-root>/" in value and not occurrences:
                    raise ValueError(f"Portable JSON has a malformed bundle-local path: {value}")
                for portable_path in occurrences:
                    relative = portable_path.removeprefix("<bundle-root>/")
                    candidate = Path(relative)
                    target = durable_dir / candidate
                    try:
                        target.resolve().relative_to(durable_dir.resolve())
                    except ValueError:
                        raise ValueError(
                            f"Portable JSON has an unresolved bundle-local path: {portable_path}"
                        ) from None
                    if candidate.is_absolute() or ".." in candidate.parts or target.is_symlink() or not target.exists():
                        raise ValueError(f"Portable JSON has an unresolved bundle-local path: {portable_path}")

    @staticmethod
    def _verify_component_compatibility(durable_dir: Path) -> None:
        compatibility_path = durable_dir / "component_compatibility.json"
        compatibility = json.loads(compatibility_path.read_text())
        if compatibility != BerniniProofBundle._expected_component_compatibility():
            raise ValueError("Portable component compatibility does not match the exact renderer contract.")
        reports = compatibility.get("parity_reports")
        for label, relative_path in reports.items():
            if not isinstance(relative_path, str):
                raise ValueError(f"Portable component compatibility link {label!r} is malformed.")
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts or not (durable_dir / candidate).is_file():
                raise ValueError(f"Portable component compatibility link {label!r} does not resolve: {relative_path}")
        for label, (schema_version, kind) in BerniniProofBundle.EXPECTED_PARITY_REPORT_HEADERS.items():
            report = json.loads((durable_dir / reports[label]).read_text())
            if (
                report.get("schema_version") != schema_version
                or report.get("kind") != kind
                or report.get("passed") is not True
            ):
                raise ValueError(f"Portable parity report {label!r} does not attest a passing exact contract.")

        diagnosis_relative = reports.get("final_latent_three_way_decode")
        review_relative = reports.get("final_latent_three_way_decode_visual_review")
        if not isinstance(diagnosis_relative, str) or not isinstance(review_relative, str):
            raise ValueError("Portable component compatibility omits the final-latent visual diagnosis.")
        diagnosis_path = durable_dir / diagnosis_relative
        review_path = durable_dir / review_relative
        diagnosis = json.loads(diagnosis_path.read_text())
        visual_review = json.loads(review_path.read_text())
        mlx_report_path = diagnosis_path.parent / "mlx_report.json"
        torch_report_path = diagnosis_path.parent / "torch_report.json"
        mlx_report = json.loads(mlx_report_path.read_text())
        torch_report = json.loads(torch_report_path.read_text())
        expected_review_path = f"<bundle-root>/{review_relative}"
        review_binding = diagnosis.get("manual_visual_review")
        if (
            diagnosis.get("schema_version") != 2
            or diagnosis.get("kind") != "bernini_identical_final_latent_three_way_decode"
            or diagnosis.get("structural_checks_passed") is not True
            or diagnosis.get("visual_disposition") != "negative_result"
            or diagnosis.get("mlx_report") != mlx_report
            or diagnosis.get("torch_report") != torch_report
            or not isinstance(review_binding, dict)
            or review_binding.get("path") != expected_review_path
            or review_binding.get("sha256") != BerniniProofBundle._sha256(review_path)
        ):
            raise ValueError("Portable final-latent diagnosis is not bound to its negative visual review.")
        BerniniProofBundle._verify_cycle17_tensor_artifacts(
            diagnosis_dir=diagnosis_path.parent,
            diagnosis=diagnosis,
            mlx_report=mlx_report,
            torch_report=torch_report,
        )
        reviewed_at = visual_review.get("reviewed_at")
        try:
            review_timestamp = datetime.fromisoformat(str(reviewed_at).replace("Z", "+00:00"))
        except ValueError:
            review_timestamp = None
        if (
            visual_review.get("schema_version") != 1
            or visual_review.get("kind") != "bernini_identical_final_latent_three_way_decode_manual_review"
            or visual_review.get("execution_mode") != "native-frame-and-high-resolution-contact-sheet-review"
            or visual_review.get("status") != "negative_result"
            or not str(visual_review.get("reviewer", "")).strip()
            or not str(visual_review.get("scope", "")).strip()
            or review_timestamp is None
            or review_timestamp.tzinfo is None
            or review_timestamp.astimezone(timezone.utc)
            > datetime.now(timezone.utc) + timedelta(seconds=BerniniProofBundle.MAX_REVIEW_CLOCK_SKEW_SECONDS)
            or (visual_review.get("findings") or {}).get("decode_implementations_materially_agree") is not True
        ):
            raise ValueError("Portable final-latent visual review must remain an explicit negative result.")

        backends = {"mlx_tiled_runtime", "mlx_untiled", "torch_diffusers_0_35_2"}
        artifact_hashes = visual_review.get("artifact_sha256")
        native_hashes = diagnosis.get("native_frame_sha256")
        reviewed_indices = visual_review.get("reviewed_native_frame_indices")
        if (
            not isinstance(artifact_hashes, dict)
            or not isinstance(native_hashes, dict)
            or set(native_hashes) != backends
            or artifact_hashes.get("native_frames") != native_hashes
            or not isinstance(reviewed_indices, dict)
        ):
            raise ValueError("Portable final-latent visual review has incomplete native-frame bindings.")
        diagnosis_dir = diagnosis_path.parent
        decoded_videos = {}
        for name in sorted(backends):
            video_path = diagnosis_dir / f"{name}.mp4"
            sheet_path = diagnosis_dir / f"{name}_all_frames_5k.png"
            if artifact_hashes.get("videos", {}).get(name) != BerniniProofBundle._sha256(video_path):
                raise ValueError(f"Portable final-latent video binding failed for {name}.")
            if artifact_hashes.get("contact_sheets", {}).get(name) != BerniniProofBundle._sha256(sheet_path):
                raise ValueError(f"Portable final-latent contact-sheet binding failed for {name}.")
            expected_hashes = native_hashes[name]
            if not isinstance(expected_hashes, list) or len(expected_hashes) != 17:
                raise ValueError(f"Portable final-latent review must bind exactly 17 native frames for {name}.")
            if reviewed_indices.get(name) != list(range(17)):
                raise ValueError(f"Portable final-latent review does not attest every native frame for {name}.")
            try:
                clip = VideoUtil.read_video_clip(video_path, max_frames=None)
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(f"Portable final-latent video is not decodable for {name}: {error}") from None
            if (
                clip.clip_frame_count != 17
                or clip.source_width != 320
                or clip.source_height != 176
                or abs(float(clip.fps) - 16.0) > 1e-6
            ):
                raise ValueError(f"Portable final-latent video geometry failed for {name}.")
            decoded_videos[name] = np.stack(
                [np.asarray(frame.convert("RGB"), dtype=np.float64) for frame in clip.frames]
            )
            native_dir = diagnosis_dir / "native_frames" / name
            expected_paths = {native_dir / f"frame_{index:03d}.png" for index in range(17)}
            actual_paths = {path for path in native_dir.iterdir() if path.is_file()}
            if actual_paths != expected_paths:
                raise ValueError(f"Portable final-latent native-frame inventory failed for {name}.")
            actual_hashes = []
            native_frames = []
            for index in range(17):
                frame_path = native_dir / f"frame_{index:03d}.png"
                with Image.open(frame_path) as frame:
                    normalized = frame.convert("RGB")
                    if normalized.size != (320, 176):
                        raise ValueError(f"Portable final-latent native-frame geometry failed for {name}.")
                    pixels = np.ascontiguousarray(np.asarray(normalized, dtype=np.uint8))
                    native_frames.append(normalized.copy())
                actual_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
            if actual_hashes != expected_hashes:
                raise ValueError(f"Portable final-latent native-frame binding failed for {name}.")
            if not np.array_equal(
                decoded_videos[name].astype(np.uint8), np.stack([np.asarray(frame) for frame in native_frames])
            ):
                raise ValueError(f"Portable final-latent video/native-frame binding failed for {name}.")
            with Image.open(sheet_path) as sheet:
                if sheet.convert("RGB").size != (5280, 4352):
                    raise ValueError(f"Portable final-latent contact-sheet geometry failed for {name}.")
                normalized_sheet = sheet.convert("RGB")
                for index, frame in enumerate(native_frames):
                    row, column = divmod(index, 4)
                    x = 32 + column * (1280 + 32)
                    y = 160 + 32 + row * (704 + 96 + 32)
                    cell = normalized_sheet.crop((x, y, x + 1280, y + 704))
                    expected_cell = frame.resize((1280, 704), Image.Resampling.NEAREST)
                    if not np.array_equal(np.asarray(cell), np.asarray(expected_cell)):
                        raise ValueError(f"Portable final-latent contact-sheet/frame binding failed for {name}.")
        expected_video_metrics = {
            "tiled_vs_untiled": BerniniProofBundle._array_comparison_metrics(
                decoded_videos["mlx_untiled"], decoded_videos["mlx_tiled_runtime"], data_range=255.0
            ),
            "torch_vs_mlx_untiled": BerniniProofBundle._array_comparison_metrics(
                decoded_videos["torch_diffusers_0_35_2"],
                decoded_videos["mlx_untiled"],
                data_range=255.0,
            ),
        }
        if not BerniniProofBundle._numeric_tree_matches(diagnosis.get("encoded_video_metrics"), expected_video_metrics):
            raise ValueError("Portable final-latent encoded-video metrics are not reproducible.")

    @staticmethod
    def _verify_portable_report_evidence(
        *,
        durable_dir: Path,
        report: dict[str, Any],
        visual_review: dict[str, Any],
        sheet_manifest: dict[str, Any],
    ) -> dict[str, bool]:
        if (
            report.get("kind") != "bernini_r_1_3b_mlx_model_backed_proof"
            or report.get("official_source_revision") != BerniniProofBundle.OFFICIAL_SOURCE_REVISION
            or report.get("official_source_root") != "<official-source-root>"
            or report.get("component_compatibility") != BerniniProofBundle.EXPECTED_COMPONENT_PROVENANCE
        ):
            raise ValueError("Portable proof report has a stale top-level evidence contract.")
        runs = report.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("Portable proof report has no model-backed runs.")
        resolved_runs = [BerniniProofBundle._verify_portable_run(durable_dir=durable_dir, run=run) for run in runs]
        summary_path = BerniniProofBundle._resolve_bundle_path(
            durable_dir=durable_dir,
            portable_path=report.get("summary_contact_sheet"),
        )
        role_path = BerniniProofBundle._resolve_bundle_path(
            durable_dir=durable_dir,
            portable_path=report.get("role_control_contact_sheet"),
        )
        sheet_manifest_path = BerniniProofBundle._resolve_bundle_path(
            durable_dir=durable_dir,
            portable_path=report.get("sheet_manifest"),
        )
        visual_review_path = BerniniProofBundle._resolve_bundle_path(
            durable_dir=durable_dir,
            portable_path=report.get("visual_review_path"),
        )
        if (
            summary_path != durable_dir / "output_summary_contact_sheet.png"
            or role_path != durable_dir / "role_control_contact_sheet.png"
            or visual_review_path != durable_dir / "visual_review.json"
            or sheet_manifest_path != durable_dir / "sheet_manifest.json"
        ):
            raise ValueError("Portable proof report points to a noncanonical overview, review, or sheet manifest.")
        overview_checks = {
            "summary": BerniniProofBundle._overview_sheet_contract_passes(
                path=summary_path,
                details=report.get("summary_contact_sheet_details"),
            ),
            "role_control": BerniniProofBundle._overview_sheet_contract_passes(
                path=role_path,
                details=report.get("role_control_contact_sheet_details"),
            ),
        }
        if report.get("overview_sheet_checks") != overview_checks or not all(overview_checks.values()):
            raise ValueError("Portable overview contact sheets fail the current high-resolution contract.")
        BerniniProofBundle._verify_overview_sheet_pixels(
            runs=resolved_runs,
            summary_path=summary_path,
            summary_details=report.get("summary_contact_sheet_details"),
            role_path=role_path,
            role_details=report.get("role_control_contact_sheet_details"),
        )
        BerniniProofBundle._verify_portable_sheet_manifest(
            durable_dir=durable_dir,
            runs=runs,
            report=report,
            sheet_manifest=sheet_manifest,
        )
        machine_contract_passed = all(run["passed"] for run in resolved_runs) and all(overview_checks.values())
        visual_cases = visual_review.get("cases")
        run_ids = {run["case"]["case_id"] for run in runs}
        if not isinstance(visual_cases, dict) or set(visual_cases) != run_ids:
            raise ValueError("Portable visual review does not contain the exact current case inventory.")
        review_checks = {
            run["case"]["case_id"]: BerniniProofBundle._visual_review_check(
                run=run,
                review=visual_cases[run["case"]["case_id"]],
            )
            for run in runs
        }
        metadata_checks = BerniniProofBundle._visual_review_metadata_checks(visual_review)
        review_bindings = {
            "summary_contact_sheet_sha256": BerniniProofBundle._sha256(summary_path),
            "role_control_contact_sheet_sha256": BerniniProofBundle._sha256(role_path),
            "sheet_manifest_sha256": BerniniProofBundle._sha256(sheet_manifest_path),
        }
        global_bindings_match = all(visual_review.get(key) == value for key, value in review_bindings.items())
        visual_review_complete = (
            all(all(checks.values()) for checks in review_checks.values())
            and all(metadata_checks.values())
            and global_bindings_match
        )
        quality_checks = BerniniProofBundle._visual_quality_checks(visual_cases=visual_cases)
        quality_passed = BerniniProofBundle._visual_quality_passes(
            visual_review_complete=visual_review_complete,
            run_ids=run_ids,
            visual_quality_checks=quality_checks,
        )
        expected_report_values = {
            "overview_sheet_checks": overview_checks,
            "visual_review_metadata_checks": metadata_checks,
            "visual_review_checks": review_checks,
            "visual_review_global_bindings": {**review_bindings, "matched": global_bindings_match},
            "machine_contract_passed": machine_contract_passed,
            "visual_review_complete": visual_review_complete,
            "visual_quality_checks": quality_checks,
            "visual_quality_passed": quality_passed,
            "visual_review_passed": quality_passed,
            "passed": machine_contract_passed and quality_passed,
        }
        for key, expected in expected_report_values.items():
            if report.get(key) != expected:
                raise ValueError(f"Portable proof report field {key!r} does not match recomputed evidence.")
        return {
            "passed": expected_report_values["passed"],
            "visual_quality_passed": quality_passed,
        }

    @staticmethod
    def _verify_overview_sheet_pixels(
        *,
        runs: list[dict[str, Any]],
        summary_path: Path,
        summary_details: Any,
        role_path: Path,
        role_details: Any,
    ) -> None:
        case_order = {case_id: index for index, case_id in enumerate(BerniniProofBundle.cases())}
        try:
            ordered_runs = sorted(runs, key=lambda run: case_order[run["case"]["case_id"]])
        except (KeyError, TypeError):
            raise ValueError("Portable overview sheets contain an unknown case ordering.") from None
        with TemporaryDirectory(prefix="mlxgen-bernini-overview-verify-") as temp_dir:
            temp_root = Path(temp_dir)
            expected_summary_path = temp_root / "output_summary_contact_sheet.png"
            expected_role_path = temp_root / "role_control_contact_sheet.png"
            expected_summary_details = BerniniProofBundle._save_summary_sheet(
                runs=ordered_runs,
                output_path=expected_summary_path,
            )
            expected_role_details = BerniniProofBundle._save_role_control_sheet(
                runs=ordered_runs,
                output_path=expected_role_path,
            )
            if summary_details != expected_summary_details or role_details != expected_role_details:
                raise ValueError("Portable overview sheet details do not match the retained decoded frames.")
            for label, actual_path, expected_path in (
                ("summary", summary_path, expected_summary_path),
                ("role-control", role_path, expected_role_path),
            ):
                if not expected_path.is_file() or not BerniniProofBundle._sheet_pixels_match(
                    actual_path=actual_path,
                    expected_path=expected_path,
                ):
                    raise ValueError(f"Portable {label} overview sheet pixel binding failed.")

    @staticmethod
    def _verify_portable_run(*, durable_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(run, dict):
            raise ValueError("Portable proof run is malformed.")
        resolved = copy.deepcopy(run)
        path_fields = ("output_path", "stdout_path", "stderr_path", "input_source_path", "official_output_path")
        for field in path_fields:
            value = run.get(field)
            resolved[field] = (
                BerniniProofBundle._resolve_bundle_path(durable_dir=durable_dir, portable_path=value)
                if value is not None
                else None
            )
        references = run.get("input_reference_paths")
        sheets = run.get("sheets")
        if not isinstance(references, list) or not isinstance(sheets, dict):
            raise ValueError("Portable proof run has malformed input or sheet paths.")
        resolved["input_reference_paths"] = [
            str(BerniniProofBundle._resolve_bundle_path(durable_dir=durable_dir, portable_path=value))
            for value in references
        ]
        resolved["input_source_path"] = str(resolved["input_source_path"]) if resolved["input_source_path"] else None
        resolved["official_output_path"] = (
            str(resolved["official_output_path"]) if resolved["official_output_path"] else None
        )
        for field in ("output_path", "stdout_path", "stderr_path"):
            resolved[field] = str(resolved[field])
        resolved["sheets"] = {
            label: str(BerniniProofBundle._resolve_bundle_path(durable_dir=durable_dir, portable_path=value))
            for label, value in sheets.items()
        }
        metadata_path = Path(resolved["output_path"]).with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text())
        if run.get("metadata") != metadata:
            raise ValueError(f"Portable proof metadata sidecar disagrees for {run.get('case', {}).get('case_id')}.")
        resolved_metadata = copy.deepcopy(metadata)
        metadata_references = resolved_metadata.get("reference_image_paths")
        if isinstance(metadata_references, list):
            resolved_metadata["reference_image_paths"] = [
                str(BerniniProofBundle._resolve_bundle_path(durable_dir=durable_dir, portable_path=value))
                for value in metadata_references
            ]
        if resolved_metadata.get("video_path") is not None:
            resolved_metadata["video_path"] = str(
                BerniniProofBundle._resolve_bundle_path(
                    durable_dir=durable_dir,
                    portable_path=resolved_metadata["video_path"],
                )
            )
        resolved["metadata"] = resolved_metadata
        BerniniProofBundle._verify_portable_artifact_hashes(durable_dir=durable_dir, run=run)
        BerniniProofBundle._verify_case_sheet_pixels(run=resolved)
        samples = run.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Portable proof memory trace is missing for {run.get('case', {}).get('case_id')}.")
        rss_values = [sample.get("rss_bytes") for sample in samples if isinstance(sample, dict)]
        physical_values = [
            sample.get("darwin_physical_footprint_bytes") for sample in samples if isinstance(sample, dict)
        ]
        if not all(isinstance(value, int) and value >= 0 for value in rss_values) or not any(
            isinstance(value, int) and value >= 0 for value in physical_values
        ):
            raise ValueError(f"Portable proof memory trace is malformed for {run.get('case', {}).get('case_id')}.")
        expected_sampler = {
            "sample_count": len(samples),
            "peak_sampled_rss_bytes": max(rss_values),
            "peak_sampled_darwin_physical_footprint_bytes": max(
                value for value in physical_values if isinstance(value, int)
            ),
        }
        if run.get("sampler") != expected_sampler:
            raise ValueError(f"Portable proof memory summary is stale for {run.get('case', {}).get('case_id')}.")
        actual_video_health = GenerationMemoryBenchmark._video_health(Path(resolved["output_path"]))
        portable_video_health = copy.deepcopy(actual_video_health)
        portable_video_health["path"] = run.get("output_path")
        if not BerniniProofBundle._numeric_tree_matches(run.get("video_health"), portable_video_health):
            raise ValueError(f"Portable proof video-health record is stale for {run.get('case', {}).get('case_id')}.")
        resolved["video_health"] = actual_video_health
        contract_checks = BerniniProofBundle._contract_checks(resolved)
        passed = actual_video_health.get("status") == "ok" and all(contract_checks.values())
        if run.get("contract_checks") != contract_checks or run.get("passed") is not passed:
            raise ValueError(
                f"Portable proof run contract is not reproducible for {run.get('case', {}).get('case_id')}."
            )
        if run.get("output_sha256") != BerniniProofBundle._sha256(Path(resolved["output_path"])):
            raise ValueError(f"Portable proof output hash is stale for {run.get('case', {}).get('case_id')}.")
        resolved["contract_checks"] = contract_checks
        resolved["passed"] = passed
        return resolved

    @staticmethod
    def _verify_case_sheet_pixels(*, run: dict[str, Any]) -> None:
        case_payload = run.get("case")
        if not isinstance(case_payload, dict):
            raise ValueError("Portable proof case sheet has no exact case contract.")
        try:
            case = BerniniProofCase(**case_payload)
        except TypeError:
            raise ValueError("Portable proof case sheet has a malformed case contract.") from None
        reference_paths = [Path(path) for path in run.get("input_reference_paths", [])]
        expected_reference_names = [
            f"reference_{index:02d}{Path(source).suffix}" for index, source in enumerate(case.reference_images)
        ]
        if [path.name for path in reference_paths] != expected_reference_names:
            raise ValueError(f"Portable proof reference ordering is not canonical for {case.case_id}.")
        source_path = Path(run["input_source_path"]) if run.get("input_source_path") else None
        official_path = Path(run["official_output_path"]) if run.get("official_output_path") else None
        output_path = Path(str(run.get("output_path", "")))
        actual_sheets = run.get("sheets")
        if not isinstance(actual_sheets, dict):
            raise ValueError(f"Portable proof case sheets are malformed for {case.case_id}.")
        BerniniProofBundle._verify_source_sheet_sampling(
            case=case,
            source_path=source_path,
            metadata=run.get("metadata") or {},
        )
        with TemporaryDirectory(prefix=f"mlxgen-bernini-{case.case_id}-sheet-verify-") as temp_dir:
            expected_sheets, expected_details = BerniniProofBundle._save_case_sheets(
                case=case,
                reference_paths=reference_paths,
                source_path=source_path,
                official_path=official_path,
                output_path=output_path,
                run_dir=Path(temp_dir),
                output_metadata=run.get("metadata") or {},
            )
            if set(actual_sheets) != set(expected_sheets) or run.get("sheet_details") != expected_details:
                raise ValueError(f"Portable proof case sheet details do not match retained frames for {case.case_id}.")
            for label, expected_path in expected_sheets.items():
                actual_path = Path(actual_sheets[label])
                if not BerniniProofBundle._sheet_pixels_match(
                    actual_path=actual_path,
                    expected_path=Path(expected_path),
                ):
                    raise ValueError(f"Portable proof case sheet pixel binding failed for {case.case_id}/{label}.")

    @staticmethod
    def _verify_source_sheet_sampling(
        *,
        case: BerniniProofCase,
        source_path: Path | None,
        metadata: dict[str, Any],
    ) -> None:
        if source_path is None:
            return
        source_info = VideoUtil.inspect_video(source_path)
        source_frame_count = source_info.source_frame_count
        source_width = source_info.source_width
        source_height = source_info.source_height
        source_fps = source_info.fps
        if (
            source_frame_count is None
            or source_frame_count < 1
            or source_width is None
            or source_height is None
            or source_fps is None
            or source_fps <= 0
        ):
            clip = VideoUtil.read_video_clip(source_path, max_frames=None)
            source_frame_count = len(clip.frames)
            source_width, source_height = clip.frames[0].size
            source_fps = clip.fps
        expected_indices = BerniniRenderer._smart_video_indices(
            total_frames=source_frame_count,
            video_fps=float(source_fps),
            fps=float(case.fps),
            frame_factor=4,
            max_frames=case.frames,
            add_one=True,
        )
        output_height, output_width = BerniniRenderer._closest_spatial_size_for_ratio(
            requested_height=case.height,
            requested_width=case.width,
            source_height=source_height,
            source_width=source_width,
            multiple_h=16,
            multiple_w=16,
        )
        condition_width, condition_height = BerniniRenderer._condition_dimensions(
            width=output_width,
            height=output_height,
            max_size=case.max_condition_size,
        )
        if (
            metadata.get("canvas_policy") != "source-aspect"
            or metadata.get("condition_resize_backend") != "pillow-bicubic"
            or metadata.get("source_width") != source_width
            or metadata.get("source_height") != source_height
            or not isinstance(metadata.get("source_fps"), (int, float))
            or not math.isclose(float(metadata["source_fps"]), float(source_fps), rel_tol=0.0, abs_tol=1e-6)
            or metadata.get("source_frame_count") != source_frame_count
            or metadata.get("source_sample_indices") != expected_indices
            or metadata.get("video_condition_width") != condition_width
            or metadata.get("video_condition_height") != condition_height
            or metadata.get("video_condition_frames") != len(expected_indices)
        ):
            raise ValueError(f"Portable proof source-sheet sampling is stale for {case.case_id}.")

    @staticmethod
    def _sheet_pixels_match(*, actual_path: Path, expected_path: Path) -> bool:
        try:
            with Image.open(actual_path) as actual_source, Image.open(expected_path) as expected_source:
                actual = actual_source.convert("RGB")
                expected = expected_source.convert("RGB")
                return actual.size == expected.size and ImageChops.difference(actual, expected).getbbox() is None
        except (OSError, ValueError):
            return False

    @staticmethod
    def _verify_portable_artifact_hashes(*, durable_dir: Path, run: dict[str, Any]) -> None:
        artifacts = run.get("artifact_hashes")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "output",
            "metadata",
            "stdout",
            "stderr",
            "references",
            "source",
            "official_output",
            "sheets",
        }:
            raise ValueError("Portable proof run has an incomplete artifact inventory.")
        output_path = str(run.get("output_path", ""))
        expected = {
            "output": output_path,
            "metadata": str(Path(output_path).with_suffix(".metadata.json")),
            "stdout": run.get("stdout_path"),
            "stderr": run.get("stderr_path"),
        }
        for label, portable_path in expected.items():
            BerniniProofBundle._verify_hashed_artifact(
                durable_dir=durable_dir,
                artifact=artifacts.get(label),
                expected_path=portable_path,
            )
        references = run.get("input_reference_paths") or []
        if not isinstance(artifacts.get("references"), list) or len(artifacts["references"]) != len(references):
            raise ValueError("Portable proof reference artifact inventory is incomplete.")
        for artifact, portable_path in zip(artifacts["references"], references):
            BerniniProofBundle._verify_hashed_artifact(
                durable_dir=durable_dir,
                artifact=artifact,
                expected_path=portable_path,
            )
        for label, run_field in (("source", "input_source_path"), ("official_output", "official_output_path")):
            portable_path = run.get(run_field)
            artifact = artifacts.get(label)
            if portable_path is None:
                if artifact is not None:
                    raise ValueError(f"Portable proof has an unexpected {label} artifact.")
            else:
                BerniniProofBundle._verify_hashed_artifact(
                    durable_dir=durable_dir,
                    artifact=artifact,
                    expected_path=portable_path,
                )
        sheets = run.get("sheets") or {}
        if not isinstance(artifacts.get("sheets"), dict) or set(artifacts["sheets"]) != set(sheets):
            raise ValueError("Portable proof sheet artifact inventory is incomplete.")
        for label, portable_path in sheets.items():
            BerniniProofBundle._verify_hashed_artifact(
                durable_dir=durable_dir,
                artifact=artifacts["sheets"][label],
                expected_path=portable_path,
            )

    @staticmethod
    def _verify_hashed_artifact(*, durable_dir: Path, artifact: Any, expected_path: Any) -> None:
        if not isinstance(artifact, dict) or artifact.get("path") != expected_path:
            raise ValueError(f"Portable proof artifact path binding failed: {expected_path!r}.")
        path = BerniniProofBundle._resolve_bundle_path(durable_dir=durable_dir, portable_path=expected_path)
        if (
            artifact.get("sha256") != BerniniProofBundle._sha256(path)
            or artifact.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Portable proof artifact hash binding failed: {expected_path!r}.")

    @staticmethod
    def _verify_portable_sheet_manifest(
        *,
        durable_dir: Path,
        runs: list[dict[str, Any]],
        report: dict[str, Any],
        sheet_manifest: dict[str, Any],
    ) -> None:
        expected_cases = {}
        for run in runs:
            case_id = run["case"]["case_id"]
            expected_cases[case_id] = {}
            for label, portable_path in sorted(run["sheets"].items()):
                path = BerniniProofBundle._resolve_bundle_path(
                    durable_dir=durable_dir,
                    portable_path=portable_path,
                )
                expected_cases[case_id][label] = {
                    "path": path.relative_to(durable_dir).as_posix(),
                    "sha256": BerniniProofBundle._sha256(path),
                    **run["sheet_details"][label],
                }
        expected_summary = {
            "path": "output_summary_contact_sheet.png",
            "sha256": BerniniProofBundle._sha256(durable_dir / "output_summary_contact_sheet.png"),
            **report["summary_contact_sheet_details"],
        }
        expected_role = {
            "path": "role_control_contact_sheet.png",
            "sha256": BerniniProofBundle._sha256(durable_dir / "role_control_contact_sheet.png"),
            **report["role_control_contact_sheet_details"],
        }
        if (
            set(sheet_manifest)
            != {
                "schema_version",
                "sheet_contract_version",
                "pillow_version",
                "font",
                "cases",
                "summary",
                "role_control",
            }
            or sheet_manifest.get("schema_version") != 1
            or sheet_manifest.get("sheet_contract_version") != BerniniProofBundle.SHEET_CONTRACT_VERSION
            or sheet_manifest.get("pillow_version") != PILLOW_VERSION
            or sheet_manifest.get("font") != "Pillow embedded scalable default"
            or sheet_manifest.get("cases") != expected_cases
            or sheet_manifest.get("summary") != expected_summary
            or sheet_manifest.get("role_control") != expected_role
        ):
            raise ValueError("Portable sheet manifest is not reproducible from the retained sheets.")

    @staticmethod
    def _resolve_bundle_path(*, durable_dir: Path, portable_path: Any) -> Path:
        if not isinstance(portable_path, str) or not portable_path.startswith("<bundle-root>/"):
            raise ValueError(f"Portable proof path is not bundle-local: {portable_path!r}.")
        relative = Path(portable_path.removeprefix("<bundle-root>/"))
        path = durable_dir / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.is_file():
            raise ValueError(f"Portable proof path does not resolve to a regular bundle artifact: {portable_path!r}.")
        try:
            path.resolve().relative_to(durable_dir.resolve())
        except ValueError:
            raise ValueError(f"Portable proof path escapes the bundle: {portable_path!r}.") from None
        return path

    @staticmethod
    def _verify_cycle17_tensor_artifacts(
        *,
        diagnosis_dir: Path,
        diagnosis: dict[str, Any],
        mlx_report: dict[str, Any],
        torch_report: dict[str, Any],
    ) -> None:
        expected_mlx_header = {
            "schema_version": 1,
            "kind": "bernini_mlx_final_latent_decode_diagnosis",
            "transformer_revision": "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce",
            "vae_revision": "ec4d2cb062b548996b179d493fdd05340de702a1",
            "latent_shape": [1, 16, 5, 22, 40],
            "untiled_decoded_shape": [1, 3, 17, 176, 320],
            "passed_structural_checks": True,
        }
        expected_torch_header = {
            "schema_version": 1,
            "kind": "bernini_torch_final_latent_decode_diagnosis",
            "diffusers_version": "0.35.2",
            "device": "mps",
            "vae_revision": "ec4d2cb062b548996b179d493fdd05340de702a1",
            "normalized_latent_shape": [1, 16, 5, 22, 40],
            "decoded_shape": [1, 3, 17, 176, 320],
            "passed_structural_checks": True,
        }
        if any(mlx_report.get(key) != value for key, value in expected_mlx_header.items()) or any(
            torch_report.get(key) != value for key, value in expected_torch_header.items()
        ):
            raise ValueError("Portable final-latent standalone decoder reports have a stale structural contract.")
        expected_mlx_tensor_keys = {"final_normalized_latents", "mlx_untiled_decoded"}
        expected_torch_tensor_keys = {"final_normalized_latents", "torch_decoded"}
        if (
            set(mlx_report.get("tensor_artifacts") or {}) != expected_mlx_tensor_keys
            or set(torch_report.get("tensor_artifacts") or {}) != expected_torch_tensor_keys
        ):
            raise ValueError("Portable final-latent standalone reports have an unexpected tensor inventory.")
        settings_keys = (
            "seed",
            "prompt",
            "negative_prompt",
            "system_prompt",
            "steps",
            "frames",
            "fps",
            "requested_width",
            "requested_height",
            "text_guidance",
            "apg_eta",
            "apg_norm_threshold",
            "apg_momentum",
            "flow_shift",
            "max_condition_size",
        )
        settings = mlx_report.get("settings") or {}
        template_path = diagnosis_dir.parent / "cycle12_source_generation" / "v2v_snowman_17f_40steps.metadata.json"
        template = json.loads(template_path.read_text())
        expected_settings = {key: template.get(key) for key in settings_keys}
        expected_prompt_hash = BerniniProofBundle.EXPECTED_UPSTREAM_PROMPT_SHA256["assets/testcases/v2v/v2v_case1.json"]
        if (
            mlx_report.get("metadata_template")
            != "<bundle-root>/diagnostics/cycle12_source_generation/v2v_snowman_17f_40steps.metadata.json"
            or mlx_report.get("source_video") != "<bundle-root>/cases/run_1/v2v_snowman/inputs/source.mp4"
            or mlx_report.get("tiled_video")
            != "<bundle-root>/diagnostics/final_latent_three_way_decode/mlx_tiled_runtime.mp4"
            or mlx_report.get("untiled_video")
            != "<bundle-root>/diagnostics/final_latent_three_way_decode/mlx_untiled.mp4"
            or torch_report.get("video")
            != "<bundle-root>/diagnostics/final_latent_three_way_decode/torch_diffusers_0_35_2.mp4"
            or settings != expected_settings
            or not isinstance(template.get("prompt"), str)
            or hashlib.sha256(template["prompt"].encode()).hexdigest() != expected_prompt_hash
        ):
            raise ValueError("Portable final-latent diagnosis is not bound to the intended Cycle12 run.")
        expected_artifacts = {
            "final_normalized_latents": ([1, 16, 5, 22, 40], "float32"),
            "mlx_untiled_decoded": ([1, 3, 17, 176, 320], "float32"),
            "torch_decoded": ([1, 3, 17, 176, 320], "float32"),
        }
        report_artifacts = {
            **(mlx_report.get("tensor_artifacts") or {}),
            **(torch_report.get("tensor_artifacts") or {}),
        }
        mlx_final = (mlx_report.get("tensor_artifacts") or {}).get("final_normalized_latents")
        torch_final = (torch_report.get("tensor_artifacts") or {}).get("final_normalized_latents")
        if mlx_final != torch_final or set(report_artifacts) != set(expected_artifacts):
            raise ValueError("Portable final-latent tensor reports do not share one exact normalized latent.")
        arrays = {}
        for label, (shape, dtype) in expected_artifacts.items():
            artifact = report_artifacts[label]
            expected_path = f"<bundle-root>/diagnostics/final_latent_three_way_decode/{label}.npy"
            path = diagnosis_dir / f"{label}.npy"
            if (
                not isinstance(artifact, dict)
                or artifact.get("path") != expected_path
                or artifact.get("sha256") != BerniniProofBundle._sha256(path)
                or artifact.get("size_bytes") != path.stat().st_size
                or artifact.get("shape") != shape
                or artifact.get("dtype") != dtype
            ):
                raise ValueError(f"Portable final-latent tensor binding failed for {label}.")
            value = np.load(path, allow_pickle=False)
            if list(value.shape) != shape or str(value.dtype) != dtype or not np.isfinite(value).all():
                raise ValueError(f"Portable final-latent tensor payload failed for {label}.")
            arrays[label] = value.astype(np.float64)
        expected_raw = BerniniProofBundle._array_comparison_metrics(
            arrays["torch_decoded"], arrays["mlx_untiled_decoded"], data_range=2.0
        )
        expected_per_frame = [
            {
                "frame_index": index,
                **BerniniProofBundle._array_comparison_metrics(
                    arrays["torch_decoded"][:, :, index],
                    arrays["mlx_untiled_decoded"][:, :, index],
                    data_range=2.0,
                ),
            }
            for index in range(17)
        ]
        if not BerniniProofBundle._numeric_tree_matches(diagnosis.get("raw_torch_vs_mlx_untiled"), expected_raw):
            raise ValueError("Portable final-latent raw decode metrics are not reproducible.")
        if not BerniniProofBundle._numeric_tree_matches(
            diagnosis.get("raw_torch_vs_mlx_untiled_per_frame"), expected_per_frame
        ):
            raise ValueError("Portable final-latent per-frame decode metrics are not reproducible.")
        for label, video_name in (
            ("mlx_untiled_decoded", "mlx_untiled.mp4"),
            ("torch_decoded", "torch_diffusers_0_35_2.mp4"),
        ):
            expected_frames = (np.clip(np.transpose(arrays[label][0], (1, 2, 3, 0)) / 2 + 0.5, 0, 1) * 255).round()
            clip = VideoUtil.read_video_clip(diagnosis_dir / video_name, max_frames=None)
            actual_frames = np.stack([np.asarray(frame.convert("RGB"), dtype=np.float64) for frame in clip.frames])
            linkage = BerniniProofBundle._array_comparison_metrics(expected_frames, actual_frames, data_range=255.0)
            if linkage["mean_absolute_error"] > 3.0 or linkage["psnr_db"] < 35.0:
                raise ValueError(f"Portable final-latent decoded tensor/video binding failed for {label}.")

    @staticmethod
    def _array_comparison_metrics(reference: np.ndarray, actual: np.ndarray, *, data_range: float) -> dict[str, float]:
        if reference.shape != actual.shape:
            raise ValueError(f"Proof metric shape mismatch: {reference.shape} != {actual.shape}.")
        delta = actual - reference
        reference_norm = float(np.linalg.norm(reference.reshape(-1)))
        delta_norm = float(np.linalg.norm(delta.reshape(-1)))
        mse = float(np.mean(np.square(delta)))
        return {
            "relative_l2": delta_norm / reference_norm if reference_norm else math.inf,
            "mean_absolute_error": float(np.mean(np.abs(delta))),
            "max_absolute_error": float(np.max(np.abs(delta))),
            "psnr_db": math.inf if mse == 0 else float(20 * math.log10(data_range / math.sqrt(mse))),
        }

    @staticmethod
    def _numeric_tree_matches(stored: Any, expected: Any) -> bool:
        if isinstance(expected, dict):
            return (
                isinstance(stored, dict)
                and set(stored) == set(expected)
                and all(BerniniProofBundle._numeric_tree_matches(stored[key], value) for key, value in expected.items())
            )
        if isinstance(expected, list):
            return (
                isinstance(stored, list)
                and len(stored) == len(expected)
                and all(BerniniProofBundle._numeric_tree_matches(left, right) for left, right in zip(stored, expected))
            )
        if isinstance(expected, float):
            return isinstance(stored, (int, float)) and math.isclose(
                float(stored), expected, rel_tol=1e-12, abs_tol=1e-12
            )
        return stored == expected

    @staticmethod
    def _sanitize_durable_text(*, durable_dir: Path, replacements: tuple[tuple[str, str], ...]) -> None:
        for path in durable_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".log", ".md"}:
                continue
            text = path.read_text(errors="replace")
            for source, replacement in replacements:
                text = text.replace(source, replacement)
            text = re.sub(
                r"/(?:private/)?tmp/(?:mlxgen-bernini-ref|bernini-review)[^/\s\"']*/(?:Bernini|repo)",
                "<official-source-root>",
                text,
            )
            path.write_text(text)

    @staticmethod
    def _refresh_portable_report(durable_dir: Path) -> None:
        report_path = durable_dir / "bernini_proof_report.json"
        report = json.loads(report_path.read_text())
        for run in report["runs"]:
            for artifact in BerniniProofBundle._iter_hashed_artifacts(run["artifact_hashes"]):
                portable_path = artifact["path"]
                if portable_path.startswith("<bundle-root>/"):
                    path = durable_dir / portable_path.removeprefix("<bundle-root>/")
                    artifact.update(BerniniProofBundle._hashed_artifact(path))
                    artifact["path"] = portable_path
            run["output_sha256"] = run["artifact_hashes"]["output"]["sha256"]
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        (durable_dir / "bernini_proof_report.md").write_text(BerniniProofBundle._markdown_report(report))

    @staticmethod
    def _iter_hashed_artifacts(value: Any):
        if isinstance(value, dict):
            if {"path", "sha256", "size_bytes"}.issubset(value):
                yield value
                return
            for child in value.values():
                yield from BerniniProofBundle._iter_hashed_artifacts(child)
        elif isinstance(value, list):
            for child in value:
                yield from BerniniProofBundle._iter_hashed_artifacts(child)

    @staticmethod
    def _write_component_compatibility(durable_dir: Path) -> None:
        compatibility = BerniniProofBundle._expected_component_compatibility()
        (durable_dir / "component_compatibility.json").write_text(json.dumps(compatibility, indent=2, sort_keys=True))

    @staticmethod
    def _expected_component_compatibility() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "profile": "bernini_r_1_3b_2026_08_04",
            "renderer_only": True,
            "runtime_precision": "bfloat16",
            "runtime_policy": BerniniProofBundle.EXPECTED_RUNTIME_POLICY,
            "quantization_supported": False,
            "components": BerniniProofBundle.EXPECTED_COMPONENT_PROVENANCE,
            "runtime_contract": BerniniProofBundle.EXPECTED_RUNTIME_CONTRACT,
            "parity_reports": BerniniProofBundle.EXPECTED_PARITY_REPORTS,
        }

    @staticmethod
    def _write_upstream_attribution(durable_dir: Path) -> None:
        text = """# Upstream Bernini evidence attribution

The files below `cases/run_1/*/inputs/` named `reference_*`, `source.*`, or
`official_output.*` are copied from ByteDance's Bernini repository at revision
`2d2b4591ac053ec25c6371b01a5a6746679e5793` solely to make this engineering proof
self-contained. The upstream repository declares Apache License 2.0; the exact license text is
retained as `UPSTREAM_BERNINI_LICENSE.txt`. The repository contained no `NOTICE` file at that
revision.

The `mlx_*.png` sheets, generated MP4s without the `official_output` name, sidecars, logs,
reports, and review records are MLX-Gen validation artifacts, not ByteDance benchmark outputs.
Official comparison clips remain labeled `official_output` and are never presented as MLX output.
The upstream repository does not attest which checkpoint or inference recipe produced those clips;
they are qualitative targets, not Bernini-R 1.3B parity baselines.
"""
        (durable_dir / "UPSTREAM_ATTRIBUTION.md").write_text(text)

    @staticmethod
    def _write_portable_manifest(durable_dir: Path) -> None:
        manifest_path = durable_dir / "portable_manifest.json"
        report = json.loads((durable_dir / "bernini_proof_report.json").read_text())
        case_fingerprints = BerniniProofBundle._portable_case_fingerprints(report)
        entries = []
        for path in sorted(durable_dir.rglob("*")):
            if not path.is_file() or path == manifest_path:
                continue
            entries.append(
                {
                    "path": path.relative_to(durable_dir).as_posix(),
                    "sha256": BerniniProofBundle._sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": 2,
            "kind": "portable_bernini_r_1_3b_proof_bundle",
            "official_source_revision": BerniniProofBundle.OFFICIAL_SOURCE_REVISION,
            "case_contract_version": BerniniProofBundle.CASE_CONTRACT_VERSION,
            "case_fingerprints": case_fingerprints,
            "proof_profile_sha256": BerniniProofBundle._proof_profile_hash(case_fingerprints),
            "path_placeholders": {
                "<bundle-root>": "the directory containing this manifest",
                "<official-source-root>": "a checkout of the pinned official Bernini source",
                "<repo-root>": "the MLX-Gen checkout used for the run",
                "<user-home>": "the validation user's home directory",
                "<validation-root>": "the ignored local Bernini validation output directory",
            },
            "entries": entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    @staticmethod
    def _portable_case_fingerprints(report: dict[str, Any]) -> dict[str, str]:
        runs = report.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("Portable proof report must contain model-backed runs.")
        fingerprints = {}
        for run in runs:
            case = run.get("case") if isinstance(run, dict) else None
            case_id = case.get("case_id") if isinstance(case, dict) else None
            fingerprint = run.get("case_fingerprint") if isinstance(run, dict) else None
            if not isinstance(case_id, str) or not isinstance(fingerprint, str):
                raise ValueError("Portable proof report contains a malformed run profile.")
            if case_id in fingerprints:
                raise ValueError(f"Portable proof report contains duplicate case {case_id!r}.")
            fingerprints[case_id] = fingerprint
        return dict(sorted(fingerprints.items()))

    @staticmethod
    def _proof_profile_hash(case_fingerprints: dict[str, str]) -> str:
        payload = {
            "case_contract_version": BerniniProofBundle.CASE_CONTRACT_VERSION,
            "case_fingerprints": case_fingerprints,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(serialized).hexdigest()

    @staticmethod
    def _verify_current_portable_profile(*, report: dict[str, Any], manifest: dict[str, Any]) -> None:
        current_cases = BerniniProofBundle.cases()
        runs = report.get("runs")
        if not isinstance(runs, list):
            raise ValueError("Portable proof report runs must be a list.")
        if len(runs) != len(current_cases):
            raise ValueError("Portable proof report does not contain the complete current case profile.")
        stored_fingerprints = BerniniProofBundle._portable_case_fingerprints(report)
        manifest_fingerprints = manifest.get("case_fingerprints")
        if manifest_fingerprints != stored_fingerprints:
            raise ValueError("Portable proof manifest is not bound to the report case fingerprints.")
        if manifest.get("proof_profile_sha256") != BerniniProofBundle._proof_profile_hash(stored_fingerprints):
            raise ValueError("Portable proof profile hash is invalid.")
        seen = set()
        for run in runs:
            case = run["case"]
            case_id = case["case_id"]
            if case_id not in current_cases or case_id in seen:
                raise ValueError(f"Portable proof report contains an unknown or duplicate case: {case_id!r}")
            seen.add(case_id)
            current_case = current_cases[case_id]
            current_case_json = json.loads(json.dumps(asdict(current_case)))
            if case != current_case_json:
                raise ValueError(f"Portable proof case {case_id!r} is stale.")
            prompt = run.get("prompt")
            if not isinstance(prompt, str):
                raise ValueError(f"Portable proof case {case_id!r} has no exact prompt binding.")
            if current_case.prompt_override is not None:
                prompt_matches_source = prompt == current_case.prompt_override
            else:
                expected_prompt_hash = BerniniProofBundle.EXPECTED_UPSTREAM_PROMPT_SHA256.get(
                    str(current_case.prompt_json)
                )
                prompt_matches_source = (
                    expected_prompt_hash is not None
                    and hashlib.sha256(prompt.encode()).hexdigest() == expected_prompt_hash
                )
            if not prompt_matches_source:
                raise ValueError(f"Portable proof case {case_id!r} does not use its pinned prompt source.")
            expected = BerniniProofBundle._case_fingerprint(case=current_case, prompt=prompt)
            if run.get("case_fingerprint") != expected:
                raise ValueError(f"Portable proof case {case_id!r} has a stale fingerprint.")
        if seen != set(current_cases):
            raise ValueError("Portable proof report is missing current cases.")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _image_sha256(image: Image.Image) -> str:
        normalized = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"RGB:{normalized.width}x{normalized.height}:".encode())
        digest.update(normalized.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        lines = [
            "# Bernini-R 1.3B MLX model-backed proof",
            "",
            f"- Created: `{report['created_at']}`",
            f"- Official source revision: `{report['official_source_revision']}`",
            f"- Machine-level contract pass: `{report['machine_contract_passed']}`",
            f"- Visual evidence review complete and hash-bound: `{report['visual_review_complete']}`",
            f"- Required visual quality cases pass: `{report['visual_quality_passed']}`",
            f"- Overall pass: `{report['passed']}`",
            "- Recorded visual inspection: see `visual_review.md` and `visual_review.json`.",
            "",
            "| Case | Mode | Size | Frames | Steps | Wall | Physical peak | Contract | Visual |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for run in report["runs"]:
            metadata = run.get("metadata") or {}
            physical = run["sampler"].get("peak_sampled_darwin_physical_footprint_bytes")
            lines.append(
                "| {case} | {mode} | {width}x{height} | {frames} | {steps} | {wall:.2f}s | {physical} GB | {passed} | {visual} |".format(
                    case=run["case"]["case_id"],
                    mode=metadata.get("bernini_guidance_mode"),
                    width=run["video_health"].get("width"),
                    height=run["video_health"].get("height"),
                    frames=run["video_health"].get("frames"),
                    steps=run["case"]["steps"],
                    wall=float(run["wall_seconds"]),
                    physical=(f"{physical / 1000**3:.3f}" if physical is not None else "n/a"),
                    passed=run["passed"],
                    visual=report["visual_review"]["cases"].get(run["case"]["case_id"], {}).get("status", "pending"),
                )
            )
        lines.extend(["", "## Artifacts", ""])
        for run in report["runs"]:
            output_path = str(run["output_path"])
            metadata_path = str(Path(output_path).with_suffix(".metadata.json"))
            lines.extend(
                [
                    f"### {run['case']['case_id']}",
                    "",
                    f"- {BerniniProofBundle._markdown_artifact_link('Generated MP4', output_path)}",
                    f"- SHA-256: `{run['output_sha256']}`",
                    f"- {BerniniProofBundle._markdown_artifact_link('Metadata', metadata_path)}",
                    "- Contact sheets:",
                    *[
                        f"  - {BerniniProofBundle._markdown_artifact_link(label, path)}"
                        for label, path in sorted(run["sheets"].items())
                    ],
                    f"- Contract checks: `{json.dumps(run['contract_checks'], sort_keys=True)}`",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _markdown_artifact_link(label: str, path: str) -> str:
        target = path.removeprefix("<bundle-root>/")
        return f"[{label}](<{target}>)"

    @staticmethod
    def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
        return {
            "passed": report["passed"],
            "machine_contract_passed": report["machine_contract_passed"],
            "visual_review_complete": report["visual_review_complete"],
            "visual_quality_passed": report["visual_quality_passed"],
            "visual_review_passed": report["visual_review_passed"],
            "report": str(Path(report["summary_contact_sheet"]).parent / "bernini_proof_report.json"),
            "summary_contact_sheet": report["summary_contact_sheet"],
            "runs": {
                run["case"]["case_id"]: {
                    "passed": run["passed"],
                    "wall_seconds": round(float(run["wall_seconds"]), 3),
                    "peak_physical_gb": (
                        round(run["sampler"]["peak_sampled_darwin_physical_footprint_bytes"] / 1000**3, 3)
                        if run["sampler"].get("peak_sampled_darwin_physical_footprint_bytes") is not None
                        else None
                    ),
                    "output": run["output_path"],
                }
                for run in report["runs"]
            },
        }


if __name__ == "__main__":
    BerniniProofBundle.main()
