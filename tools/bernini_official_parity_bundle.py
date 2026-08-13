import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class BerniniOfficialParityBundle:
    # Keep this in lockstep with OfficialPublicBerniniCases.MANIFEST: these are
    # the public Bernini-R 1.3B rows documented by the upstream 1.3B release,
    # not the larger mixed testcase inventory in the Bernini repo.
    CASE_IDS = (
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
    DEFAULT_SEARCH_ROOTS = (
        "validation_outputs/bernini_r_1_3b_2026_08_11",
        "validation_outputs/bernini_r_1_3b_2026_08_10",
    )
    COPY_FILES = ("README.md", "input_sheet.png", "official_sheet.png", "mlx_sheet.png")
    REQUIRED_PROOF_FILES = ("proof.json", "README.md", "input_sheet.png", "official_sheet.png", "mlx_sheet.png")
    SUMMARY_SHEET_NAME = "official_public_summary_contact_sheet.png"
    SUMMARY_PREVIEW_NAME = "official_public_summary_contact_sheet_preview.png"
    SUMMARY_ROW_TARGET_WIDTH = 1080
    SUMMARY_LABEL_WIDTH = 220
    SUMMARY_PADDING = 28
    SUMMARY_GAP = 16
    SHEET_PREVIEW_MAX_WIDTH = 1200
    SHEET_IMAGE_MARKDOWN: tuple[tuple[str, str, str], ...] = (
        ("input_sheet.png", "input_sheet_preview.png", "Input contact sheet"),
        ("official_sheet.png", "official_sheet_preview.png", "Official reference contact sheet"),
        ("mlx_sheet.png", "mlx_sheet_preview.png", "mlx-gen output contact sheet"),
    )
    DISPOSITIONED_ONLY_CASE_IDS = frozenset({"r2v_case2", "v2v_case3"})
    # Pinned accepted-case sources for the committed 2026-08-11 bundle. Heuristic
    # discovery can pick newer ablations or rejected release candidates.
    CANONICAL_CASE_SOURCES: dict[str, str] = {
        "t2i": "validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_t2i/t2i",
        "i2i": "validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_i2i/i2i",
        "t2v": "validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_t2v/t2v",
        "v2v_case1": "validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_segmented_v2v_case1_launchd_round3/v2v_case1",
        "mv2v": "validation_outputs/bernini_r_1_3b_2026_08_11/head_canvasfix_mv2v_full_v2/mv2v",
        "r2v": "validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_segmented_r2v_40step_launchd_round7/r2v",
        "rv2v_case1": "validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1/rv2v_case1",
        "ads2v": "validation_outputs/bernini_r_1_3b_2026_08_11/head_ads2v_mid480_61f/ads2v",
    }
    # Oracle-dispositioned rows and tuned recovery recipes that sit outside the public
    # 1.3B model-card manifest but are documented in the parity matrix.
    DISPOSITIONED_VARIANTS: tuple[tuple[str, str], ...] = (
        ("r2v_case2_official", "validation_outputs/bernini_r_1_3b_2026_08_11/head_r2v_case2_full_v2/r2v_case2"),
        ("r2v_case2_tuned", "validation_outputs/bernini_r_1_3b_2026_08_11/exp_r2v_case2_full_refg6_s43/r2v_case2"),
        ("v2v_case3_official", "validation_outputs/bernini_r_1_3b_2026_08_11/head_v2v_case3_full_v2/v2v_case3"),
        ("v2v_case3_mv2vprefix", "validation_outputs/bernini_r_1_3b_2026_08_11/exp_v2v_case3_mv2vprefix"),
    )

    @staticmethod
    def main() -> None:
        args = BerniniOfficialParityBundle._parse_args()
        workspace_root = args.workspace_root.resolve()
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_case_ids = tuple(args.case_id) if args.case_id else BerniniOfficialParityBundle.CASE_IDS
        if args.include_dispositioned:
            selected_case_ids = tuple(
                case_id
                for case_id in selected_case_ids
                if case_id not in BerniniOfficialParityBundle.DISPOSITIONED_ONLY_CASE_IDS
            )
        discovered = BerniniOfficialParityBundle._discover_cases(
            workspace_root=workspace_root,
            case_ids=selected_case_ids,
            search_roots=[workspace_root / root for root in args.search_root],
            require_reviewed=bool(args.require_reviewed),
        )

        manifest: dict[str, Any] = {
            "workspace_root": str(workspace_root),
            "search_roots": [str((workspace_root / root).resolve()) for root in args.search_root],
            "cases": [],
        }
        copied_cases: list[dict[str, Any]] = []
        for case_id in selected_case_ids:
            case_source_dir = discovered.get(case_id)
            if case_source_dir is None:
                continue
            proof_path = case_source_dir / "proof.json"
            proof = json.loads(proof_path.read_text())
            case_target_dir = output_dir / case_id
            case_target_dir.mkdir(parents=True, exist_ok=True)
            for filename in BerniniOfficialParityBundle.COPY_FILES:
                source_path = case_source_dir / filename
                if source_path.exists():
                    shutil.copy2(source_path, case_target_dir / filename)
            bundled_artifacts = BerniniOfficialParityBundle._copy_case_artifacts(
                workspace_root=workspace_root,
                case_source_dir=case_source_dir,
                case_target_dir=case_target_dir,
                proof=proof,
            )
            proof_copy = dict(proof)
            if bundled_artifacts:
                proof_copy["bundled_artifacts"] = bundled_artifacts
            BerniniOfficialParityBundle._finalize_case_bundle(
                case_target_dir=case_target_dir,
                proof=proof_copy,
                bundled_artifacts=bundled_artifacts,
                case_source_dir=case_source_dir,
            )
            (case_target_dir / "proof.json").write_text(json.dumps(proof_copy, indent=2) + "\n")
            copied_case = {
                "id": case_id,
                "title": proof.get("title", case_id),
                "task_type": proof.get("task_type"),
                "source_dir": str(case_source_dir),
                "bundle_dir": str(case_target_dir),
                "output": proof.get("output"),
                "official_output": proof.get("official_output"),
                "observed_result": proof.get("observed_result"),
                "bundle_output": bundled_artifacts.get("output"),
                "bundle_metadata": bundled_artifacts.get("metadata"),
            }
            manifest["cases"].append(copied_case)
            copied_cases.append(copied_case)

        if args.include_dispositioned:
            for bundle_id, source_rel in BerniniOfficialParityBundle.DISPOSITIONED_VARIANTS:
                copied_case = BerniniOfficialParityBundle._copy_dispositioned_variant(
                    workspace_root=workspace_root,
                    output_dir=output_dir,
                    bundle_id=bundle_id,
                    source_rel=source_rel,
                )
                if copied_case is None:
                    continue
                manifest["cases"].append(copied_case)
                copied_cases.append(copied_case)

        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report = BerniniOfficialParityBundle._build_report(
            selected_case_ids=selected_case_ids,
            copied_cases=copied_cases,
        )
        (output_dir / "bernini_proof_report.json").write_text(json.dumps(report, indent=2) + "\n")
        BerniniOfficialParityBundle._write_readme(output_dir=output_dir, copied_cases=copied_cases)
        BerniniOfficialParityBundle._write_summary_contact_sheet(
            output_dir=output_dir,
            copied_cases=copied_cases,
        )

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--workspace-root",
            type=Path,
            default=Path("."),
        )
        parser.add_argument("--case-id", action="append")
        parser.add_argument(
            "--search-root",
            action="append",
            default=list(BerniniOfficialParityBundle.DEFAULT_SEARCH_ROOTS),
        )
        parser.add_argument("--require-reviewed", action="store_true", default=True)
        parser.add_argument("--allow-unreviewed", action="store_false", dest="require_reviewed")
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("docs/assets/validation/bernini-r-1.3b-2026-08-11"),
        )
        parser.add_argument(
            "--include-dispositioned",
            action="store_true",
            help="Also export oracle-dispositioned rows and tuned recovery recipes.",
        )
        return parser.parse_args()

    @classmethod
    def _discover_cases(
        cls,
        *,
        workspace_root: Path,
        case_ids: tuple[str, ...],
        search_roots: list[Path],
        require_reviewed: bool,
    ) -> dict[str, Path]:
        requested = set(case_ids)
        candidates: dict[str, list[tuple[tuple[int, float], Path]]] = {case_id: [] for case_id in case_ids}
        for root in search_roots:
            if not root.exists():
                continue
            for proof_path in root.rglob("proof.json"):
                try:
                    proof = json.loads(proof_path.read_text())
                except Exception:
                    continue
                case_id = str(proof.get("id") or "")
                if case_id not in requested:
                    continue
                case_dir = proof_path.parent
                if not cls._is_complete_case_dir(case_dir):
                    continue
                if require_reviewed and not cls._is_accepted_for_bundle(proof):
                    continue
                native_score = 1 if cls._is_native_case_dir(case_dir, proof) else 0
                score = (native_score, proof_path.stat().st_mtime)
                candidates[case_id].append((score, case_dir))
        selected: dict[str, Path] = {}
        for case_id in case_ids:
            canonical_dir = cls._resolve_canonical_case_dir(workspace_root=workspace_root, case_id=case_id)
            if canonical_dir is not None:
                selected[case_id] = canonical_dir
                continue
            options = candidates.get(case_id, [])
            if not options:
                continue
            options.sort(key=lambda item: item[0], reverse=True)
            selected[case_id] = options[0][1]
        return selected

    @classmethod
    def _resolve_canonical_case_dir(cls, *, workspace_root: Path, case_id: str) -> Path | None:
        source_rel = cls.CANONICAL_CASE_SOURCES.get(case_id)
        if source_rel is None:
            return None
        case_dir = (workspace_root / source_rel).resolve()
        if not cls._is_complete_case_dir(case_dir):
            return None
        return case_dir

    @classmethod
    def _is_complete_case_dir(cls, case_dir: Path) -> bool:
        return all((case_dir / name).exists() for name in cls.REQUIRED_PROOF_FILES)

    @staticmethod
    def _observed_result_text(proof: dict[str, Any]) -> str:
        observed = proof.get("observed_result")
        if isinstance(observed, list):
            return " ".join(str(item) for item in observed)
        if isinstance(observed, str):
            return observed
        return ""

    @classmethod
    def _is_accepted_for_bundle(cls, proof: dict[str, Any]) -> bool:
        text = cls._observed_result_text(proof).strip().lower()
        if not text:
            return False
        if "not yet manually reviewed" in text:
            return False
        if "not acceptable" in text:
            return False
        return True

    @staticmethod
    def _is_reviewed(proof: dict[str, Any]) -> bool:
        return BerniniOfficialParityBundle._is_accepted_for_bundle(proof)

    @staticmethod
    def _is_native_case_dir(case_dir: Path, proof: dict[str, Any]) -> bool:
        output_value = proof.get("output")
        metadata_value = proof.get("metadata")
        if output_value is None or metadata_value is None:
            return False
        try:
            output_path = Path(str(output_value)).resolve()
            metadata_path = Path(str(metadata_value)).resolve()
        except Exception:
            return False
        try:
            resolved_case_dir = case_dir.resolve()
        except Exception:
            return False
        return output_path.parent == resolved_case_dir and metadata_path.parent == resolved_case_dir

    @staticmethod
    def _copy_case_artifacts(
        *,
        workspace_root: Path,
        case_source_dir: Path,
        case_target_dir: Path,
        proof: dict[str, Any],
    ) -> dict[str, str]:
        bundled: dict[str, str] = {}
        output_path = BerniniOfficialParityBundle._resolve_existing_path(
            value=proof.get("output"),
            workspace_root=workspace_root,
            case_source_dir=case_source_dir,
        )
        if output_path is not None:
            target_output_path = case_target_dir / output_path.name
            shutil.copy2(output_path, target_output_path)
            bundled["output"] = target_output_path.name
        metadata_path = BerniniOfficialParityBundle._resolve_metadata_path(
            output_path=output_path,
            metadata_value=proof.get("metadata"),
            workspace_root=workspace_root,
            case_source_dir=case_source_dir,
        )
        if metadata_path is not None:
            target_metadata_path = case_target_dir / metadata_path.name
            shutil.copy2(metadata_path, target_metadata_path)
            bundled["metadata"] = target_metadata_path.name
        return bundled

    @staticmethod
    def _resolve_metadata_path(
        *,
        output_path: Path | None,
        metadata_value: Any,
        workspace_root: Path,
        case_source_dir: Path,
    ) -> Path | None:
        metadata_path = BerniniOfficialParityBundle._resolve_existing_path(
            value=metadata_value,
            workspace_root=workspace_root,
            case_source_dir=case_source_dir,
        )
        if metadata_path is not None:
            return metadata_path
        if output_path is None:
            return None
        sibling_metadata = output_path.with_suffix(".metadata.json")
        if sibling_metadata.exists():
            return sibling_metadata
        return None

    @staticmethod
    def _resolve_existing_path(
        *,
        value: Any,
        workspace_root: Path,
        case_source_dir: Path,
    ) -> Path | None:
        if value is None:
            return None
        try:
            raw_path = Path(str(value))
        except Exception:
            return None
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(workspace_root / raw_path)
            candidates.append(case_source_dir / raw_path)
            candidates.append(case_source_dir / raw_path.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _copy_dispositioned_variant(
        cls,
        *,
        workspace_root: Path,
        output_dir: Path,
        bundle_id: str,
        source_rel: str,
    ) -> dict[str, Any] | None:
        source_dir = (workspace_root / source_rel).resolve()
        if not source_dir.exists():
            return None
        case_target_dir = output_dir / bundle_id
        case_target_dir.mkdir(parents=True, exist_ok=True)
        proof_path = source_dir / "proof.json"
        if proof_path.exists() and cls._is_complete_case_dir(source_dir):
            proof = json.loads(proof_path.read_text())
            for filename in cls.COPY_FILES:
                source_path = source_dir / filename
                if source_path.exists():
                    shutil.copy2(source_path, case_target_dir / filename)
            bundled_artifacts = cls._copy_case_artifacts(
                workspace_root=workspace_root,
                case_source_dir=source_dir,
                case_target_dir=case_target_dir,
                proof=proof,
            )
            proof_copy = dict(proof)
            proof_copy["bundle_id"] = bundle_id
            if bundled_artifacts:
                proof_copy["bundled_artifacts"] = bundled_artifacts
            cls._finalize_case_bundle(
                case_target_dir=case_target_dir,
                proof=proof_copy,
                bundled_artifacts=bundled_artifacts,
                case_source_dir=source_dir,
            )
            (case_target_dir / "proof.json").write_text(json.dumps(proof_copy, indent=2) + "\n")
            return {
                "id": bundle_id,
                "title": proof.get("title", bundle_id),
                "task_type": proof.get("task_type"),
                "source_dir": str(source_dir),
                "bundle_dir": str(case_target_dir),
                "output": proof.get("output"),
                "official_output": proof.get("official_output"),
                "observed_result": proof.get("observed_result"),
                "bundle_output": bundled_artifacts.get("output"),
                "bundle_metadata": bundled_artifacts.get("metadata"),
            }
        return cls._copy_sparse_mv2v_recovery_case(
            workspace_root=workspace_root,
            source_dir=source_dir,
            case_target_dir=case_target_dir,
            bundle_id=bundle_id,
        )

    @staticmethod
    def _copy_sparse_mv2v_recovery_case(
        *,
        workspace_root: Path,
        source_dir: Path,
        case_target_dir: Path,
        bundle_id: str,
    ) -> dict[str, Any] | None:
        mlx_sheet = source_dir / "mlx_sheet.png"
        output_candidates = sorted(source_dir.glob("*.mp4"))
        metadata_candidates = sorted(source_dir.glob("*.metadata.json"))
        if not mlx_sheet.exists() or not output_candidates:
            return None
        shutil.copy2(mlx_sheet, case_target_dir / "mlx_sheet.png")
        output_path = output_candidates[0]
        shutil.copy2(output_path, case_target_dir / output_path.name)
        metadata_path = metadata_candidates[0] if metadata_candidates else None
        if metadata_path is not None:
            shutil.copy2(metadata_path, case_target_dir / metadata_path.name)
            metadata = json.loads(metadata_path.read_text())
        else:
            metadata = {}
        prompt = str(metadata.get("prompt") or "")
        proof = {
            "id": bundle_id,
            "title": "V2V robot to robotic dog (mv2v-prefix recovery)",
            "task_type": "mv2v",
            "prompt": prompt,
            "bundle_id": bundle_id,
            "output": str(output_path.relative_to(workspace_root)),
            "bundled_artifacts": {
                "output": output_path.name,
                "metadata": metadata_path.name if metadata_path is not None else None,
            },
        }
        (case_target_dir / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
        readme_lines = [
            "# V2V robot to robotic dog (mv2v-prefix recovery)",
            "",
            "This row uses the official `mv2v` task prefix with text guidance `5.0` to recover the",
            "quadruped robotic dog outcome that the official `v2v` recipe fails to produce at 1.3B.",
            "",
            "## Request",
            "",
            f"- task: `mv2v`",
            f"- prompt: {prompt}",
            "",
            "## mlx-gen output",
            "",
            "![mlx-gen](mlx_sheet.png)",
            "",
            "## Artifacts",
            "",
            f"- output: `{output_path.name}`",
        ]
        if metadata_path is not None:
            readme_lines.append(f"- metadata: `{metadata_path.name}`")
        (case_target_dir / "README.md").write_text("\n".join(readme_lines) + "\n")
        bundled_artifacts = dict(proof["bundled_artifacts"])
        BerniniOfficialParityBundle._finalize_case_bundle(
            case_target_dir=case_target_dir,
            proof=proof,
            bundled_artifacts=bundled_artifacts,
            case_source_dir=source_dir,
        )
        return {
            "id": bundle_id,
            "title": proof["title"],
            "task_type": proof["task_type"],
            "source_dir": str(source_dir),
            "bundle_dir": str(case_target_dir),
            "output": proof.get("output"),
            "official_output": None,
            "observed_result": None,
            "bundle_output": output_path.name,
            "bundle_metadata": metadata_path.name if metadata_path is not None else None,
        }

    @classmethod
    def _finalize_case_bundle(
        cls,
        *,
        case_target_dir: Path,
        proof: dict[str, Any],
        bundled_artifacts: dict[str, str],
        case_source_dir: Path | None,
    ) -> None:
        cls._create_case_sheet_previews(case_target_dir)
        cls._normalize_case_readme(
            case_target_dir=case_target_dir,
            proof=proof,
            bundled_artifacts=bundled_artifacts,
            case_source_dir=case_source_dir,
        )

    @classmethod
    def _create_case_sheet_previews(cls, case_target_dir: Path) -> None:
        for stem in ("input_sheet", "official_sheet", "mlx_sheet"):
            source_path = case_target_dir / f"{stem}.png"
            if not source_path.exists():
                continue
            target_path = case_target_dir / f"{stem}_preview.png"
            cls._create_sheet_preview(source_path=source_path, target_path=target_path)

    @staticmethod
    def _create_sheet_preview(*, source_path: Path, target_path: Path) -> None:
        max_width = BerniniOfficialParityBundle.SHEET_PREVIEW_MAX_WIDTH
        with Image.open(source_path) as image:
            rgb_image = image.convert("RGB")
            if rgb_image.width <= max_width:
                preview = rgb_image
            else:
                height = max(1, round(rgb_image.height * max_width / rgb_image.width))
                preview = rgb_image.resize((max_width, height), Image.Resampling.LANCZOS)
            preview.save(target_path, format="PNG", optimize=True)

    @classmethod
    def _normalize_case_readme(
        cls,
        *,
        case_target_dir: Path,
        proof: dict[str, Any],
        bundled_artifacts: dict[str, str],
        case_source_dir: Path | None,
    ) -> None:
        readme_path = case_target_dir / "README.md"
        if not readme_path.exists():
            return
        text = readme_path.read_text()
        for full_name, preview_name, alt in cls.SHEET_IMAGE_MARKDOWN:
            pattern = re.compile(rf"!\[[^\]]*\]\({re.escape(full_name)}\)")
            replacement = (
                f'<img src="{preview_name}" alt="{alt}" width="100%" />\n\n'
                f"Full resolution: [{full_name}]({full_name})"
            )
            text = pattern.sub(replacement, text)
        text = cls._rewrite_artifacts_section(
            text=text,
            proof=proof,
            bundled_artifacts=bundled_artifacts,
            case_source_dir=case_source_dir,
            case_target_dir=case_target_dir,
        )
        readme_path.write_text(text)

    @staticmethod
    def _rewrite_artifacts_section(
        *,
        text: str,
        proof: dict[str, Any],
        bundled_artifacts: dict[str, str],
        case_source_dir: Path | None,
        case_target_dir: Path,
    ) -> str:
        marker = "## Artifacts"
        if marker not in text:
            return text
        prefix, _ = text.split(marker, 1)
        lines = [marker, ""]
        if bundled_artifacts.get("output"):
            lines.append(f"- output: `{bundled_artifacts['output']}`")
        if bundled_artifacts.get("metadata"):
            lines.append(f"- metadata: `{bundled_artifacts['metadata']}`")
        for sheet_name in ("input_sheet.png", "official_sheet.png", "mlx_sheet.png"):
            if not (case_target_dir / sheet_name).exists():
                continue
            label = sheet_name.removesuffix(".png").replace("_", " ")
            lines.append(f"- {label}: [{sheet_name}]({sheet_name})")
        if case_source_dir is not None and (case_source_dir / "initial_noise.npy").exists():
            lines.append(
                "- initial noise: `initial_noise.npy` in the source validation run (not bundled)"
            )
        official_output = proof.get("official_output")
        if official_output:
            lines.append(f"- official output: `{official_output}`")
        if case_source_dir is not None:
            lines.append(f"- source validation run: `{case_source_dir}` (local harness only)")
        return prefix.rstrip() + "\n\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _write_readme(*, output_dir: Path, copied_cases: list[dict[str, Any]]) -> None:
        lines = [
            "# Bernini-R 1.3B official public parity bundle",
            "",
            "This bundle assembles the current official public-case proof rows from the local Bernini 1.3B validation runs.",
            "Each case directory contains the human-readable case README, prompt, expected result, actual result,",
            "reproduce command, GitHub-friendly preview contact sheets plus full-resolution sheets, and the generated",
            "mlx artifact (video or image) with metadata when available.",
            "",
            "Dispositioned rows (`r2v_case2_*`, `v2v_case3_*`) document oracle-proven 1.3B limits and the",
            "tuned recovery recipes described in the parity matrix.",
            "",
            "## Overview",
            "",
            f'<img src="{BerniniOfficialParityBundle.SUMMARY_PREVIEW_NAME}" '
            'alt="Official public parity mlx-gen overview contact sheet" width="100%" />',
            "",
            f"Full resolution: [{BerniniOfficialParityBundle.SUMMARY_SHEET_NAME}]"
            f"({BerniniOfficialParityBundle.SUMMARY_SHEET_NAME})",
            "",
            "Each row shows input, official reference, and mlx-gen contact sheets from the pinned accepted-case",
            "validation runs in this bundle (not the older 2026-08-04 schema-v3 FAIL profile).",
            "",
            "## Included rows",
            "",
            "| Row | Task | Proof |",
            "| --- | --- | --- |",
        ]
        for case in copied_cases:
            case_id = str(case["id"])
            title = str(case["title"])
            lines.append(f"| `{case_id}` | {title} | [{case_id}/README.md]({case_id}/README.md) |")
        lines.extend(
            [
                "",
                "## Manifest",
                "",
                "- [manifest.json](manifest.json)",
                "- [bernini_proof_report.json](bernini_proof_report.json)",
                "",
            ]
        )
        (output_dir / "README.md").write_text("\n".join(lines) + "\n")

    @classmethod
    def _write_summary_contact_sheet(
        cls,
        *,
        output_dir: Path,
        copied_cases: list[dict[str, Any]],
    ) -> None:
        panel_names = (
            ("input", "input_sheet_preview.png"),
            ("official", "official_sheet_preview.png"),
            ("mlx", "mlx_sheet_preview.png"),
        )
        rows: list[tuple[str, str, list[tuple[str, Image.Image]]]] = []
        for case in copied_cases:
            case_id = str(case["id"])
            title = str(case.get("title") or case_id)
            case_dir = output_dir / case_id
            panels: list[tuple[str, Image.Image]] = []
            for panel_label, filename in panel_names:
                sheet_path = case_dir / filename
                if not sheet_path.exists():
                    continue
                with Image.open(sheet_path) as image:
                    panels.append((panel_label, image.convert("RGB")))
            if not panels:
                continue
            rows.append((case_id, title, panels))
        if not rows:
            return

        label_font = cls._summary_font(size=22, bold=True)
        title_font = cls._summary_font(size=34, bold=True)
        subtitle_font = cls._summary_font(size=20, bold=False)
        header_title = "Bernini-R 1.3B official public parity"
        header_subtitle = (
            "Pinned accepted-case validation runs · input / official / mlx-gen contact sheets"
        )
        content_width = cls.SUMMARY_LABEL_WIDTH + cls.SUMMARY_GAP + cls.SUMMARY_ROW_TARGET_WIDTH
        sheet_width = content_width + (cls.SUMMARY_PADDING * 2)

        row_heights: list[int] = []
        for _case_id, _title, panels in rows:
            scaled = cls._scale_summary_panels(panels=panels, target_width=cls.SUMMARY_ROW_TARGET_WIDTH)
            row_heights.append(max(image.height for _label, image in scaled) + 34)

        header_height = 110
        sheet_height = (
            cls.SUMMARY_PADDING
            + header_height
            + cls.SUMMARY_GAP
            + sum(row_heights)
            + ((len(rows) - 1) * cls.SUMMARY_GAP)
            + cls.SUMMARY_PADDING
        )
        canvas = Image.new("RGB", (sheet_width, sheet_height), "#ffffff")
        draw = ImageDraw.Draw(canvas)
        y = cls.SUMMARY_PADDING
        draw.text((cls.SUMMARY_PADDING, y), header_title, fill="#111827", font=title_font)
        y += 42
        draw.text((cls.SUMMARY_PADDING, y), header_subtitle, fill="#4b5563", font=subtitle_font)
        y += header_height - 42

        for row_index, (case_id, title, panels) in enumerate(rows):
            scaled = cls._scale_summary_panels(panels=panels, target_width=cls.SUMMARY_ROW_TARGET_WIDTH)
            row_height = max(image.height for _label, image in scaled)
            label = f"{case_id}\n{title}"
            draw.multiline_text(
                (cls.SUMMARY_PADDING, y + 4),
                label,
                fill="#111827",
                font=label_font,
                spacing=4,
            )
            panel_x = cls.SUMMARY_PADDING + cls.SUMMARY_LABEL_WIDTH + cls.SUMMARY_GAP
            cursor_x = panel_x
            for panel_label, image in scaled:
                canvas.paste(image, (cursor_x, y))
                draw.text((cursor_x, y + row_height + 6), panel_label, fill="#6b7280", font=subtitle_font)
                cursor_x += image.width + cls.SUMMARY_GAP
            y += row_height + 34
            if row_index < len(rows) - 1:
                y += cls.SUMMARY_GAP

        summary_path = output_dir / cls.SUMMARY_SHEET_NAME
        preview_path = output_dir / cls.SUMMARY_PREVIEW_NAME
        canvas.save(summary_path, format="PNG", optimize=True)
        cls._create_sheet_preview(source_path=summary_path, target_path=preview_path)

    @classmethod
    def _scale_summary_panels(
        cls,
        *,
        panels: list[tuple[str, Image.Image]],
        target_width: int,
    ) -> list[tuple[str, Image.Image]]:
        gap_total = cls.SUMMARY_GAP * max(0, len(panels) - 1)
        panel_width = max(1, (target_width - gap_total) // max(1, len(panels)))
        scaled: list[tuple[str, Image.Image]] = []
        for panel_label, image in panels:
            if image.width <= panel_width:
                scaled_image = image
            else:
                height = max(1, round(image.height * panel_width / image.width))
                scaled_image = image.resize((panel_width, height), Image.Resampling.LANCZOS)
            scaled.append((panel_label, scaled_image))
        return scaled

    @staticmethod
    def _summary_font(*, size: int, bold: bool) -> ImageFont.ImageFont:
        names = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        for name in names:
            path = Path(name)
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    @staticmethod
    def _build_report(*, selected_case_ids: tuple[str, ...], copied_cases: list[dict[str, Any]]) -> dict[str, Any]:
        included_case_ids = [str(case["id"]) for case in copied_cases]
        missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in included_case_ids]
        reviewed_case_ids = [
            str(case["id"])
            for case in copied_cases
            if BerniniOfficialParityBundle._is_accepted_for_bundle(
                {"observed_result": case.get("observed_result")}
            )
        ]
        return {
            "kind": "bernini_r_1_3b_official_public_parity_bundle",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "required_case_ids": list(selected_case_ids),
            "included_case_ids": included_case_ids,
            "missing_case_ids": missing_case_ids,
            "reviewed_case_ids": reviewed_case_ids,
            "passed": len(missing_case_ids) == 0,
            "cases": copied_cases,
        }


if __name__ == "__main__":
    BerniniOfficialParityBundle.main()
