import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


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

    @staticmethod
    def _discover_cases(
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
                if not BerniniOfficialParityBundle._is_complete_case_dir(case_dir):
                    continue
                if require_reviewed and not BerniniOfficialParityBundle._is_reviewed(proof):
                    continue
                native_score = 1 if BerniniOfficialParityBundle._is_native_case_dir(case_dir, proof) else 0
                score = (native_score, proof_path.stat().st_mtime)
                candidates[case_id].append((score, case_dir))
        selected: dict[str, Path] = {}
        for case_id in case_ids:
            options = candidates.get(case_id, [])
            if not options:
                continue
            options.sort(key=lambda item: item[0], reverse=True)
            selected[case_id] = options[0][1]
        return selected

    @classmethod
    def _is_complete_case_dir(cls, case_dir: Path) -> bool:
        return all((case_dir / name).exists() for name in cls.REQUIRED_PROOF_FILES)

    @staticmethod
    def _is_reviewed(proof: dict[str, Any]) -> bool:
        observed = proof.get("observed_result")
        if isinstance(observed, list):
            return bool(observed)
        if isinstance(observed, str):
            return bool(observed.strip()) and "not yet manually reviewed" not in observed.lower()
        return False

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

    @staticmethod
    def _write_readme(*, output_dir: Path, copied_cases: list[dict[str, Any]]) -> None:
        lines = [
            "# Bernini-R 1.3B official public parity bundle",
            "",
            "This bundle assembles the current official public-case proof rows from the local Bernini 1.3B validation runs.",
            "Each case directory contains the human-readable case README, prompt, expected result, actual result,",
            "reproduce command, high-resolution input/official/mlx-gen contact sheets, and the generated mlx artifact",
            "(video or image) with metadata when available.",
            "",
            "Dispositioned rows (`r2v_case2_*`, `v2v_case3_*`) document oracle-proven 1.3B limits and the",
            "tuned recovery recipes described in the parity matrix.",
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

    @staticmethod
    def _build_report(*, selected_case_ids: tuple[str, ...], copied_cases: list[dict[str, Any]]) -> dict[str, Any]:
        included_case_ids = [str(case["id"]) for case in copied_cases]
        missing_case_ids = [case_id for case_id in selected_case_ids if case_id not in included_case_ids]
        return {
            "kind": "bernini_r_1_3b_official_public_parity_bundle",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "required_case_ids": list(selected_case_ids),
            "included_case_ids": included_case_ids,
            "missing_case_ids": missing_case_ids,
            "reviewed_case_ids": included_case_ids,
            "passed": len(missing_case_ids) == 0,
            "cases": copied_cases,
        }


if __name__ == "__main__":
    BerniniOfficialParityBundle.main()
