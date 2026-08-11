import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class BerniniSegmentQueue:
    RUNNER = Path("validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/run_official_public_case_segmented.py")
    CASES = (
        {
            "case_id": "mv2v",
            "out_dir": "validation_outputs/bernini_r_1_3b_2026_08_11/mv2v_official_release_lowram_v1",
            "segment_steps": 1,
            "low_ram": True,
            "guidance_mode": None,
            "mlx_cache_limit_gb": None,
            "clear_cache_each_transformer_block": False,
        },
        {
            "case_id": "v2v_case3",
            "out_dir": "validation_outputs/bernini_r_1_3b_2026_08_11/v2v_case3_official_release_v1",
            "segment_steps": 1,
            "low_ram": True,
            "guidance_mode": None,
            "mlx_cache_limit_gb": None,
            "clear_cache_each_transformer_block": False,
        },
        {
            "case_id": "r2v_case2",
            "out_dir": "validation_outputs/bernini_r_1_3b_2026_08_11/r2v_case2_official_release_v1",
            "segment_steps": 1,
            "low_ram": False,
            "guidance_mode": None,
            "mlx_cache_limit_gb": None,
            "clear_cache_each_transformer_block": False,
        },
        {
            "case_id": "ads2v",
            "out_dir": "validation_outputs/bernini_r_1_3b_2026_08_11/ads2v_official_release_v1",
            "segment_steps": 1,
            "low_ram": False,
            "guidance_mode": "v2v_apg",
            "mlx_cache_limit_gb": None,
            "clear_cache_each_transformer_block": False,
        },
    )

    @staticmethod
    def main() -> None:
        args = BerniniSegmentQueue._parse_args()
        workspace_root = args.workspace_root.resolve()
        if args.wait_for_proof is not None:
            BerniniSegmentQueue._wait_for_proof(workspace_root / args.wait_for_proof, poll_seconds=args.poll_seconds)
        selected = BerniniSegmentQueue._selected_cases(start_case=args.start_case, only_case=args.only_case)
        for spec in selected:
            BerniniSegmentQueue._run_case_until_proof(
                workspace_root=workspace_root,
                official_root=args.official_root,
                case_id=str(spec["case_id"]),
                out_dir=workspace_root / str(spec["out_dir"]),
                segment_steps=int(spec["segment_steps"]),
                seed=args.seed,
                steps=args.steps,
                poll_seconds=args.poll_seconds,
                low_ram=bool(spec.get("low_ram", True)),
                guidance_mode=spec.get("guidance_mode"),
                mlx_cache_limit_gb=spec.get("mlx_cache_limit_gb", args.mlx_cache_limit_gb),
                clear_cache_each_transformer_block=bool(
                    spec.get("clear_cache_each_transformer_block", args.clear_cache_each_transformer_block)
                ),
            )

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--workspace-root", type=Path, default=Path("."))
        parser.add_argument("--official-root", type=Path, default=Path("/private/tmp/bernini_official_20260810"))
        parser.add_argument("--start-case", default="mv2v")
        parser.add_argument("--only-case")
        parser.add_argument("--wait-for-proof", type=Path)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--steps", type=int, default=40)
        parser.add_argument("--poll-seconds", type=int, default=20)
        parser.add_argument("--mlx-cache-limit-gb", type=float, default=None)
        parser.add_argument("--clear-cache-each-transformer-block", action="store_true")
        return parser.parse_args()

    @staticmethod
    def _selected_cases(*, start_case: str, only_case: str | None) -> tuple[dict, ...]:
        if only_case is not None:
            for spec in BerniniSegmentQueue.CASES:
                if spec["case_id"] == only_case:
                    return (spec,)
            raise ValueError(f"Unknown only case: {only_case}")
        seen = False
        selected: list[dict] = []
        for spec in BerniniSegmentQueue.CASES:
            if spec["case_id"] == start_case:
                seen = True
            if seen:
                selected.append(spec)
        if not selected:
            raise ValueError(f"Unknown start case: {start_case}")
        return tuple(selected)

    @staticmethod
    def _run_case_until_proof(
        *,
        workspace_root: Path,
        official_root: Path,
        case_id: str,
        out_dir: Path,
        segment_steps: int,
        seed: int,
        steps: int,
        poll_seconds: int,
        low_ram: bool,
        guidance_mode: str | None,
        mlx_cache_limit_gb: float | None,
        clear_cache_each_transformer_block: bool,
    ) -> None:
        proof_path = out_dir / case_id / "proof.json"
        lock_path = out_dir / case_id / ".segment_run.lock"
        while not proof_path.exists():
            owner_pid = BerniniSegmentQueue._lock_owner_pid(lock_path)
            if owner_pid is not None and BerniniSegmentQueue._pid_alive(owner_pid):
                time.sleep(poll_seconds)
                continue
            command = [
                "uv",
                "run",
                "python",
                str(BerniniSegmentQueue.RUNNER),
                "--official-root",
                str(official_root),
                "--case",
                case_id,
                "--out-dir",
                str(out_dir.relative_to(workspace_root)),
                "--seed",
                str(seed),
                "--steps",
                str(steps),
                "--segment-steps",
                str(segment_steps),
                "--no-checkpoint-preview",
            ]
            command.append("--low-ram" if low_ram else "--full-ram")
            if guidance_mode is not None:
                command.extend(["--guidance-mode", str(guidance_mode)])
            if mlx_cache_limit_gb is not None:
                command.extend(["--mlx-cache-limit-gb", str(mlx_cache_limit_gb)])
            if clear_cache_each_transformer_block:
                command.append("--clear-cache-each-transformer-block")
            try:
                subprocess.run(command, cwd=workspace_root, check=True)
            except subprocess.CalledProcessError as error:
                print(
                    f"[bernini_segment_queue] runner failed case={case_id} "
                    f"returncode={error.returncode}; retrying after short delay",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(5)
                continue
            time.sleep(2)

    @staticmethod
    def _wait_for_proof(proof_path: Path, *, poll_seconds: int) -> None:
        while not proof_path.exists():
            time.sleep(poll_seconds)

    @staticmethod
    def _lock_owner_pid(lock_path: Path) -> int | None:
        if not lock_path.exists():
            return None
        try:
            payload = json.loads(lock_path.read_text())
        except json.JSONDecodeError:
            return None
        pid = payload.get("pid")
        return pid if isinstance(pid, int) else None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            subprocess.run(["kill", "-0", str(pid)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return False
        return True


if __name__ == "__main__":
    BerniniSegmentQueue.main()
