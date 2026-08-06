import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np


class BerniniAPGParity:
    OFFICIAL_SOURCE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
    OFFICIAL_NORMALIZE_DIFF_SHA256 = "8cde4c7094bc9b2882e969b175b366ab3a5146f2e02edf5a26c861c255210754"
    REDUCTION_AXES = (1, 3, 4)
    DEFAULT_SEED = 8104

    @staticmethod
    def main() -> None:
        args = BerniniAPGParity._parse_args()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = output_dir / "inputs.npz"
        manifest_path = output_dir / "inputs.json"
        if not inputs_path.exists() or not manifest_path.exists():
            BerniniAPGParity._write_inputs(inputs_path, manifest_path=manifest_path, seed=args.seed)

        if args.stage in {"torch", "all"}:
            BerniniAPGParity._run_torch(
                inputs_path=inputs_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
                reference_root=args.reference_root.resolve(),
            )
        if args.stage in {"mlx", "all"}:
            BerniniAPGParity._run_mlx(
                inputs_path=inputs_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
            )
        if args.stage in {"compare", "all"}:
            BerniniAPGParity._compare(output_dir=output_dir, manifest_path=manifest_path)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Export and compare the pinned official Torch and MLX Bernini APG calculations."
        )
        parser.add_argument("--stage", choices=("torch", "mlx", "compare", "all"), required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--reference-root", type=Path, default=Path("/tmp/Bernini"))
        parser.add_argument("--seed", type=int, default=BerniniAPGParity.DEFAULT_SEED)
        return parser.parse_args()

    @staticmethod
    def _write_inputs(path: Path, *, manifest_path: Path, seed: int) -> None:
        generator = np.random.default_rng(seed)
        shape = (1, 3, 2, 2, 2)
        normal_diff = generator.standard_normal(shape, dtype=np.float32)
        normal_base = generator.standard_normal(shape, dtype=np.float32)
        collinear_base = generator.standard_normal(shape, dtype=np.float32)
        orthogonal_diff = np.zeros(shape, dtype=np.float32)
        orthogonal_base = np.zeros(shape, dtype=np.float32)
        orthogonal_diff[:, 1] = 1.0
        orthogonal_base[:, 0] = 1.0
        momentum_first = generator.standard_normal(shape, dtype=np.float32)
        momentum_second = generator.standard_normal(shape, dtype=np.float32)
        realistic_shape = (1, 16, 2, 16, 24)
        arrays = {
            "zero__base": np.zeros(shape, dtype=np.float32),
            "zero__diff_0": np.zeros(shape, dtype=np.float32),
            "tiny_base_norm__base": normal_base * np.float32(1e-14),
            "tiny_base_norm__diff_0": normal_diff,
            "normal__base": normal_base,
            "normal__diff_0": normal_diff,
            "large_base_norm__base": normal_base * np.float32(1e20),
            "large_base_norm__diff_0": normal_diff,
            "collinear__base": collinear_base,
            "collinear__diff_0": collinear_base * np.float32(2.5),
            "orthogonal__base": orthogonal_base,
            "orthogonal__diff_0": orthogonal_diff,
            "finite_clip__base": normal_base,
            "finite_clip__diff_0": normal_diff * np.float32(1e10),
            "overflow_clip__base": normal_base,
            "overflow_clip__diff_0": normal_diff * np.float32(1e20),
            "momentum_step_2__base": normal_base,
            "momentum_step_2__diff_0": momentum_first,
            "momentum_step_2__diff_1": momentum_second,
            "realistic_reduction__base": generator.standard_normal(realistic_shape, dtype=np.float32),
            "realistic_reduction__diff_0": generator.standard_normal(realistic_shape, dtype=np.float32),
        }
        cases = {
            "zero": BerniniAPGParity._case(eta=0.5, norm_threshold=50.0, momentum=0.0, updates=1),
            "tiny_base_norm": BerniniAPGParity._case(eta=0.37, norm_threshold=50.0, momentum=0.0, updates=1),
            "normal": BerniniAPGParity._case(eta=0.37, norm_threshold=50.0, momentum=0.0, updates=1),
            "large_base_norm": BerniniAPGParity._case(eta=0.37, norm_threshold=50.0, momentum=0.0, updates=1),
            "collinear": BerniniAPGParity._case(eta=0.37, norm_threshold=0.0, momentum=0.0, updates=1),
            "orthogonal": BerniniAPGParity._case(eta=0.37, norm_threshold=0.0, momentum=0.0, updates=1),
            "finite_clip": BerniniAPGParity._case(eta=0.37, norm_threshold=1.25, momentum=0.0, updates=1),
            "overflow_clip": BerniniAPGParity._case(
                eta=0.37,
                norm_threshold=1.25,
                momentum=0.0,
                updates=1,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            ),
            "momentum_step_2": BerniniAPGParity._case(eta=0.37, norm_threshold=50.0, momentum=-0.5, updates=2),
            "realistic_reduction": BerniniAPGParity._case(
                eta=0.37,
                norm_threshold=50.0,
                momentum=0.0,
                updates=1,
                absolute_tolerance=2e-5,
                relative_tolerance=2e-5,
            ),
        }
        np.savez(path, **arrays)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "bernini_official_apg_inputs",
                    "seed": seed,
                    "dtype": "float32",
                    "reduction_axes": list(BerniniAPGParity.REDUCTION_AXES),
                    "official_source_revision": BerniniAPGParity.OFFICIAL_SOURCE_REVISION,
                    "official_normalize_diff_sha256": BerniniAPGParity.OFFICIAL_NORMALIZE_DIFF_SHA256,
                    "cases": cases,
                    "input_sha256": BerniniAPGParity._archive_hash(arrays),
                },
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _case(
        *,
        eta: float,
        norm_threshold: float,
        momentum: float,
        updates: int,
        absolute_tolerance: float = 5e-7,
        relative_tolerance: float = 2e-7,
    ) -> dict:
        return {
            "eta": eta,
            "norm_threshold": norm_threshold,
            "momentum": momentum,
            "updates": updates,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
        }

    @staticmethod
    def _run_torch(*, inputs_path: Path, manifest_path: Path, output_dir: Path, reference_root: Path) -> None:
        import torch
        import torch.nn.functional as torch_functional

        source_revision, function_hash = BerniniAPGParity._validate_reference_source(reference_root)
        inputs = np.load(inputs_path)
        manifest = json.loads(manifest_path.read_text())
        outputs = {}
        for case_name, case in manifest["cases"].items():
            base = torch.from_numpy(inputs[f"{case_name}__base"]).to(torch.float32)
            running_average: torch.Tensor | int = 0
            for update_index in range(case["updates"]):
                diff = torch.from_numpy(inputs[f"{case_name}__diff_{update_index}"]).to(torch.float32)
                running_average = diff + case["momentum"] * running_average
                outputs[f"{case_name}__state_{update_index}"] = running_average.numpy(force=True)
                clipped = running_average
                if case["norm_threshold"] > 0:
                    ones = torch.ones_like(clipped)
                    diff_norm = clipped.norm(p=2, dim=[-1, -2, -4], keepdim=True)
                    scale_factor = torch.minimum(ones, case["norm_threshold"] / diff_norm)
                    clipped = clipped * scale_factor
                v0, v1 = clipped.double(), base.double()
                v1 = torch_functional.normalize(v1, dim=[-1, -2, -4])
                parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
                orthogonal = v0 - parallel
                output = orthogonal.to(clipped.dtype) + case["eta"] * parallel.to(clipped.dtype)
                outputs[f"{case_name}__output_{update_index}"] = output.numpy(force=True)
        np.savez(output_dir / "torch_outputs.npz", **outputs)
        BerniniAPGParity._write_stage_report(
            output_dir / "torch_report.json",
            backend="official-pytorch-formula",
            arrays=outputs,
            extra={
                "torch_version": torch.__version__,
                "reference_root": str(reference_root),
                "official_source_revision": source_revision,
                "official_normalize_diff_sha256": function_hash,
                "projection_precision": "float64",
                "clip_precision": "float32",
            },
        )

    @staticmethod
    def _run_mlx(*, inputs_path: Path, manifest_path: Path, output_dir: Path) -> None:
        import mlx.core as mx

        from mflux.models.wan.variants.wan_bernini import BerniniRenderer, _MomentumBuffer

        inputs = np.load(inputs_path)
        manifest = json.loads(manifest_path.read_text())
        outputs = {}
        for case_name, case in manifest["cases"].items():
            base = mx.array(inputs[f"{case_name}__base"], dtype=mx.float32)
            buffer = _MomentumBuffer(float(case["momentum"]))
            for update_index in range(case["updates"]):
                diff = mx.array(inputs[f"{case_name}__diff_{update_index}"], dtype=mx.float32)
                output = BerniniRenderer._normalize_diff(
                    diff,
                    base,
                    momentum_buffer=buffer,
                    eta=float(case["eta"]),
                    norm_threshold=float(case["norm_threshold"]),
                )
                mx.eval(output, buffer.running_average)
                outputs[f"{case_name}__state_{update_index}"] = np.asarray(buffer.running_average.astype(mx.float32))
                outputs[f"{case_name}__output_{update_index}"] = np.asarray(output.astype(mx.float32))
        np.savez(output_dir / "mlx_outputs.npz", **outputs)
        BerniniAPGParity._write_stage_report(
            output_dir / "mlx_report.json",
            backend="mlx",
            arrays=outputs,
            extra={
                "mlx_version": getattr(mx, "__version__", "unknown"),
                "projection_precision": "stable-float32",
                "clip_precision": "float32",
            },
        )

    @staticmethod
    def _compare(*, output_dir: Path, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        torch_outputs = np.load(output_dir / "torch_outputs.npz")
        mlx_outputs = np.load(output_dir / "mlx_outputs.npz")
        expected_keys = sorted(torch_outputs.files)
        if expected_keys != sorted(mlx_outputs.files):
            raise ValueError(f"APG output keys differ: Torch {expected_keys!r}, MLX {sorted(mlx_outputs.files)!r}.")
        comparisons = {}
        passed = True
        for key in expected_keys:
            case_name = key.split("__", maxsplit=1)[0]
            case = manifest["cases"][case_name]
            reference = torch_outputs[key].astype(np.float64)
            actual = mlx_outputs[key].astype(np.float64)
            if reference.shape != actual.shape:
                raise ValueError(f"APG shape mismatch for {key}: Torch {reference.shape}, MLX {actual.shape}.")
            delta = actual - reference
            reference_norm = float(np.linalg.norm(reference.reshape(-1)))
            delta_norm = float(np.linalg.norm(delta.reshape(-1)))
            relative_l2 = 0.0 if reference_norm == 0.0 and delta_norm == 0.0 else delta_norm / reference_norm
            array_passed = bool(
                np.allclose(
                    actual,
                    reference,
                    atol=float(case["absolute_tolerance"]),
                    rtol=float(case["relative_tolerance"]),
                )
            )
            comparisons[key] = {
                "shape": list(reference.shape),
                "max_absolute_error": float(np.max(np.abs(delta))),
                "mean_absolute_error": float(np.mean(np.abs(delta))),
                "relative_l2": relative_l2,
                "absolute_tolerance": float(case["absolute_tolerance"]),
                "relative_tolerance": float(case["relative_tolerance"]),
                "torch_sha256": BerniniAPGParity._array_hash(torch_outputs[key]),
                "mlx_sha256": BerniniAPGParity._array_hash(mlx_outputs[key]),
                "passed": array_passed,
            }
            passed = passed and array_passed
        report = {
            "schema_version": 1,
            "kind": "bernini_official_apg_parity",
            "official_source_revision": BerniniAPGParity.OFFICIAL_SOURCE_REVISION,
            "official_normalize_diff_sha256": BerniniAPGParity.OFFICIAL_NORMALIZE_DIFF_SHA256,
            "comparisons": comparisons,
            "passed": passed,
        }
        (output_dir / "parity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)

    @staticmethod
    def _validate_reference_source(reference_root: Path) -> tuple[str, str]:
        source_path = reference_root / "bernini" / "models" / "wan_diffusion.py"
        if not source_path.is_file():
            raise FileNotFoundError(f"Official Bernini source not found below {reference_root}.")
        source = source_path.read_text()
        start = source.index("def _normalize_diff(")
        end = source.index("\ndef normalized_guidance(", start)
        function_hash = hashlib.sha256(source[start:end].encode()).hexdigest()
        if function_hash != BerniniAPGParity.OFFICIAL_NORMALIZE_DIFF_SHA256:
            raise ValueError(
                "Official Bernini _normalize_diff source does not match the pinned revision: "
                f"expected {BerniniAPGParity.OFFICIAL_NORMALIZE_DIFF_SHA256}, got {function_hash}."
            )
        revision = subprocess.run(
            ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != BerniniAPGParity.OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                f"Official Bernini revision mismatch: expected {BerniniAPGParity.OFFICIAL_SOURCE_REVISION}, "
                f"got {revision}."
            )
        return revision, function_hash

    @staticmethod
    def _write_stage_report(path: Path, *, backend: str, arrays: dict[str, np.ndarray], extra: dict) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "backend": backend,
                    "platform": platform.platform(),
                    "arrays": {
                        key: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "finite": bool(np.isfinite(value).all()),
                            "min": float(value.min()),
                            "max": float(value.max()),
                            "sha256": BerniniAPGParity._array_hash(value),
                        }
                        for key, value in sorted(arrays.items())
                    },
                    **extra,
                },
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _archive_hash(arrays: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(arrays.items()):
            digest.update(key.encode())
            digest.update(BerniniAPGParity._canonical_bytes(value))
        return digest.hexdigest()

    @staticmethod
    def _array_hash(value: np.ndarray) -> str:
        return hashlib.sha256(BerniniAPGParity._canonical_bytes(value)).hexdigest()

    @staticmethod
    def _canonical_bytes(value: np.ndarray) -> bytes:
        return np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False)).tobytes()


if __name__ == "__main__":
    BerniniAPGParity.main()
