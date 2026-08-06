import argparse
import gc
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np


class BerniniTransformerParity:
    SOURCE_IDS = [1.0, 2.0, 0.0]
    OFFICIAL_SOURCE_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
    CHECKPOINT_REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"
    PROFILE_THRESHOLDS = {
        "float32": {"min_cosine_similarity": 0.99999, "max_relative_l2": 0.001},
        "runtime": {"min_cosine_similarity": 0.9995, "max_relative_l2": 0.025},
    }

    @staticmethod
    def main() -> None:
        args = BerniniTransformerParity._parse_args()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = output_dir / "inputs.npz"
        if not inputs_path.exists():
            BerniniTransformerParity._write_inputs(
                inputs_path,
                seed=args.seed,
                target_frames=args.target_frames,
                target_height=args.target_height,
                target_width=args.target_width,
            )

        if args.stage in {"torch", "all"}:
            BerniniTransformerParity._run_torch(
                inputs_path=inputs_path,
                output_dir=output_dir,
                checkpoint_root=args.checkpoint_root.resolve(),
                reference_root=args.reference_root.resolve(),
                precision=args.precision,
            )
        if args.stage in {"mlx", "all"}:
            BerniniTransformerParity._run_mlx(
                inputs_path=inputs_path,
                output_dir=output_dir,
                checkpoint_root=args.checkpoint_root.resolve(),
                precision=args.precision,
            )
        if args.stage in {"compare", "all"}:
            BerniniTransformerParity._compare(
                output_dir=output_dir,
                min_cosine=args.min_cosine,
                max_relative_l2=args.max_relative_l2,
                precision=args.precision,
            )

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Export and compare one exact official-vs-MLX Bernini packed-transformer forward pass."
        )
        parser.add_argument("--stage", choices=("torch", "mlx", "compare", "all"), required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--checkpoint-root", type=Path, required=True)
        parser.add_argument("--reference-root", type=Path, default=Path("/tmp/Bernini"))
        parser.add_argument("--seed", type=int, default=8103)
        parser.add_argument("--target-frames", type=int, default=1)
        parser.add_argument("--target-height", type=int, default=4)
        parser.add_argument("--target-width", type=int, default=4)
        parser.add_argument("--precision", choices=("float32", "runtime"), default="float32")
        parser.add_argument("--min-cosine", type=float)
        parser.add_argument("--max-relative-l2", type=float)
        return parser.parse_args()

    @staticmethod
    def _write_inputs(
        path: Path,
        *,
        seed: int,
        target_frames: int = 1,
        target_height: int = 4,
        target_width: int = 4,
    ) -> None:
        if target_frames < 1 or target_height % 2 or target_width % 2:
            raise ValueError("Target parity shape requires positive frames and spatial dimensions divisible by two.")
        generator = np.random.default_rng(seed)
        target_tokens = target_frames * (target_height // 2) * (target_width // 2)
        total_tokens = 1 + 2 + target_tokens
        arrays = {
            "condition_1": generator.standard_normal((1, 16, 1, 2, 2), dtype=np.float32),
            "condition_2": generator.standard_normal((1, 16, 1, 2, 4), dtype=np.float32),
            "target": generator.standard_normal(
                (1, 16, target_frames, target_height, target_width),
                dtype=np.float32,
            ),
            "encoder_hidden_states": generator.standard_normal((1, 3, 4096), dtype=np.float32),
            "timestep": np.array([500.0], dtype=np.float32),
            "rotary_query": generator.standard_normal((1, total_tokens, 12, 128), dtype=np.float32),
        }
        np.savez(path, **arrays)
        (path.parent / "inputs.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "source_ids": BerniniTransformerParity.SOURCE_IDS,
                    "shapes": {name: list(value.shape) for name, value in arrays.items()},
                    "dtype": "float32",
                    "target_segment_index": 2,
                    "official_source_revision": BerniniTransformerParity.OFFICIAL_SOURCE_REVISION,
                    "checkpoint_revision": BerniniTransformerParity.CHECKPOINT_REVISION,
                    "input_sha256": BerniniTransformerParity._archive_hash(arrays),
                },
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _run_torch(
        *,
        inputs_path: Path,
        output_dir: Path,
        checkpoint_root: Path,
        reference_root: Path,
        precision: str,
    ) -> None:
        import torch

        transformer_module, source_revision = BerniniTransformerParity._official_transformer_module(reference_root)
        transformer_class = transformer_module.WanTransformer3DModel
        inputs = np.load(inputs_path)
        started = time.perf_counter()
        torch_dtype = torch.float32 if precision == "float32" else torch.bfloat16
        model = transformer_class.from_pretrained(
            str(checkpoint_root),
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=torch_dtype,
            use_src_id_rotary_emb=True,
        )
        model.eval()
        segments = [
            torch.from_numpy(inputs["condition_1"]).to(torch_dtype),
            torch.from_numpy(inputs["condition_2"]).to(torch_dtype),
            torch.from_numpy(inputs["target"]).to(torch_dtype),
        ]
        with torch.no_grad():
            patched = []
            rotary = []
            segment_lengths = []
            for segment, source_id in zip(segments, BerniniTransformerParity.SOURCE_IDS, strict=True):
                segment_tokens, segment_rotary = model.patch_vae_latent(segment, source_id=source_id)
                patched.append(segment_tokens)
                rotary.append(segment_rotary)
                segment_lengths.append(int(segment_tokens.shape[1]))
            hidden_states = torch.cat(patched, dim=1)
            rotary_emb = torch.cat(rotary, dim=2)
            rotary_phase = rotary_emb.transpose(1, 2)
            rotary_query = torch.from_numpy(inputs["rotary_query"]).to(torch_dtype)
            rotary_output = transformer_module._apply_rotary_emb(rotary_query, rotary_phase)
            packed_output = model(
                hidden_states=hidden_states,
                timestep=torch.from_numpy(inputs["timestep"]),
                encoder_hidden_states=torch.from_numpy(inputs["encoder_hidden_states"]).to(torch_dtype),
                rotary_emb=rotary_emb,
                batch_image_vae_seqlen=[int(hidden_states.shape[1])],
                text_features_length=[int(inputs["encoder_hidden_states"].shape[1])],
                return_dict=False,
            )[0]
            target_start = sum(segment_lengths[:2])
            target_tokens = packed_output[:, target_start : target_start + segment_lengths[2]]
            target_output = BerniniTransformerParity._torch_unpatch(target_tokens, segments[2].shape)
        output = target_output.detach().float().cpu().numpy()
        rotary_output_array = rotary_output.detach().float().cpu().numpy()
        rotary_cos = np.repeat(rotary_phase.real.detach().cpu().numpy(), repeats=2, axis=-1)
        rotary_sin = np.repeat(rotary_phase.imag.detach().cpu().numpy(), repeats=2, axis=-1)
        suffix = BerniniTransformerParity._precision_suffix(precision)
        np.save(output_dir / f"torch{suffix}_output.npy", output)
        np.save(output_dir / f"torch{suffix}_rotary_output.npy", rotary_output_array)
        np.savez(
            output_dir / f"torch{suffix}_rotary_phase.npz",
            cos=rotary_cos,
            sin=rotary_sin,
        )
        BerniniTransformerParity._write_stage_report(
            output_dir / f"torch{suffix}_report.json",
            backend="official-pytorch",
            elapsed_seconds=time.perf_counter() - started,
            output=output,
            extra={
                "torch_version": torch.__version__,
                "checkpoint_root": str(checkpoint_root),
                "reference_root": str(reference_root),
                "official_source_revision": source_revision,
                "checkpoint_revision": BerniniTransformerParity.CHECKPOINT_REVISION,
                "attention_backend": importlib.import_module("bernini.attention").get_attention_backend(),
                "weight_precision": str(torch_dtype),
                "execution_device": str(next(model.parameters()).device),
                "rotary_phase_precision": str(rotary_phase.dtype),
                "rotary_input_precision": str(rotary_query.dtype),
                "rotary_output_precision": str(rotary_output.dtype),
                "rotary_output_sha256": BerniniTransformerParity._array_hash(rotary_output_array),
            },
        )

    @staticmethod
    def _official_transformer_module(reference_root: Path):
        package_root = reference_root / "bernini"
        if not (package_root / "models" / "transformer_wan.py").is_file():
            raise FileNotFoundError(f"Official Bernini source not found below {reference_root}.")
        revision = subprocess.run(
            ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != BerniniTransformerParity.OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                f"Official Bernini revision mismatch: expected {BerniniTransformerParity.OFFICIAL_SOURCE_REVISION}, "
                f"got {revision}."
            )
        bernini_package = types.ModuleType("bernini")
        bernini_package.__path__ = [str(package_root)]
        sys.modules["bernini"] = bernini_package
        models_package = types.ModuleType("bernini.models")
        models_package.__path__ = [str(package_root / "models")]
        sys.modules["bernini.models"] = models_package
        module = importlib.import_module("bernini.models.transformer_wan")
        return module, revision

    @staticmethod
    def _torch_unpatch(tokens, latent_shape):
        batch_size, channels, frames, height, width = latent_shape
        tokens = tokens.reshape(batch_size, frames, height // 2, width // 2, 1, 2, 2, channels)
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return tokens.reshape(batch_size, channels, frames, height, width)

    @staticmethod
    def _run_mlx(*, inputs_path: Path, output_dir: Path, checkpoint_root: Path, precision: str) -> None:
        import mlx.core as mx

        from mflux.models.common.config import ModelConfig
        from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.wan.model.wan_transformer import WanTransformer
        from mflux.models.wan.model.wan_transformer.wan_attention import WanAttention
        from mflux.models.wan.wan_initializer import WanInitializer
        from mflux.models.wan.weights import WanWeightDefinition

        inputs = np.load(inputs_path)
        started = time.perf_counter()
        config = ModelConfig.bernini_r_1_3b()
        definition = WanWeightDefinition.for_config(config)
        component = next(item for item in definition.get_components() if item.name == "transformer")
        input_precision = mx.float32 if precision == "float32" else ModelConfig.precision
        if precision == "float32":
            component = replace(component, precision=mx.float32)
        transformer = WanTransformer(**WanInitializer._transformer_kwargs(config))
        component_weights, quantization_level, version = WeightLoader._load_component(
            root_path=checkpoint_root,
            component=component,
        )
        loaded = LoadedWeights(
            components={"transformer": component_weights},
            meta_data=MetaData(quantization_level=quantization_level, mflux_version=version),
        )
        WeightApplier.apply_and_quantize_single(
            weights=loaded,
            model=transformer,
            component=component,
            quantize_arg=None,
            quantization_predicate=definition.quantization_predicate,
        )
        del loaded, component_weights
        gc.collect()
        segments = [
            mx.array(inputs["condition_1"], dtype=input_precision),
            mx.array(inputs["condition_2"], dtype=input_precision),
            mx.array(inputs["target"], dtype=input_precision),
        ]
        rotary_pairs = [
            transformer.rope(segment, source_id=source_id)
            for segment, source_id in zip(segments, BerniniTransformerParity.SOURCE_IDS, strict=True)
        ]
        rotary_cos = mx.concatenate([pair[0] for pair in rotary_pairs], axis=1)
        rotary_sin = mx.concatenate([pair[1] for pair in rotary_pairs], axis=1)
        rotary_query = mx.array(inputs["rotary_query"], dtype=input_precision)
        rotary_output = WanAttention._apply_rotary_emb(rotary_query, rotary_cos, rotary_sin)
        output_array = transformer.forward_packed(
            latent_segments=segments,
            source_ids=BerniniTransformerParity.SOURCE_IDS,
            timestep=mx.array(inputs["timestep"], dtype=mx.float32),
            encoder_hidden_states=mx.array(inputs["encoder_hidden_states"], dtype=input_precision),
            target_segment_index=2,
            clear_cache_each_block=True,
        )
        mx.eval(output_array, rotary_output, rotary_cos, rotary_sin)
        output = np.asarray(output_array.astype(mx.float32))
        rotary_output_array = np.asarray(rotary_output.astype(mx.float32))
        rotary_cos_array = np.asarray(rotary_cos.astype(mx.float32))
        rotary_sin_array = np.asarray(rotary_sin.astype(mx.float32))
        suffix = BerniniTransformerParity._precision_suffix(precision)
        np.save(output_dir / f"mlx{suffix}_output.npy", output)
        np.save(output_dir / f"mlx{suffix}_rotary_output.npy", rotary_output_array)
        np.savez(
            output_dir / f"mlx{suffix}_rotary_phase.npz",
            cos=rotary_cos_array,
            sin=rotary_sin_array,
        )
        BerniniTransformerParity._write_stage_report(
            output_dir / f"mlx{suffix}_report.json",
            backend="mlx",
            elapsed_seconds=time.perf_counter() - started,
            output=output,
            extra={
                "mlx_version": getattr(mx, "__version__", "unknown"),
                "checkpoint_root": str(checkpoint_root),
                "checkpoint_revision": BerniniTransformerParity.CHECKPOINT_REVISION,
                "weight_precision": str(component.precision),
                "rotary_phase_precision": str(rotary_cos.dtype),
                "rotary_input_precision": str(rotary_query.dtype),
                "rotary_output_precision": str(rotary_output.dtype),
                "rotary_output_sha256": BerniniTransformerParity._array_hash(rotary_output_array),
            },
        )

    @staticmethod
    def _compare(
        *,
        output_dir: Path,
        min_cosine: float | None,
        max_relative_l2: float | None,
        precision: str,
    ) -> None:
        suffix = BerniniTransformerParity._precision_suffix(precision)
        torch_output = np.load(output_dir / f"torch{suffix}_output.npy").astype(np.float64)
        mlx_output = np.load(output_dir / f"mlx{suffix}_output.npy").astype(np.float64)
        if torch_output.shape != mlx_output.shape:
            raise ValueError(f"Output shape mismatch: Torch {torch_output.shape}, MLX {mlx_output.shape}.")
        profile_thresholds = BerniniTransformerParity.PROFILE_THRESHOLDS[precision]
        min_cosine = profile_thresholds["min_cosine_similarity"] if min_cosine is None else float(min_cosine)
        max_relative_l2 = profile_thresholds["max_relative_l2"] if max_relative_l2 is None else float(max_relative_l2)
        output_metrics = BerniniTransformerParity._metrics(reference=torch_output, actual=mlx_output)
        per_frame_metrics = [
            {
                "frame_index": frame_index,
                **BerniniTransformerParity._metrics(
                    reference=torch_output[:, :, frame_index],
                    actual=mlx_output[:, :, frame_index],
                ),
            }
            for frame_index in range(torch_output.shape[2])
        ]
        torch_rotary_output = np.load(output_dir / f"torch{suffix}_rotary_output.npy").astype(np.float64)
        mlx_rotary_output = np.load(output_dir / f"mlx{suffix}_rotary_output.npy").astype(np.float64)
        rotary_output_metrics = BerniniTransformerParity._metrics(
            reference=torch_rotary_output,
            actual=mlx_rotary_output,
        )
        torch_phase = np.load(output_dir / f"torch{suffix}_rotary_phase.npz")
        mlx_phase = np.load(output_dir / f"mlx{suffix}_rotary_phase.npz")
        rotary_phase_metrics = {
            name: BerniniTransformerParity._metrics(
                reference=torch_phase[name].astype(np.float64),
                actual=mlx_phase[name].astype(np.float64),
            )
            for name in ("cos", "sin")
        }
        torch_stage = json.loads((output_dir / f"torch{suffix}_report.json").read_text())
        mlx_stage = json.loads((output_dir / f"mlx{suffix}_report.json").read_text())
        phase_max_error = max(value["max_absolute_error"] for value in rotary_phase_metrics.values())
        rotary_output_max_relative_l2 = 0.005 if precision == "runtime" else 2e-6
        rotary_passed = (
            phase_max_error <= 2e-6 and rotary_output_metrics["relative_l2"] <= rotary_output_max_relative_l2
        )
        output_passed = (
            output_metrics["cosine_similarity"] >= min_cosine and output_metrics["relative_l2"] <= max_relative_l2
        )
        report = {
            "schema_version": 2,
            "kind": "bernini_official_full_transformer_parity",
            "precision_profile": precision,
            "shape": list(torch_output.shape),
            **output_metrics,
            "per_frame": per_frame_metrics,
            "rotary_phase": rotary_phase_metrics,
            "rotary_application": {
                **rotary_output_metrics,
                "official_input_precision": torch_stage["rotary_input_precision"],
                "official_output_precision": torch_stage["rotary_output_precision"],
                "mlx_input_precision": mlx_stage["rotary_input_precision"],
                "mlx_output_precision": mlx_stage["rotary_output_precision"],
            },
            "precision_semantics": {
                "official_phase": torch_stage["rotary_phase_precision"],
                "official_attention_qk": torch_stage["rotary_output_precision"],
                "mlx_phase": mlx_stage["rotary_phase_precision"],
                "mlx_attention_qk": mlx_stage["rotary_output_precision"],
            },
            "thresholds": {
                "min_cosine_similarity": min_cosine,
                "max_relative_l2": max_relative_l2,
                "max_rotary_phase_absolute_error": 2e-6,
                "max_rotary_application_relative_l2": rotary_output_max_relative_l2,
            },
            "passed": output_passed and rotary_passed,
        }
        report_path = output_dir / f"parity{suffix}_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)

    @staticmethod
    def _precision_suffix(precision: str) -> str:
        return "" if precision == "float32" else "_runtime"

    @staticmethod
    def _metrics(*, reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        if reference.shape != actual.shape:
            raise ValueError(f"Shape mismatch: reference {reference.shape}, actual {actual.shape}.")
        delta = actual - reference
        reference_flat = reference.reshape(-1)
        actual_flat = actual.reshape(-1)
        reference_norm = float(np.linalg.norm(reference_flat))
        actual_norm = float(np.linalg.norm(actual_flat))
        delta_norm = float(np.linalg.norm(delta.reshape(-1)))
        cosine = (
            1.0
            if reference_norm == 0.0 and actual_norm == 0.0
            else float(np.dot(reference_flat, actual_flat) / (reference_norm * actual_norm))
        )
        relative_l2 = 0.0 if reference_norm == 0.0 and delta_norm == 0.0 else delta_norm / reference_norm
        return {
            "cosine_similarity": cosine,
            "relative_l2": relative_l2,
            "mean_absolute_error": float(np.mean(np.abs(delta))),
            "max_absolute_error": float(np.max(np.abs(delta))),
        }

    @staticmethod
    def _write_stage_report(
        path: Path,
        *,
        backend: str,
        elapsed_seconds: float,
        output: np.ndarray,
        extra: dict,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "backend": backend,
                    "elapsed_seconds": elapsed_seconds,
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    "output_min": float(output.min()),
                    "output_max": float(output.max()),
                    "output_sha256": BerniniTransformerParity._array_hash(output),
                    "platform": platform.platform(),
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
            digest.update(BerniniTransformerParity._canonical_bytes(value))
        return digest.hexdigest()

    @staticmethod
    def _array_hash(value: np.ndarray) -> str:
        return hashlib.sha256(BerniniTransformerParity._canonical_bytes(value)).hexdigest()

    @staticmethod
    def _canonical_bytes(value: np.ndarray) -> bytes:
        return np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False)).tobytes()


if __name__ == "__main__":
    BerniniTransformerParity.main()
