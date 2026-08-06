import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


class BerniniVaeParity:
    OFFICIAL_COMPONENT_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
    PINNED_DIFFUSERS_VERSION = "0.35.2"

    @staticmethod
    def main() -> None:
        args = BerniniVaeParity._parse_args()
        if args.tile_spatial and args.height <= 256 and args.width <= 256:
            raise ValueError("--tile-spatial parity requires height or width greater than 256 pixels.")
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        pixels_path = output_dir / "pixels.npy"
        expected_shape = (1, 3, args.frames, args.height, args.width)
        if not pixels_path.exists() or tuple(np.load(pixels_path, mmap_mode="r").shape) != expected_shape:
            BerniniVaeParity._write_pixels(
                pixels_path,
                frames=args.frames,
                height=args.height,
                width=args.width,
            )

        if args.stage == "torch":
            BerniniVaeParity._run_torch(
                pixels_path=pixels_path,
                output_dir=output_dir,
                checkpoint_root=args.checkpoint_root.resolve(),
                device=args.device,
                tile_spatial=args.tile_spatial,
            )
        elif args.stage == "mlx":
            BerniniVaeParity._run_mlx(
                pixels_path=pixels_path,
                output_dir=output_dir,
                checkpoint_root=args.checkpoint_root.resolve(),
                tile_spatial=args.tile_spatial,
            )
        else:
            BerniniVaeParity._compare(output_dir, tile_spatial=args.tile_spatial)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Export and compare the official FP32 and MLX runtime-BF16 Wan2.1 VAE paths used by Bernini."
        )
        parser.add_argument("--stage", choices=("torch", "mlx", "compare"), required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--checkpoint-root", type=Path, required=True)
        parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
        parser.add_argument("--frames", type=int, default=17)
        parser.add_argument("--height", type=int, default=64)
        parser.add_argument("--width", type=int, default=80)
        parser.add_argument("--tile-spatial", action="store_true")
        return parser.parse_args()

    @staticmethod
    def _write_pixels(path: Path, *, frames: int, height: int, width: int) -> None:
        if frames < 1 or height < 16 or width < 16:
            raise ValueError("Parity input requires at least one frame and 16x16 pixels.")
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, None, :]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[None, :, None]
        t = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None, None]
        red = np.clip(x + np.float32(0.45) * np.sin(t * np.float32(math.pi)), -1.0, 1.0)
        green = np.clip(y - np.float32(0.35) * np.sin(t * np.float32(math.pi)), -1.0, 1.0)
        blue = np.sin(x * np.float32(math.pi)) * np.cos(y * np.float32(math.pi)) * np.cos(t * np.float32(math.pi / 2))
        pixels = np.stack(
            [
                np.broadcast_to(red, (frames, height, width)),
                np.broadcast_to(green, (frames, height, width)),
                np.broadcast_to(blue, (frames, height, width)),
            ],
            axis=0,
        )[None, ...].astype(np.float32)
        np.save(path, pixels)
        (path.parent / "pixels.json").write_text(
            json.dumps(
                {
                    "shape": list(pixels.shape),
                    "dtype": str(pixels.dtype),
                    "range": [float(pixels.min()), float(pixels.max())],
                    "construction": "three-channel x/y/time analytic pattern",
                },
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _run_torch(
        *,
        pixels_path: Path,
        output_dir: Path,
        checkpoint_root: Path,
        device: str,
        tile_spatial: bool,
    ) -> None:
        import diffusers
        import torch
        from diffusers import AutoencoderKLWan

        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested for the VAE reference stage but is unavailable.")
        if tile_spatial and diffusers.__version__ != BerniniVaeParity.PINNED_DIFFUSERS_VERSION:
            raise ValueError(
                "Tiled VAE parity requires the pinned diffusers==0.35.2 reference, "
                f"got {diffusers.__version__} from {diffusers.__file__}."
            )
        pixels = torch.from_numpy(np.load(pixels_path)).to(device=device, dtype=torch.float32)
        started = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint_root),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)
        vae.eval()
        if tile_spatial:
            with torch.no_grad():
                untiled_latents = vae.encode(pixels).latent_dist.mode()
                z_dim = int(vae.config.z_dim)
                untiled_mean = torch.tensor(
                    vae.config.latents_mean,
                    dtype=untiled_latents.dtype,
                    device=device,
                ).reshape(1, z_dim, 1, 1, 1)
                untiled_std = torch.tensor(
                    vae.config.latents_std,
                    dtype=untiled_latents.dtype,
                    device=device,
                ).reshape(1, z_dim, 1, 1, 1)
                untiled_normalized = (untiled_latents - untiled_mean) / untiled_std
                untiled_decoded = vae.decode(untiled_latents, return_dict=False)[0].clamp(-1.0, 1.0)
            np.save(
                output_dir / "torch_untiled_normalized_latents.npy",
                untiled_normalized.detach().float().cpu().numpy(),
            )
            np.save(
                output_dir / "torch_untiled_decoded.npy",
                untiled_decoded.detach().float().cpu().numpy(),
            )
            del untiled_latents, untiled_normalized, untiled_decoded, untiled_mean, untiled_std
            if device == "mps":
                torch.mps.empty_cache()
            vae.enable_tiling()
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.mode()
            z_dim = int(vae.config.z_dim)
            mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=device).reshape(1, z_dim, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=device).reshape(1, z_dim, 1, 1, 1)
            normalized = (latents - mean) / std
            decoded = vae.decode(latents, return_dict=False)[0].clamp(-1.0, 1.0)
        normalized_np = normalized.detach().float().cpu().numpy()
        decoded_np = decoded.detach().float().cpu().numpy()
        np.save(output_dir / "torch_normalized_latents.npy", normalized_np)
        np.save(output_dir / "torch_decoded.npy", decoded_np)
        BerniniVaeParity._write_stage_report(
            output_dir / "torch_report.json",
            backend="official-diffusers",
            elapsed_seconds=time.perf_counter() - started,
            pixels=np.load(pixels_path),
            normalized=normalized_np,
            decoded=decoded_np,
            extra={
                "device": device,
                "torch_version": torch.__version__,
                "diffusers_version": diffusers.__version__,
                "diffusers_path": str(diffusers.__file__),
                "checkpoint_root": str(checkpoint_root),
                "weight_precision": "torch.float32",
                "tile_spatial": tile_spatial,
            },
        )

    @staticmethod
    def _run_mlx(
        *,
        pixels_path: Path,
        output_dir: Path,
        checkpoint_root: Path,
        tile_spatial: bool,
    ) -> None:
        import mlx.core as mx

        from mflux.models.common.config import ModelConfig
        from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.wan.model.wan_vae import Wan2_2_VAE
        from mflux.models.wan.wan_initializer import WanInitializer
        from mflux.models.wan.weights import WanWeightDefinition

        pixels_np = np.load(pixels_path)
        official_normalized = np.load(output_dir / "torch_normalized_latents.npy")
        started = time.perf_counter()
        config = ModelConfig.bernini_r_1_3b()
        definition = WanWeightDefinition.for_config(config)
        component = next(item for item in definition.get_components() if item.name == "vae")
        vae = Wan2_2_VAE(**WanInitializer._vae_kwargs(config))
        component_weights, quantization_level, version = WeightLoader._load_component(
            root_path=checkpoint_root,
            component=component,
        )
        loaded = LoadedWeights(
            components={"vae": component_weights},
            meta_data=MetaData(quantization_level=quantization_level, mflux_version=version),
        )
        WeightApplier.apply_and_quantize_single(
            weights=loaded,
            model=vae,
            component=component,
            quantize_arg=None,
            quantization_predicate=definition.quantization_predicate,
        )
        del loaded, component_weights
        gc.collect()

        pixels = mx.array(pixels_np, dtype=mx.float32)
        if tile_spatial:
            untiled_normalized = vae.encode_normalized(
                pixels,
                clear_cache_each_slice=True,
            )
            untiled_shared = vae.decode_normalized_latents(
                mx.array(official_normalized, dtype=ModelConfig.precision),
                clear_cache_each_slice=True,
            )
            untiled_e2e = vae.decode_normalized_latents(
                untiled_normalized,
                clear_cache_each_slice=True,
            )
            mx.eval(untiled_normalized, untiled_shared, untiled_e2e)
            np.save(
                output_dir / "mlx_untiled_normalized_latents.npy",
                np.asarray(untiled_normalized, dtype=np.float32),
            )
            np.save(
                output_dir / "mlx_untiled_decode_official_latent.npy",
                np.asarray(untiled_shared, dtype=np.float32),
            )
            np.save(
                output_dir / "mlx_untiled_e2e.npy",
                np.asarray(untiled_e2e, dtype=np.float32),
            )
            del untiled_normalized, untiled_shared, untiled_e2e
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        normalized = vae.encode_normalized(
            pixels,
            clear_cache_each_slice=tile_spatial,
            tile_spatial=tile_spatial,
        )
        decoded_shared = vae.decode_normalized_latents(
            mx.array(official_normalized, dtype=ModelConfig.precision),
            clear_cache_each_slice=True,
            tile_spatial=tile_spatial,
        )
        decoded_e2e = vae.decode_normalized_latents(
            normalized,
            clear_cache_each_slice=True,
            tile_spatial=tile_spatial,
        )
        mx.eval(normalized, decoded_shared, decoded_e2e)
        normalized_np = np.asarray(normalized, dtype=np.float32)
        decoded_shared_np = np.asarray(decoded_shared, dtype=np.float32)
        decoded_e2e_np = np.asarray(decoded_e2e, dtype=np.float32)
        np.save(output_dir / "mlx_bf16_normalized_latents.npy", normalized_np)
        np.save(output_dir / "mlx_bf16_decode_official_latent.npy", decoded_shared_np)
        np.save(output_dir / "mlx_bf16_e2e.npy", decoded_e2e_np)
        BerniniVaeParity._write_stage_report(
            output_dir / "mlx_report.json",
            backend="mlx",
            elapsed_seconds=time.perf_counter() - started,
            pixels=pixels_np,
            normalized=normalized_np,
            decoded=decoded_e2e_np,
            extra={
                "mlx_version": getattr(mx, "__version__", "unknown"),
                "checkpoint_root": str(checkpoint_root),
                "weight_precision": str(component.precision),
                "tile_spatial": tile_spatial,
            },
        )

    @staticmethod
    def _compare(output_dir: Path, *, tile_spatial: bool) -> None:
        torch_report = json.loads((output_dir / "torch_report.json").read_text())
        mlx_report = json.loads((output_dir / "mlx_report.json").read_text())
        BerniniVaeParity._validate_stage_reports(
            torch_report=torch_report,
            mlx_report=mlx_report,
            tile_spatial=tile_spatial,
        )
        torch_latents = np.load(output_dir / "torch_normalized_latents.npy")
        torch_decoded = np.load(output_dir / "torch_decoded.npy")
        mlx_latents = np.load(output_dir / "mlx_bf16_normalized_latents.npy")
        mlx_shared = np.load(output_dir / "mlx_bf16_decode_official_latent.npy")
        mlx_e2e = np.load(output_dir / "mlx_bf16_e2e.npy")
        if torch_latents.shape != mlx_latents.shape:
            raise ValueError(f"VAE parity latent shape mismatch: {torch_latents.shape} != {mlx_latents.shape}.")
        if torch_decoded.shape != mlx_shared.shape or torch_decoded.shape != mlx_e2e.shape:
            raise ValueError(
                "VAE parity decoded shape mismatch: "
                f"torch={torch_decoded.shape}, shared={mlx_shared.shape}, e2e={mlx_e2e.shape}."
            )
        metrics = {
            "encode": BerniniVaeParity._metrics(torch_latents, mlx_latents),
            "decode_shared_official_latent": BerniniVaeParity._metrics(torch_decoded, mlx_shared),
            "end_to_end": BerniniVaeParity._metrics(torch_decoded, mlx_e2e),
        }
        per_frame = {
            "decode_shared_official_latent": BerniniVaeParity._per_frame_metrics(torch_decoded, mlx_shared),
            "end_to_end": BerniniVaeParity._per_frame_metrics(torch_decoded, mlx_e2e),
        }
        seam_metrics = {
            "decode_shared_official_latent": BerniniVaeParity._seam_metrics(torch_decoded, mlx_shared),
            "end_to_end": BerniniVaeParity._seam_metrics(torch_decoded, mlx_e2e),
        }
        tiling_quality = None
        if tile_spatial:
            torch_untiled_latents = np.load(output_dir / "torch_untiled_normalized_latents.npy")
            torch_untiled = np.load(output_dir / "torch_untiled_decoded.npy")
            mlx_untiled_latents = np.load(output_dir / "mlx_untiled_normalized_latents.npy")
            mlx_untiled_shared = np.load(output_dir / "mlx_untiled_decode_official_latent.npy")
            mlx_untiled_e2e = np.load(output_dir / "mlx_untiled_e2e.npy")
            tiling_quality = {
                "torch_encode_tiled_vs_untiled": {
                    **BerniniVaeParity._metrics(torch_untiled_latents, torch_latents),
                    "latent_seams": BerniniVaeParity._seam_metrics(
                        torch_untiled_latents,
                        torch_latents,
                        stride=24,
                        band=1,
                    ),
                },
                "torch_end_to_end_tiled_vs_untiled": {
                    **BerniniVaeParity._metrics(torch_untiled, torch_decoded),
                    "seams": BerniniVaeParity._seam_metrics(torch_untiled, torch_decoded),
                },
                "mlx_encode_tiled_vs_untiled": {
                    **BerniniVaeParity._metrics(mlx_untiled_latents, mlx_latents),
                    "latent_seams": BerniniVaeParity._seam_metrics(
                        mlx_untiled_latents,
                        mlx_latents,
                        stride=24,
                        band=1,
                    ),
                },
                "mlx_decode_same_official_latent_tiled_vs_untiled": {
                    **BerniniVaeParity._metrics(mlx_untiled_shared, mlx_shared),
                    "seams": BerniniVaeParity._seam_metrics(mlx_untiled_shared, mlx_shared),
                },
                "mlx_end_to_end_tiled_vs_untiled": {
                    **BerniniVaeParity._metrics(mlx_untiled_e2e, mlx_e2e),
                    "seams": BerniniVaeParity._seam_metrics(mlx_untiled_e2e, mlx_e2e),
                },
            }
        thresholds = {
            "encode": {"min_cosine_similarity": 0.9999, "max_relative_l2": 0.01, "max_absolute_error": 0.03},
            "decode_shared_official_latent": {
                "min_cosine_similarity": 0.9999,
                "max_relative_l2": 0.01,
                "max_absolute_error": 0.03,
            },
            "end_to_end": {
                "min_cosine_similarity": 0.9998,
                "max_relative_l2": 0.015,
                "min_psnr_peak_2": 45.0,
            },
        }
        passed = all(BerniniVaeParity._passes(metrics[name], gate) for name, gate in thresholds.items())
        report = {
            "schema_version": 2,
            "kind": (
                "bernini_wan21_vae_tiled_bf16_vs_official_fp32"
                if tile_spatial
                else "bernini_wan21_vae_default_bf16_vs_official_fp32"
            ),
            "official_component_base_revision": BerniniVaeParity.OFFICIAL_COMPONENT_REVISION,
            "official_diffusers_tiling_reference": "diffusers-0.35.2" if tile_spatial else None,
            "tile_spatial": tile_spatial,
            "input_shape": list(np.load(output_dir / "pixels.npy").shape),
            "component_precision": "mlx.core.bfloat16",
            **metrics,
            "per_frame": per_frame,
            "seam_metrics": seam_metrics,
            "tiling_quality": tiling_quality,
            "thresholds": thresholds,
            "passed": passed,
        }
        (output_dir / "vae_parity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(report, indent=2, sort_keys=True))
        if not passed:
            raise SystemExit(1)

    @staticmethod
    def _validate_stage_reports(*, torch_report: dict, mlx_report: dict, tile_spatial: bool) -> None:
        for backend, report in (("torch", torch_report), ("mlx", mlx_report)):
            if report.get("tile_spatial") is not tile_spatial:
                raise ValueError(
                    f"{backend} stage tile_spatial={report.get('tile_spatial')!r} does not match "
                    f"compare request {tile_spatial!r}."
                )
        if torch_report.get("input_shape") != mlx_report.get("input_shape"):
            raise ValueError("Torch and MLX VAE parity stages used different input shapes.")
        if torch_report.get("checkpoint_root") != mlx_report.get("checkpoint_root"):
            raise ValueError("Torch and MLX VAE parity stages used different checkpoint roots.")
        checkpoint_revision = Path(str(torch_report.get("checkpoint_root", ""))).name
        if checkpoint_revision != BerniniVaeParity.OFFICIAL_COMPONENT_REVISION:
            raise ValueError(
                "VAE parity requires the pinned component revision "
                f"{BerniniVaeParity.OFFICIAL_COMPONENT_REVISION}, got {checkpoint_revision!r}."
            )
        if tile_spatial and torch_report.get("diffusers_version") != BerniniVaeParity.PINNED_DIFFUSERS_VERSION:
            raise ValueError(
                f"Tiled VAE parity stage did not use pinned diffusers {BerniniVaeParity.PINNED_DIFFUSERS_VERSION}."
            )

    @staticmethod
    def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
        reference64 = reference.astype(np.float64)
        actual64 = actual.astype(np.float64)
        delta = actual64 - reference64
        reference_flat = reference64.reshape(-1)
        actual_flat = actual64.reshape(-1)
        rmse = float(np.sqrt(np.mean(np.square(delta))))
        return {
            "cosine_similarity": float(
                np.dot(reference_flat, actual_flat) / (np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat))
            ),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference64)),
            "mean_absolute_error": float(np.mean(np.abs(delta))),
            "max_absolute_error": float(np.max(np.abs(delta))),
            "rmse": rmse,
            "psnr_peak_2": float(20 * math.log10(2.0 / rmse)) if rmse > 0 else math.inf,
        }

    @staticmethod
    def _per_frame_metrics(reference: np.ndarray, actual: np.ndarray) -> list[dict[str, float | int]]:
        return [
            {
                "frame_index": frame_index,
                **BerniniVaeParity._metrics(reference[:, :, frame_index], actual[:, :, frame_index]),
            }
            for frame_index in range(reference.shape[2])
        ]

    @staticmethod
    def _seam_metrics(reference: np.ndarray, actual: np.ndarray, *, stride: int = 192, band: int = 4) -> dict:
        height, width = reference.shape[-2:]
        horizontal_boundaries = list(range(stride, height, stride))
        vertical_boundaries = list(range(stride, width, stride))
        horizontal_mask = np.zeros((height, width), dtype=bool)
        vertical_mask = np.zeros((height, width), dtype=bool)
        for boundary in horizontal_boundaries:
            horizontal_mask[max(0, boundary - band) : min(height, boundary + band), :] = True
        for boundary in vertical_boundaries:
            vertical_mask[:, max(0, boundary - band) : min(width, boundary + band)] = True
        seam_mask = horizontal_mask | vertical_mask
        absolute_error = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
        global_mae = float(np.mean(absolute_error))
        seam_mae = BerniniVaeParity._masked_mean(absolute_error, seam_mask)
        interior_mae = BerniniVaeParity._masked_mean(absolute_error, ~seam_mask)
        return {
            "stride": stride,
            "band": band,
            "horizontal_boundaries": horizontal_boundaries,
            "vertical_boundaries": vertical_boundaries,
            "global_mae": global_mae,
            "horizontal_seam_mae": BerniniVaeParity._masked_mean(absolute_error, horizontal_mask),
            "vertical_seam_mae": BerniniVaeParity._masked_mean(absolute_error, vertical_mask),
            "combined_seam_mae": seam_mae,
            "interior_mae": interior_mae,
            "seam_to_interior_ratio": (
                float(seam_mae / interior_mae) if seam_mae is not None and interior_mae not in {None, 0.0} else None
            ),
            "max_absolute_error": float(np.max(absolute_error)),
        }

    @staticmethod
    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.mean(values[..., mask]))

    @staticmethod
    def _passes(metrics: dict[str, float], thresholds: dict[str, float]) -> bool:
        return (
            metrics["cosine_similarity"] >= thresholds["min_cosine_similarity"]
            and metrics["relative_l2"] <= thresholds["max_relative_l2"]
            and metrics.get("max_absolute_error", 0.0) <= thresholds.get("max_absolute_error", math.inf)
            and metrics.get("psnr_peak_2", math.inf) >= thresholds.get("min_psnr_peak_2", -math.inf)
        )

    @staticmethod
    def _write_stage_report(
        path: Path,
        *,
        backend: str,
        elapsed_seconds: float,
        pixels: np.ndarray,
        normalized: np.ndarray,
        decoded: np.ndarray,
        extra: dict,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "backend": backend,
                    "elapsed_seconds": elapsed_seconds,
                    "platform": platform.platform(),
                    "input_shape": list(pixels.shape),
                    "normalized_latent_shape": list(normalized.shape),
                    "decoded_shape": list(decoded.shape),
                    **extra,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    BerniniVaeParity.main()
