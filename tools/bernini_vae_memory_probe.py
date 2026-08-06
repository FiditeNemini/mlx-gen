from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from generation_memory_benchmark import ProcessTreeSampler
from PIL import Image


class BerniniVaeMemoryProbe:
    COMPONENT_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"

    @staticmethod
    def main() -> None:
        args = BerniniVaeMemoryProbe._parse_args()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.child:
            BerniniVaeMemoryProbe._run_child(
                output_dir=output_dir,
                checkpoint_root=args.checkpoint_root.resolve(),
                frames=args.frames,
                height=args.height,
                width=args.width,
            )
            return
        BerniniVaeMemoryProbe._run_parent(args=args, output_dir=output_dir)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Measure a full tiled Bernini Wan2.1 VAE encode/decode in a sampled subprocess."
        )
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--checkpoint-root", type=Path, required=True)
        parser.add_argument("--frames", type=int, default=81)
        parser.add_argument("--height", type=int, default=480)
        parser.add_argument("--width", type=int, default=848)
        parser.add_argument("--sample-interval-ms", type=int, default=250)
        parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
        args = parser.parse_args()
        if args.frames < 1 or (args.frames - 1) % 4 != 0:
            raise ValueError("Wan2.1 probe frames must satisfy 4n+1.")
        if args.height <= 256 and args.width <= 256:
            raise ValueError("The memory probe must exercise spatial tiling above 256 pixels.")
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        return args

    @staticmethod
    def _run_parent(*, args: argparse.Namespace, output_dir: Path) -> None:
        child_report_path = output_dir / "vae_memory_child_report.json"
        argv = [
            "/usr/bin/time",
            "-l",
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--output-dir",
            str(output_dir),
            "--checkpoint-root",
            str(args.checkpoint_root.resolve()),
            "--frames",
            str(args.frames),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--sample-interval-ms",
            str(args.sample_interval_ms),
        ]
        env = os.environ.copy()
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING": "1",
            }
        )
        previous_sampling = os.environ.get("MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING")
        os.environ["MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING"] = "1"
        started = time.perf_counter()
        process = subprocess.Popen(
            argv,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with ProcessTreeSampler(process.pid, args.sample_interval_ms / 1000) as sampler:
                stdout, stderr = process.communicate()
        finally:
            if previous_sampling is None:
                os.environ.pop("MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING", None)
            else:
                os.environ["MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING"] = previous_sampling
        (output_dir / "stdout.log").write_text(stdout)
        (output_dir / "stderr.log").write_text(stderr)
        if process.returncode != 0:
            raise RuntimeError(f"Bernini VAE memory child failed; see {output_dir / 'stderr.log'}")
        if not child_report_path.is_file():
            raise RuntimeError("Bernini VAE memory child exited without a report.")
        child_report = json.loads(child_report_path.read_text())
        report = {
            "schema_version": 1,
            "kind": "bernini_wan2_1_tiled_vae_production_geometry_memory",
            "command": argv,
            "wall_seconds": time.perf_counter() - started,
            "sampler": sampler.summary(),
            "samples": sampler.samples,
            "child": child_report,
            "passed": (
                child_report.get("passed") is True
                and sampler.summary().get("peak_sampled_darwin_physical_footprint_bytes") is not None
            ),
        }
        (output_dir / "vae_memory_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps({key: report[key] for key in ("kind", "wall_seconds", "sampler", "passed")}, indent=2))
        if not report["passed"]:
            raise SystemExit(1)

    @staticmethod
    def _run_child(
        *,
        output_dir: Path,
        checkpoint_root: Path,
        frames: int,
        height: int,
        width: int,
    ) -> None:
        import mlx.core as mx

        from mflux.models.common.config import ModelConfig
        from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.wan.model.wan_vae import Wan2_2_VAE
        from mflux.models.wan.wan_initializer import WanInitializer
        from mflux.models.wan.weights import WanWeightDefinition
        from mflux.utils.runtime_memory import RuntimeMemory

        if checkpoint_root.name != BerniniVaeMemoryProbe.COMPONENT_REVISION:
            raise ValueError(
                "Bernini VAE memory probe requires the pinned component revision "
                f"{BerniniVaeMemoryProbe.COMPONENT_REVISION}, got {checkpoint_root.name}."
            )
        RuntimeMemory.apply_mlx_cache_limit(1.0, low_ram=True)
        mx.reset_peak_memory()
        started = time.perf_counter()
        snapshots = []
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
        mx.eval(vae.parameters())
        mx.synchronize()
        mx.clear_cache()
        snapshots.append(RuntimeMemory.snapshot("vae-weights-materialized", synchronize=True).to_metadata())

        tile_sample_size = [vae.tile_sample_min_height, vae.tile_sample_min_width]
        tile_sample_stride = [vae.tile_sample_stride_height, vae.tile_sample_stride_width]
        policy_id = vae.SPATIAL_TILING_POLICY_ID

        pixels = BerniniVaeMemoryProbe._analytic_pixels(
            frames=frames,
            height=height,
            width=width,
        )
        mx.eval(pixels)
        input_dtype = str(pixels.dtype)
        snapshots.append(
            RuntimeMemory.snapshot("pixels-materialized", tensors=(pixels,), synchronize=True).to_metadata()
        )
        latents = vae.encode_normalized(
            pixels,
            clear_cache_each_slice=True,
            tile_spatial=True,
        )
        latents = mx.contiguous(latents.astype(ModelConfig.precision))
        mx.eval(latents)
        latent_dtype = str(latents.dtype)
        snapshots.append(
            RuntimeMemory.snapshot("tiled-encode-complete", tensors=(latents,), synchronize=True).to_metadata()
        )
        input_shape = list(pixels.shape)
        latent_shape = list(latents.shape)
        del pixels
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        frame_hasher = hashlib.sha256()
        decoded_frame_count = 0
        decoded_shape = None
        decoded_dtype = None
        selected_indices = {0, frames // 2, frames - 1}
        selected_paths = {}
        for decoded_slice in vae.iter_decode_normalized_latent_slices(
            latents,
            clear_cache_each_slice=True,
            tile_spatial=True,
        ):
            decoded_dtype = str(decoded_slice.dtype)
            decoded_np = np.asarray(decoded_slice, dtype=np.float32)
            if not np.isfinite(decoded_np).all():
                raise RuntimeError(f"Non-finite tiled VAE output at decoded frame {decoded_frame_count}.")
            if decoded_shape is None:
                decoded_shape = [
                    decoded_np.shape[0],
                    decoded_np.shape[1],
                    frames,
                    decoded_np.shape[-2],
                    decoded_np.shape[-1],
                ]
            for local_index in range(decoded_np.shape[2]):
                frame_index = decoded_frame_count + local_index
                frame = BerniniVaeMemoryProbe._uint8_frame(decoded_np[0, :, local_index])
                frame_hasher.update(frame.tobytes())
                if frame_index in selected_indices:
                    frame_path = output_dir / f"decoded_frame_{frame_index:03d}.png"
                    Image.fromarray(frame, mode="RGB").save(frame_path)
                    selected_paths[str(frame_index)] = str(frame_path)
            decoded_frame_count += decoded_np.shape[2]
            del decoded_np, decoded_slice
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
        snapshots.append(RuntimeMemory.snapshot("tiled-decode-consumed", synchronize=True).to_metadata())
        del latents, vae
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        snapshots.append(RuntimeMemory.snapshot("probe-complete", synchronize=True).to_metadata())

        expected_latent_shape = [1, 16, 1 + (frames - 1) // 4, height // 8, width // 8]
        expected_decoded_shape = [1, 3, frames, height, width]
        report = {
            "schema_version": 1,
            "kind": "bernini_wan2_1_tiled_vae_memory_child",
            "component_source": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
            "component_revision": BerniniVaeMemoryProbe.COMPONENT_REVISION,
            "checkpoint_root": str(checkpoint_root),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "weight_precision": str(component.precision),
            "input_dtype": input_dtype,
            "latent_dtype": latent_dtype,
            "decoded_dtype": decoded_dtype,
            "mlx_cache_limit_bytes": 1000**3,
            "policy_id": policy_id,
            "tile_sample_size": tile_sample_size,
            "tile_sample_stride": tile_sample_stride,
            "input_shape": input_shape,
            "latent_shape": latent_shape,
            "expected_latent_shape": expected_latent_shape,
            "decoded_shape": decoded_shape,
            "expected_decoded_shape": expected_decoded_shape,
            "decoded_frame_count": decoded_frame_count,
            "decoded_uint8_sha256": frame_hasher.hexdigest(),
            "selected_frame_paths": selected_paths,
            "snapshots": snapshots,
            "elapsed_seconds": time.perf_counter() - started,
            "passed": (
                latent_shape == expected_latent_shape
                and decoded_shape == expected_decoded_shape
                and decoded_frame_count == frames
                and set(selected_paths) == {str(index) for index in selected_indices}
            ),
        }
        (output_dir / "vae_memory_child_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    @staticmethod
    def _analytic_pixels(*, frames: int, height: int, width: int):
        import mlx.core as mx

        x = mx.linspace(-1.0, 1.0, width).reshape(1, 1, 1, 1, width)
        y = mx.linspace(-1.0, 1.0, height).reshape(1, 1, 1, height, 1)
        t = mx.linspace(0.0, 1.0, frames).reshape(1, 1, frames, 1, 1)
        red = mx.broadcast_to(mx.clip(x + 0.35 * mx.sin(t * mx.pi), -1.0, 1.0), (1, 1, frames, height, width))
        green = mx.broadcast_to(mx.clip(y - 0.25 * mx.sin(t * mx.pi), -1.0, 1.0), (1, 1, frames, height, width))
        blue = mx.broadcast_to(
            mx.sin(x * mx.pi) * mx.cos(y * mx.pi) * mx.cos(t * mx.pi / 2), (1, 1, frames, height, width)
        )
        return mx.concatenate([red, green, blue], axis=1).astype(mx.float32)

    @staticmethod
    def _uint8_frame(frame: np.ndarray) -> np.ndarray:
        return np.clip((np.transpose(frame, (1, 2, 0)) + 1.0) * 127.5, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    BerniniVaeMemoryProbe.main()
