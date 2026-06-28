import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2StreamedVideoNoiseProvider
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.runtime_memory import RuntimeMemory
from mflux.utils.video_util import VideoUtil

SCENARIOS = {
    "wan_save_streaming": {
        "item": "0059",
        "metric": "process_peak_rss_bytes",
        "threshold_percent": 15.0,
        "description": "Decoded-video save residency: full materialized PIL frames versus streamed frame batches.",
    },
    "runtime_snapshot_overhead": {
        "item": "0060",
        "metric": "wall_seconds",
        "threshold_percent": -10.0,
        "description": "Telemetry overhead: no runtime snapshots versus repeated runtime snapshots.",
    },
    "prompt_materialization": {
        "item": "0061",
        "metric": "mlx_peak_memory_bytes",
        "threshold_percent": 15.0,
        "description": "MLX lazy prompt graph release: lazy encoder graph versus materialized inference tree.",
    },
    "seedvr2_noise": {
        "item": "0062",
        "metric": "mlx_peak_memory_bytes",
        "threshold_percent": 40.0,
        "description": "SeedVR2 streamed-video noise residency: full clip-global noise versus chunk-bounded slices.",
    },
    "cache_limit_policy": {
        "item": "0063",
        "metric": "mlx_cache_memory_bytes",
        "threshold_percent": 40.0,
        "description": "Pre-load cache policy: unconstrained MLX cache versus low cache limit.",
    },
    "stepwise_retention": {
        "item": "0064",
        "metric": "process_peak_rss_bytes",
        "threshold_percent": 1.0,
        "description": "Stepwise host retention: GeneratedImage objects versus retained PIL images only.",
    },
}


@dataclass(frozen=True)
class ChildConfig:
    scenario: str
    variant: str
    run: int
    output_json: Path
    work_dir: Path
    sample_interval_ms: int


class MemorySampler:
    def __init__(self, interval_seconds: float):
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.capture("final", synchronize=True)

    def capture(self, phase: str, *, synchronize: bool = False) -> None:
        snapshot = RuntimeMemory.snapshot(phase, synchronize=synchronize).to_metadata()
        self.samples.append(snapshot)

    def summary(self) -> dict[str, int | None]:
        return {
            "process_peak_rss_bytes": self._max("process_rss_bytes"),
            "darwin_peak_physical_footprint_bytes": self._max("darwin_physical_footprint_bytes"),
            "mlx_active_memory_bytes": self._max("mlx_active_memory_bytes"),
            "mlx_peak_memory_bytes": self._max("mlx_peak_memory_bytes"),
            "mlx_cache_memory_bytes": self._max("mlx_cache_memory_bytes"),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.capture("sample")
            self._stop.wait(self.interval_seconds)

    def _max(self, key: str) -> int | None:
        values = [sample.get(key) for sample in self.samples if sample.get(key) is not None]
        return int(max(values)) if values else None


class MemoryOptimizationBenchmark:
    @staticmethod
    def main() -> None:
        args = MemoryOptimizationBenchmark._parse_args()
        if args.child:
            MemoryOptimizationBenchmark._run_child(
                ChildConfig(
                    scenario=args.scenario,
                    variant=args.variant,
                    run=args.run,
                    output_json=args.output_json,
                    work_dir=args.work_dir,
                    sample_interval_ms=args.sample_interval_ms,
                )
            )
            return

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.work_dir.mkdir(parents=True, exist_ok=True)
        scenarios = args.scenarios or list(SCENARIOS)
        raw_results: list[dict[str, Any]] = []
        for scenario in scenarios:
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}")
            for run in range(1, args.runs + 1):
                raw_results.extend(
                    [
                        MemoryOptimizationBenchmark._run_parent_child(
                            scenario=scenario,
                            variant=variant,
                            run=run,
                            args=args,
                        )
                        for variant in ("baseline", "optimized")
                    ]
                )

        report = MemoryOptimizationBenchmark._build_report(raw_results)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(MemoryOptimizationBenchmark._compact_report(report), indent=2, sort_keys=True))

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-json", type=Path, default=Path("validation_outputs/memory/memory_stats.json"))
        parser.add_argument("--work-dir", type=Path, default=Path("validation_outputs/memory/work"))
        parser.add_argument("--runs", type=int, default=3)
        parser.add_argument("--sample-interval-ms", type=int, default=20)
        parser.add_argument("--scenarios", nargs="*", choices=sorted(SCENARIOS))
        parser.add_argument("--keep-artifacts", action="store_true")
        parser.add_argument("--child", action="store_true")
        parser.add_argument("--scenario", choices=sorted(SCENARIOS))
        parser.add_argument("--variant", choices=["baseline", "optimized"])
        parser.add_argument("--run", type=int, default=1)
        args = parser.parse_args()
        if args.runs <= 0:
            raise ValueError("--runs must be greater than zero.")
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        if args.child and (args.scenario is None or args.variant is None):
            raise ValueError("--child requires --scenario and --variant.")
        return args

    @staticmethod
    def _run_parent_child(*, scenario: str, variant: str, run: int, args: argparse.Namespace) -> dict[str, Any]:
        child_output = args.work_dir / f"{scenario}_{variant}_run{run}.json"
        child_work_dir = args.work_dir / f"{scenario}_{variant}_run{run}"
        if child_work_dir.exists():
            shutil.rmtree(child_work_dir)
        child_work_dir.mkdir(parents=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--scenario",
            scenario,
            "--variant",
            variant,
            "--run",
            str(run),
            "--output-json",
            str(child_output),
            "--work-dir",
            str(child_work_dir),
            "--sample-interval-ms",
            str(args.sample_interval_ms),
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"{scenario}/{variant}/run{run} failed with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(child_output.read_text())
        result["parent_wall_seconds"] = elapsed
        if not args.keep_artifacts and child_work_dir.exists():
            shutil.rmtree(child_work_dir)
        return result

    @staticmethod
    def _run_child(config: ChildConfig) -> None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        mx.clear_cache()
        mx.reset_peak_memory()
        gc.collect()
        started = time.perf_counter()
        with MemorySampler(config.sample_interval_ms / 1000) as sampler:
            sampler.capture("start", synchronize=True)
            scenario_result = getattr(MemoryOptimizationBenchmark, f"_scenario_{config.scenario}")(
                variant=config.variant,
                run=config.run,
                work_dir=config.work_dir,
            )
            mx.synchronize()
            sampler.capture("after-scenario", synchronize=True)
        wall_seconds = time.perf_counter() - started
        summary = sampler.summary()
        result = {
            "scenario": config.scenario,
            "variant": config.variant,
            "run": config.run,
            "pid": os.getpid(),
            "wall_seconds": wall_seconds,
            "metrics": summary,
            "scenario_result": scenario_result,
            "sample_count": len(sampler.samples),
            "samples": sampler.samples,
        }
        config.output_json.write_text(json.dumps(result, indent=2, sort_keys=True))

    @staticmethod
    def _scenario_prompt_materialization(*, variant: str, run: int, work_dir: Path) -> dict[str, Any]:
        del work_dir
        hidden = 4096
        token_count = 64
        denoise_elements = (140 * 1000 * 1000) // 4
        weights = mx.ones((hidden, hidden), dtype=mx.float32)
        tokens = mx.ones((token_count, hidden), dtype=mx.float32)
        mx.eval(weights, tokens)
        prompt = tokens @ weights
        if variant == "optimized":
            prompt = RuntimeMemory.materialize_inference_tree(prompt)
        del weights
        del tokens
        mx.clear_cache()
        denoise = mx.ones((denoise_elements,), dtype=mx.float32)
        mx.eval(prompt, denoise)
        checksum = float(mx.sum(prompt).item())
        return {
            "hidden": hidden,
            "token_count": token_count,
            "denoise_bytes": denoise_elements * 4,
            "prompt_checksum": checksum,
            "proof": "optimized materializes prompt before simulated encoder release; baseline leaves lazy graph live.",
        }

    @staticmethod
    def _scenario_wan_save_streaming(*, variant: str, run: int, work_dir: Path) -> dict[str, Any]:
        frames = 49
        height = 320
        width = 512
        fps = 16
        output_path = work_dir / f"{variant}_{run}.mp4"
        model_config = ModelConfig.wan2_2_ti2v_5b()
        if variant == "baseline":
            latents = MemoryOptimizationBenchmark._decoded_latents(
                frame_start=0,
                frame_count=frames,
                height=height,
                width=width,
            )
            mx.eval(latents)
            video = VideoUtil.to_video(
                decoded_latents=latents,
                fps=fps,
                model_config=model_config,
                seed=run,
                prompt="memory benchmark",
                steps=1,
                guidance=None,
                quantization=8,
                generation_time=0.0,
                materialize_frames=True,
                extra_metadata={"benchmark_variant": variant},
            )
        else:

            def decoded_slices():
                for start in range(0, frames, 7):
                    count = min(7, frames - start)
                    yield MemoryOptimizationBenchmark._decoded_latents(
                        frame_start=start,
                        frame_count=count,
                        height=height,
                        width=width,
                    )

            def frame_batches_factory():
                return VideoUtil.decoded_latent_slices_to_frame_batches(
                    decoded_slices(),
                    batch_size=8,
                    total_frames=frames,
                )

            video = VideoUtil.to_video_from_frame_batches(
                frame_batches_factory=frame_batches_factory,
                fps=fps,
                model_config=model_config,
                seed=run,
                prompt="memory benchmark",
                steps=1,
                guidance=None,
                quantization=8,
                generation_time=0.0,
                height=height,
                width=width,
                frame_count=frames,
                extra_metadata={"benchmark_variant": variant, "wan_decode_mode": "streamed_vae_slices"},
            )
        video.save(output_path, export_json_metadata=False, overwrite=True, validate_health=False)
        info = VideoUtil.inspect_video(output_path)
        return {
            "output_path": str(output_path),
            "output_sha256": MemoryOptimizationBenchmark._sha256(output_path),
            "output_size_bytes": output_path.stat().st_size,
            "frames": info.frame_count,
            "width": info.width,
            "height": info.height,
            "fps": fps,
            "proof": "baseline materializes all decoded frames; optimized writes replayable frame batches.",
        }

    @staticmethod
    def _scenario_seedvr2_noise(*, variant: str, run: int, work_dir: Path) -> dict[str, Any]:
        del work_dir
        seed = 9000 + run
        latent_height = 128
        latent_width = 128
        latent_channels = 16
        total_latent_frames = 160
        target_input_frame_count = 29
        chunk_starts = list(range(0, 96, 16))
        estimated_bytes = latent_channels * total_latent_frames * latent_height * latent_width * 4
        global_noise = None
        max_global_noise_bytes = SeedVR2StreamedVideoNoiseProvider.DEFAULT_MAX_GLOBAL_NOISE_BYTES
        if variant == "baseline":
            global_noise = SeedVR2LatentCreator.create_noise_latents(
                seed=seed,
                height=latent_height,
                width=latent_width,
                num_frames=total_latent_frames,
                latent_channels=latent_channels,
            )
            mx.eval(global_noise)
        else:
            max_global_noise_bytes = 1
        provider = SeedVR2StreamedVideoNoiseProvider(
            seed=seed,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_channels=latent_channels,
            total_latent_frames=total_latent_frames,
            estimated_global_noise_bytes=estimated_bytes,
            max_global_noise_bytes=max_global_noise_bytes,
            global_noise=global_noise,
        )
        checksums = []
        overlap_equal = True
        previous = None
        previous_start = None
        for start in chunk_starts:
            current = provider.slice(chunk_start_frame=start, target_input_frame_count=target_input_frame_count)
            mx.eval(current)
            checksums.append(round(float(mx.sum(current).item()), 4))
            if previous is not None and previous_start is not None:
                previous_abs_start = previous_start // 4
                current_abs_start = start // 4
                overlap = max(0, previous_abs_start + int(previous.shape[2]) - current_abs_start)
                if overlap > 0:
                    lhs = previous[:, :, -overlap:, :, :]
                    rhs = current[:, :, :overlap, :, :]
                    mx.eval(lhs, rhs)
                    overlap_equal = overlap_equal and bool(mx.allclose(lhs, rhs).item())
            previous = current
            previous_start = start
            if variant == "optimized":
                del current
                mx.clear_cache()
        metadata = provider.metadata()
        del provider
        if global_noise is not None:
            del global_noise
        return {
            "metadata": metadata,
            "estimated_global_noise_bytes": estimated_bytes,
            "chunk_starts": chunk_starts,
            "target_input_frame_count": target_input_frame_count,
            "checksums": checksums,
            "overlap_equal": overlap_equal,
            "proof": "baseline keeps full clip-global noise; optimized creates only requested absolute-frame slices.",
        }

    @staticmethod
    def _scenario_cache_limit_policy(*, variant: str, run: int, work_dir: Path) -> dict[str, Any]:
        del work_dir
        del run
        if variant == "optimized":
            RuntimeMemory.apply_mlx_cache_limit(0.02, low_ram=False)
        else:
            mx.set_cache_limit(4 * 1000**3)
            mx.clear_cache()
            mx.reset_peak_memory()
        elements = (64 * 1000 * 1000) // 4
        for _ in range(8):
            tensor = mx.ones((elements,), dtype=mx.float32)
            mx.eval(tensor)
            del tensor
        cache_memory = int(mx.get_cache_memory())
        return {
            "cache_limit_mode": "20MB" if variant == "optimized" else "4GB",
            "allocated_temporary_bytes": elements * 4 * 8,
            "cache_memory_bytes": cache_memory,
            "proof": "optimized applies a small MLX cache limit before repeated temporary allocations.",
        }

    @staticmethod
    def _scenario_stepwise_retention(*, variant: str, run: int, work_dir: Path) -> dict[str, Any]:
        del work_dir
        model_config = ModelConfig.dev()
        retained = []
        frames = 48
        width = 768
        height = 512
        for index in range(frames):
            image = MemoryOptimizationBenchmark._pil_frame(index, width=width, height=height)
            if variant == "baseline":
                retained.append(
                    GeneratedImage(
                        image=image,
                        model_config=model_config,
                        seed=run,
                        prompt="memory benchmark",
                        steps=frames,
                        guidance=3.5,
                        precision=ModelConfig.precision,
                        quantization=8,
                        generation_time=0.0,
                        extra_metadata={"step": index, "debug": "x" * 2048},
                    )
                )
            else:
                retained.append(image.copy())
        digest = hashlib.sha256()
        for item in retained:
            image = item.image if isinstance(item, GeneratedImage) else item
            digest.update(np.asarray(image.getdata(), dtype=np.uint8).tobytes())
        return {
            "retained_items": len(retained),
            "width": width,
            "height": height,
            "retained_mode": "GeneratedImage" if variant == "baseline" else "PIL.Image",
            "content_sha256": digest.hexdigest(),
            "proof": "baseline mirrors old StepwiseHandler object retention; optimized retains only PIL images.",
        }

    @staticmethod
    def _scenario_runtime_snapshot_overhead(*, variant: str, run: int, work_dir: Path) -> dict[str, Any]:
        del work_dir
        del run
        iterations = 80
        payload = mx.ones((4_000_000,), dtype=mx.float32)
        mx.eval(payload)
        started = time.perf_counter()
        for index in range(iterations):
            payload = payload + 0.0001
            if variant == "optimized":
                RuntimeMemory.snapshot(f"telemetry-{index}", tensors=(payload,), synchronize=False)
            elif index % 10 == 0:
                mx.eval(payload)
        mx.eval(payload)
        loop_seconds = time.perf_counter() - started
        checksum = float(mx.sum(payload).item())
        return {
            "iterations": iterations,
            "loop_seconds": loop_seconds,
            "checksum": checksum,
            "proof": "baseline performs no telemetry snapshots; optimized measures snapshot overhead.",
        }

    @staticmethod
    def _decoded_latents(*, frame_start: int, frame_count: int, height: int, width: int) -> mx.array:
        elements = frame_count * height * width * 3
        offset = frame_start * height * width * 3
        values = mx.arange(offset, offset + elements, dtype=mx.float32)
        values = ((values % 511) / 255.5) - 1.0
        return mx.reshape(values, (1, 3, frame_count, height, width))

    @staticmethod
    def _pil_frame(index: int, *, width: int, height: int) -> PIL.Image.Image:
        x = np.arange(width, dtype=np.uint16)[None, :]
        y = np.arange(height, dtype=np.uint16)[:, None]
        red = ((x + index * 3) % 256).astype(np.uint8)
        green = ((y + index * 5) % 256).astype(np.uint8)
        blue = (((x // 2) + (y // 3) + index * 7) % 256).astype(np.uint8)
        rgb = np.stack(
            [
                np.broadcast_to(red, (height, width)),
                np.broadcast_to(green, (height, width)),
                blue,
            ],
            axis=2,
        )
        return PIL.Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _build_report(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for result in raw_results:
            grouped.setdefault(result["scenario"], {}).setdefault(result["variant"], []).append(result)

        scenarios = {}
        for scenario, variants in grouped.items():
            scenario_meta = SCENARIOS[scenario]
            baseline = variants["baseline"]
            optimized = variants["optimized"]
            metrics = sorted({key for result in baseline + optimized for key in result["metrics"]})
            metric_summary = {}
            for metric in metrics:
                baseline_values = MemoryOptimizationBenchmark._metric_values(baseline, metric)
                optimized_values = MemoryOptimizationBenchmark._metric_values(optimized, metric)
                metric_summary[metric] = MemoryOptimizationBenchmark._compare_values(baseline_values, optimized_values)
            primary_metric = scenario_meta["metric"]
            primary = metric_summary[primary_metric]
            threshold = scenario_meta["threshold_percent"]
            if threshold >= 0:
                passed = primary["improvement_percent"] is not None and primary["improvement_percent"] >= threshold
            else:
                passed = primary["improvement_percent"] is not None and primary["improvement_percent"] >= threshold
            scenarios[scenario] = {
                "item": scenario_meta["item"],
                "description": scenario_meta["description"],
                "primary_metric": primary_metric,
                "threshold_percent": threshold,
                "passed": passed,
                "metrics": metric_summary,
                "runs": {
                    "baseline": baseline,
                    "optimized": optimized,
                },
            }

        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "method": {
                "process_isolation": "Each scenario/variant/run executes in a fresh Python process.",
                "sampling": "Child process samples RuntimeMemory snapshots at a fixed interval and records MLX peak counters.",
                "scope": "Quantifies representative memory mechanisms; full-model public claims still require model-backed generation runs.",
            },
            "scenarios": scenarios,
        }

    @staticmethod
    def _metric_values(results: list[dict[str, Any]], metric: str) -> list[float]:
        values = []
        for result in results:
            if metric == "wall_seconds":
                value = result.get("wall_seconds")
            else:
                value = result["metrics"].get(metric)
            if value is not None:
                values.append(float(value))
        return values

    @staticmethod
    def _compare_values(baseline_values: list[float], optimized_values: list[float]) -> dict[str, Any]:
        baseline_mean = mean(baseline_values) if baseline_values else None
        optimized_mean = mean(optimized_values) if optimized_values else None
        improvement = None
        if baseline_mean and optimized_mean is not None:
            improvement = ((baseline_mean - optimized_mean) / baseline_mean) * 100
        return {
            "baseline": MemoryOptimizationBenchmark._value_stats(baseline_values),
            "optimized": MemoryOptimizationBenchmark._value_stats(optimized_values),
            "improvement_percent": round(improvement, 2) if improvement is not None and math.isfinite(improvement) else None,
        }

    @staticmethod
    def _value_stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": round(mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    @staticmethod
    def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for scenario, data in report["scenarios"].items():
            primary_metric = data["primary_metric"]
            primary = data["metrics"][primary_metric]
            compact[scenario] = {
                "item": data["item"],
                "primary_metric": primary_metric,
                "baseline_mean": primary["baseline"]["mean"],
                "optimized_mean": primary["optimized"]["mean"],
                "improvement_percent": primary["improvement_percent"],
                "threshold_percent": data["threshold_percent"],
                "passed": data["passed"],
            }
        return compact


if __name__ == "__main__":
    MemoryOptimizationBenchmark.main()
