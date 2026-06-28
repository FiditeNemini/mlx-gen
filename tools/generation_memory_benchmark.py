import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import PIL.Image

from mflux.utils.video_util import VideoUtil

BYTES_PER_GB = 1000**3


@dataclass(frozen=True)
class CommandVariant:
    name: str
    argv: list[str]
    output_path: Path
    cwd: Path
    env: dict[str, str] | None = None


class ProcessTreeSampler:
    def __init__(self, root_pid: int, interval_seconds: float):
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ProcessTreeSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.sample("final")

    def sample(self, phase: str = "sample") -> None:
        processes = self._process_rows()
        descendants = self._descendants(processes)
        physical_by_pid = {
            pid: self._darwin_physical_footprint_bytes(pid)
            for pid in descendants
        }
        rss_bytes = sum(processes[pid]["rss_bytes"] for pid in descendants)
        physical_bytes = sum(physical_by_pid[pid] or 0 for pid in descendants)
        self.samples.append(
            {
                "phase": phase,
                "timestamp": time.time(),
                "pids": sorted(descendants),
                "process_count": len(descendants),
                "rss_bytes": rss_bytes,
                "darwin_physical_footprint_bytes": physical_bytes or None,
                "processes": {
                    str(pid): {
                        "rss_bytes": processes[pid]["rss_bytes"],
                        "darwin_physical_footprint_bytes": physical_by_pid[pid],
                    }
                    for pid in sorted(descendants)
                },
            }
        )

    def summary(self) -> dict[str, int | None]:
        return {
            "peak_sampled_rss_bytes": self._max("rss_bytes"),
            "peak_sampled_darwin_physical_footprint_bytes": self._max("darwin_physical_footprint_bytes"),
            "sample_count": len(self.samples),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval_seconds)

    def _max(self, key: str) -> int | None:
        values = [sample[key] for sample in self.samples if sample.get(key) is not None]
        return int(max(values)) if values else None

    def _process_rows(self) -> dict[int, dict[str, int | None]]:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
        rows = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                pid, ppid, rss_kb = (int(part) for part in parts)
            except ValueError:
                continue
            rows[pid] = {
                "ppid": ppid,
                "rss_bytes": rss_kb * 1024,
            }
        return rows

    def _descendants(self, processes: dict[int, dict[str, int | None]]) -> set[int]:
        descendants = {self.root_pid}
        changed = True
        while changed:
            changed = False
            for pid, row in processes.items():
                if pid not in descendants and row["ppid"] in descendants:
                    descendants.add(pid)
                    changed = True
        return {pid for pid in descendants if pid in processes}

    @staticmethod
    def _darwin_physical_footprint_bytes(pid: int) -> int | None:
        if sys.platform != "darwin" or os.environ.get("MFLUX_BENCHMARK_PARENT_PHYSICAL_SAMPLING") != "1":
            return None
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    ProcessTreeSampler._DARWIN_PHYSICAL_FOOTPRINT_HELPER,
                    str(pid),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError, TimeoutError, TypeError, ValueError):
            return None
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    _DARWIN_PHYSICAL_FOOTPRINT_HELPER = r"""
import ctypes
import sys


class RUsageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


info = RUsageInfoV4()
rc = ctypes.CDLL("libproc.dylib").proc_pid_rusage(int(sys.argv[1]), 4, ctypes.byref(info))
if rc != 0:
    raise SystemExit(rc)
print(int(info.ri_phys_footprint))
"""

class GenerationMemoryBenchmark:
    @staticmethod
    def main() -> None:
        args = GenerationMemoryBenchmark._parse_args()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_names = args.profiles or ["zimage-cache-policy"]
        variants_by_profile = {
            profile: GenerationMemoryBenchmark._profile_variants(profile, output_dir=output_dir)
            for profile in profile_names
        }
        results = []
        for profile, variants in variants_by_profile.items():
            for run_index in range(1, args.runs + 1):
                results.extend(
                    [
                        GenerationMemoryBenchmark._run_variant(
                            profile=profile,
                            variant=variant,
                            run_index=run_index,
                            output_dir=output_dir,
                            sample_interval_ms=args.sample_interval_ms,
                        )
                        for variant in variants
                    ]
                )

        report = GenerationMemoryBenchmark._build_report(results)
        report_path = output_dir / "generation_memory_benchmark.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(json.dumps(GenerationMemoryBenchmark._compact_report(report), indent=2, sort_keys=True))
        print(f"wrote {report_path}")

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("validation_outputs/memory/real_generation"))
        parser.add_argument("--runs", type=int, default=1)
        parser.add_argument("--sample-interval-ms", type=int, default=200)
        parser.add_argument(
            "--profiles",
            nargs="*",
            choices=[
                "wan-streamed-decode",
                "zimage-cache-policy",
                "flux2-lowram-stepwise",
                "seedvr2-noise-mode",
                "seedvr2-image-3b-1280",
                "seedvr2-image-7b-1280",
                "seedvr2-image-3b-1280-vae-tiling",
                "seedvr2-image-7b-1280-vae-tiling",
                "telemetry-overhead",
                "flux2-prompt-materialization",
                "ernie-prompt-materialization",
                "flux2-hidden-state-retention",
                "flux2-stepwise-retention",
                "seedvr2-video-noise-29f",
                "zimage-runtime-telemetry",
                "zimage-hidden-state-retention",
                "seedvr2-video-3b-bounded-scaling",
                "seedvr2-video-3b-mode-149",
                "seedvr2-video-7b-bounded-scaling",
                "seedvr2-video-3b-149-restoration",
                "seedvr2-video-7b-149-restoration",
            ],
        )
        args = parser.parse_args()
        if args.runs <= 0:
            raise ValueError("--runs must be greater than zero.")
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be greater than zero.")
        return args

    @staticmethod
    def _profile_variants(profile: str, *, output_dir: Path) -> list[CommandVariant]:
        cwd = Path.cwd()
        if profile == "wan-streamed-decode":
            base = [
                "uv",
                "run",
                "mflux-generate-wan",
                "--model",
                "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
                "--prompt",
                "A tiny clockwork robot walks across a wooden desk, stable camera",
                "--width",
                "320",
                "--height",
                "192",
                "--frames",
                "9",
                "--fps",
                "8",
                "--steps",
                "1",
                "--guidance",
                "1",
                "--flow-shift",
                "3",
                "--seed",
                "4217",
                "--metadata",
                "--replace",
                "--no-progress",
            ]
            return [
                CommandVariant(
                    name="eager",
                    argv=base + ["--output", str(output_dir / "wan_eager.mp4")],
                    output_path=output_dir / "wan_eager.mp4",
                    cwd=cwd,
                ),
                CommandVariant(
                    name="streamed_low_ram",
                    argv=base + ["--low-ram", "--output", str(output_dir / "wan_streamed_low_ram.mp4")],
                    output_path=output_dir / "wan_streamed_low_ram.mp4",
                    cwd=cwd,
                ),
            ]
        if profile == "zimage-cache-policy":
            base = [
                "uv",
                "run",
                "mflux-generate-z-image-turbo",
                "--model",
                "AbstractFramework/z-image-turbo-8bit",
                "--prompt",
                "A red enamel kettle on a steel counter, product photo",
                "--width",
                "384",
                "--height",
                "384",
                "--steps",
                "2",
                "--seed",
                "5151",
                "--metadata",
                "--replace",
                "--no-progress",
            ]
            return [
                CommandVariant(
                    name="cache_default",
                    argv=base + ["--output", str(output_dir / "zimage_cache_default.png")],
                    output_path=output_dir / "zimage_cache_default.png",
                    cwd=cwd,
                ),
                CommandVariant(
                    name="cache_limited",
                    argv=base
                    + ["--mlx-cache-limit-gb", "1", "--output", str(output_dir / "zimage_cache_limited.png")],
                    output_path=output_dir / "zimage_cache_limited.png",
                    cwd=cwd,
                ),
            ]
        if profile in {"telemetry-overhead", "zimage-runtime-telemetry"}:
            base = [
                "uv",
                "run",
                "mflux-generate-z-image-turbo",
                "--model",
                "AbstractFramework/z-image-turbo-8bit",
                "--prompt",
                "A red enamel kettle on a steel counter, product photo",
                "--width",
                "384",
                "--height",
                "384",
                "--steps",
                "2",
                "--seed",
                "5151",
                "--metadata",
                "--replace",
                "--no-progress",
            ]
            return [
                CommandVariant(
                    name="telemetry_disabled",
                    argv=base + ["--output", str(output_dir / "telemetry_disabled.png")],
                    output_path=output_dir / "telemetry_disabled.png",
                    cwd=cwd,
                    env={"MFLUX_RUNTIME_MEMORY_TELEMETRY": "0"},
                ),
                CommandVariant(
                    name="telemetry_enabled",
                    argv=base + ["--output", str(output_dir / "telemetry_enabled.png")],
                    output_path=output_dir / "telemetry_enabled.png",
                    cwd=cwd,
                ),
            ]
        if profile == "flux2-lowram-stepwise":
            return GenerationMemoryBenchmark._flux2_profile_variants(
                output_dir=output_dir,
                output_prefix="flux2",
                candidate_name="low_ram_stepwise",
                candidate_args=[
                    "--low-ram",
                    "--stepwise-image-output-dir",
                    str(output_dir / "flux2_stepwise"),
                ],
            )
        if profile == "flux2-prompt-materialization":
            return GenerationMemoryBenchmark._flux2_profile_variants(
                output_dir=output_dir,
                output_prefix="flux2_prompt_materialization",
                base_args=["--low-ram"],
                default_name="materialized",
                candidate_name="legacy_lazy_prompt",
                candidate_args=["--low-ram"],
                candidate_env=GenerationMemoryBenchmark._internal_benchmark_env(
                    "disable_prompt_materialization"
                ),
            )
        if profile == "ernie-prompt-materialization":
            return GenerationMemoryBenchmark._ernie_prompt_materialization_variants(output_dir=output_dir)
        if profile == "flux2-hidden-state-retention":
            return GenerationMemoryBenchmark._flux2_profile_variants(
                output_dir=output_dir,
                output_prefix="flux2_hidden_retention",
                default_name="selected_hidden_states",
                candidate_name="legacy_all_hidden_states",
                candidate_env=GenerationMemoryBenchmark._internal_benchmark_env(
                    "legacy_hidden_state_retention"
                ),
            )
        if profile == "flux2-stepwise-retention":
            return GenerationMemoryBenchmark._flux2_profile_variants(
                output_dir=output_dir,
                output_prefix="flux2_stepwise_retention",
                prompt="A ceramic teapot on a wooden table, soft studio light",
                width=512,
                height=512,
                steps=12,
                seed=6464,
                default_name="pil_retention",
                base_args=[
                    "--stepwise-image-output-dir",
                    str(output_dir / "flux2_stepwise_pil"),
                ],
                candidate_name="legacy_generated_image_retention",
                candidate_args=[
                    "--stepwise-image-output-dir",
                    str(output_dir / "flux2_stepwise_legacy"),
                ],
                candidate_env=GenerationMemoryBenchmark._internal_benchmark_env(
                    "legacy_stepwise_retention"
                ),
            )
        if profile == "zimage-hidden-state-retention":
            return GenerationMemoryBenchmark._zimage_hidden_state_variants(output_dir=output_dir)
        if profile == "seedvr2-noise-mode":
            source_video = Path(
                "validation_outputs/seedvr2_video_2026_06_21_pass1/"
                "eiffel_74s_29f_3b_chunk29_overlap0_wavelet_2x_tailprobe.mp4"
            ).resolve()
            base = [
                "uv",
                "run",
                "mlxgen",
                "upscale",
                "--model",
                "AbstractFramework/seedvr2-3b-4bit",
                "--video-path",
                str(source_video),
                "--resolution",
                "1x",
                "--max-frames",
                "9",
                "--temporal-chunk-size",
                "9",
                "--temporal-chunk-overlap",
                "0",
                "--seed",
                "7171",
                "--metadata",
                "--replace",
                "--no-progress",
            ]
            return [
                CommandVariant(
                    name="global_noise",
                    argv=base + ["--output", str(output_dir / "seedvr2_global_noise.mp4")],
                    output_path=output_dir / "seedvr2_global_noise.mp4",
                    cwd=cwd,
                    env=GenerationMemoryBenchmark._seedvr2_noise_env(10**12),
                ),
                CommandVariant(
                    name="bounded_noise",
                    argv=base + ["--output", str(output_dir / "seedvr2_bounded_noise.mp4")],
                    output_path=output_dir / "seedvr2_bounded_noise.mp4",
                    cwd=cwd,
                    env=GenerationMemoryBenchmark._seedvr2_noise_env(0),
                ),
            ]
        if profile == "seedvr2-video-noise-29f":
            source_video = Path(
                "validation_outputs/seedvr2_video_2026_06_21_pass1/"
                "eiffel_74s_29f_3b_chunk29_overlap0_wavelet_2x_tailprobe.mp4"
            ).resolve()
            base = [
                "uv",
                "run",
                "mlxgen",
                "upscale",
                "--model",
                "AbstractFramework/seedvr2-3b-4bit",
                "--video-path",
                str(source_video),
                "--resolution",
                "1x",
                "--max-frames",
                "29",
                "--temporal-chunk-size",
                "17",
                "--temporal-chunk-overlap",
                "4",
                "--seed",
                "7272",
                "--metadata",
                "--replace",
                "--no-progress",
                "--drop-audio",
            ]
            return [
                CommandVariant(
                    name="global_noise",
                    argv=base + ["--output", str(output_dir / "seedvr2_29f_global_noise.mp4")],
                    output_path=output_dir / "seedvr2_29f_global_noise.mp4",
                    cwd=cwd,
                    env={**GenerationMemoryBenchmark._seedvr2_noise_env(10**12), "HF_HUB_OFFLINE": "1"},
                ),
                CommandVariant(
                    name="bounded_noise",
                    argv=base + ["--output", str(output_dir / "seedvr2_29f_bounded_noise.mp4")],
                    output_path=output_dir / "seedvr2_29f_bounded_noise.mp4",
                    cwd=cwd,
                    env={**GenerationMemoryBenchmark._seedvr2_noise_env(0), "HF_HUB_OFFLINE": "1"},
                ),
            ]
        if profile == "seedvr2-video-3b-bounded-scaling":
            return GenerationMemoryBenchmark._seedvr2_video_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-3b-4bit",
                output_prefix="seedvr2_3b_bounded_scaling",
                variants=[
                    ("bounded_53f", 53, 29, 8, GenerationMemoryBenchmark._seedvr2_noise_env(0)),
                    ("bounded_149f", 149, 29, 8, GenerationMemoryBenchmark._seedvr2_noise_env(0)),
                ],
            )
        if profile == "seedvr2-video-3b-mode-149":
            return GenerationMemoryBenchmark._seedvr2_video_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-3b-4bit",
                output_prefix="seedvr2_3b_mode_149",
                variants=[
                    ("global_149f", 149, 29, 8, GenerationMemoryBenchmark._seedvr2_noise_env(10**12)),
                    ("bounded_149f", 149, 29, 8, GenerationMemoryBenchmark._seedvr2_noise_env(0)),
                ],
            )
        if profile == "seedvr2-video-7b-bounded-scaling":
            return GenerationMemoryBenchmark._seedvr2_video_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-7b-4bit",
                output_prefix="seedvr2_7b_bounded_scaling",
                variants=[
                    ("bounded_53f", 53, 29, 8, GenerationMemoryBenchmark._seedvr2_noise_env(0)),
                    ("bounded_101f", 101, 29, 8, GenerationMemoryBenchmark._seedvr2_noise_env(0)),
                ],
            )
        if profile == "seedvr2-video-3b-149-restoration":
            return GenerationMemoryBenchmark._seedvr2_video_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-3b-4bit",
                output_prefix="seedvr2_3b_149_restoration",
                variants=[
                    ("restored_149f", 149, 29, 8, {}),
                ],
            )
        if profile == "seedvr2-video-7b-149-restoration":
            return GenerationMemoryBenchmark._seedvr2_video_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-7b-4bit",
                output_prefix="seedvr2_7b_149_restoration",
                variants=[
                    ("restored_149f", 149, 29, 8, {}),
                ],
            )
        if profile == "seedvr2-image-3b-1280":
            return GenerationMemoryBenchmark._seedvr2_image_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-3b-4bit",
                output_prefix="seedvr2_3b_1280",
            )
        if profile == "seedvr2-image-7b-1280":
            return GenerationMemoryBenchmark._seedvr2_image_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-7b-4bit",
                output_prefix="seedvr2_7b_1280",
            )
        if profile == "seedvr2-image-3b-1280-vae-tiling":
            return GenerationMemoryBenchmark._seedvr2_image_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-3b-4bit",
                output_prefix="seedvr2_3b_1280_vae_tiling",
                candidate_name="vae_tiling",
                candidate_args=["--vae-tiling"],
            )
        if profile == "seedvr2-image-7b-1280-vae-tiling":
            return GenerationMemoryBenchmark._seedvr2_image_variants(
                output_dir=output_dir,
                model_name="AbstractFramework/seedvr2-7b-4bit",
                output_prefix="seedvr2_7b_1280_vae_tiling",
                candidate_name="vae_tiling",
                candidate_args=["--vae-tiling"],
            )
        raise ValueError(f"Unknown profile: {profile}")

    @staticmethod
    def _flux2_profile_variants(
        *,
        output_dir: Path,
        output_prefix: str,
        default_name: str = "default",
        base_args: list[str] | None = None,
        candidate_name: str = "candidate",
        candidate_args: list[str] | None = None,
        candidate_env: dict[str, str] | None = None,
        prompt: str = "A ceramic teapot on a wooden table, soft studio light",
        width: int = 384,
        height: int = 384,
        steps: int = 4,
        seed: int = 6161,
    ) -> list[CommandVariant]:
        cwd = Path.cwd()
        offline_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        base = [
            "uv",
            "run",
            "mflux-generate-flux2",
            "--model",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--steps",
            str(steps),
            "--guidance",
            "1",
            "--seed",
            str(seed),
            "--metadata",
            "--replace",
            "--no-progress",
        ]
        return [
            CommandVariant(
                name=default_name,
                argv=base + (base_args or []) + ["--output", str(output_dir / f"{output_prefix}_{default_name}.png")],
                output_path=output_dir / f"{output_prefix}_{default_name}.png",
                cwd=cwd,
                env=offline_env,
            ),
            CommandVariant(
                name=candidate_name,
                argv=base
                + (candidate_args or [])
                + ["--output", str(output_dir / f"{output_prefix}_{candidate_name}.png")],
                output_path=output_dir / f"{output_prefix}_{candidate_name}.png",
                cwd=cwd,
                env={**offline_env, **(candidate_env or {})},
            ),
        ]

    @staticmethod
    def _zimage_hidden_state_variants(*, output_dir: Path) -> list[CommandVariant]:
        cwd = Path.cwd()
        offline_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        prompt = (
            "highly detailed studio product photograph, clean typography, complex materials, "
            "transparent glass, brushed steel, readable labels, layered reflections, "
        ) * 48
        base = [
            "uv",
            "run",
            "mflux-generate-z-image-turbo",
            "--model",
            "AbstractFramework/z-image-turbo-8bit",
            "--prompt",
            prompt,
            "--negative-prompt",
            " ",
            "--width",
            "384",
            "--height",
            "384",
            "--steps",
            "2",
            "--guidance",
            "1",
            "--seed",
            "6465",
            "--metadata",
            "--replace",
            "--no-progress",
        ]
        return [
            CommandVariant(
                name="previous_hidden_state",
                argv=base + ["--output", str(output_dir / "zimage_previous_hidden_state.png")],
                output_path=output_dir / "zimage_previous_hidden_state.png",
                cwd=cwd,
                env=offline_env,
            ),
            CommandVariant(
                name="legacy_all_hidden_states",
                argv=base + ["--output", str(output_dir / "zimage_legacy_all_hidden_states.png")],
                output_path=output_dir / "zimage_legacy_all_hidden_states.png",
                cwd=cwd,
                env={
                    **offline_env,
                    **GenerationMemoryBenchmark._internal_benchmark_env("legacy_hidden_state_retention"),
                },
            ),
        ]

    @staticmethod
    def _ernie_prompt_materialization_variants(*, output_dir: Path) -> list[CommandVariant]:
        cwd = Path.cwd()
        offline_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        base = [
            "uv",
            "run",
            "mflux-generate-ernie-image",
            "--model",
            "AbstractFramework/ernie-image-turbo-8bit",
            "--prompt",
            "A detailed editorial product image of a translucent mechanical keyboard on a dark acrylic table",
            "--width",
            "384",
            "--height",
            "384",
            "--steps",
            "2",
            "--guidance",
            "1",
            "--seed",
            "6262",
            "--low-ram",
            "--metadata",
            "--replace",
            "--no-progress",
        ]
        return [
            CommandVariant(
                name="materialized",
                argv=base + ["--output", str(output_dir / "ernie_materialized.png")],
                output_path=output_dir / "ernie_materialized.png",
                cwd=cwd,
                env=offline_env,
            ),
            CommandVariant(
                name="legacy_lazy_prompt",
                argv=base + ["--output", str(output_dir / "ernie_legacy_lazy_prompt.png")],
                output_path=output_dir / "ernie_legacy_lazy_prompt.png",
                cwd=cwd,
                env={
                    **offline_env,
                    **GenerationMemoryBenchmark._internal_benchmark_env("disable_prompt_materialization"),
                },
            ),
        ]

    @staticmethod
    def _internal_benchmark_env(*flags: str) -> dict[str, str]:
        return {
            "MFLUX_INTERNAL_MEMORY_BENCHMARK_MODE": "1",
            "MFLUX_INTERNAL_MEMORY_BENCHMARK_FLAGS": ",".join(flags),
        }

    @staticmethod
    def _seedvr2_noise_env(max_global_noise_bytes: int) -> dict[str, str]:
        return {
            **GenerationMemoryBenchmark._internal_benchmark_env("seedvr2_max_global_noise_bytes"),
            "MFLUX_INTERNAL_SEEDVR2_MAX_GLOBAL_NOISE_BYTES": str(max_global_noise_bytes),
        }

    @staticmethod
    def _seedvr2_video_variants(
        *,
        output_dir: Path,
        model_name: str,
        output_prefix: str,
        variants: list[tuple[str, int, int, int, dict[str, str]]],
    ) -> list[CommandVariant]:
        cwd = Path.cwd()
        source_video = Path("validation_outputs/seedvr2_video_2026_06_20/eiffel_70s_149f_source.mp4").resolve()
        offline_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        commands = []
        for name, max_frames, chunk_size, chunk_overlap, env in variants:
            base = [
                "uv",
                "run",
                "mlxgen",
                "upscale",
                "--model",
                model_name,
                "--video-path",
                str(source_video),
                "--resolution",
                "1x",
                "--max-frames",
                str(max_frames),
                "--temporal-chunk-size",
                str(chunk_size),
                "--temporal-chunk-overlap",
                str(chunk_overlap),
                "--seed",
                "7272",
                "--metadata",
                "--replace",
                "--no-progress",
                "--drop-audio",
            ]
            commands.append(
                CommandVariant(
                    name=name,
                    argv=base + ["--output", str(output_dir / f"{output_prefix}_{name}.mp4")],
                    output_path=output_dir / f"{output_prefix}_{name}.mp4",
                    cwd=cwd,
                    env={**offline_env, **env},
                )
            )
        return commands

    @staticmethod
    def _seedvr2_image_variants(
        *,
        output_dir: Path,
        model_name: str,
        output_prefix: str,
        candidate_name: str = "low_ram",
        candidate_args: list[str] | None = None,
    ) -> list[CommandVariant]:
        cwd = Path.cwd()
        source_image = GenerationMemoryBenchmark._seedvr2_source_image(output_dir)
        offline_env = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        base = [
            "uv",
            "run",
            "mlxgen",
            "upscale",
            "--model",
            model_name,
            "--image-path",
            str(source_image),
            "--resolution",
            "1280",
            "--seed",
            "8181",
            "--metadata",
            "--replace",
            "--no-progress",
        ]
        return [
            CommandVariant(
                name="default",
                argv=base + ["--output", str(output_dir / f"{output_prefix}_default.png")],
                output_path=output_dir / f"{output_prefix}_default.png",
                cwd=cwd,
                env=offline_env,
            ),
            CommandVariant(
                name=candidate_name,
                argv=base
                + (candidate_args or ["--low-ram"])
                + ["--output", str(output_dir / f"{output_prefix}_{candidate_name}.png")],
                output_path=output_dir / f"{output_prefix}_{candidate_name}.png",
                cwd=cwd,
                env=offline_env,
            ),
        ]

    @staticmethod
    def _seedvr2_source_image(output_dir: Path) -> Path:
        source_path = output_dir / "seedvr2_1280_source.png"
        if source_path.exists():
            return source_path
        size = 960
        y, x = np.mgrid[0:size, 0:size]
        image = np.empty((size, size, 3), dtype=np.uint8)
        image[..., 0] = (x * 255 // (size - 1)).astype(np.uint8)
        image[..., 1] = (y * 255 // (size - 1)).astype(np.uint8)
        image[..., 2] = (((x // 24 + y // 24) % 2) * 96 + 80).astype(np.uint8)
        PIL.Image.fromarray(image, mode="RGB").save(source_path)
        return source_path

    @staticmethod
    def _run_variant(
        *,
        profile: str,
        variant: CommandVariant,
        run_index: int,
        output_dir: Path,
        sample_interval_ms: int,
    ) -> dict[str, Any]:
        run_dir = output_dir / profile / f"run_{run_index}" / variant.name
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / variant.output_path.name
        argv = [str(part) for part in variant.argv]
        argv = GenerationMemoryBenchmark._replace_output_arg(argv, output_path)
        argv = GenerationMemoryBenchmark._replace_stepwise_output_dir(argv, run_dir / "stepwise")
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        started = time.perf_counter()
        timed_argv = ["/usr/bin/time", "-l", *argv]
        env = os.environ.copy()
        if variant.env:
            env.update(variant.env)
        process = subprocess.Popen(
            timed_argv,
            cwd=variant.cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with ProcessTreeSampler(process.pid, sample_interval_ms / 1000) as sampler:
            stdout, stderr = process.communicate()
        wall_seconds = time.perf_counter() - started
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        time_metrics = GenerationMemoryBenchmark._parse_time_l_output(stderr)
        result = {
            "profile": profile,
            "variant": variant.name,
            "run_index": run_index,
            "argv": timed_argv,
            "env_overrides": variant.env or {},
            "cwd": str(variant.cwd),
            "returncode": process.returncode,
            "wall_seconds": wall_seconds,
            "sampler": sampler.summary(),
            "samples": sampler.samples,
            "time_l": time_metrics,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "output_path": str(output_path),
            "output_exists": output_path.exists(),
            "output_sha256": GenerationMemoryBenchmark._sha256(output_path) if output_path.exists() else None,
            "metadata": GenerationMemoryBenchmark._load_metadata(output_path),
            "environment": GenerationMemoryBenchmark._environment(),
        }
        if process.returncode != 0:
            raise RuntimeError(f"{profile}/{variant.name} failed; see {stderr_path}")
        return result

    @staticmethod
    def _replace_output_arg(argv: list[str], output_path: Path) -> list[str]:
        replaced = list(argv)
        for index, value in enumerate(replaced):
            if value == "--output" and index + 1 < len(replaced):
                replaced[index + 1] = str(output_path)
                return replaced
        return replaced + ["--output", str(output_path)]

    @staticmethod
    def _replace_stepwise_output_dir(argv: list[str], output_dir: Path) -> list[str]:
        replaced = list(argv)
        for index, value in enumerate(replaced):
            if value == "--stepwise-image-output-dir" and index + 1 < len(replaced):
                replaced[index + 1] = str(output_dir)
                return replaced
        return replaced

    @staticmethod
    def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for result in results:
            grouped.setdefault(result["profile"], {}).setdefault(result["variant"], []).append(result)
        profiles = {}
        for profile, variants in grouped.items():
            comparisons = {}
            variant_names = list(variants)
            if len(variant_names) == 2:
                comparisons["quality"] = GenerationMemoryBenchmark._quality_comparison(
                    profile=profile,
                    left=Path(variants[variant_names[0]][0]["output_path"]),
                    right=Path(variants[variant_names[1]][0]["output_path"]),
                )
                comparisons["memory"] = GenerationMemoryBenchmark._memory_comparison(
                    variants[variant_names[0]],
                    variants[variant_names[1]],
                )
            elif len(variant_names) == 1:
                output_path = Path(variants[variant_names[0]][0]["output_path"])
                if output_path.suffix.lower() in {".mp4", ".mov"}:
                    comparisons["quality"] = {
                        "status": "ok",
                        "kind": "single_video",
                        "variant": variant_names[0],
                        "video_health": GenerationMemoryBenchmark._video_health(output_path),
                    }
            profiles[profile] = {
                "variants": {
                    name: GenerationMemoryBenchmark._summarize_variant(runs)
                    for name, runs in variants.items()
                },
                "comparisons": comparisons,
                "raw_runs": variants,
            }
        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "schema_version": 1,
            "method": {
                "process_isolation": "Each variant runs as an external CLI process.",
                "memory": (
                    "Parent samples process-tree RSS while the CLI runs; child metadata records MLX memory, "
                    "process RSS, and Darwin physical footprint at the runtime snapshot point."
                ),
                "quality": "Image/video outputs are compared for dimensions, frame count, MAE/RMSE/max error, and PSNR.",
            },
            "profiles": profiles,
        }

    @staticmethod
    def _summarize_variant(runs: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = {
            "wall_seconds": [run["wall_seconds"] for run in runs],
            "peak_sampled_rss_bytes": [run["sampler"]["peak_sampled_rss_bytes"] for run in runs],
            "peak_sampled_darwin_physical_footprint_bytes": [
                run["sampler"].get("peak_sampled_darwin_physical_footprint_bytes") for run in runs
            ],
            "time_l_peak_memory_footprint_bytes": [
                run["time_l"].get("peak_memory_footprint_bytes") for run in runs
            ],
            "time_l_maximum_resident_set_size_bytes": [
                run["time_l"].get("maximum_resident_set_size_bytes") for run in runs
            ],
            "mlx_peak_memory_bytes": [
                ((run.get("metadata") or {}).get("runtime_memory") or {}).get("mlx_peak_memory_bytes")
                for run in runs
            ],
            "mlx_cache_memory_bytes": [
                ((run.get("metadata") or {}).get("runtime_memory") or {}).get("mlx_cache_memory_bytes")
                for run in runs
            ],
            "metadata_process_rss_bytes": [
                ((run.get("metadata") or {}).get("runtime_memory") or {}).get("process_rss_bytes")
                for run in runs
            ],
            "metadata_darwin_physical_footprint_bytes": [
                ((run.get("metadata") or {}).get("runtime_memory") or {}).get("darwin_physical_footprint_bytes")
                for run in runs
            ],
        }
        return {
            key: GenerationMemoryBenchmark._stats([value for value in values if value is not None])
            for key, values in metrics.items()
        }

    @staticmethod
    def _stats(values: list[float | int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": round(float(mean(values)), 4),
            "median": round(float(median(values)), 4),
            "min": round(float(min(values)), 4),
            "max": round(float(max(values)), 4),
        }

    @staticmethod
    def _quality_comparison(*, profile: str, left: Path, right: Path) -> dict[str, Any]:
        if profile in {
            "seedvr2-video-3b-bounded-scaling",
            "seedvr2-video-7b-bounded-scaling",
        }:
            return {
                "status": "ok",
                "kind": "video_scaling",
                "comparison": "skipped_intentional_frame_count_difference",
                "variants": {
                    "left": GenerationMemoryBenchmark._video_health(left),
                    "right": GenerationMemoryBenchmark._video_health(right),
                },
            }
        if left.suffix.lower() in {".mp4", ".mov"} or right.suffix.lower() in {".mp4", ".mov"}:
            return GenerationMemoryBenchmark._video_quality(left, right)
        return GenerationMemoryBenchmark._image_quality(left, right)

    @staticmethod
    def _image_quality(left: Path, right: Path) -> dict[str, Any]:
        left_image = PIL.Image.open(left).convert("RGB")
        right_image = PIL.Image.open(right).convert("RGB")
        if left_image.size != right_image.size:
            return {"status": "failed", "reason": "image_size_mismatch", "left_size": left_image.size, "right_size": right_image.size}
        return {"status": "ok", "kind": "image", **GenerationMemoryBenchmark._array_delta(
            np.asarray(left_image, dtype=np.float32),
            np.asarray(right_image, dtype=np.float32),
        )}

    @staticmethod
    def _video_quality(left: Path, right: Path) -> dict[str, Any]:
        left_clip = VideoUtil.read_video_clip(left)
        right_clip = VideoUtil.read_video_clip(right)
        if left_clip.clip_frame_count != right_clip.clip_frame_count:
            return {
                "status": "failed",
                "reason": "frame_count_mismatch",
                "left_frames": left_clip.clip_frame_count,
                "right_frames": right_clip.clip_frame_count,
            }
        if (left_clip.source_width, left_clip.source_height) != (right_clip.source_width, right_clip.source_height):
            return {
                "status": "failed",
                "reason": "video_size_mismatch",
                "left_size": [left_clip.source_width, left_clip.source_height],
                "right_size": [right_clip.source_width, right_clip.source_height],
            }
        deltas = [
            GenerationMemoryBenchmark._array_delta(
                np.asarray(left_frame.convert("RGB"), dtype=np.float32),
                np.asarray(right_frame.convert("RGB"), dtype=np.float32),
            )
            for left_frame, right_frame in zip(left_clip.frames, right_clip.frames)
        ]
        return {
            "status": "ok",
            "kind": "video",
            "frames": left_clip.clip_frame_count,
            "width": left_clip.source_width,
            "height": left_clip.source_height,
            "mae": round(float(mean(item["mae"] for item in deltas)), 4),
            "rmse": round(float(mean(item["rmse"] for item in deltas)), 4),
            "max_abs": round(float(max(item["max_abs"] for item in deltas)), 4),
            "psnr_db": round(float(mean(item["psnr_db"] for item in deltas if item["psnr_db"] is not None)), 4)
            if any(item["psnr_db"] is not None for item in deltas)
            else None,
            "left_temporal_continuity": GenerationMemoryBenchmark._video_temporal_continuity(left, left_clip),
            "right_temporal_continuity": GenerationMemoryBenchmark._video_temporal_continuity(right, right_clip),
        }

    @staticmethod
    def _video_health(path: Path) -> dict[str, Any]:
        clip = VideoUtil.read_video_clip(path)
        metadata = GenerationMemoryBenchmark._load_metadata(path) or {}
        expected_frames = GenerationMemoryBenchmark._expected_frame_count(path)
        status = "ok"
        reason = None
        if expected_frames is not None and clip.clip_frame_count != expected_frames:
            status = "failed"
            reason = "frame_count_mismatch"
        result = {
            "status": status,
            "kind": "video",
            "path": str(path),
            "frames": clip.clip_frame_count,
            "expected_frames": expected_frames,
            "width": clip.source_width,
            "height": clip.source_height,
            "fps": clip.fps,
            "metadata_temporal_chunk_count": metadata.get("temporal_chunk_count"),
            "metadata_temporal_chunk_size": metadata.get("temporal_chunk_size"),
            "metadata_temporal_chunk_overlap": metadata.get("temporal_chunk_overlap"),
            "temporal_continuity": GenerationMemoryBenchmark._video_temporal_continuity(path, clip),
        }
        if reason is not None:
            result["reason"] = reason
        return result

    @staticmethod
    def _expected_frame_count(path: Path) -> int | None:
        match = re.search(r"_(\d+)f(?:\.|_)", path.name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _video_temporal_continuity(path: Path, clip) -> dict[str, Any]:
        metadata = GenerationMemoryBenchmark._load_metadata(path) or {}
        plan = metadata.get("temporal_chunk_plan") or []
        frames = [np.asarray(frame.convert("L"), dtype=np.float32) for frame in clip.frames]
        if len(frames) < 2:
            return {"status": "skipped", "reason": "too_few_frames"}
        deltas = [
            float(np.mean(np.abs(frames[index] - frames[index - 1])))
            for index in range(1, len(frames))
        ]
        boundary_indices = []
        cursor = 0
        for chunk in plan[:-1]:
            cursor += int(chunk.get("output_frame_count", 0))
            if 0 < cursor < len(frames):
                boundary_indices.append(cursor)
        boundary_deltas = [deltas[index - 1] for index in boundary_indices if 0 <= index - 1 < len(deltas)]
        boundary_set = set(boundary_indices)
        non_boundary_deltas = [
            value
            for index, value in enumerate(deltas, start=1)
            if index not in boundary_set
        ]
        non_boundary_median = float(median(non_boundary_deltas)) if non_boundary_deltas else None
        non_boundary_p95 = float(np.percentile(non_boundary_deltas, 95)) if non_boundary_deltas else None
        boundary_median = float(median(boundary_deltas)) if boundary_deltas else None
        boundary_max = float(max(boundary_deltas)) if boundary_deltas else None
        return {
            "status": "ok",
            "frame_count": len(frames),
            "chunk_count": int(metadata.get("temporal_chunk_count") or len(plan) or 0),
            "boundary_indices": boundary_indices,
            "mean_temporal_mae": round(float(mean(deltas)), 4),
            "max_temporal_mae": round(float(max(deltas)), 4),
            "non_boundary_median_mae": round(non_boundary_median, 4) if non_boundary_median is not None else None,
            "non_boundary_p95_mae": round(non_boundary_p95, 4) if non_boundary_p95 is not None else None,
            "boundary_median_mae": round(boundary_median, 4) if boundary_median is not None else None,
            "boundary_max_mae": round(boundary_max, 4) if boundary_max is not None else None,
            "boundary_to_non_boundary_median_ratio": round(boundary_median / non_boundary_median, 4)
            if boundary_median is not None and non_boundary_median and non_boundary_median > 0
            else None,
            "boundary_max_to_non_boundary_p95_ratio": round(boundary_max / non_boundary_p95, 4)
            if boundary_max is not None and non_boundary_p95 and non_boundary_p95 > 0
            else None,
        }

    @staticmethod
    def _memory_comparison(left_runs: list[dict[str, Any]], right_runs: list[dict[str, Any]]) -> dict[str, Any]:
        left_summary = GenerationMemoryBenchmark._summarize_variant(left_runs)
        right_summary = GenerationMemoryBenchmark._summarize_variant(right_runs)
        keys = [
            "wall_seconds",
            "peak_sampled_rss_bytes",
            "peak_sampled_darwin_physical_footprint_bytes",
            "time_l_maximum_resident_set_size_bytes",
            "mlx_peak_memory_bytes",
            "mlx_cache_memory_bytes",
            "metadata_process_rss_bytes",
            "metadata_darwin_physical_footprint_bytes",
        ]
        deltas = {}
        for key in keys:
            left_value = left_summary[key]["median"]
            right_value = right_summary[key]["median"]
            deltas[key] = GenerationMemoryBenchmark._relative_delta(left_value, right_value)
        return {
            "left_variant": left_runs[0]["variant"],
            "right_variant": right_runs[0]["variant"],
            "median_deltas": deltas,
            "metadata_agreement": {
                f"{run['variant']}_run_{run['run_index']}": GenerationMemoryBenchmark._metadata_agreement(run)
                for run in [*left_runs, *right_runs]
            },
        }

    @staticmethod
    def _relative_delta(left: float | int | None, right: float | int | None) -> dict[str, float | None]:
        if left is None or right is None:
            return {"left": left, "right": right, "absolute": None, "percent": None}
        absolute = float(right) - float(left)
        percent = (absolute / float(left) * 100) if left else None
        return {
            "left": round(float(left), 4),
            "right": round(float(right), 4),
            "absolute": round(absolute, 4),
            "percent": round(percent, 4) if percent is not None else None,
        }

    @staticmethod
    def _metadata_agreement(run: dict[str, Any]) -> dict[str, Any]:
        runtime_memory = ((run.get("metadata") or {}).get("runtime_memory") or {})
        metadata_pid = runtime_memory.get("pid")
        metadata_timestamp = runtime_memory.get("timestamp")
        nearest_sample = GenerationMemoryBenchmark._nearest_sample_for_pid(
            run.get("samples") or [],
            metadata_pid,
            metadata_timestamp,
        )
        return {
            "metadata_pid": metadata_pid,
            "metadata_timestamp": metadata_timestamp,
            "process_peak_rss_vs_time_l": GenerationMemoryBenchmark._relative_delta(
                runtime_memory.get("process_peak_rss_bytes"),
                run.get("time_l", {}).get("maximum_resident_set_size_bytes"),
            ),
            "nearest_sample": nearest_sample,
            "process_rss_vs_nearest_sample": GenerationMemoryBenchmark._relative_delta(
                runtime_memory.get("process_rss_bytes"),
                (nearest_sample or {}).get("rss_bytes"),
            ),
            "physical_footprint_vs_nearest_sample": GenerationMemoryBenchmark._relative_delta(
                runtime_memory.get("darwin_physical_footprint_bytes"),
                (nearest_sample or {}).get("darwin_physical_footprint_bytes"),
            ),
        }

    @staticmethod
    def _nearest_sample_for_pid(
        samples: list[dict[str, Any]],
        pid: int | None,
        timestamp: float | None,
    ) -> dict[str, Any] | None:
        if pid is None or timestamp is None:
            return None
        pid_key = str(pid)
        candidates = [
            sample
            for sample in samples
            if pid_key in (sample.get("processes") or {})
        ]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda sample: abs(float(sample["timestamp"]) - float(timestamp)))
        process = nearest["processes"][pid_key]
        return {
            "timestamp": nearest["timestamp"],
            "timestamp_delta_seconds": round(abs(float(nearest["timestamp"]) - float(timestamp)), 4),
            "rss_bytes": process.get("rss_bytes"),
            "darwin_physical_footprint_bytes": process.get("darwin_physical_footprint_bytes"),
        }

    @staticmethod
    def _array_delta(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
        delta = left - right
        mae = float(np.mean(np.abs(delta)))
        mse = float(np.mean(delta * delta))
        rmse = float(np.sqrt(mse))
        max_abs = float(np.max(np.abs(delta)))
        psnr = None if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "max_abs": round(max_abs, 4),
            "psnr_db": round(float(psnr), 4) if psnr is not None else None,
        }

    @staticmethod
    def _parse_time_l_output(stderr: str) -> dict[str, int | float | None]:
        metrics = {}
        patterns = {
            "maximum_resident_set_size_bytes": r"(\d+)\s+maximum resident set size",
            "peak_memory_footprint_bytes": r"(\d+)\s+peak memory footprint",
            "user_seconds": r"([\d.]+)\s+user",
            "system_seconds": r"([\d.]+)\s+system",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, stderr)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)
        return metrics

    @staticmethod
    def _load_metadata(output_path: Path) -> dict | None:
        metadata_path = output_path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text())

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _environment() -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "python": sys.version,
            "machine": platform.machine(),
            "macos_hw_memsize": GenerationMemoryBenchmark._sysctl_int("hw.memsize"),
        }

    @staticmethod
    def _sysctl_int(name: str) -> int | None:
        try:
            return int(subprocess.check_output(["sysctl", "-n", name], text=True).strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    @staticmethod
    def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
        compact = {}
        for profile, data in report["profiles"].items():
            compact[profile] = {
                "variants": {
                    name: {
                        "rss_gb": GenerationMemoryBenchmark._gb(summary["peak_sampled_rss_bytes"]["median"]),
                        "sampled_physical_gb": GenerationMemoryBenchmark._gb(
                            summary["peak_sampled_darwin_physical_footprint_bytes"]["median"]
                        ),
                        "metadata_physical_gb": GenerationMemoryBenchmark._gb(
                            summary["metadata_darwin_physical_footprint_bytes"]["median"]
                        ),
                        "metadata_rss_gb": GenerationMemoryBenchmark._gb(summary["metadata_process_rss_bytes"]["median"]),
                        "wall_seconds": summary["wall_seconds"]["median"],
                        "mlx_peak_gb": GenerationMemoryBenchmark._gb(summary["mlx_peak_memory_bytes"]["median"]),
                    }
                    for name, summary in data["variants"].items()
                },
                "quality": data["comparisons"].get("quality"),
            }
        return compact

    @staticmethod
    def _gb(value: float | int | None) -> float | None:
        return round(float(value) / BYTES_PER_GB, 4) if value is not None else None


if __name__ == "__main__":
    GenerationMemoryBenchmark.main()
