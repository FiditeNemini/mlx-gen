import json
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.download_policy import DownloadRequiredError, is_huggingface_repo_id
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.scale_factor import ScaleFactor
from mflux.utils.video_util import VideoUtil

SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

SUPPORTED_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB = 8.0
SEEDVR2_DIRECT_VIDEO_PIXEL_VOLUME_LIMITS = {
    "3b": 4_500_000,
    "7b": 4_250_000,
    "7b-sharp": 4_250_000,
}
SEEDVR2_STREAMING_CHUNK_PIXEL_VOLUME_LIMITS = {
    "3b": 4_500_000,
    "7b": 4_250_000,
    "7b-sharp": 4_250_000,
}
SEEDVR2_MIN_STREAMING_CHUNK_FRAMES = 5


@dataclass(frozen=True)
class SeedVR2VideoRestorePlan:
    variant: str
    requested_frames: int
    target_height: int
    target_width: int
    restore_mode: str
    route_reason: str
    effective_chunk_size: int | None
    effective_chunk_overlap: int | None
    chunk_frame_limit: int | None
    chunk_pixel_volume: int | None
    low_ram_required: bool
    low_ram_effective: bool
    cache_limit_gb: float | None
    risk_level: str
    warnings: tuple[str, ...]


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def _resolve_seedvr2_model(model_arg: str | None, model_path: str | None) -> tuple[ModelConfig, str | None]:
    if model_arg is None:
        return ModelConfig.seedvr2_3b(), model_path

    normalized = model_arg.lower()
    if normalized in {"seedvr2", "seedvr2-3b"}:
        return ModelConfig.seedvr2_3b(), None
    if normalized == "seedvr2-7b":
        return ModelConfig.seedvr2_7b(), None
    if normalized == "seedvr2-7b-sharp":
        return ModelConfig.seedvr2_7b_sharp(), None
    if normalized in {"bytedance-seed/seedvr2-3b", "bytedance-seed/seedvr2_3b"}:
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized in {"bytedance-seed/seedvr2-7b", "bytedance-seed/seedvr2_7b"}:
        return ModelConfig.seedvr2_7b(), model_arg
    if normalized == "numz/seedvr2_comfyui":
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized.startswith("abstractframework/seedvr2-3b-"):
        return ModelConfig.seedvr2_3b(), model_arg
    if normalized.startswith("abstractframework/seedvr2-7b-"):
        return ModelConfig.seedvr2_7b(), model_arg

    requested_model_path = model_path
    if requested_model_path is None and Path(model_arg).expanduser().exists():
        requested_model_path = model_arg

    if requested_model_path is not None:
        path = Path(requested_model_path).expanduser()
        if path.is_dir():
            has_3b = (path / "seedvr2_ema_3b_fp16.safetensors").exists()
            has_official_3b = (path / "seedvr2_ema_3b.pth").exists()
            has_7b = (path / "seedvr2_ema_7b_fp16.safetensors").exists()
            has_official_7b = (path / "seedvr2_ema_7b.pth").exists()
            has_official_7b_sharp = (path / "seedvr2_ema_7b_sharp.pth").exists()
            if has_official_7b_sharp and not (has_3b or has_official_3b or has_7b or has_official_7b):
                return ModelConfig.seedvr2_7b_sharp(), requested_model_path
            if (has_7b or has_official_7b or has_official_7b_sharp) and not (has_3b or has_official_3b):
                if has_official_7b_sharp and (
                    "seedvr2-7b-sharp" in normalized or not (has_7b or has_official_7b)
                ):
                    return ModelConfig.seedvr2_7b_sharp(), requested_model_path
                return ModelConfig.seedvr2_7b(), requested_model_path
            if (has_3b or has_official_3b) and not (has_7b or has_official_7b):
                return ModelConfig.seedvr2_3b(), requested_model_path
            if (path / "transformer" / "model.safetensors.index.json").exists():
                if "seedvr2-7b-sharp" in normalized or "7b-sharp" in path.name.lower() or "7b_sharp" in path.name.lower():
                    return ModelConfig.seedvr2_7b_sharp(), requested_model_path
                if "seedvr2-7b" in normalized or "7b" in path.name.lower():
                    return ModelConfig.seedvr2_7b(), requested_model_path
                return ModelConfig.seedvr2_3b(), requested_model_path

    if is_huggingface_repo_id(model_arg):
        raise ValueError(
            "Unsupported SeedVR2 model handle "
            f"{model_arg!r}. Use seedvr2, seedvr2-3b, seedvr2-7b, seedvr2-7b-sharp, "
            "ByteDance-Seed/SeedVR2-3B, ByteDance-Seed/SeedVR2-7B, "
            "AbstractFramework/seedvr2-3b-8bit, AbstractFramework/seedvr2-3b-4bit, "
            "AbstractFramework/seedvr2-7b-8bit, AbstractFramework/seedvr2-7b-4bit, "
            "or an explicit local SeedVR2 path."
        )

    source = (requested_model_path or model_arg).lower()
    if "seedvr2_ema_7b_sharp" in source or "seedvr2-7b-sharp" in source or "seedvr2_7b_sharp" in source:
        return ModelConfig.seedvr2_7b_sharp(), requested_model_path
    if "seedvr2_ema_7b" in source or "seedvr2-7b" in source:
        return ModelConfig.seedvr2_7b(), requested_model_path
    return ModelConfig.seedvr2_3b(), requested_model_path


def _expand_image_paths(image_paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for image_path in image_paths:
        if image_path.is_dir():
            dir_images = sorted(
                [path for path in image_path.iterdir() if _is_image_file(path)],
                key=lambda path: path.name.lower(),
            )
            if not dir_images:
                print(f"No images found in directory: {image_path}")
            expanded.extend(dir_images)
        else:
            expanded.append(image_path)
    return expanded


def _expand_video_paths(video_paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for video_path in video_paths:
        if video_path.is_dir():
            dir_videos = sorted(
                [path for path in video_path.iterdir() if _is_video_file(path)],
                key=lambda path: path.name.lower(),
            )
            if not dir_videos:
                print(f"No videos found in directory: {video_path}")
            expanded.extend(dir_videos)
        else:
            expanded.append(video_path)
    return expanded


def _default_output_for_inputs(output: str, *, is_video: bool) -> str:
    if output != "image.png":
        return output
    return "video.mp4" if is_video else output


def _provided_options(argv: list[str]) -> set[str]:
    provided = set()
    aliases = {
        "-m": "--model",
        "-i": "--image-path",
        "-s": "--seed",
        "-r": "--resolution",
        "-q": "--quantize",
    }
    expects_value = {
        "--model",
        "--image-path",
        "--video-path",
        "--seed",
        "--resolution",
        "--quantize",
        "--softness",
        "--color-correction",
        "--start-seconds",
        "--max-frames",
        "--temporal-chunk-size",
        "--temporal-chunk-overlap",
        "--mlx-cache-limit-gb",
        "--output",
    }
    index = 0
    while index < len(argv):
        arg = aliases.get(argv[index], argv[index])
        if not arg.startswith("-"):
            index += 1
            continue
        provided.add(arg)
        if arg in expects_value:
            index += 2
            continue
        index += 1
    return provided


def _requested_video_frame_count(source_probe, max_frames: int | None) -> int:
    if source_probe.source_frame_count is not None:
        available_frames = max(0, source_probe.source_frame_count - source_probe.clip_start_frame)
    elif source_probe.source_duration_seconds is not None:
        available_frames = max(1, int(round(source_probe.source_duration_seconds * source_probe.fps)) - source_probe.clip_start_frame)
    else:
        raise ValueError("SeedVR2 video restore requires a finite source frame count or duration.")
    return min(max_frames, available_frames) if max_frames is not None else available_frames


def _seedvr2_variant_name(model_config: ModelConfig) -> str:
    aliases = {alias.lower() for alias in (model_config.aliases or [])}
    if "seedvr2-7b-sharp" in aliases:
        return "7b-sharp"
    if "seedvr2-7b" in aliases:
        return "7b"
    return "3b"


def _seedvr2_resolution_scale(resolution: int | ScaleFactor, min_side: int) -> float:
    if isinstance(resolution, ScaleFactor):
        return float(resolution.get_scaled_value(min_side)) / float(min_side)
    return float(resolution) / float(min_side)


def _estimate_seedvr2_output_size(
    *,
    source_width: int,
    source_height: int,
    resolution: int | ScaleFactor,
) -> tuple[int, int]:
    min_side = min(source_width, source_height)
    if isinstance(resolution, ScaleFactor):
        target_res = resolution.get_scaled_value(min_side)
    else:
        target_res = resolution
    scale = target_res / min_side
    width = max(2, (int(source_width * scale) // 2) * 2)
    height = max(2, (int(source_height * scale) // 2) * 2)
    return height, width


def _aligned_chunk_size(max_frames: int) -> int:
    if max_frames <= 1:
        return 1
    candidate = max_frames
    while candidate > 1 and (candidate - 1) % 4 != 0:
        candidate -= 1
    return max(candidate, 1)


def _safe_chunk_frame_limit(*, area: int, variant: str) -> int:
    if area <= 0:
        raise ValueError("area must be greater than zero.")
    return max(1, SEEDVR2_STREAMING_CHUNK_PIXEL_VOLUME_LIMITS[variant] // area)


def _plan_seedvr2_video_restore(
    *,
    model_config: ModelConfig,
    source_probe,
    requested_frames: int,
    resolution: int | ScaleFactor,
    temporal_chunk_size: int,
    temporal_chunk_overlap: int,
    low_ram_requested: bool,
    cache_limit_gb: float | None,
    force_unsafe_memory_profile: bool,
) -> SeedVR2VideoRestorePlan:
    variant = _seedvr2_variant_name(model_config)
    height, width = _estimate_seedvr2_output_size(
        source_width=source_probe.source_width,
        source_height=source_probe.source_height,
        resolution=resolution,
    )
    area = height * width
    scale = _seedvr2_resolution_scale(resolution, min(source_probe.source_width, source_probe.source_height))
    direct_pixel_volume = requested_frames * area
    safe_chunk_limit = _safe_chunk_frame_limit(area=area, variant=variant)
    warnings: list[str] = []
    low_ram_required = variant in {"7b", "7b-sharp"}

    if low_ram_required and not low_ram_requested and not force_unsafe_memory_profile:
        warnings.append("SeedVR2 7B video restore requires --low-ram for the supported safe profile.")

    direct_allowed = requested_frames <= 121 and direct_pixel_volume <= SEEDVR2_DIRECT_VIDEO_PIXEL_VOLUME_LIMITS[variant]
    if direct_allowed:
        restore_mode = "direct"
        route_reason = "direct_safe"
        effective_chunk_size = None
        effective_chunk_overlap = None
        chunk_pixel_volume = None
    else:
        requested_chunk_size = min(temporal_chunk_size, requested_frames)
        safe_chunk_size = min(requested_chunk_size, safe_chunk_limit)
        effective_chunk_size = _aligned_chunk_size(safe_chunk_size)
        if effective_chunk_size < SEEDVR2_MIN_STREAMING_CHUNK_FRAMES and not force_unsafe_memory_profile:
            warnings.append(
                "This SeedVR2 video request exceeds the supported safe memory profile. "
                "Reduce resolution or use a shorter clip, or override with --force-unsafe-video-memory."
            )
        restore_mode = "streaming"
        if low_ram_requested:
            route_reason = "low_ram"
        elif requested_frames > 121:
            route_reason = "frame_count"
        elif effective_chunk_size < requested_chunk_size:
            route_reason = "chunk_volume_clamped"
        else:
            route_reason = "pixel_volume"
        overlap_cap = max(0, effective_chunk_size // 3)
        effective_chunk_overlap = min(temporal_chunk_overlap, overlap_cap)
        chunk_pixel_volume = effective_chunk_size * area

    risk_level = "low"
    if scale > 1.0 or variant in {"7b", "7b-sharp"}:
        risk_level = "medium"
    if warnings:
        risk_level = "high"

    return SeedVR2VideoRestorePlan(
        variant=variant,
        requested_frames=requested_frames,
        target_height=height,
        target_width=width,
        restore_mode=restore_mode,
        route_reason=route_reason,
        effective_chunk_size=effective_chunk_size,
        effective_chunk_overlap=effective_chunk_overlap,
        chunk_frame_limit=safe_chunk_limit,
        chunk_pixel_volume=chunk_pixel_volume,
        low_ram_required=low_ram_required,
        low_ram_effective=low_ram_requested,
        cache_limit_gb=cache_limit_gb,
        risk_level=risk_level,
        warnings=tuple(warnings),
    )


def _apply_cache_limit_if_needed(mlx_cache_limit_gb: float | None) -> None:
    if mlx_cache_limit_gb is None:
        return
    mx.set_cache_limit(int(mlx_cache_limit_gb * (1000**3)))
    mx.clear_cache()
    mx.reset_peak_memory()


def _seedvr2_video_runtime_diagnostics() -> dict[str, int | None]:
    def _mlx_memory(name: str) -> int | None:
        try:
            return int(getattr(mx, name)())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    return {
        "mlx_active_memory_bytes": _mlx_memory("get_active_memory"),
        "mlx_peak_memory_bytes": _mlx_memory("get_peak_memory"),
        "mlx_cache_memory_bytes": _mlx_memory("get_cache_memory"),
    }


def _write_seedvr2_failure_manifest(
    *,
    output_path: str | Path,
    video_path: Path,
    model: str | None,
    seed: int,
    resolution: int | ScaleFactor,
    softness: float,
    color_correction_mode: str,
    start_seconds: float,
    max_frames: int | None,
    plan: SeedVR2VideoRestorePlan,
    error: BaseException,
) -> Path:
    failure_path = Path(output_path).with_suffix(".failure.json")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(),
                "status": "failed",
                "error_type": error.__class__.__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "run": {
                    "model": model,
                    "task": "video-to-video",
                    "seed": seed,
                    "video_path": str(video_path),
                    "resolution": str(resolution),
                    "softness": round(float(softness), 3),
                    "color_correction_mode": color_correction_mode,
                    "start_seconds": round(float(start_seconds), 3),
                    "max_frames": max_frames,
                    "restore_plan": asdict(plan),
                    "output": str(output_path),
                },
                "runtime_diagnostics": _seedvr2_video_runtime_diagnostics(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return failure_path


def _print_seedvr2_video_preflight(video_path: Path, source_probe, plan: SeedVR2VideoRestorePlan) -> None:
    print(
        "SeedVR2 video preflight: "
        f"model={plan.variant} "
        f"source={source_probe.source_width}x{source_probe.source_height} "
        f"target={plan.target_width}x{plan.target_height} "
        f"frames={plan.requested_frames} "
        f"mode={plan.restore_mode} "
        f"reason={plan.route_reason} "
        f"low_ram={plan.low_ram_effective} "
        f"cache_limit_gb={plan.cache_limit_gb or 'none'} "
        f"video={video_path.name}"
    )
    if plan.restore_mode == "streaming":
        print(
            "SeedVR2 streaming plan: "
            f"chunk_size={plan.effective_chunk_size} "
            f"chunk_overlap={plan.effective_chunk_overlap} "
            f"safe_chunk_frame_limit={plan.chunk_frame_limit} "
            f"chunk_pixel_volume={plan.chunk_pixel_volume}"
        )
    for warning in plan.warnings:
        print(f"SeedVR2 warning: {warning}")


def main():
    # 1. Parse command line arguments
    parser = CommandLineParser(description="Upscale an image using SeedVR2 diffusion-based super-resolution.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_seedvr2_upscale_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()
    provided_options = _provided_options(sys.argv[1:])

    image_paths = _expand_image_paths(args.image_path) if args.image_path else []
    video_paths = _expand_video_paths(args.video_path) if args.video_path else []
    if not image_paths and not video_paths:
        print("No images or videos to upscale.")
        return
    if image_paths and (args.start_seconds != 0.0 or args.max_frames is not None):
        parser.error("--start-seconds and --max-frames are only supported with --video-path.")
    if video_paths and args.vae_tiling:
        parser.error("--vae-tiling is not supported for SeedVR2 video restore. Use --low-ram and temporal chunking instead.")
    if args.temporal_chunk_size <= 0:
        parser.error("--temporal-chunk-size must be greater than zero.")
    if args.temporal_chunk_overlap < 0:
        parser.error("--temporal-chunk-overlap must be greater than or equal to zero.")
    if args.temporal_chunk_overlap >= args.temporal_chunk_size:
        parser.error("--temporal-chunk-overlap must be smaller than --temporal-chunk-size.")
    if video_paths and "--resolution" not in provided_options:
        args.resolution = ScaleFactor(1)

    try:
        model_config, resolved_model_path = _resolve_seedvr2_model(args.model, args.model_path)
    except ValueError as exc:
        parser.error(str(exc))

    video_probes: dict[Path, object] = {}
    video_restore_plans: dict[Path, SeedVR2VideoRestorePlan] = {}
    if video_paths:
        cache_limit_gb = args.mlx_cache_limit_gb or DEFAULT_SEEDVR2_VIDEO_CACHE_LIMIT_GB
        for video_path in video_paths:
            source_probe = VideoUtil.read_video_clip(
                video_path,
                start_seconds=args.start_seconds,
                max_frames=1,
            )
            video_probes[video_path] = source_probe
            requested_frames = _requested_video_frame_count(source_probe, args.max_frames)
            plan = _plan_seedvr2_video_restore(
                model_config=model_config,
                source_probe=source_probe,
                requested_frames=requested_frames,
                resolution=args.resolution,
                temporal_chunk_size=args.temporal_chunk_size,
                temporal_chunk_overlap=args.temporal_chunk_overlap,
                low_ram_requested=args.low_ram,
                cache_limit_gb=cache_limit_gb,
                force_unsafe_memory_profile=args.force_unsafe_video_memory,
            )
            video_restore_plans[video_path] = plan
            if plan.low_ram_required and not args.low_ram and not args.force_unsafe_video_memory:
                parser.error(
                    "SeedVR2 7B video restore requires --low-ram on the supported safe profile. "
                    "Use --low-ram --mlx-cache-limit-gb 8, or override with --force-unsafe-video-memory."
                )
            if plan.warnings and not args.force_unsafe_video_memory:
                parser.error(plan.warnings[0])
        if args.mlx_cache_limit_gb is None:
            args.mlx_cache_limit_gb = cache_limit_gb
        _apply_cache_limit_if_needed(args.mlx_cache_limit_gb)

    # 3. Load the SeedVR2 model
    try:
        model = SeedVR2(
            quantize=args.quantize,
            model_path=resolved_model_path,
            model_config=model_config,
        )
    except DownloadRequiredError as exc:
        parser.error(str(exc))
    if args.vae_tiling:
        model.tiling_config = TilingConfig()

    # 4. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=SeedVR2LatentCreator,
    )
    if memory_saver is not None and video_paths:
        memory_saver.keep_transformer = True

    try:
        # 5. Upscale the image for each seed
        output_pattern = _default_output_for_inputs(args.output, is_video=bool(video_paths))
        for image_path in image_paths:
            for seed in args.seed:
                result = model.generate_image(
                    seed=seed,
                    image_path=image_path,
                    resolution=args.resolution,
                    softness=args.softness,
                    color_correction_mode=args.color_correction,
                )

                result.save(
                    output_pattern.format(seed=seed, image_name=image_path.stem),
                    export_json_metadata=args.metadata,
                    overwrite=args.replace,
                )
        for video_path in video_paths:
            for seed in args.seed:
                source_probe = video_probes[video_path]
                plan = video_restore_plans[video_path]
                output_path = output_pattern.format(seed=seed, image_name=video_path.stem)
                _print_seedvr2_video_preflight(video_path, source_probe, plan)
                if source_probe.audio_present:
                    print("SeedVR2 warning: source audio will not be copied; output will be a silent MP4.")
                runtime_metadata = {
                    "restore_mode": plan.restore_mode,
                    "restore_mode_reason": plan.route_reason,
                    "low_ram_requested": bool(args.low_ram),
                    "low_ram_effective": bool(plan.low_ram_effective),
                    "mlx_cache_limit_gb": args.mlx_cache_limit_gb,
                    "requested_temporal_chunk_size": args.temporal_chunk_size,
                    "requested_temporal_chunk_overlap": args.temporal_chunk_overlap,
                    "effective_temporal_chunk_size": plan.effective_chunk_size,
                    "effective_temporal_chunk_overlap": plan.effective_chunk_overlap,
                    "seedvr2_risk_level": plan.risk_level,
                    "seedvr2_target_height": plan.target_height,
                    "seedvr2_target_width": plan.target_width,
                }
                try:
                    if plan.restore_mode == "streaming":
                        result_path = model.restore_video_to_path(
                            seed=seed,
                            video_path=video_path,
                            resolution=args.resolution,
                            softness=args.softness,
                            start_seconds=args.start_seconds,
                            max_frames=args.max_frames,
                            output_path=output_path,
                            export_json_metadata=args.metadata,
                            overwrite=args.replace,
                            temporal_chunk_size=plan.effective_chunk_size or args.temporal_chunk_size,
                            temporal_chunk_overlap=plan.effective_chunk_overlap or args.temporal_chunk_overlap,
                            color_correction_mode=args.color_correction,
                            restore_metadata=runtime_metadata,
                        )
                    else:
                        result = model.generate_video(
                            seed=seed,
                            video_path=video_path,
                            resolution=args.resolution,
                            softness=args.softness,
                            start_seconds=args.start_seconds,
                            max_frames=args.max_frames,
                            color_correction_mode=args.color_correction,
                            restore_metadata=runtime_metadata,
                        )
                        result_path = result.save(
                            output_path,
                            export_json_metadata=args.metadata,
                            overwrite=args.replace,
                        )
                except Exception as exc:
                    failure_path = _write_seedvr2_failure_manifest(
                        output_path=output_path,
                        video_path=video_path,
                        model=args.model,
                        seed=seed,
                        resolution=args.resolution,
                        softness=args.softness,
                        color_correction_mode=args.color_correction,
                        start_seconds=args.start_seconds,
                        max_frames=args.max_frames,
                        plan=plan,
                        error=exc,
                    )
                    print(f"SeedVR2 failure manifest saved at: {failure_path}")
                    raise
                print(f"Video saved successfully at: {result_path}")
    except StopImageGenerationException as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
