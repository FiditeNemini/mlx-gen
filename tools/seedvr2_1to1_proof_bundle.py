import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from mflux.utils.video_util import DecodedVideoClip, VideoUtil


class SeedVR2OneToOneProofBundle:
    PROFILES = {
        "3b": {
            "profile": "seedvr2-video-3b-149-restoration",
            "label": "SeedVR2 3B",
            "dir": "3b",
            "restored_name": "seedvr2_3b_149f_restored.mp4",
        },
        "7b": {
            "profile": "seedvr2-video-7b-149-restoration",
            "label": "SeedVR2 7B",
            "dir": "7b",
            "restored_name": "seedvr2_7b_149f_restored.mp4",
        },
    }

    @staticmethod
    def main() -> None:
        args = SeedVR2OneToOneProofBundle._parse_args()
        report = json.loads(args.benchmark_report.read_text())
        source_path = args.source_video.resolve()
        output_dir = args.output_dir.resolve()
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_bundle_path = source_dir / source_path.name
        shutil.copy2(source_path, source_bundle_path)

        source_clip = VideoUtil.read_video_clip(source_bundle_path)
        source_probe = SeedVR2OneToOneProofBundle._probe_video(source_bundle_path, source_clip)
        source_probe["sha256"] = SeedVR2OneToOneProofBundle._sha256(source_bundle_path)
        (source_dir / f"{source_path.stem}_probe.json").write_text(json.dumps(source_probe, indent=2))

        manifest: dict[str, Any] = {
            "created_from": {
                "benchmark_report": str(args.benchmark_report.resolve()),
                "source_video": str(source_bundle_path),
            },
            "quality_policy": {
                "normal_package_seedvr2_noise_mode": "global",
                "production_temporal_chunk_floor": "29 frames with 8 frames of overlap when chunked",
                "reason": (
                    "This bundle proves the current package default 1:1 restoration path, records memory stats, "
                    "and includes all production chunk-boundary frames for the 149-frame SeedVR2 profile."
                ),
            },
            "source": source_probe,
            "models": {},
        }

        for model_key, config in SeedVR2OneToOneProofBundle.PROFILES.items():
            model_manifest = SeedVR2OneToOneProofBundle._bundle_model(
                report=report,
                source_clip=source_clip,
                source_bundle_path=source_bundle_path,
                output_dir=output_dir,
                model_key=model_key,
                config=config,
            )
            manifest["models"][model_key] = model_manifest

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--benchmark-report", type=Path, required=True)
        parser.add_argument(
            "--source-video",
            type=Path,
            default=Path("validation_outputs/seedvr2_video_2026_06_20/eiffel_70s_149f_source.mp4"),
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("validation_outputs/seedvr2_1to1_memory_quality_20260628"),
        )
        return parser.parse_args()

    @staticmethod
    def _bundle_model(
        *,
        report: dict,
        source_clip: DecodedVideoClip,
        source_bundle_path: Path,
        output_dir: Path,
        model_key: str,
        config: dict,
    ) -> dict:
        profile_name = config["profile"]
        profile = report["profiles"][profile_name]
        variant_name = "restored_149f"
        run = SeedVR2OneToOneProofBundle._first_successful_run(profile["raw_runs"][variant_name])
        restored_source_path = Path(run["output_path"])
        model_dir = output_dir / config["dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        restored_bundle_path = model_dir / config["restored_name"]
        shutil.copy2(restored_source_path, restored_bundle_path)
        metadata_source = restored_source_path.with_suffix(".metadata.json")
        metadata_path = restored_bundle_path.with_suffix(".metadata.json")
        if metadata_source.exists():
            shutil.copy2(metadata_source, metadata_path)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else run.get("metadata") or {}

        restored_clip = VideoUtil.read_video_clip(restored_bundle_path)
        memory_stats = SeedVR2OneToOneProofBundle._memory_stats(profile=profile, variant_name=variant_name)
        quality_stats = SeedVR2OneToOneProofBundle._quality_stats(
            source_clip=source_clip,
            restored_clip=restored_clip,
            metadata=metadata,
        )
        restored_probe = SeedVR2OneToOneProofBundle._probe_video(restored_bundle_path, restored_clip)
        restored_probe["sha256"] = SeedVR2OneToOneProofBundle._sha256(restored_bundle_path)
        restored_probe["metadata"] = {
            key: metadata.get(key)
            for key in [
                "model",
                "seed",
                "steps",
                "frames",
                "source_video_frames",
                "source_video_fps",
                "temporal_chunk_count",
                "temporal_chunk_size",
                "temporal_chunk_overlap",
                "seedvr2_noise_mode",
                "seedvr2_noise_version",
                "seedvr2_noise_estimated_global_bytes",
            ]
        }

        memory_stats_path = model_dir / f"seedvr2_{model_key}_149f_memory_stats.json"
        quality_stats_path = model_dir / f"seedvr2_{model_key}_149f_quality_stats.json"
        memory_stats_path.write_text(json.dumps(memory_stats, indent=2))
        quality_stats_path.write_text(json.dumps(quality_stats, indent=2))

        timeline_sheet = model_dir / f"seedvr2_{model_key}_149f_source_vs_restored_timeline.jpg"
        boundary_sheet = model_dir / f"seedvr2_{model_key}_149f_source_vs_restored_boundaries.jpg"
        comparison_video = model_dir / f"seedvr2_{model_key}_149f_source_vs_restored_side_by_side.mp4"

        SeedVR2OneToOneProofBundle._save_contact_sheet(
            source_clip=source_clip,
            restored_clip=restored_clip,
            frame_indices=SeedVR2OneToOneProofBundle._timeline_indices(source_clip.clip_frame_count),
            output_path=timeline_sheet,
            title=f"{config['label']} 149-frame 1:1 timeline",
            subtitle="Rows: source, restored, absolute RGB difference, restored adjacent-frame difference.",
        )
        SeedVR2OneToOneProofBundle._save_contact_sheet(
            source_clip=source_clip,
            restored_clip=restored_clip,
            frame_indices=SeedVR2OneToOneProofBundle._boundary_indices(
                frame_count=source_clip.clip_frame_count,
                metadata=metadata,
            ),
            output_path=boundary_sheet,
            title=f"{config['label']} 149-frame chunk-boundary review",
            subtitle="Columns cover every production chunk boundary with before/boundary/after frames.",
        )
        SeedVR2OneToOneProofBundle._save_comparison_video(
            source_clip=source_clip,
            restored_clip=restored_clip,
            output_path=comparison_video,
            label=config["label"],
        )

        return {
            "profile": profile_name,
            "selected_run_index": run["run_index"],
            "restored_video": str(restored_bundle_path),
            "restored_probe": restored_probe,
            "source_vs_restored_comparison_video": str(comparison_video),
            "timeline_contact_sheet": str(timeline_sheet),
            "boundary_contact_sheet": str(boundary_sheet),
            "memory_stats": str(memory_stats_path),
            "quality_stats": str(quality_stats_path),
        }

    @staticmethod
    def _first_successful_run(runs: list[dict]) -> dict:
        for run in runs:
            output_path = Path(run["output_path"])
            if run.get("returncode") == 0 and run.get("output_exists") and output_path.exists():
                return run
        raise ValueError("No successful run with an existing restored output was found.")

    @staticmethod
    def _memory_stats(*, profile: dict, variant_name: str) -> dict:
        raw_runs = profile["raw_runs"][variant_name]
        return {
            "variant": variant_name,
            "aggregate": profile["variants"][variant_name],
            "single_video_health": (profile.get("comparisons", {}).get("quality") or {}).get("video_health"),
            "runs": [
                {
                    "run_index": run["run_index"],
                    "returncode": run["returncode"],
                    "wall_seconds": run["wall_seconds"],
                    "output_path": run["output_path"],
                    "output_sha256": run.get("output_sha256"),
                    "sampler": run.get("sampler"),
                    "time_l": run.get("time_l"),
                    "runtime_memory": (run.get("metadata") or {}).get("runtime_memory"),
                    "argv": run.get("argv"),
                }
                for run in raw_runs
            ],
        }

    @staticmethod
    def _probe_video(path: Path, clip: DecodedVideoClip) -> dict:
        return {
            "path": str(path),
            "frames": clip.clip_frame_count,
            "source_frame_count": clip.source_frame_count,
            "width": clip.source_width,
            "height": clip.source_height,
            "fps": clip.fps,
            "duration_seconds": clip.source_duration_seconds,
            "audio_present": clip.audio_present,
        }

    @staticmethod
    def _quality_stats(
        *,
        source_clip: DecodedVideoClip,
        restored_clip: DecodedVideoClip,
        metadata: dict,
    ) -> dict:
        if source_clip.clip_frame_count != restored_clip.clip_frame_count:
            raise ValueError(
                f"Frame count mismatch: source={source_clip.clip_frame_count}, restored={restored_clip.clip_frame_count}."
            )
        source_size = (source_clip.source_width, source_clip.source_height)
        restored_frames = [
            frame.resize(source_size, PIL.Image.Resampling.LANCZOS)
            if frame.size != source_size
            else frame
            for frame in restored_clip.frames
        ]
        source_luma = [SeedVR2OneToOneProofBundle._to_luma(frame) for frame in source_clip.frames]
        restored_luma = [SeedVR2OneToOneProofBundle._to_luma(frame) for frame in restored_frames]
        rgb_deltas = [
            SeedVR2OneToOneProofBundle._array_delta(
                np.asarray(source_frame.convert("RGB"), dtype=np.float32),
                np.asarray(restored_frame.convert("RGB"), dtype=np.float32),
            )
            for source_frame, restored_frame in zip(source_clip.frames, restored_frames)
        ]
        source_sharpness = SeedVR2OneToOneProofBundle._mean_sharpness(source_luma)
        restored_sharpness = SeedVR2OneToOneProofBundle._mean_sharpness(restored_luma)
        source_contrast = float(np.mean([np.std(frame) for frame in source_luma]))
        restored_contrast = float(np.mean([np.std(frame) for frame in restored_luma]))
        source_temporal = SeedVR2OneToOneProofBundle._temporal_delta(source_luma)
        restored_temporal = SeedVR2OneToOneProofBundle._temporal_delta(restored_luma)
        boundary_temporal = SeedVR2OneToOneProofBundle._boundary_temporal_stats(
            source_luma=source_luma,
            restored_luma=restored_luma,
            metadata=metadata,
        )
        return {
            "same_frame_count": True,
            "source_frames": source_clip.clip_frame_count,
            "restored_frames": restored_clip.clip_frame_count,
            "same_resolution": source_size == (restored_clip.source_width, restored_clip.source_height),
            "source_resolution": list(source_size),
            "restored_resolution": [restored_clip.source_width, restored_clip.source_height],
            "fps_delta": abs(source_clip.fps - restored_clip.fps),
            "seedvr2_noise_mode": metadata.get("seedvr2_noise_mode"),
            "seedvr2_noise_version": metadata.get("seedvr2_noise_version"),
            "temporal_chunk_count": metadata.get("temporal_chunk_count"),
            "temporal_chunk_size": metadata.get("temporal_chunk_size"),
            "temporal_chunk_overlap": metadata.get("temporal_chunk_overlap"),
            "source_vs_restored_rgb_delta": {
                "mae": round(float(np.mean([item["mae"] for item in rgb_deltas])), 4),
                "rmse": round(float(np.mean([item["rmse"] for item in rgb_deltas])), 4),
                "max_abs": int(max(item["max_abs"] for item in rgb_deltas)),
                "psnr_db": round(float(np.mean([item["psnr_db"] for item in rgb_deltas])), 4),
            },
            "source_resolution_metrics": {
                "source_sharpness": round(source_sharpness, 6),
                "restored_sharpness": round(restored_sharpness, 6),
                "sharpness_gain": round(restored_sharpness / max(source_sharpness, 1e-8), 4),
                "source_contrast": round(source_contrast, 6),
                "restored_contrast": round(restored_contrast, 6),
                "contrast_gain": round(restored_contrast / max(source_contrast, 1e-8), 4),
                "source_temporal_delta": round(source_temporal, 6),
                "restored_temporal_delta": round(restored_temporal, 6),
                "temporal_ratio": round(restored_temporal / max(source_temporal, 1e-8), 4),
            },
            "chunk_boundary_temporal_delta": boundary_temporal,
            "interpretation": (
                "These are source-to-restored metrics for the production default restore, including every "
                "chunk boundary reported by the output metadata."
            ),
        }

    @staticmethod
    def _save_contact_sheet(
        *,
        source_clip: DecodedVideoClip,
        restored_clip: DecodedVideoClip,
        frame_indices: list[int],
        output_path: Path,
        title: str,
        subtitle: str,
    ) -> None:
        panel_width = 160
        panel_height = 120
        label_width = 220
        header_height = 94
        row_gap = 12
        col_gap = 8
        rows = [
            ("Source", source_clip.frames),
            ("Restored", restored_clip.frames),
            ("Abs RGB diff x4", SeedVR2OneToOneProofBundle._diff_frames(source_clip.frames, restored_clip.frames, scale=4.0)),
            ("Restored temporal diff x4", SeedVR2OneToOneProofBundle._temporal_diff_frames(restored_clip.frames, scale=4.0)),
        ]
        width = label_width + len(frame_indices) * panel_width + max(0, len(frame_indices) - 1) * col_gap + 28
        height = header_height + len(rows) * panel_height + (len(rows) - 1) * row_gap + 28
        canvas = PIL.Image.new("RGB", (width, height), "white")
        draw = PIL.ImageDraw.Draw(canvas)
        title_font = PIL.ImageFont.load_default(size=26)
        body_font = PIL.ImageFont.load_default(size=17)
        small_font = PIL.ImageFont.load_default(size=14)
        draw.text((18, 14), title, fill="black", font=title_font)
        draw.text((18, 48), subtitle, fill=(70, 70, 70), font=body_font)

        x = label_width
        for frame_index in frame_indices:
            draw.text((x + 4, 72), f"f{frame_index}", fill=(55, 55, 55), font=small_font)
            x += panel_width + col_gap

        y = header_height
        for label, frames in rows:
            draw.rectangle((18, y, label_width - 16, y + panel_height), fill=(32, 70, 105))
            SeedVR2OneToOneProofBundle._draw_centered_text(
                draw=draw,
                box=(28, y + 14, label_width - 26, y + panel_height - 14),
                text=label,
                font=body_font,
                fill="white",
            )
            x = label_width
            for frame_index in frame_indices:
                image = frames[frame_index].resize((panel_width, panel_height), PIL.Image.Resampling.LANCZOS)
                canvas.paste(image, (x, y))
                x += panel_width + col_gap
            y += panel_height + row_gap

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)

    @staticmethod
    def _save_comparison_video(
        *,
        source_clip: DecodedVideoClip,
        restored_clip: DecodedVideoClip,
        output_path: Path,
        label: str,
    ) -> None:
        width = max(source_clip.source_width, restored_clip.source_width)
        height = max(source_clip.source_height, restored_clip.source_height)
        label_height = 38
        font = PIL.ImageFont.load_default(size=18)
        frames = []
        for index, (source_frame, restored_frame) in enumerate(zip(source_clip.frames, restored_clip.frames)):
            source_panel = source_frame.resize((width, height), PIL.Image.Resampling.LANCZOS)
            restored_panel = restored_frame.resize((width, height), PIL.Image.Resampling.LANCZOS)
            canvas = PIL.Image.new("RGB", (width * 2, height + label_height), "white")
            draw = PIL.ImageDraw.Draw(canvas)
            draw.text((10, 10), f"Source f{index}", fill="black", font=font)
            draw.text((width + 10, 10), f"{label} restored f{index}", fill="black", font=font)
            canvas.paste(source_panel, (0, label_height))
            canvas.paste(restored_panel, (width, label_height))
            frames.append(canvas)
        VideoUtil.save_video(
            frames=frames,
            path=output_path,
            fps=source_clip.fps,
            metadata={"note": "SeedVR2 1:1 source/restored comparison video.", "model": label},
            export_json_metadata=True,
            overwrite=True,
            validate_health=True,
        )

    @staticmethod
    def _timeline_indices(frame_count: int) -> list[int]:
        return SeedVR2OneToOneProofBundle._unique_indices(
            int(round(value)) for value in np.linspace(0, frame_count - 1, num=min(frame_count, 9))
        )

    @staticmethod
    def _boundary_indices(*, frame_count: int, metadata: dict) -> list[int]:
        selected_boundaries = SeedVR2OneToOneProofBundle._chunk_boundaries(
            frame_count=frame_count,
            metadata=metadata,
        )
        indices = [0]
        for boundary in selected_boundaries:
            indices.extend([boundary - 1, boundary, boundary + 1])
        indices.append(frame_count - 1)
        return SeedVR2OneToOneProofBundle._unique_indices(
            min(max(index, 0), frame_count - 1) for index in indices
        )

    @staticmethod
    def _chunk_boundaries(*, frame_count: int, metadata: dict) -> list[int]:
        boundaries = []
        output_cursor = 0
        for chunk in metadata.get("temporal_chunk_plan") or []:
            output_cursor += int(chunk.get("output_frame_count") or 0)
            if 0 < output_cursor < frame_count:
                boundaries.append(output_cursor)
        return boundaries

    @staticmethod
    def _boundary_temporal_stats(
        *,
        source_luma: list[np.ndarray],
        restored_luma: list[np.ndarray],
        metadata: dict,
    ) -> dict:
        frame_count = min(len(source_luma), len(restored_luma))
        boundaries = SeedVR2OneToOneProofBundle._chunk_boundaries(
            frame_count=frame_count,
            metadata=metadata,
        )
        per_boundary = []
        for boundary in boundaries:
            source_left = SeedVR2OneToOneProofBundle._frame_delta(source_luma[boundary - 1], source_luma[boundary])
            restored_left = SeedVR2OneToOneProofBundle._frame_delta(
                restored_luma[boundary - 1],
                restored_luma[boundary],
            )
            source_right = None
            restored_right = None
            right_ratio = None
            if boundary + 1 < frame_count:
                source_right = SeedVR2OneToOneProofBundle._frame_delta(source_luma[boundary], source_luma[boundary + 1])
                restored_right = SeedVR2OneToOneProofBundle._frame_delta(
                    restored_luma[boundary],
                    restored_luma[boundary + 1],
                )
                right_ratio = restored_right / max(source_right, 1e-8)
            per_boundary.append(
                {
                    "boundary_frame": boundary,
                    "source_left_delta": round(source_left, 6),
                    "restored_left_delta": round(restored_left, 6),
                    "left_ratio": round(restored_left / max(source_left, 1e-8), 4),
                    "source_right_delta": round(source_right, 6) if source_right is not None else None,
                    "restored_right_delta": round(restored_right, 6) if restored_right is not None else None,
                    "right_ratio": round(right_ratio, 4) if right_ratio is not None else None,
                }
            )
        left_ratios = [item["left_ratio"] for item in per_boundary]
        right_ratios = [item["right_ratio"] for item in per_boundary if item["right_ratio"] is not None]
        return {
            "boundary_count": len(per_boundary),
            "boundary_frames": boundaries,
            "left_ratio_mean": round(float(np.mean(left_ratios)), 4) if left_ratios else None,
            "left_ratio_max": round(float(np.max(left_ratios)), 4) if left_ratios else None,
            "right_ratio_mean": round(float(np.mean(right_ratios)), 4) if right_ratios else None,
            "right_ratio_max": round(float(np.max(right_ratios)), 4) if right_ratios else None,
            "per_boundary": per_boundary,
        }

    @staticmethod
    def _unique_indices(indices) -> list[int]:
        unique = []
        seen = set()
        for index in indices:
            if index not in seen:
                unique.append(index)
                seen.add(index)
        return unique

    @staticmethod
    def _diff_frames(
        source_frames: list[PIL.Image.Image],
        restored_frames: list[PIL.Image.Image],
        *,
        scale: float,
    ) -> list[PIL.Image.Image]:
        frames = []
        for source_frame, restored_frame in zip(source_frames, restored_frames):
            restored = restored_frame.resize(source_frame.size, PIL.Image.Resampling.LANCZOS)
            diff = np.abs(
                np.asarray(source_frame.convert("RGB"), dtype=np.int16)
                - np.asarray(restored.convert("RGB"), dtype=np.int16)
            )
            frames.append(PIL.Image.fromarray(np.clip(diff * scale, 0, 255).astype(np.uint8), mode="RGB"))
        return frames

    @staticmethod
    def _temporal_diff_frames(frames: list[PIL.Image.Image], *, scale: float) -> list[PIL.Image.Image]:
        if not frames:
            return []
        result = [PIL.Image.new("RGB", frames[0].size, "black")]
        previous = np.asarray(frames[0].convert("RGB"), dtype=np.int16)
        for frame in frames[1:]:
            current = np.asarray(frame.convert("RGB"), dtype=np.int16)
            diff = np.abs(current - previous)
            result.append(PIL.Image.fromarray(np.clip(diff * scale, 0, 255).astype(np.uint8), mode="RGB"))
            previous = current
        return result

    @staticmethod
    def _to_luma(frame: PIL.Image.Image) -> np.ndarray:
        rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722

    @staticmethod
    def _mean_sharpness(frames: list[np.ndarray]) -> float:
        return float(np.mean([SeedVR2OneToOneProofBundle._sharpness(frame) for frame in frames]))

    @staticmethod
    def _sharpness(frame: np.ndarray) -> float:
        dx = np.abs(np.diff(frame, axis=1))
        dy = np.abs(np.diff(frame, axis=0))
        return float((np.mean(dx) + np.mean(dy)) / 2.0)

    @staticmethod
    def _temporal_delta(frames: list[np.ndarray]) -> float:
        if len(frames) < 2:
            return 0.0
        return float(np.mean([np.mean(np.abs(right - left)) for left, right in zip(frames, frames[1:])]))

    @staticmethod
    def _frame_delta(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean(np.abs(right - left)))

    @staticmethod
    def _array_delta(left: np.ndarray, right: np.ndarray) -> dict:
        diff = left - right
        mse = float(np.mean(np.square(diff)))
        return {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": math.sqrt(mse),
            "max_abs": int(np.max(np.abs(diff))),
            "psnr_db": 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else 99.0,
        }

    @staticmethod
    def _draw_centered_text(draw, box: tuple[int, int, int, int], text: str, font, fill) -> None:
        lines = text.split("\n")
        line_height = font.size + 4
        y = box[1] + max(0, (box[3] - box[1] - len(lines) * line_height) // 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x = box[0] + max(0, (box[2] - box[0] - (bbox[2] - bbox[0])) // 2)
            draw.text((x, y), line, fill=fill, font=font)
            y += line_height

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


if __name__ == "__main__":
    SeedVR2OneToOneProofBundle.main()
