import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from mflux.utils.video_util import DecodedVideoClip, VideoUtil


@dataclass(frozen=True)
class CandidateVideo:
    label: str
    video_path: Path
    metadata: dict | None
    clip: DecodedVideoClip


class SeedVR2VideoRestoreMetrics:
    @staticmethod
    def main() -> None:
        args = SeedVR2VideoRestoreMetrics._parse_args()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)

        source_clip = VideoUtil.read_video_clip(
            args.source_video,
            start_seconds=args.start_seconds,
            max_frames=args.max_frames,
        )
        if args.source_clip_output is not None:
            VideoUtil.save_video(
                frames=source_clip.frames,
                path=args.source_clip_output,
                fps=source_clip.fps,
                metadata={
                    "video_path": str(args.source_video),
                    "source_clip_start_seconds": args.start_seconds,
                    "source_clip_frames": source_clip.clip_frame_count,
                    "audio_present": source_clip.audio_present,
                    "audio_copied": False,
                    "note": "Source excerpt for SeedVR2 restore comparison.",
                },
                export_json_metadata=True,
                overwrite=True,
                validate_health=True,
            )

        candidates = SeedVR2VideoRestoreMetrics._load_candidates(
            labels=args.labels,
            paths=args.restored_videos,
        )
        results = {
            "source": {
                "video_path": str(args.source_video),
                "start_seconds": args.start_seconds,
                "frames": source_clip.clip_frame_count,
                "fps": source_clip.fps,
                "width": source_clip.source_width,
                "height": source_clip.source_height,
                "audio_present": source_clip.audio_present,
            },
            "metrics_method": {
                "comparison_space": "Restored clips are downscaled back to the original source resolution before scoring.",
                "metrics": {
                    "sharpness": "Mean absolute horizontal/vertical luminance gradient.",
                    "contrast": "Luminance standard deviation.",
                    "drift_mae": "Mean absolute luma difference between source and downscaled restore.",
                    "temporal_delta": "Mean absolute luma delta between adjacent frames.",
                    "temporal_ratio": "restore_temporal_delta / source_temporal_delta",
                    "heuristic_score": "Clip-local heuristic score for ranking candidates on the same source clip only.",
                },
            },
            "candidates": [],
        }

        for candidate in candidates:
            results["candidates"].append(
                SeedVR2VideoRestoreMetrics._score_candidate(
                    source_clip=source_clip,
                    candidate=candidate,
                )
            )

        with open(args.output_json, "w") as handle:
            json.dump(results, handle, indent=2)

        if args.comparison_video_output is not None:
            SeedVR2VideoRestoreMetrics._save_comparison_video(
                source_clip=source_clip,
                candidates=candidates,
                scored_candidates=results["candidates"],
                output_path=args.comparison_video_output,
            )

        if args.contact_sheet_output is not None:
            SeedVR2VideoRestoreMetrics._save_contact_sheet(
                source_clip=source_clip,
                candidates=candidates,
                scored_candidates=results["candidates"],
                output_path=args.contact_sheet_output,
            )

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-video", type=Path, required=True)
        parser.add_argument("--start-seconds", type=float, default=0.0)
        parser.add_argument("--max-frames", type=int, required=True)
        parser.add_argument("--restored-videos", type=Path, nargs="+", required=True)
        parser.add_argument("--labels", nargs="+", required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--source-clip-output", type=Path)
        parser.add_argument("--comparison-video-output", type=Path)
        parser.add_argument("--contact-sheet-output", type=Path)
        args = parser.parse_args()
        if len(args.restored_videos) != len(args.labels):
            raise ValueError("--labels must provide one label per restored video.")
        return args

    @staticmethod
    def _load_candidates(labels: list[str], paths: list[Path]) -> list[CandidateVideo]:
        candidates: list[CandidateVideo] = []
        for label, path in zip(labels, paths):
            metadata = None
            metadata_path = path.with_suffix(".metadata.json")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
            clip = VideoUtil.read_video_clip(path)
            candidates.append(CandidateVideo(label=label, video_path=path, metadata=metadata, clip=clip))
        return candidates

    @staticmethod
    def _score_candidate(source_clip: DecodedVideoClip, candidate: CandidateVideo) -> dict:
        if candidate.clip.clip_frame_count != source_clip.clip_frame_count:
            raise ValueError(
                f"{candidate.label} has {candidate.clip.clip_frame_count} frames, expected {source_clip.clip_frame_count}."
            )

        source_size = (source_clip.source_width, source_clip.source_height)
        source_luma = [SeedVR2VideoRestoreMetrics._to_luma(frame) for frame in source_clip.frames]
        candidate_downscaled = [
            frame.resize(source_size, PIL.Image.Resampling.LANCZOS)
            for frame in candidate.clip.frames
        ]
        candidate_luma = [SeedVR2VideoRestoreMetrics._to_luma(frame) for frame in candidate_downscaled]

        source_sharpness = SeedVR2VideoRestoreMetrics._mean_sharpness(source_luma)
        source_contrast = SeedVR2VideoRestoreMetrics._mean_contrast(source_luma)
        source_temporal = SeedVR2VideoRestoreMetrics._temporal_delta(source_luma)

        restored_sharpness = SeedVR2VideoRestoreMetrics._mean_sharpness(candidate_luma)
        restored_contrast = SeedVR2VideoRestoreMetrics._mean_contrast(candidate_luma)
        restored_temporal = SeedVR2VideoRestoreMetrics._temporal_delta(candidate_luma)
        drift_mae = SeedVR2VideoRestoreMetrics._mean_drift(source_luma, candidate_luma)

        sharpness_gain = restored_sharpness / max(source_sharpness, 1e-8)
        contrast_gain = restored_contrast / max(source_contrast, 1e-8)
        temporal_ratio = restored_temporal / max(source_temporal, 1e-8)
        heuristic_score = SeedVR2VideoRestoreMetrics._heuristic_score(
            sharpness_gain=sharpness_gain,
            contrast_gain=contrast_gain,
            drift_mae=drift_mae,
            temporal_ratio=temporal_ratio,
        )

        return {
            "label": candidate.label,
            "video_path": str(candidate.video_path),
            "model": candidate.metadata.get("model") if candidate.metadata else None,
            "generation_time_seconds": candidate.metadata.get("generation_time_seconds") if candidate.metadata else None,
            "restored_width": candidate.clip.source_width,
            "restored_height": candidate.clip.source_height,
            "source_resolution_metrics": {
                "source_sharpness": round(source_sharpness, 6),
                "restored_sharpness": round(restored_sharpness, 6),
                "sharpness_gain": round(sharpness_gain, 4),
                "source_contrast": round(source_contrast, 6),
                "restored_contrast": round(restored_contrast, 6),
                "contrast_gain": round(contrast_gain, 4),
                "source_temporal_delta": round(source_temporal, 6),
                "restored_temporal_delta": round(restored_temporal, 6),
                "temporal_ratio": round(temporal_ratio, 4),
                "drift_mae": round(drift_mae, 6),
                "heuristic_score": round(heuristic_score, 2),
            },
            "heuristic_read": SeedVR2VideoRestoreMetrics._heuristic_read(
                sharpness_gain=sharpness_gain,
                temporal_ratio=temporal_ratio,
                drift_mae=drift_mae,
            ),
        }

    @staticmethod
    def _save_comparison_video(
        source_clip: DecodedVideoClip,
        candidates: list[CandidateVideo],
        scored_candidates: list[dict],
        output_path: Path,
    ) -> None:
        labels = {
            item["label"]: SeedVR2VideoRestoreMetrics._video_label_text(item)
            for item in scored_candidates
        }
        panel_width = max([frame.width for frame in candidates[0].clip.frames] + [source_clip.source_width * 2])
        panel_height = max([frame.height for frame in candidates[0].clip.frames] + [source_clip.source_height * 2])
        source_frames = [
            frame.resize((panel_width, panel_height), PIL.Image.Resampling.NEAREST) for frame in source_clip.frames
        ]

        rendered_frames: list[PIL.Image.Image] = []
        for frame_index in range(source_clip.clip_frame_count):
            panels = [("Source x2 nearest", source_frames[frame_index])]
            for candidate in candidates:
                frame = candidate.clip.frames[frame_index]
                if frame.size != (panel_width, panel_height):
                    frame = frame.resize((panel_width, panel_height), PIL.Image.Resampling.LANCZOS)
                panels.append((labels[candidate.label], frame))
            rendered_frames.append(SeedVR2VideoRestoreMetrics._stack_panels_horizontally(panels))

        VideoUtil.save_video(
            frames=rendered_frames,
            path=output_path,
            fps=source_clip.fps,
            metadata={"note": "SeedVR2 restore comparison video."},
            export_json_metadata=True,
            overwrite=True,
            validate_health=True,
        )

    @staticmethod
    def _save_contact_sheet(
        source_clip: DecodedVideoClip,
        candidates: list[CandidateVideo],
        scored_candidates: list[dict],
        output_path: Path,
    ) -> None:
        frame_indices = SeedVR2VideoRestoreMetrics._sample_indices(source_clip.clip_frame_count, max_samples=5)
        panel_width = 240
        panel_height = 180
        row_gap = 16
        col_gap = 10
        left_label_width = 220
        header_height = 110
        rows = 1 + len(candidates)
        canvas_width = left_label_width + len(frame_indices) * panel_width + (len(frame_indices) - 1) * col_gap + 40
        canvas_height = header_height + rows * panel_height + (rows - 1) * row_gap + 60
        canvas = PIL.Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = PIL.ImageDraw.Draw(canvas)
        title_font = PIL.ImageFont.load_default(size=28)
        body_font = PIL.ImageFont.load_default(size=18)
        small_font = PIL.ImageFont.load_default(size=15)

        draw.text((24, 18), "SeedVR2 Eiffel bounded restore proof", fill="black", font=title_font)
        draw.text(
            (24, 54),
            "Frames sampled from the same 6-frame source excerpt. Videos are the primary proof; this sheet supports review.",
            fill=(60, 60, 60),
            font=body_font,
        )

        x = left_label_width
        for frame_index in frame_indices:
            label = f"Frame {frame_index}"
            draw.text((x + 8, 84), label, fill=(70, 70, 70), font=body_font)
            x += panel_width + col_gap

        y = header_height
        SeedVR2VideoRestoreMetrics._draw_contact_row(
            canvas=canvas,
            top=y,
            left_label="Source\n320x240 excerpt",
            frames=source_clip.frames,
            frame_indices=frame_indices,
            panel_width=panel_width,
            panel_height=panel_height,
            left_label_width=left_label_width,
            font=body_font,
            small_font=small_font,
            footer="",
        )
        y += panel_height + row_gap

        scored_by_label = {item["label"]: item for item in scored_candidates}
        for candidate in candidates:
            item = scored_by_label[candidate.label]
            left_label = SeedVR2VideoRestoreMetrics._contact_label_text(item)
            SeedVR2VideoRestoreMetrics._draw_contact_row(
                canvas=canvas,
                top=y,
                left_label=left_label,
                frames=candidate.clip.frames,
                frame_indices=frame_indices,
                panel_width=panel_width,
                panel_height=panel_height,
                left_label_width=left_label_width,
                font=body_font,
                small_font=small_font,
                footer="",
            )
            y += panel_height + row_gap

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)

    @staticmethod
    def _draw_contact_row(
        *,
        canvas: PIL.Image.Image,
        top: int,
        left_label: str,
        frames: list[PIL.Image.Image],
        frame_indices: list[int],
        panel_width: int,
        panel_height: int,
        left_label_width: int,
        font,
        small_font,
        footer: str,
    ) -> None:
        draw = PIL.ImageDraw.Draw(canvas)
        draw.rounded_rectangle((24, top + 12, left_label_width - 10, top + panel_height - 12), radius=10, fill=(25, 79, 140))
        SeedVR2VideoRestoreMetrics._multiline_centered(
            draw=draw,
            box=(24, top + 22, left_label_width - 10, top + panel_height - 40),
            text=left_label,
            fill="white",
            font=font,
        )
        if footer:
            draw.text((24, top + panel_height - 26), footer, fill=(90, 90, 90), font=small_font)

        x = left_label_width
        for frame_index in frame_indices:
            image = frames[frame_index].resize((panel_width, panel_height), PIL.Image.Resampling.LANCZOS)
            canvas.paste(image, (x, top))
            x += panel_width + 10

    @staticmethod
    def _stack_panels_horizontally(panels: list[tuple[str, PIL.Image.Image]]) -> PIL.Image.Image:
        label_height = 70
        gap = 12
        width = sum(frame.width for _, frame in panels) + gap * (len(panels) - 1)
        height = max(frame.height for _, frame in panels) + label_height
        canvas = PIL.Image.new("RGB", (width, height), "white")
        draw = PIL.ImageDraw.Draw(canvas)
        font = PIL.ImageFont.load_default(size=22)
        small_font = PIL.ImageFont.load_default(size=16)

        x = 0
        for label, frame in panels:
            draw.rounded_rectangle((x, 0, x + frame.width, 58), radius=10, fill=(233, 239, 246))
            title, subtitle = SeedVR2VideoRestoreMetrics._split_label(label)
            draw.text((x + 12, 10), title, fill="black", font=font)
            if subtitle:
                draw.text((x + 12, 34), subtitle, fill=(80, 80, 80), font=small_font)
            canvas.paste(frame, (x, label_height))
            x += frame.width + gap
        return canvas

    @staticmethod
    def _multiline_centered(draw, box: tuple[int, int, int, int], text: str, fill, font) -> None:
        lines = text.split("\n")
        line_height = font.size + 6
        total_height = len(lines) * line_height
        y = box[1] + max(0, (box[3] - box[1] - total_height) // 2)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            x = box[0] + max(0, (box[2] - box[0] - width) // 2)
            draw.text((x, y), line, fill=fill, font=font)
            y += line_height

    @staticmethod
    def _contact_label_text(item: dict) -> str:
        return f"{item['label']}\n{item['generation_time_seconds']}s"

    @staticmethod
    def _video_label_text(item: dict) -> str:
        metrics = item["source_resolution_metrics"]
        return (
            f"{item['label']}\n"
            f"{item['generation_time_seconds']}s | score {metrics['heuristic_score']:.1f}\n"
            f"sharp {metrics['sharpness_gain']:.2f}x | drift {metrics['drift_mae']:.3f}"
        )

    @staticmethod
    def _split_label(label: str) -> tuple[str, str]:
        lines = label.split("\n")
        if len(lines) == 1:
            return lines[0], ""
        return lines[0], " | ".join(lines[1:])

    @staticmethod
    def _sample_indices(frame_count: int, max_samples: int) -> list[int]:
        if frame_count <= max_samples:
            return list(range(frame_count))
        positions = np.linspace(0, frame_count - 1, num=max_samples)
        return [int(round(value)) for value in positions]

    @staticmethod
    def _to_luma(frame: PIL.Image.Image) -> np.ndarray:
        rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722

    @staticmethod
    def _mean_sharpness(frames: list[np.ndarray]) -> float:
        values = [SeedVR2VideoRestoreMetrics._sharpness(frame) for frame in frames]
        return float(np.mean(values))

    @staticmethod
    def _sharpness(frame: np.ndarray) -> float:
        dx = np.abs(np.diff(frame, axis=1)).mean()
        dy = np.abs(np.diff(frame, axis=0)).mean()
        return float((dx + dy) / 2.0)

    @staticmethod
    def _mean_contrast(frames: list[np.ndarray]) -> float:
        return float(np.mean([frame.std() for frame in frames]))

    @staticmethod
    def _temporal_delta(frames: list[np.ndarray]) -> float:
        if len(frames) < 2:
            return 0.0
        values = [
            np.abs(frames[index + 1] - frames[index]).mean()
            for index in range(len(frames) - 1)
        ]
        return float(np.mean(values))

    @staticmethod
    def _mean_drift(source_frames: list[np.ndarray], restored_frames: list[np.ndarray]) -> float:
        values = [
            np.abs(restored - source).mean()
            for source, restored in zip(source_frames, restored_frames)
        ]
        return float(np.mean(values))

    @staticmethod
    def _heuristic_score(
        *,
        sharpness_gain: float,
        contrast_gain: float,
        drift_mae: float,
        temporal_ratio: float,
    ) -> float:
        sharpness_component = min(max((sharpness_gain - 1.0) / 0.25, 0.0), 1.0)
        contrast_component = min(max((contrast_gain - 1.0) / 0.10, 0.0), 1.0)
        drift_component = min(max(1.0 - drift_mae / 0.08, 0.0), 1.0)
        temporal_component = min(
            max(1.0 - abs(math.log(max(temporal_ratio, 1e-8))) / math.log(1.35), 0.0),
            1.0,
        )
        return 100.0 * (
            0.4 * sharpness_component
            + 0.1 * contrast_component
            + 0.3 * drift_component
            + 0.2 * temporal_component
        )

    @staticmethod
    def _heuristic_read(
        *,
        sharpness_gain: float,
        temporal_ratio: float,
        drift_mae: float,
    ) -> str:
        if sharpness_gain < 1.02 or temporal_ratio < 0.75:
            return "likely over-smoothed"
        if drift_mae > 0.08 or temporal_ratio > 1.35:
            return "aggressive drift or flicker risk"
        return "balanced restore candidate"


def main() -> None:
    SeedVR2VideoRestoreMetrics.main()


if __name__ == "__main__":
    main()
