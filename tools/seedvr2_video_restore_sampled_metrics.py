import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from mflux.utils.video_util import VideoUtil


@dataclass(frozen=True)
class SampledVideo:
    label: str
    video_path: Path
    metadata: dict | None
    fps: float
    width: int
    height: int
    frame_count: int
    sample_indices: list[int]
    sample_frames: list[PIL.Image.Image]


class SeedVR2VideoRestoreSampledMetrics:
    @staticmethod
    def main() -> None:
        args = SeedVR2VideoRestoreSampledMetrics._parse_args()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        source = SeedVR2VideoRestoreSampledMetrics._load_source(
            source_video=args.source_video,
            start_seconds=args.start_seconds,
            max_frames=args.max_frames,
            sample_count=args.sample_count,
        )
        candidates = [
            SeedVR2VideoRestoreSampledMetrics._load_candidate(
                label=label,
                path=path,
                source=source,
            )
            for label, path in zip(args.labels, args.restored_videos)
        ]

        results = {
            "source": {
                "video_path": str(source.video_path),
                "start_seconds": args.start_seconds,
                "frames": source.frame_count,
                "fps": source.fps,
                "width": source.width,
                "height": source.height,
                "sample_count": len(source.sample_indices),
                "sample_indices": source.sample_indices,
            },
            "metrics_method": {
                "comparison_space": "Sampled restored frames are downscaled back to the original source resolution before scoring.",
                "metrics": {
                    "sharpness": "Mean absolute horizontal/vertical luminance gradient.",
                    "contrast": "Luminance standard deviation.",
                    "drift_mae": "Mean absolute luma difference between sampled source and restore.",
                    "temporal_delta": "Mean absolute luma delta between adjacent sampled frames.",
                    "temporal_ratio": "restore_temporal_delta / source_temporal_delta",
                    "heuristic_score": "Sampled clip-local heuristic score for ranking candidates on the same source clip only.",
                },
            },
            "candidates": [
                SeedVR2VideoRestoreSampledMetrics._score_candidate(source=source, candidate=candidate)
                for candidate in candidates
            ],
        }

        args.output_json.write_text(json.dumps(results, indent=2))

        if args.contact_sheet_output is not None:
            SeedVR2VideoRestoreSampledMetrics._save_contact_sheet(
                source=source,
                candidates=candidates,
                scored_candidates=results["candidates"],
                output_path=args.contact_sheet_output,
                frame_count=args.contact_sheet_frame_count,
            )

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("--source-video", type=Path, required=True)
        parser.add_argument("--start-seconds", type=float, default=0.0)
        parser.add_argument("--max-frames", type=int)
        parser.add_argument("--sample-count", type=int, default=48)
        parser.add_argument("--contact-sheet-frame-count", type=int, default=8)
        parser.add_argument("--restored-videos", type=Path, nargs="+", required=True)
        parser.add_argument("--labels", nargs="+", required=True)
        parser.add_argument("--output-json", type=Path, required=True)
        parser.add_argument("--contact-sheet-output", type=Path)
        args = parser.parse_args()
        if len(args.restored_videos) != len(args.labels):
            raise ValueError("--labels must provide one label per restored video.")
        if args.sample_count <= 0:
            raise ValueError("--sample-count must be greater than zero.")
        if args.contact_sheet_frame_count <= 0:
            raise ValueError("--contact-sheet-frame-count must be greater than zero.")
        return args

    @staticmethod
    def _load_source(
        *,
        source_video: Path,
        start_seconds: float,
        max_frames: int | None,
        sample_count: int,
    ) -> SampledVideo:
        probe = VideoUtil.read_video_clip(source_video, start_seconds=start_seconds, max_frames=1)
        if probe.source_frame_count is not None:
            available_frames = max(0, probe.source_frame_count - probe.clip_start_frame)
        elif probe.source_duration_seconds is not None:
            available_frames = max(1, int(round(probe.source_duration_seconds * probe.fps)) - probe.clip_start_frame)
        else:
            raise ValueError("SeedVR2 sampled metrics require a finite source frame count or duration.")
        frame_count = min(max_frames, available_frames) if max_frames is not None else available_frames
        sample_indices = SeedVR2VideoRestoreSampledMetrics._sample_indices(
            frame_count,
            max_samples=min(sample_count, frame_count),
        )
        sample_frames = SeedVR2VideoRestoreSampledMetrics._read_sampled_frames(
            source_video,
            clip_start_frame=probe.clip_start_frame,
            sample_indices=sample_indices,
        )
        return SampledVideo(
            label="source",
            video_path=source_video,
            metadata=None,
            fps=probe.fps,
            width=probe.source_width,
            height=probe.source_height,
            frame_count=frame_count,
            sample_indices=sample_indices,
            sample_frames=sample_frames,
        )

    @staticmethod
    def _load_candidate(
        *,
        label: str,
        path: Path,
        source: SampledVideo,
    ) -> SampledVideo:
        metadata = None
        metadata_path = path.with_suffix(".metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        probe = VideoUtil.read_video_clip(path, max_frames=1)
        frame_count = probe.source_frame_count or source.frame_count
        if frame_count < source.frame_count:
            raise ValueError(f"{label} has only {frame_count} frames, expected at least {source.frame_count}.")
        sample_frames = SeedVR2VideoRestoreSampledMetrics._read_sampled_frames(
            path,
            clip_start_frame=0,
            sample_indices=source.sample_indices,
        )
        return SampledVideo(
            label=label,
            video_path=path,
            metadata=metadata,
            fps=probe.fps,
            width=probe.source_width,
            height=probe.source_height,
            frame_count=frame_count,
            sample_indices=source.sample_indices,
            sample_frames=sample_frames,
        )

    @staticmethod
    def _read_sampled_frames(path: Path, *, clip_start_frame: int, sample_indices: list[int]) -> list[PIL.Image.Image]:
        clips = VideoUtil.iter_video_frame_windows(
            path,
            start_frame=clip_start_frame,
            windows=[(index, index + 1) for index in sample_indices],
        )
        return [clip.frames[0] for clip in clips]

    @staticmethod
    def _score_candidate(source: SampledVideo, candidate: SampledVideo) -> dict:
        source_size = (source.width, source.height)
        source_luma = [SeedVR2VideoRestoreSampledMetrics._to_luma(frame) for frame in source.sample_frames]
        candidate_downscaled = [
            frame.resize(source_size, PIL.Image.Resampling.LANCZOS)
            for frame in candidate.sample_frames
        ]
        candidate_luma = [SeedVR2VideoRestoreSampledMetrics._to_luma(frame) for frame in candidate_downscaled]

        source_sharpness = SeedVR2VideoRestoreSampledMetrics._mean_sharpness(source_luma)
        source_contrast = SeedVR2VideoRestoreSampledMetrics._mean_contrast(source_luma)
        source_temporal = SeedVR2VideoRestoreSampledMetrics._temporal_delta(source_luma)

        restored_sharpness = SeedVR2VideoRestoreSampledMetrics._mean_sharpness(candidate_luma)
        restored_contrast = SeedVR2VideoRestoreSampledMetrics._mean_contrast(candidate_luma)
        restored_temporal = SeedVR2VideoRestoreSampledMetrics._temporal_delta(candidate_luma)
        drift_mae = SeedVR2VideoRestoreSampledMetrics._mean_drift(source_luma, candidate_luma)

        sharpness_gain = restored_sharpness / max(source_sharpness, 1e-8)
        contrast_gain = restored_contrast / max(source_contrast, 1e-8)
        temporal_ratio = restored_temporal / max(source_temporal, 1e-8)
        heuristic_score = SeedVR2VideoRestoreSampledMetrics._heuristic_score(
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
            "restored_width": candidate.width,
            "restored_height": candidate.height,
            "sample_resolution_metrics": {
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
            "heuristic_read": SeedVR2VideoRestoreSampledMetrics._heuristic_read(
                sharpness_gain=sharpness_gain,
                temporal_ratio=temporal_ratio,
                drift_mae=drift_mae,
            ),
        }

    @staticmethod
    def _save_contact_sheet(
        *,
        source: SampledVideo,
        candidates: list[SampledVideo],
        scored_candidates: list[dict],
        output_path: Path,
        frame_count: int,
    ) -> None:
        frame_positions = SeedVR2VideoRestoreSampledMetrics._sample_indices(
            len(source.sample_indices),
            max_samples=min(frame_count, len(source.sample_indices)),
        )
        panel_width = 220
        panel_height = 165
        row_gap = 16
        col_gap = 10
        left_label_width = 240
        header_height = 116
        rows = 1 + len(candidates)
        canvas_width = left_label_width + len(frame_positions) * panel_width + (len(frame_positions) - 1) * col_gap + 40
        canvas_height = header_height + rows * panel_height + (rows - 1) * row_gap + 60
        canvas = PIL.Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = PIL.ImageDraw.Draw(canvas)
        title_font = PIL.ImageFont.load_default(size=28)
        body_font = PIL.ImageFont.load_default(size=18)

        draw.text((24, 18), "SeedVR2 Eiffel full restore proof", fill="black", font=title_font)
        draw.text(
            (24, 54),
            "Full-video proof sampled across the clip. The restored MP4 files are the primary evidence.",
            fill=(60, 60, 60),
            font=body_font,
        )

        x = left_label_width
        for position in frame_positions:
            label = f"Frame {source.sample_indices[position]}"
            draw.text((x + 8, 86), label, fill=(70, 70, 70), font=body_font)
            x += panel_width + col_gap

        y = header_height
        SeedVR2VideoRestoreSampledMetrics._draw_contact_row(
            canvas=canvas,
            top=y,
            left_label="Source\nsampled frames",
            frames=[source.sample_frames[position] for position in frame_positions],
            panel_width=panel_width,
            panel_height=panel_height,
            left_label_width=left_label_width,
            font=body_font,
        )
        y += panel_height + row_gap

        scored_by_label = {item["label"]: item for item in scored_candidates}
        for candidate in candidates:
            item = scored_by_label[candidate.label]
            left_label = (
                f"{candidate.label}\n"
                f"{item['generation_time_seconds']}s\n"
                f"score {item['sample_resolution_metrics']['heuristic_score']:.1f}"
            )
            SeedVR2VideoRestoreSampledMetrics._draw_contact_row(
                canvas=canvas,
                top=y,
                left_label=left_label,
                frames=[candidate.sample_frames[position] for position in frame_positions],
                panel_width=panel_width,
                panel_height=panel_height,
                left_label_width=left_label_width,
                font=body_font,
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
        panel_width: int,
        panel_height: int,
        left_label_width: int,
        font,
    ) -> None:
        draw = PIL.ImageDraw.Draw(canvas)
        draw.rounded_rectangle((24, top + 12, left_label_width - 10, top + panel_height - 12), radius=10, fill=(25, 79, 140))
        SeedVR2VideoRestoreSampledMetrics._multiline_centered(
            draw=draw,
            box=(24, top + 22, left_label_width - 10, top + panel_height - 18),
            text=left_label,
            fill="white",
            font=font,
        )

        x = left_label_width
        for frame in frames:
            image = frame.resize((panel_width, panel_height), PIL.Image.Resampling.LANCZOS)
            canvas.paste(image, (x, top))
            x += panel_width + 10

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
        values = [SeedVR2VideoRestoreSampledMetrics._sharpness(frame) for frame in frames]
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
    SeedVR2VideoRestoreSampledMetrics.main()


if __name__ == "__main__":
    main()
