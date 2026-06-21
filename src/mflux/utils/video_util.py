import logging
import shutil
import subprocess
from contextlib import suppress
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Iterator

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.config import ModelConfig
from mflux.utils.image_util import ImageUtil
from mflux.utils.tensor_health import TensorHealth
from mflux.utils.video_health import VideoHealth

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecodedVideoClip:
    frames: list[PIL.Image.Image]
    fps: float
    source_width: int
    source_height: int
    source_frame_count: int | None
    source_duration_seconds: float | None
    audio_present: bool
    clip_start_frame: int
    clip_frame_count: int


@dataclass(frozen=True)
class SourceVideoInfo:
    fps: float
    source_width: int
    source_height: int
    source_frame_count: int | None
    source_duration_seconds: float | None
    audio_present: bool


@dataclass(frozen=True)
class AudioCopyResult:
    audio_present: bool
    audio_copied: bool
    copy_mode: str | None
    reason: str | None


class VideoStreamWriter:
    def __init__(
        self,
        *,
        path: str | Path,
        fps: int | float,
        width: int,
        height: int,
        overwrite: bool = True,
    ):
        import av

        self.file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self._should_replace = False

        with NamedTemporaryFile(
            suffix=self.file_path.suffix or ".mp4",
            prefix=f".{self.file_path.stem}-",
            dir=self.file_path.parent,
            delete=False,
        ) as temp_file:
            self.temp_path = Path(temp_file.name)

        self.container = av.open(str(self.temp_path), mode="w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=VideoUtil._fps_to_rate(fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "medium"}
        self._closed = False

    def write_frames(self, frames: list[PIL.Image.Image]) -> None:
        import av

        for frame in frames:
            rgb = frame.convert("RGB")
            if rgb.size != (self.width, self.height):
                rgb = rgb.resize((self.width, self.height), PIL.Image.Resampling.LANCZOS)
            video_frame = av.VideoFrame.from_ndarray(np.array(rgb), format="rgb24")
            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)

    def write_frame_arrays(self, frames: np.ndarray) -> None:
        import av

        for frame in frames:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                rgb = PIL.Image.fromarray(frame, mode="RGB").resize((self.width, self.height), PIL.Image.Resampling.LANCZOS)
                frame = np.array(rgb, dtype=np.uint8)
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)

    def close(self) -> Path:
        if self._closed:
            return self.file_path
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
            self._should_replace = True
            self._closed = True
        finally:
            if self._should_replace:
                self.temp_path.replace(self.file_path)
            elif self.temp_path.exists():
                self.temp_path.unlink()
        return self.file_path

    def abort(self) -> None:
        if self._closed:
            return
        with suppress(OSError, RuntimeError, ValueError):
            self.container.close()
        self._closed = True
        if self.temp_path.exists():
            self.temp_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class VideoUtil:
    @staticmethod
    def copy_source_audio_to_video(
        *,
        source_video_path: str | Path,
        restored_video_path: str | Path,
        clip_start_seconds: float,
        clip_duration_seconds: float,
    ) -> AudioCopyResult:
        if clip_start_seconds < 0:
            raise ValueError("clip_start_seconds must be greater than or equal to zero.")
        if clip_duration_seconds <= 0:
            raise ValueError("clip_duration_seconds must be greater than zero.")

        source_info = VideoUtil.inspect_video(source_video_path)
        if not source_info.audio_present:
            return AudioCopyResult(
                audio_present=False,
                audio_copied=False,
                copy_mode=None,
                reason="no_source_audio",
            )

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="ffmpeg_not_found",
            )

        restored_path = Path(restored_video_path)
        restored_info = VideoUtil.inspect_video(restored_path)
        alignment_tolerance = VideoUtil._media_alignment_tolerance_seconds(restored_info.fps)
        if restored_info.source_duration_seconds is None:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="restored_duration_unknown",
            )
        if abs(restored_info.source_duration_seconds - clip_duration_seconds) > alignment_tolerance:
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="restored_duration_mismatch",
            )
        if (
            source_info.source_duration_seconds is not None
            and clip_start_seconds + clip_duration_seconds > source_info.source_duration_seconds + alignment_tolerance
        ):
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="source_clip_out_of_range",
            )

        with NamedTemporaryFile(
            suffix=restored_path.suffix or ".mp4",
            prefix=f".{restored_path.stem}-audio-",
            dir=restored_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        command = VideoUtil._build_audio_copy_command(
            ffmpeg_path=ffmpeg_path,
            restored_video_path=restored_path,
            source_video_path=Path(source_video_path),
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            output_path=temp_path,
        )

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            with suppress(FileNotFoundError):
                temp_path.unlink()
            stderr = (error.stderr or "").strip()
            if stderr:
                log.warning("Audio copy-through failed for %s: %s", restored_path, stderr.splitlines()[-1])
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason="ffmpeg_mux_failed",
            )

        validation_error = VideoUtil._validate_copied_audio_output(
            output_path=temp_path,
            expected_video=restored_info,
            expected_audio_duration_seconds=clip_duration_seconds,
        )
        if validation_error is not None:
            with suppress(FileNotFoundError):
                temp_path.unlink()
            log.warning("Audio copy-through rejected for %s: %s", restored_path, validation_error)
            return AudioCopyResult(
                audio_present=True,
                audio_copied=False,
                copy_mode=None,
                reason=validation_error,
            )

        temp_path.replace(restored_path)
        return AudioCopyResult(
            audio_present=True,
            audio_copied=True,
            copy_mode="ffmpeg_copy_video_aac_audio",
            reason=None,
        )

    @staticmethod
    def to_video(
        decoded_latents: mx.array,
        fps: int | float,
        model_config: ModelConfig,
        seed: int,
        prompt: str,
        steps: int,
        guidance: float | None,
        quantization: int,
        generation_time: float,
        flow_shift: float | None = None,
        solver: str | None = None,
        guidance_2: float | None = None,
        task: str = "text-to-video",
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        negative_prompt: str | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        requested_width: int | None = None,
        requested_height: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        extra_metadata: dict | None = None,
    ):
        from mflux.utils.generated_video import GeneratedVideo

        frames = VideoUtil._latents_to_frames(decoded_latents)
        first_frame = frames[0]
        return GeneratedVideo(
            frames=frames,
            fps=fps,
            model_config=model_config,
            seed=seed,
            prompt=prompt,
            steps=steps,
            guidance=guidance,
            flow_shift=flow_shift,
            solver=solver,
            guidance_2=guidance_2,
            precision=ModelConfig.precision,
            quantization=quantization,
            generation_time=generation_time,
            height=first_frame.height,
            width=first_frame.width,
            task=task,
            image_path=image_path,
            video_path=video_path,
            negative_prompt=negative_prompt,
            source_width=source_width,
            source_height=source_height,
            requested_width=requested_width,
            requested_height=requested_height,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def save_video(
        frames: list[PIL.Image.Image],
        path: str | Path,
        fps: int | float,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
    ) -> Path:
        if not frames:
            raise ValueError("Cannot save a video without frames.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frames[0].size
        frame_health = None
        file_health = None
        if validate_health:
            frame_health = VideoHealth.validate_frames(
                frames,
                fps=fps,
                expected_width=width,
                expected_height=height,
                strict_visual=True,
            )
        VideoUtil._save_video_with_pyav(frames=frames, file_path=file_path, fps=fps, width=width, height=height)
        if validate_health:
            file_health = VideoHealth.validate_file(
                file_path,
                expected_width=width,
                expected_height=height,
                expected_frames=len(frames),
                expected_fps=fps,
                strict_visual=True,
            )

        if metadata is not None and validate_health:
            metadata = dict(metadata)
            metadata["video_health"] = {
                "frames": asdict(frame_health),
                "file": asdict(file_health),
            }

        VideoUtil._save_metadata(
            file_path=file_path,
            metadata=metadata,
            export_json_metadata=export_json_metadata,
        )

        log.info(f"Video saved successfully at: {file_path}")
        return file_path

    @staticmethod
    def save_video_batches(
        frame_batches: Iterable[list[PIL.Image.Image]],
        path: str | Path,
        fps: int | float,
        metadata: dict | None = None,
        export_json_metadata: bool = False,
        overwrite: bool = True,
        validate_health: bool = True,
    ) -> Path:
        batch_iterator = iter(frame_batches)
        first_batch = next(batch_iterator, None)
        if first_batch is None or not first_batch:
            raise ValueError("Cannot save a video without frames.")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        first_frame = first_batch[0]
        width, height = first_frame.size
        file_path = ImageUtil.resolve_output_path(path=path, overwrite=overwrite)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        VideoUtil._save_video_batches_with_pyav(
            frame_batches=chain([first_batch], batch_iterator),
            file_path=file_path,
            fps=fps,
            width=width,
            height=height,
        )

        if validate_health:
            file_health = VideoHealth.validate_file(
                file_path,
                expected_width=width,
                expected_height=height,
                expected_fps=fps,
                strict_visual=True,
            )
            if metadata is not None:
                metadata = dict(metadata)
                metadata["video_health"] = {
                    "file": asdict(file_health),
                }

        VideoUtil._save_metadata(
            file_path=file_path,
            metadata=metadata,
            export_json_metadata=export_json_metadata,
        )

        log.info(f"Video saved successfully at: {file_path}")
        return file_path

    @staticmethod
    def extract_frame(path: str | Path, index: int = 0) -> PIL.Image.Image:
        if index < 0:
            raise ValueError("Frame index must be greater than or equal to zero.")

        import av

        with av.open(str(path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")
            video_stream = container.streams.video[0]
            for frame_number, frame in enumerate(container.decode(video_stream)):
                if frame_number == index:
                    return PIL.Image.fromarray(frame.to_ndarray(format="rgb24"))
        raise RuntimeError(f"Could not read frame {index} from {path}")

    @staticmethod
    def read_video_clip(
        path: str | Path,
        *,
        start_seconds: float = 0.0,
        max_frames: int | None = None,
    ) -> DecodedVideoClip:
        if start_seconds < 0:
            raise ValueError("start_seconds must be greater than or equal to zero.")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be greater than zero when provided.")

        import av

        file_path = Path(path)
        with av.open(str(file_path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")

            video_stream = container.streams.video[0]
            fps = VideoUtil._rate_to_float(video_stream.average_rate or video_stream.base_rate)
            if fps is None or fps <= 0:
                raise RuntimeError(f"Could not determine a valid video fps for {path}")

            source_width = int(video_stream.width)
            source_height = int(video_stream.height)
            source_frame_count = int(video_stream.frames) if video_stream.frames else None
            source_duration_seconds = VideoUtil._duration_seconds(container=container, video_stream=video_stream, fps=fps, source_frame_count=source_frame_count)  # fmt: off
            audio_present = len(container.streams.audio) > 0

            frames: list[PIL.Image.Image] = []
            clip_start_frame: int | None = None
            for frame_index, frame in enumerate(container.decode(video_stream)):
                frame_time = frame.time
                if frame_time is None:
                    frame_time = frame_index / fps
                if frame_time + 1e-9 < start_seconds:
                    continue
                if clip_start_frame is None:
                    clip_start_frame = frame_index
                frames.append(PIL.Image.fromarray(frame.to_ndarray(format="rgb24")))
                if max_frames is not None and len(frames) >= max_frames:
                    break

        if not frames:
            raise RuntimeError(f"Could not decode any video frames from {path}")

        return DecodedVideoClip(
            frames=frames,
            fps=fps,
            source_width=source_width,
            source_height=source_height,
            source_frame_count=source_frame_count,
            source_duration_seconds=source_duration_seconds,
            audio_present=audio_present,
            clip_start_frame=clip_start_frame or 0,
            clip_frame_count=len(frames),
        )

    @staticmethod
    def inspect_video(path: str | Path) -> SourceVideoInfo:
        import av

        file_path = Path(path)
        with av.open(str(file_path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")

            video_stream = container.streams.video[0]
            fps = VideoUtil._rate_to_float(video_stream.average_rate or video_stream.base_rate)
            if fps is None or fps <= 0:
                raise RuntimeError(f"Could not determine a valid video fps for {path}")

            source_width = int(video_stream.width)
            source_height = int(video_stream.height)
            source_frame_count = int(video_stream.frames) if video_stream.frames else None
            source_duration_seconds = VideoUtil._duration_seconds(
                container=container,
                video_stream=video_stream,
                fps=fps,
                source_frame_count=source_frame_count,
            )
            audio_present = len(container.streams.audio) > 0

        return SourceVideoInfo(
            fps=fps,
            source_width=source_width,
            source_height=source_height,
            source_frame_count=source_frame_count,
            source_duration_seconds=source_duration_seconds,
            audio_present=audio_present,
        )

    @staticmethod
    def read_video_frame_window(
        path: str | Path,
        *,
        start_frame: int = 0,
        max_frames: int | None = None,
    ) -> DecodedVideoClip:
        if start_frame < 0:
            raise ValueError("start_frame must be greater than or equal to zero.")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be greater than zero when provided.")

        import av

        source_info = VideoUtil.inspect_video(path)
        file_path = Path(path)
        with av.open(str(file_path)) as container:
            video_stream = container.streams.video[0]
            frames: list[PIL.Image.Image] = []
            for frame_index, frame in enumerate(container.decode(video_stream)):
                if frame_index < start_frame:
                    continue
                frames.append(PIL.Image.fromarray(frame.to_ndarray(format="rgb24")))
                if max_frames is not None and len(frames) >= max_frames:
                    break

        if not frames:
            raise RuntimeError(f"Could not decode any video frames from {path}")

        return DecodedVideoClip(
            frames=frames,
            fps=source_info.fps,
            source_width=source_info.source_width,
            source_height=source_info.source_height,
            source_frame_count=source_info.source_frame_count,
            source_duration_seconds=source_info.source_duration_seconds,
            audio_present=source_info.audio_present,
            clip_start_frame=start_frame,
            clip_frame_count=len(frames),
        )

    @staticmethod
    def iter_video_frame_windows(
        path: str | Path,
        *,
        start_frame: int = 0,
        windows: list[tuple[int, int]],
    ) -> Iterator[DecodedVideoClip]:
        if start_frame < 0:
            raise ValueError("start_frame must be greater than or equal to zero.")
        if not windows:
            raise ValueError("windows must contain at least one frame range.")

        normalized_windows: list[tuple[int, int]] = []
        previous_start = -1
        for window_start, window_end in windows:
            if window_start < 0:
                raise ValueError("window start_frame must be greater than or equal to zero.")
            if window_end <= window_start:
                raise ValueError("window end_frame must be greater than start_frame.")
            if window_start < previous_start:
                raise ValueError("windows must be sorted by start_frame.")
            normalized_windows.append((window_start, window_end))
            previous_start = window_start

        import av

        source_info = VideoUtil.inspect_video(path)
        absolute_windows = [
            (start_frame + window_start, start_frame + window_end)
            for window_start, window_end in normalized_windows
        ]
        file_path = Path(path)
        with av.open(str(file_path)) as container:
            if len(container.streams.video) == 0:
                raise RuntimeError(f"Could not find a video stream in {path}")

            video_stream = container.streams.video[0]
            active_windows: list[dict] = []
            next_window_index = 0
            for frame_index, frame in enumerate(container.decode(video_stream)):
                while (
                    next_window_index < len(absolute_windows)
                    and absolute_windows[next_window_index][0] == frame_index
                ):
                    absolute_start, absolute_end = absolute_windows[next_window_index]
                    relative_start, _ = normalized_windows[next_window_index]
                    active_windows.append(
                        {
                            "absolute_end": absolute_end,
                            "relative_start": relative_start,
                            "frames": [],
                        }
                    )
                    next_window_index += 1

                if not active_windows:
                    if next_window_index >= len(absolute_windows):
                        break
                    if frame_index < absolute_windows[next_window_index][0]:
                        continue

                pil_frame = PIL.Image.fromarray(frame.to_ndarray(format="rgb24"))
                for active_window in active_windows:
                    if frame_index < active_window["absolute_end"]:
                        active_window["frames"].append(pil_frame)

                completed_windows: list[dict] = []
                while active_windows and frame_index + 1 >= active_windows[0]["absolute_end"]:
                    completed_windows.append(active_windows.pop(0))

                for completed_window in completed_windows:
                    yield DecodedVideoClip(
                        frames=completed_window["frames"],
                        fps=source_info.fps,
                        source_width=source_info.source_width,
                        source_height=source_info.source_height,
                        source_frame_count=source_info.source_frame_count,
                        source_duration_seconds=source_info.source_duration_seconds,
                        audio_present=source_info.audio_present,
                        clip_start_frame=completed_window["relative_start"],
                        clip_frame_count=len(completed_window["frames"]),
                    )

        if active_windows or next_window_index < len(absolute_windows):
            raise RuntimeError(f"Could not decode all requested video windows from {path}")

    @staticmethod
    def _latents_to_frames(decoded_latents: mx.array) -> list[PIL.Image.Image]:
        video_np = VideoUtil._latents_to_frame_arrays(decoded_latents)
        return [PIL.Image.fromarray(frame) for frame in video_np]

    @staticmethod
    def _latents_to_frame_arrays(decoded_latents: mx.array) -> np.ndarray:
        if decoded_latents.ndim != 5:
            raise ValueError(f"Expected decoded video latents with shape [B, C, F, H, W], got {decoded_latents.shape}")
        video = decoded_latents[0]
        video = mx.transpose(video, (1, 2, 3, 0)).astype(mx.float32)
        video_np = np.array(video, dtype=np.float32)
        for frame_index in range(video_np.shape[0]):
            frame_np = video_np[frame_index]
            TensorHealth.ensure_finite(
                frame_np,
                name="decoded_video_frame",
                phase="video-frame-conversion",
                frame=frame_index + 1,
                total_frames=video_np.shape[0],
            )
        return (np.clip(video_np / 2 + 0.5, 0, 1) * 255).round().astype("uint8")

    @staticmethod
    def _save_video_with_pyav(
        frames: list[PIL.Image.Image],
        file_path: Path,
        fps: int | float,
        width: int,
        height: int,
    ) -> None:
        import av

        with NamedTemporaryFile(
            suffix=file_path.suffix or ".mp4",
            prefix=f".{file_path.stem}-",
            dir=file_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        should_replace = False
        try:
            container = av.open(str(temp_path), mode="w", options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=VideoUtil._fps_to_rate(fps))
            with container:
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "medium"}
                for frame in frames:
                    rgb = frame.convert("RGB")
                    if rgb.size != (width, height):
                        rgb = rgb.resize((width, height), PIL.Image.Resampling.LANCZOS)
                    video_frame = av.VideoFrame.from_ndarray(np.array(rgb), format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            should_replace = True
        finally:
            if should_replace:
                temp_path.replace(file_path)
            elif temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _save_video_batches_with_pyav(
        frame_batches: Iterable[list[PIL.Image.Image]],
        file_path: Path,
        fps: int | float,
        width: int,
        height: int,
    ) -> None:
        import av

        with NamedTemporaryFile(
            suffix=file_path.suffix or ".mp4",
            prefix=f".{file_path.stem}-",
            dir=file_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        should_replace = False
        try:
            container = av.open(str(temp_path), mode="w", options={"movflags": "+faststart"})
            stream = container.add_stream("libx264", rate=VideoUtil._fps_to_rate(fps))
            with container:
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "medium"}
                for batch in frame_batches:
                    for frame in batch:
                        rgb = frame.convert("RGB")
                        if rgb.size != (width, height):
                            rgb = rgb.resize((width, height), PIL.Image.Resampling.LANCZOS)
                        video_frame = av.VideoFrame.from_ndarray(np.array(rgb), format="rgb24")
                        for packet in stream.encode(video_frame):
                            container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            should_replace = True
        finally:
            if should_replace:
                temp_path.replace(file_path)
            elif temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _save_metadata(file_path: Path, metadata: dict | None, export_json_metadata: bool) -> None:
        if export_json_metadata and metadata is not None:
            from mflux.utils.generated_video import GeneratedVideo

            GeneratedVideo.save_metadata(file_path, metadata)

    @staticmethod
    def _build_audio_copy_command(
        *,
        ffmpeg_path: str,
        restored_video_path: Path,
        source_video_path: Path,
        clip_start_seconds: float,
        clip_duration_seconds: float,
        output_path: Path,
    ) -> list[str]:
        return [
            ffmpeg_path,
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(restored_video_path),
            "-ss",
            f"{clip_start_seconds:.6f}",
            "-t",
            f"{clip_duration_seconds:.6f}",
            "-i",
            str(source_video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            "aresample=async=1:first_pts=0",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]

    @staticmethod
    def _validate_copied_audio_output(
        *,
        output_path: Path,
        expected_video: SourceVideoInfo,
        expected_audio_duration_seconds: float,
    ) -> str | None:
        output_info = VideoUtil.inspect_video(output_path)
        if not output_info.audio_present:
            return "output_missing_audio"
        if output_info.source_width != expected_video.source_width or output_info.source_height != expected_video.source_height:
            return "output_video_dimensions_mismatch"
        if abs(output_info.fps - expected_video.fps) > 1e-6:
            return "output_video_fps_mismatch"
        if (
            expected_video.source_frame_count is not None
            and output_info.source_frame_count is not None
            and output_info.source_frame_count != expected_video.source_frame_count
        ):
            return "output_video_frame_count_mismatch"
        if (
            expected_video.source_duration_seconds is not None
            and output_info.source_duration_seconds is not None
            and abs(output_info.source_duration_seconds - expected_video.source_duration_seconds)
            > VideoUtil._media_alignment_tolerance_seconds(expected_video.fps)
        ):
            return "output_video_duration_mismatch"

        audio_duration_seconds = VideoUtil._audio_duration_seconds(output_path)
        if audio_duration_seconds is None:
            return "output_audio_duration_unknown"
        if abs(audio_duration_seconds - expected_audio_duration_seconds) > VideoUtil._media_alignment_tolerance_seconds(expected_video.fps):
            return "output_audio_duration_mismatch"
        return None

    @staticmethod
    def _duration_seconds(
        *,
        container,
        video_stream,
        fps: float,
        source_frame_count: int | None,
    ) -> float | None:
        if video_stream.duration is not None and video_stream.time_base is not None:
            return float(video_stream.duration * video_stream.time_base)
        if container.duration is not None:
            return float(container.duration) / 1_000_000.0
        if source_frame_count is not None and fps > 0:
            return source_frame_count / fps
        return None

    @staticmethod
    def _fps_to_rate(fps: int | float) -> Fraction:
        if float(fps).is_integer():
            return Fraction(int(fps), 1)
        return Fraction(str(float(fps))).limit_denominator(1001)

    @staticmethod
    def _rate_to_float(rate: Fraction | int | float | None) -> float | None:
        if rate is None:
            return None
        return float(rate)

    @staticmethod
    def _audio_duration_seconds(path: str | Path) -> float | None:
        import av

        with av.open(str(path)) as container:
            if len(container.streams.audio) == 0:
                return None
            audio_stream = container.streams.audio[0]
            if audio_stream.duration is not None and audio_stream.time_base is not None:
                return float(audio_stream.duration * audio_stream.time_base)
            if container.duration is not None:
                return float(container.duration) / 1_000_000.0
        return None

    @staticmethod
    def _media_alignment_tolerance_seconds(fps: float) -> float:
        return max(1.0 / max(float(fps), 1.0), 0.05)
