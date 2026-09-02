"""The ffmpeg frame-window decoder must only use options every supported FFmpeg accepts.

FFmpeg 9.0 removed the deprecated ``-vsync`` option (deprecated since 5.1 in favour of
``-fps_mode``). Homebrew moved to 9.0.1 in 2026, and every restore route died at decode
time with "Unrecognized option 'vsync'". This pins the command line so the contract holds
regardless of which FFmpeg the test machine happens to have installed.
"""

import io
from pathlib import Path

import pytest

from mflux.utils import video_util
from mflux.utils.video_util import SourceVideoInfo, VideoUtil

WIDTH = 4
HEIGHT = 3
FRAME_SIZE = WIDTH * HEIGHT * 3


class _FakeProcess:
    def __init__(self, frame_count: int):
        self.stdout = io.BytesIO(bytes(range(FRAME_SIZE)) * frame_count)
        self.stderr = io.BytesIO(b"")

    def wait(self) -> int:
        return 0


@pytest.fixture
def captured_commands(monkeypatch):
    commands: list[list[str]] = []

    def fake_popen(command, **kwargs):
        commands.append(list(command))
        return _FakeProcess(frame_count=6)

    monkeypatch.setattr(video_util.subprocess, "Popen", fake_popen)
    return commands


def _info() -> SourceVideoInfo:
    return SourceVideoInfo(
        fps=24.0,
        source_width=WIDTH,
        source_height=HEIGHT,
        source_frame_count=6,
        source_duration_seconds=0.25,
        audio_present=False,
    )


def test_frame_window_decode_uses_fps_mode_passthrough_not_vsync(captured_commands):
    clips = list(
        VideoUtil._iter_video_frame_windows_ffmpeg(
            path=Path("source.mp4"),
            ffmpeg_path="ffmpeg",
            source_info=_info(),
            start_frame=0,
            normalized_windows=[(0, 6)],
        )
    )

    assert len(clips) == 1 and clips[0].clip_frame_count == 6
    (command,) = captured_commands
    assert "-vsync" not in command, "-vsync was removed in FFmpeg 9.0"
    fps_mode_index = command.index("-fps_mode")
    assert command[fps_mode_index + 1] == "passthrough"
    # fps_mode is an output option: it must follow the input and precede the output target.
    assert command.index("-i") < fps_mode_index < command.index("pipe:1")
