"""Source frame counts must reflect what a stream can actually deliver.

Some MP4 containers advertise more video samples than the stream will decode. The case
that reached users was an edit-list trim: every sample stays in ``stsz`` so ``nb_frames``
counts the untrimmed source, while the edit list shortens playback. Restore routes plan
their whole chunk schedule from that number up front, so an over-reported count schedules
work for frames that never arrive and the run dies late with an opaque decode error.

These tests build real files with ffmpeg and decode them for real; nothing here is mocked.
"""

import shutil
import subprocess

import pytest

from mflux.utils.video_util import VideoUtil

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required to build the fixtures",
)

FPS = 24


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)


def _decodable_frames(path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def _declared_frames(path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


@pytest.fixture(scope="module")
def clean_video(tmp_path_factory):
    """A well-formed 10 second clip whose metadata tells the truth."""
    path = tmp_path_factory.mktemp("frame_count") / "clean.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=160x120:rate={FPS}:duration=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )
    return path


@pytest.fixture(scope="module")
def edit_list_video(clean_video):
    """Stream-copy trim: all 240 samples retained, playback shortened to 7 seconds.

    This is the shape that failed in the field - ``nb_frames`` counts the untrimmed
    source while ``duration`` reflects the edit list.
    """
    path = clean_video.parent / "edit_list.mp4"
    _run(["ffmpeg", "-y", "-v", "error", "-ss", "3", "-i", str(clean_video), "-c", "copy", str(path)])
    return path


@pytest.fixture(scope="module")
def truncated_video(clean_video):
    """A partially written file: metadata is self-consistent, the stream stops early."""
    path = clean_video.parent / "truncated.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(clean_video),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(path.with_name("faststart.mp4")),
        ]
    )
    payload = path.with_name("faststart.mp4").read_bytes()
    path.write_bytes(payload[: int(len(payload) * 0.6)])
    return path


def test_clean_video_frame_count_is_reported_unchanged(clean_video):
    info = VideoUtil.inspect_video(clean_video)
    assert info.source_frame_count == _decodable_frames(clean_video) == 240
    assert info.fps == pytest.approx(FPS)


def test_edit_list_video_declares_more_frames_than_it_delivers(edit_list_video):
    # Guards the fixture itself: if a future ffmpeg stops producing the edit list, the
    # regression below would silently stop testing anything.
    assert _declared_frames(edit_list_video) == 240
    assert _decodable_frames(edit_list_video) == 168


def test_over_reported_frame_count_is_corrected_to_what_decodes(edit_list_video):
    info = VideoUtil.inspect_video(edit_list_video)
    assert info.source_frame_count == 168, (
        "inspect_video must report the deliverable frame count, not the container's stale "
        f"nb_frames of {_declared_frames(edit_list_video)}"
    )


def test_correction_is_logged_once_per_file(edit_list_video, caplog):
    VideoUtil._resolved_frame_count.cache_clear()
    with caplog.at_level("WARNING"):
        for _ in range(4):
            VideoUtil.inspect_video(edit_list_video)
    corrections = [record for record in caplog.records if "declares 240 frames" in record.getMessage()]
    assert len(corrections) == 1, "the exact frame count decodes the whole source and must be cached"


def test_exact_count_is_not_paid_for_well_formed_files(clean_video):
    VideoUtil._resolved_frame_count.cache_clear()
    VideoUtil.inspect_video(clean_video)
    assert VideoUtil._resolved_frame_count.cache_info().misses == 0, (
        "a self-consistent container must not trigger a full decode pass"
    )


def test_truncated_stream_fails_with_an_actionable_message(truncated_video):
    decodable = _decodable_frames(truncated_video)
    # The container is self-consistent here (nb_frames and duration both say 240), so the
    # reconciliation cannot catch this one without decoding every source. Ask for more than
    # the stream holds and require the failure to be legible.
    requested = decodable + 24
    with pytest.raises(RuntimeError) as error:
        list(VideoUtil.iter_video_frame_windows(path=truncated_video, windows=[(0, requested)]))
    message = str(error.value)
    # Fails closed rather than restoring a shorter clip than asked for (ADR 0002), but the
    # message must name the mismatch and give the remedy.
    assert f"planned for {requested} frames" in message
    assert f"stream delivered {decodable}" in message
    assert f"--max-frames {decodable}" in message


def test_pyav_short_stream_reports_the_exact_delivered_count(clean_video):
    """The pyav fallback must count frames, not reuse the last loop index.

    An enumerate index is count - 1 after the loop and unbound when nothing decodes, so
    a 10-frame stream asked for 20 frames used to report "stopped after 9" and suggest a
    cap one frame short of what the source can actually deliver.
    """
    from mflux.utils.video_util import VideoUtil

    info = VideoUtil.inspect_video(clean_video)
    with pytest.raises(RuntimeError) as error:
        list(
            VideoUtil._iter_video_frame_windows_pyav(
                path=clean_video,
                source_info=info,
                start_frame=0,
                normalized_windows=[(0, 500)],
            )
        )
    message = str(error.value)
    assert "stream delivered 240" in message, message
    assert "--max-frames 240" in message, message


def test_short_stream_remedy_is_clip_relative_under_a_start_offset(truncated_video):
    """--max-frames counts frames from the requested start, not from the stream head.

    Suggesting an absolute count would over-cap a run that used --start-seconds, and the
    retried command would fail the same way.
    """
    from mflux.utils.video_util import VideoUtil

    decodable = _decodable_frames(truncated_video)
    offset = 24
    requested = decodable  # more than the stream holds once the offset is consumed
    with pytest.raises(RuntimeError) as error:
        list(
            VideoUtil.iter_video_frame_windows(
                path=truncated_video,
                windows=[(0, requested)],
                start_frame=offset,
            )
        )
    message = str(error.value)
    assert f"--max-frames {decodable - offset}" in message, message
    assert f"relative to the requested start frame {offset}" in message, message
