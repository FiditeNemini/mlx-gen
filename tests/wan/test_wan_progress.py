from mflux.models.wan.variants import Wan2_2_TI2V, WanProgressEvent


def test_wan_progress_event_exposes_fraction():
    event = WanProgressEvent(phase="denoise", frame=5, total_frames=20, step=2, total_steps=10)

    assert event.progress == 0.25


def test_wan_progress_frame_for_step_is_frame_based_and_leaves_final_frame_for_completion():
    frames = [
        Wan2_2_TI2V._progress_frame_for_step(step_index=step, total_steps=50, total_frames=121)
        for step in range(50)
    ]

    assert frames[0] == 2
    assert frames[-1] == 120
    assert frames == sorted(frames)
    assert max(frames) == 120


def test_wan_emit_progress_callback_receives_structured_event():
    events = []

    Wan2_2_TI2V._emit_progress(
        events.append,
        phase="complete",
        frame=121,
        total_frames=121,
        step=50,
        total_steps=50,
    )

    assert events == [WanProgressEvent(phase="complete", frame=121, total_frames=121, step=50, total_steps=50)]
