from mflux.utils.runtime_timer import RuntimeTimer


def test_runtime_timer_elapsed_seconds_is_non_negative():
    timer = RuntimeTimer()

    elapsed = timer.elapsed_seconds()

    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
