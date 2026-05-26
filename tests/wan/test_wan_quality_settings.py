from mflux.models.wan.variants import Wan2_2_TI2V


def test_wan_warns_when_settings_are_smoke_only(capsys):
    Wan2_2_TI2V._warn_if_smoke_settings(
        height=128,
        width=128,
        num_frames=5,
        num_inference_steps=4,
        fps=8,
    )

    captured = capsys.readouterr()

    assert "smoke tests, not quality validation" in captured.err
    assert "1280x704 or 704x1280" in captured.err


def test_wan_does_not_warn_for_recommended_settings(capsys):
    Wan2_2_TI2V._warn_if_smoke_settings(
        height=704,
        width=1280,
        num_frames=121,
        num_inference_steps=50,
        fps=24,
    )

    captured = capsys.readouterr()

    assert captured.err == ""
