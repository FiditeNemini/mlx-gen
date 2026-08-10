import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from mflux.callbacks import ProgressEvent


def test_wan_cli_selects_bernini_and_forwards_role_aware_guidance(monkeypatch, tmp_path):
    from mflux.models.wan.cli import wan_generate

    observed = {}
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 48), "red").save(reference)

    class FakeVideo:
        task = "text-to-video"
        fps = 8
        width = 64
        height = 48
        num_frames = 5
        steps = 2

        def save(self, path, **kwargs):
            observed["save"] = {"path": Path(path), **kwargs}
            return Path(path)

    class FakeBernini:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "BerniniRenderer", FakeBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "place the referenced sculpture in a gallery",
            "--reference-image",
            str(reference),
            "--width",
            "64",
            "--height",
            "48",
            "--frames",
            "5",
            "--steps",
            "2",
            "--fps",
            "8",
            "--seed",
            "91",
            "--quantize",
            "4",
            "--reference-guidance",
            "2.5",
            "--source-guidance",
            "2.0",
            "--apg-eta",
            "0.8",
            "--apg-norm-threshold",
            "40",
            "--apg-momentum",
            "-0.25",
            "--max-condition-size",
            "640",
            "--system-prompt",
            "custom-renderer-prefix:",
            "--output",
            str(tmp_path / "out.mp4"),
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["init"]["quantize"] == 4
    assert observed["generate"]["reference_image_paths"] == [str(reference)]
    assert observed["generate"]["reference_guidance"] == 2.5
    assert observed["generate"]["source_guidance"] == 2.0
    assert observed["generate"]["apg_eta"] == 0.8
    assert observed["generate"]["apg_norm_threshold"] == 40.0
    assert observed["generate"]["apg_momentum"] == -0.25
    assert observed["generate"]["max_condition_size"] == 640
    assert observed["generate"]["system_prompt"] == "custom-renderer-prefix:"
    assert observed["generate"]["flow_shift"] == 5.0
    assert observed["generate"]["video_strength"] is None


def test_wan_cli_applies_bernini_renderer_role_guidance_defaults_when_not_explicit(monkeypatch, tmp_path):
    from mflux.models.wan.cli import wan_generate

    observed = {}
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 48), "red").save(reference)

    class FakeVideo:
        task = "text-to-video"
        fps = 8
        width = 64
        height = 48
        num_frames = 5
        steps = 2

        def save(self, path, **kwargs):
            observed["save"] = {"path": Path(path), **kwargs}
            return Path(path)

    class FakeBernini:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "BerniniRenderer", FakeBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "place the referenced sculpture in a gallery",
            "--reference-image",
            str(reference),
            "--width",
            "64",
            "--height",
            "48",
            "--frames",
            "5",
            "--steps",
            "2",
            "--fps",
            "8",
            "--seed",
            "91",
            "--quantize",
            "4",
            "--output",
            str(tmp_path / "out.mp4"),
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["generate"]["reference_guidance"] == 4.5
    assert observed["generate"]["source_guidance"] == 1.25
    assert observed["generate"]["flow_shift"] == 5.0


def test_wan_cli_bernini_metadata_replay_restores_reference_contract(tmp_path):
    from mflux.models.wan.cli import wan_generate

    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "prompt": "replay",
                "reference_image_paths": ["ref_a.png", "ref_b.png"],
                "reference_guidance": 2.75,
                "source_guidance": 2.25,
                "apg_eta": 0.7,
                "apg_norm_threshold": 35.0,
                "apg_momentum": -0.1,
                "max_condition_size": 704,
                "system_prompt": "replayed-prefix:",
            }
        )
    )
    args = wan_generate._parser().parse_args(
        ["--model", "bernini-r-1.3b", "--config-from-metadata", str(metadata_path)]
    )

    provided = wan_generate._apply_metadata_defaults(args)

    assert args.reference_image_paths == ["ref_a.png", "ref_b.png"]
    assert args.reference_guidance == 2.75
    assert args.source_guidance == 2.25
    assert args.apg_eta == 0.7
    assert args.apg_norm_threshold == 35.0
    assert args.apg_momentum == -0.1
    assert args.max_condition_size == 704
    assert args.system_prompt == "replayed-prefix:"
    assert "--reference-image" in provided
    assert "--apg-norm-threshold" in provided
    assert "--max-condition-size" in provided
    assert "--system-prompt" in provided


def test_bernini_failure_manifest_preserves_component_provenance(monkeypatch, tmp_path):
    from mflux.models.wan.cli import wan_generate

    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "blue").save(reference)
    output = tmp_path / "failed.mp4"

    class FailingBernini:
        component_source_provenance = {
            "transformer": {
                "source": "ByteDance/Bernini-R-1.3B-Diffusers",
                "revision": "ff4c5d4",
                "source_role": "transformer",
            }
        }

        def __init__(self, **kwargs):
            pass

        def generate_video(self, **kwargs):
            raise RuntimeError("synthetic Bernini failure")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", FailingBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "test",
            "--reference-image",
            str(reference),
            "--output",
            str(output),
            "--no-progress",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    manifest = json.loads(output.with_suffix(".failure.json").read_text())
    assert manifest["run"]["component_source_provenance"] == FailingBernini.component_source_provenance
    assert "root" not in manifest["run"]["component_source_provenance"]["transformer"]


def test_ordinary_wan_rejects_bernini_guidance_before_loading(monkeypatch, capsys):
    from mflux.models.wan.cli import wan_generate

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "wan2.2-ti2v-5b",
            "--prompt",
            "x",
            "--reference-guidance",
            "3",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert "require a Bernini-R model config" in capsys.readouterr().err


def test_bernini_cli_rejects_non_official_prompt_length_before_loading(monkeypatch, capsys):
    from mflux.models.wan.cli import wan_generate

    class UnexpectedBernini:
        def __init__(self, **kwargs):
            raise AssertionError("model loading must not start")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", UnexpectedBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "test",
            "--max-sequence-length",
            "256",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert "requires --max-sequence-length 512" in capsys.readouterr().err


def test_bernini_cli_rejects_missing_source_roles_before_loading(monkeypatch, capsys):
    from mflux.models.wan.cli import wan_generate

    class UnexpectedBernini:
        def __init__(self, **kwargs):
            raise AssertionError("model loading must not start")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", UnexpectedBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlxgen-generate-wan", "--model", "bernini-r-1.3b", "--prompt", "test"],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert "requires one source role before loading weights" in capsys.readouterr().err


def test_bernini_cli_rejects_source_aspect_on_r2v_before_loading(monkeypatch, tmp_path, capsys):
    from mflux.models.wan.cli import wan_generate

    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "blue").save(reference)

    class UnexpectedBernini:
        def __init__(self, **kwargs):
            raise AssertionError("model loading must not start")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", UnexpectedBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "test",
            "--reference-image",
            str(reference),
            "--canvas-policy",
            "source-aspect",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert "has no source canvas" in capsys.readouterr().err


def test_bernini_cli_rejects_ninth_reference_before_loading(monkeypatch, tmp_path, capsys):
    from mflux.models.wan.cli import wan_generate

    references = []
    for index in range(9):
        reference = tmp_path / f"reference-{index}.png"
        Image.new("RGB", (16, 16), "blue").save(reference)
        references.extend(["--reference-image", str(reference)])

    class UnexpectedBernini:
        def __init__(self, **kwargs):
            raise AssertionError("model loading must not start")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", UnexpectedBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        ["mlxgen-generate-wan", "--model", "bernini-r-1.3b", "--prompt", "test", *references],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert "at most 8 ordered --reference-image values" in capsys.readouterr().err


@pytest.mark.parametrize(("max_condition_size", "message"), [(15, "at least 16"), (31, "multiple of 16")])
def test_bernini_cli_rejects_invalid_condition_cap_before_loading(
    monkeypatch, tmp_path, capsys, max_condition_size, message
):
    from mflux.models.wan.cli import wan_generate

    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "blue").save(reference)

    class UnexpectedBernini:
        def __init__(self, **kwargs):
            raise AssertionError("model loading must not start")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", UnexpectedBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "test",
            "--reference-image",
            str(reference),
            "--max-condition-size",
            str(max_condition_size),
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert message in capsys.readouterr().err


def test_bernini_cli_allows_short_source_and_defers_frame_clamping_to_renderer(monkeypatch, tmp_path):
    from mflux.models.wan.cli import wan_generate

    observed = {}
    source = tmp_path / "short.mp4"
    source.touch()

    class FakeVideo:
        task = "video-to-video"
        fps = 8
        width = 64
        height = 48
        num_frames = 5
        steps = 1

        def save(self, path, **kwargs):
            observed["saved"] = Path(path)
            return Path(path)

    class FakeBernini:
        def __init__(self, **kwargs):
            pass

        def generate_video(self, **kwargs):
            observed["generate"] = kwargs
            return FakeVideo()

    def fake_probe(**kwargs):
        observed["probe"] = kwargs

    monkeypatch.setattr(wan_generate, "BerniniRenderer", FakeBernini)
    monkeypatch.setattr(wan_generate, "_probe_source_video", fake_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "make it snow",
            "--video-path",
            str(source),
            "--frames",
            "81",
            "--output",
            str(tmp_path / "out.mp4"),
            "--no-progress",
        ],
    )

    wan_generate.main()

    assert observed["probe"]["requested_frames"] is None
    assert observed["generate"]["num_frames"] == 81
    assert observed["generate"]["video_path"] == str(source)
    assert observed["saved"] == tmp_path / "out.mp4"


def test_bernini_lazy_decode_failure_emits_one_terminal_failed_event(monkeypatch, tmp_path, capsys):
    from mflux.models.wan.cli import wan_generate

    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "green").save(reference)
    output = tmp_path / "decode-fails.mp4"

    class FakeVideo:
        task = "text-to-video"
        fps = 8
        width = 64
        height = 48
        num_frames = 5
        steps = 1

        def save(self, path, **kwargs):
            raise RuntimeError("synthetic lazy decode failure")

    class FakeBernini:
        def __init__(self, **kwargs):
            pass

        def generate_video(self, **kwargs):
            progress = kwargs["progress_callback"]
            progress(
                ProgressEvent(
                    phase="start",
                    task="text-to-video",
                    frame=0,
                    total_frames=5,
                    step=0,
                    total_steps=1,
                )
            )
            progress(
                ProgressEvent(
                    phase="generated",
                    task="text-to-video",
                    frame=5,
                    total_frames=5,
                    step=1,
                    total_steps=1,
                )
            )
            return FakeVideo()

    monkeypatch.setattr(wan_generate, "BerniniRenderer", FakeBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "animate the subject",
            "--reference-image",
            str(reference),
            "--output",
            str(output),
            "--json-events",
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    phases = [event["phase"] for event in events]
    assert phases == ["start", "generated", "save", "failed"]
    assert phases.count("failed") == 1
    assert events[-1]["error_type"] == "RuntimeError"
    assert events[-1]["diagnostics_path"] == str(output.with_suffix(".failure.json"))


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--canvas-policy", "exact-resize", "only --canvas-policy source-aspect"),
        ("--resize-mode", "crop", "only --resize-mode resize"),
        ("--resize-mode", "pad", "only --resize-mode resize"),
    ],
)
def test_bernini_cli_rejects_unproven_video_canvas_extensions_before_loading(
    monkeypatch,
    tmp_path,
    capsys,
    option,
    value,
    message,
):
    from mflux.models.wan.cli import wan_generate

    source = tmp_path / "source.mp4"
    source.touch()

    class UnexpectedBernini:
        def __init__(self, **kwargs):
            raise AssertionError("model loading must not start")

    monkeypatch.setattr(wan_generate, "BerniniRenderer", UnexpectedBernini)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen-generate-wan",
            "--model",
            "bernini-r-1.3b",
            "--prompt",
            "edit",
            "--video-path",
            str(source),
            option,
            value,
        ],
    )

    with pytest.raises(SystemExit):
        wan_generate.main()

    assert message in capsys.readouterr().err
