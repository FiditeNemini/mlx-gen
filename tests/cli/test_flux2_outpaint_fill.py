import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.cli.completions.generator import CompletionGenerator
from mflux.cli.parser.parsers import rgb_color_value
from mflux.models.flux2.cli import flux2_edit_generate

GREEN_BORDER_LORA_REQUEST = "fal/flux-2-klein-4B-outpaint-lora"
# What LoraResolution.resolve() turns the repo request into: a Hugging Face snapshot file. The
# repo identity survives only as the "models--org--repo" cache directory name.
GREEN_BORDER_LORA_RESOLVED = (
    "/Users/x/.cache/mflux/lora/models--fal--flux-2-klein-4B-outpaint-lora/snapshots/"
    "abc123/LyNiaZ53Tudg0J6sT8Xbx_pytorch_lora_weights_comfy_converted.safetensors"
)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0,255,0", (0, 255, 0)),
        (" 12 , 34 , 56 ", (12, 34, 56)),
        ("#00ff00", (0, 255, 0)),
        ("#0A0B0C", (10, 11, 12)),
    ],
)
def test_rgb_color_value_accepts_tuple_and_hex_forms(value, expected):
    assert rgb_color_value(value) == expected


@pytest.mark.fast
@pytest.mark.parametrize("value", ["0,255", "0,255,0,0", "#00ff0", "#00ff0g", "0,256,0", "-1,0,0", "green"])
def test_rgb_color_value_rejects_bad_forms(value):
    with pytest.raises(argparse.ArgumentTypeError):
        rgb_color_value(value)


@pytest.mark.fast
def test_completion_generator_advertises_the_outpaint_fill_contract():
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command("mflux-generate-flux2-edit")
    script = generator.generate_command_function("mflux-generate-flux2-edit", parser)

    assert "--outpaint-fill" in script
    assert "--outpaint-fill-color" in script
    assert "(auto edge neutral solid blur)" in script


@pytest.mark.fast
def test_completion_generator_does_not_advertise_outpaint_fill_on_qwen_edit():
    # The Qwen edit backend has no --outpaint-fill option; completion must not offer one.
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command("mflux-generate-qwen-edit")
    script = generator.generate_command_function("mflux-generate-qwen-edit", parser)

    assert "--outpaint-fill" not in script


class _FakeGeneratedImage:
    def __init__(self, size: tuple[int, int]):
        self.image = Image.new("RGB", size, (10, 20, 30))
        self.image_path = None
        self.image_paths = None
        self.extra_metadata: dict = {}
        self.saved_to: Path | None = None

    def save(self, path, export_json_metadata=False, overwrite=True, embed_metadata=False):
        self.saved_to = Path(path)
        Path(path).touch()


def _two_tone_source(size: tuple[int, int]) -> Image.Image:
    # Deliberately not uniform: edge fill stretches the bottom border strip outward, so a
    # left/right split makes an edge-filled padded region distinguishable from a flat one.
    image = Image.new("RGB", size, (200, 20, 20))
    image.paste((20, 20, 200), (size[0] // 2, 0, size[0], size[1]))
    return image


def _run_outpaint_cli(
    monkeypatch,
    tmp_path: Path,
    *,
    source_size: tuple[int, int],
    extra_argv: list[str],
    source_image: Image.Image | None = None,
):
    source = tmp_path / "source.png"
    (source_image or Image.new("RGB", source_size, (40, 90, 160))).save(source)
    output = tmp_path / "out.png"
    generated: list[_FakeGeneratedImage] = []

    class FakeOutpaintModel:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        def generate_image(self, **kwargs):
            canvas = Image.open(kwargs["canvas"].canvas_path)
            image = _FakeGeneratedImage(canvas.size)
            image.conditioning_canvas = canvas.copy()
            generated.append(image)
            return image

    monkeypatch.setattr(flux2_edit_generate, "Flux2KleinOutpaint", FakeOutpaintModel)
    monkeypatch.setattr(flux2_edit_generate.PromptUtil, "read_prompt", lambda args: "extend the scene")
    monkeypatch.setattr(flux2_edit_generate.CallbackManager, "register_callbacks", lambda **kwargs: None)
    monkeypatch.setattr(flux2_edit_generate.CallbackManager, "apply_runtime_memory_options", lambda args: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "mflux-generate-flux2-edit",
            "--model",
            "flux2-klein-base-4b",
            "--prompt",
            "extend the scene",
            "--image-paths",
            str(source),
            "--steps",
            "1",
            "--seed",
            "1234",
            "--output",
            str(output),
            *extra_argv,
        ],
    )

    flux2_edit_generate.main()
    return generated


@pytest.mark.fast
def test_cli_prints_the_resolved_fill_notice_without_a_lora(monkeypatch, tmp_path, capsys):
    _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(768, 766),
        extra_argv=["--outpaint-padding", "0%,10%,100%,10%"],
    )

    stderr = capsys.readouterr().err
    assert "Outpaint: fill=neutral, canvas 928x1536 from source 768x766" in stderr
    assert "padding top=0 right=76 bottom=766 left=76." in stderr
    assert "Outpaint: --outpaint-fill auto selected neutral because" in stderr
    assert "2.0x the 384px edge-fill reach" in stderr


@pytest.mark.fast
def test_cli_prints_the_resolved_fill_notice_with_the_green_border_lora(monkeypatch, tmp_path, capsys):
    # Resolution rewrites the repo request into a cache file whose basename carries no marker.
    monkeypatch.setattr(
        "mflux.models.common.resolution.lora_resolution.LoraResolution.resolve",
        lambda path: GREEN_BORDER_LORA_RESOLVED,
    )

    _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(768, 766),
        extra_argv=[
            "--outpaint-padding",
            "0%,10%,100%,10%",
            "--lora-paths",
            GREEN_BORDER_LORA_REQUEST,
        ],
    )

    stderr = capsys.readouterr().err
    assert "Outpaint: fill=solid color=0,255,0, canvas 928x1536 from source 768x766" in stderr
    assert "Outpaint: --outpaint-fill auto selected solid because a green-border outpaint LoRA is loaded" in stderr


@pytest.mark.fast
def test_cli_builds_a_flat_conditioning_canvas_outside_the_source_box(monkeypatch, tmp_path):
    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(768, 766),
        source_image=_two_tone_source((768, 766)),
        extra_argv=["--outpaint-padding", "0%,10%,100%,10%"],
    )

    canvas = generated[0].conditioning_canvas
    row = canvas.height - 8
    left_sample = canvas.getpixel((canvas.width // 4, row))
    right_sample = canvas.getpixel((canvas.width * 3 // 4, row))

    # Edge fill would stretch the two-tone bottom border outward and leave these two samples
    # different. A blank canvas is flat, so the model has to generate rather than continue.
    assert left_sample == right_sample


@pytest.mark.fast
def test_cli_explicit_edge_fill_still_stretches_the_source_border(monkeypatch, tmp_path):
    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(768, 766),
        source_image=_two_tone_source((768, 766)),
        extra_argv=["--outpaint-padding", "0%,10%,100%,10%", "--outpaint-fill", "edge"],
    )

    canvas = generated[0].conditioning_canvas
    row = canvas.height - 8
    assert canvas.getpixel((canvas.width // 4, row)) != canvas.getpixel((canvas.width * 3 // 4, row))
    assert generated[0].extra_metadata["outpaint_fill"] == "edge"


@pytest.mark.fast
def test_cli_records_the_resolved_fill_in_metadata(monkeypatch, tmp_path):
    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(768, 766),
        extra_argv=["--outpaint-padding", "0%,10%,100%,10%"],
    )

    metadata = generated[0].extra_metadata
    assert metadata["outpaint_padding"] == "0%,10%,100%,10%"
    assert metadata["outpaint_fill"] == "neutral"
    assert metadata["outpaint_fill_color"] is None
    assert metadata["outpaint_fill_requested"] == "auto"
    assert metadata["outpaint_max_side_padding"] == "bottom"
    assert metadata["outpaint_max_side_padding_px"] == 766
    assert metadata["outpaint_max_side_padding_ratio"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.fast
def test_cli_replays_the_resolved_fill_from_prior_metadata(monkeypatch, tmp_path):
    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "flux2-klein-base-4b",
                "prompt": "extend the scene",
                "outpaint_padding": "0%,10%,100%,10%",
                "outpaint_fill": "edge",
                "outpaint_fill_color": [255, 255, 255],
            }
        )
    )

    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(768, 766),
        extra_argv=["--outpaint-padding", "0%,10%,100%,10%", "-C", str(metadata_path)],
    )

    # The replayed run must reproduce the recorded edge canvas, not re-run `auto`.
    assert generated[0].extra_metadata["outpaint_fill"] == "edge"
    assert generated[0].extra_metadata["outpaint_fill_requested"] == "edge"


@pytest.mark.fast
def test_cli_rejects_outpaint_fill_without_outpaint_padding(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run_outpaint_cli(
            monkeypatch,
            tmp_path,
            source_size=(432, 240),
            extra_argv=["--outpaint-fill", "solid"],
        )

    message = capsys.readouterr().err
    assert "--outpaint-fill and --outpaint-fill-color configure the --outpaint-padding" in message
    assert "Pass --outpaint-padding, or drop these options." in message


@pytest.mark.fast
def test_cli_splits_a_deep_two_axis_request_into_passes_and_records_them(monkeypatch, tmp_path, capsys):
    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(432, 240),
        extra_argv=["--outpaint-padding", "0,0,148,256"],
    )

    stderr = capsys.readouterr().err
    assert "Warning: --outpaint-padding expands both axes deeply" not in stderr
    assert "Outpaint: 2 passes because the request pads bottom 148px (62% of the source height)" in stderr
    assert "Pass 1 pads 0,0,148,0 to 432x400; pass 2 pads 0,0,0,256 to 688x400." in stderr
    assert "Outpaint: fill=edge, canvas 432x400 from source 432x240" in stderr
    assert "Outpaint: pass 2 fill=edge, canvas 688x400 from source 432x400" in stderr
    # Two denoises, one saved artifact carrying the final geometry.
    assert [image.image.size for image in generated] == [(432, 400), (688, 400)]
    assert generated[0].saved_to is None
    assert generated[1].saved_to == tmp_path / "out.png"
    metadata = generated[1].extra_metadata
    assert metadata["outpaint_padding"] == "0,0,148,256"
    assert metadata["outpaint_passes"] == 2
    assert metadata["outpaint_passes_requested"] == "auto"
    assert metadata["outpaint_pass_paddings"] == ["0,0,148,0", "0,0,0,256"]
    assert metadata["outpaint_pass_fills"] == ["edge", "edge"]
    assert (metadata["outpaint_target_width"], metadata["outpaint_target_height"]) == (688, 400)
    assert (metadata["outpaint_source_paste_left"], metadata["outpaint_source_paste_top"]) == (256, 0)


@pytest.mark.fast
def test_cli_explicit_single_pass_warns_on_a_deep_corner_and_runs_one_canvas(monkeypatch, tmp_path, capsys):
    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(432, 240),
        extra_argv=["--outpaint-padding", "0,0,148,256", "--outpaint-passes", "1"],
    )

    stderr = capsys.readouterr().err
    assert "Warning: --outpaint-padding expands both axes deeply" in stderr
    assert "because --outpaint-passes 1 was passed" in stderr
    assert "Outpaint: 1 pass because --outpaint-passes 1 was passed explicitly." in stderr
    assert [image.image.size for image in generated] == [(688, 400)]
    assert generated[0].extra_metadata["outpaint_passes"] == 1
    assert generated[0].extra_metadata["outpaint_passes_requested"] == "1"


@pytest.mark.fast
def test_cli_replays_the_resolved_pass_count_from_prior_metadata(monkeypatch, tmp_path):
    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "flux2-klein-base-4b",
                "prompt": "extend the scene",
                "outpaint_padding": "0,0,148,256",
                "outpaint_fill": "edge",
                "outpaint_passes": 1,
            }
        )
    )

    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(432, 240),
        extra_argv=["--outpaint-padding", "0,0,148,256", "-C", str(metadata_path)],
    )

    # The recorded run was one pass; the replay must not re-decide `auto` into two.
    assert len(generated) == 1
    assert generated[0].extra_metadata["outpaint_passes"] == 1
    assert generated[0].extra_metadata["outpaint_passes_requested"] == "1"


@pytest.mark.fast
def test_cli_replays_the_fill_request_when_a_split_run_resolved_different_fills(monkeypatch, tmp_path):
    # A recorded split whose passes chose different fills cannot be reproduced by one explicit
    # fill, so the replay falls back to the recorded request (`auto` is deterministic).
    metadata_path = tmp_path / "prior.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model": "flux2-klein-base-4b",
                "prompt": "extend the scene",
                "outpaint_padding": "0,0,148,256",
                "outpaint_fill": "edge",
                "outpaint_fill_requested": "auto",
                "outpaint_passes": 2,
                "outpaint_pass_fills": ["edge", "neutral"],
            }
        )
    )

    generated = _run_outpaint_cli(
        monkeypatch,
        tmp_path,
        source_size=(432, 240),
        extra_argv=["--outpaint-padding", "0,0,148,256", "-C", str(metadata_path)],
    )

    assert len(generated) == 2
    assert generated[1].extra_metadata["outpaint_fill_requested"] == "auto"
    assert generated[1].extra_metadata["outpaint_passes_requested"] == "2"


@pytest.mark.fast
def test_cli_rejects_outpaint_passes_without_outpaint_padding(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run_outpaint_cli(
            monkeypatch,
            tmp_path,
            source_size=(64, 64),
            extra_argv=["--outpaint-passes", "2"],
        )

    assert "--outpaint-passes configures the --outpaint-padding run" in capsys.readouterr().err


@pytest.mark.fast
@pytest.mark.parametrize("command", ["mflux-generate-flux2-edit", "mflux-generate-qwen-edit"])
def test_completion_generator_advertises_outpaint_passes_on_every_outpaint_backend(command):
    # The pass planner is route-independent, so both expanded-canvas backends complete it.
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command(command)
    script = generator.generate_command_function(command, parser)

    assert "--outpaint-passes" in script
    assert "(auto 1 2)" in script
