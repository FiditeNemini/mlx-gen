import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from mflux.cli.completions.generator import CompletionGenerator
from mflux.cli.parser.parsers import rgb_color_value
from mflux.models.flux2.cli import flux2_edit_generate
from mflux.models.flux2.cli.flux2_edit_generate import (
    FLUX2_GREEN_BORDER_FILL_COLOR,
    resolve_outpaint_fill_plan,
)
from mflux.utils.box_values import BoxValues
from mflux.utils.outpaint_util import OutpaintUtil

GREEN_BORDER_LORA_REQUEST = "fal/flux-2-klein-4B-outpaint-lora"
# What LoraResolution.resolve() turns the repo request into: a Hugging Face snapshot file. The
# repo identity survives only as the "models--org--repo" cache directory name.
GREEN_BORDER_LORA_RESOLVED = (
    "/Users/x/.cache/mflux/lora/models--fal--flux-2-klein-4B-outpaint-lora/snapshots/"
    "abc123/LyNiaZ53Tudg0J6sT8Xbx_pytorch_lora_weights_comfy_converted.safetensors"
)
GREEN_BORDER_LORA_RESOLVED_LEGACY_BASENAME = "/Users/x/.cache/mflux/lora/flux-outpaint-lora.safetensors"


def _args(**overrides) -> argparse.Namespace:
    values = {
        "outpaint_fill": "auto",
        "outpaint_fill_color": None,
        "lora_paths": None,
        "requested_lora_paths": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _plan(*, source_size: tuple[int, int], padding: str, **overrides):
    source = Image.new("RGB", source_size, (40, 90, 160))
    box = BoxValues.parse(padding).normalize_to_dimensions(width=source.width, height=source.height)
    return resolve_outpaint_fill_plan(args=_args(**overrides), source=source, padding=box)


@pytest.mark.fast
def test_auto_selects_edge_fill_for_small_padding():
    plan = _plan(source_size=(432, 240), padding="5%,10%,5%,10%")

    assert plan.mode == "edge"
    assert plan.requested == "auto"
    assert plan.uses_green_border_lora is False
    assert plan.max_side == "right"
    assert plan.max_side_ratio == pytest.approx(0.1, abs=0.01)
    assert plan.edge_fill_within_reach is True
    assert "edge-fill reach" in plan.reason


@pytest.mark.fast
def test_auto_selects_blank_fill_for_large_padding_without_lora():
    # The measured defect: a 768x766 portrait grown by a full source height on the bottom edge
    # used to fall through to edge fill and return the conditioning canvas back as smear.
    plan = _plan(source_size=(768, 766), padding="0%,10%,100%,10%")

    # "neutral" is OutpaintUtil's flat per-side border-color canvas: blank, so there is nothing
    # for the model to continue, but without a hard chroma step at the source boundary.
    assert plan.mode == "neutral"
    assert plan.fill_color is None
    assert plan.max_side == "bottom"
    assert plan.max_side_ratio == pytest.approx(1.0, abs=0.01)
    assert plan.edge_fill_within_reach is False
    assert "blank canvas" in plan.reason


@pytest.mark.fast
def test_auto_threshold_boundary_is_inclusive():
    # `auto` switches at exactly the depth edge fill covers, so the boundary is read from
    # OutpaintUtil rather than restated here.
    reach = OutpaintUtil.edge_fill_reach(400)
    at_limit = _plan(source_size=(400, 400), padding=f"0,{reach},0,0")
    over_limit = _plan(source_size=(400, 400), padding=f"0,{reach + 1},0,0")

    assert at_limit.mode == "edge"
    assert over_limit.mode == "neutral"


@pytest.mark.fast
def test_auto_keeps_the_published_validation_profile_on_edge_fill():
    # The published profile runs 80% single-side padding on a 432x240 source at a 10.9x stretch
    # and is validated with edge fill. Selecting by stretch rather than by padding ratio is what
    # keeps that command on the fill mode its recorded artifacts were produced with.
    plan = _plan(source_size=(432, 240), padding="5%,80%,5%,60%")

    assert plan.mode == "edge"
    assert plan.edge_fill_within_reach is True


@pytest.mark.fast
@pytest.mark.parametrize(
    ("lora_field", "lora_value"),
    [
        ("requested_lora_paths", [GREEN_BORDER_LORA_REQUEST]),
        ("requested_lora_paths", [f"{GREEN_BORDER_LORA_REQUEST}:pytorch_lora_weights_comfy_converted.safetensors"]),
        ("lora_paths", [GREEN_BORDER_LORA_RESOLVED]),
        ("lora_paths", [GREEN_BORDER_LORA_RESOLVED_LEGACY_BASENAME]),
    ],
)
@pytest.mark.parametrize("padding", ["5%,10%,5%,10%", "0%,10%,100%,10%"])
def test_green_border_lora_selects_green_in_requested_and_resolved_forms(lora_field, lora_value, padding):
    plan = _plan(source_size=(768, 766), padding=padding, **{lora_field: lora_value})

    assert plan.uses_green_border_lora is True
    assert plan.mode == "solid"
    assert plan.fill_color == FLUX2_GREEN_BORDER_FILL_COLOR == (0, 255, 0)
    assert "green" in plan.reason


@pytest.mark.fast
def test_unrelated_lora_does_not_select_green():
    plan = _plan(
        source_size=(432, 240),
        padding="5%,10%,5%,10%",
        requested_lora_paths=["fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"],
        lora_paths=["/Users/x/.cache/mflux/lora/multi_angles.safetensors"],
    )

    assert plan.uses_green_border_lora is False
    assert plan.mode == "edge"


@pytest.mark.fast
@pytest.mark.parametrize("padding", ["5%,10%,5%,10%", "0%,10%,100%,10%"])
@pytest.mark.parametrize("requested", ["edge", "neutral", "solid", "blur"])
@pytest.mark.parametrize("with_green_lora", [False, True])
def test_explicit_fill_overrides_auto_in_every_case(padding, requested, with_green_lora):
    lora = {"requested_lora_paths": [GREEN_BORDER_LORA_REQUEST]} if with_green_lora else {}
    plan = _plan(source_size=(768, 766), padding=padding, outpaint_fill=requested, **lora)

    assert plan.mode == requested
    assert plan.requested == requested
    assert plan.is_explicit is True
    assert plan.reason == f"--outpaint-fill {requested} was passed explicitly."


@pytest.mark.fast
def test_explicit_fill_color_overrides_the_default_solid_color():
    plan = _plan(
        source_size=(768, 766),
        padding="0%,10%,100%,10%",
        outpaint_fill="solid",
        outpaint_fill_color=(12, 34, 56),
        requested_lora_paths=[GREEN_BORDER_LORA_REQUEST],
    )

    assert plan.fill_color == (12, 34, 56)


@pytest.mark.fast
def test_explicit_solid_without_a_color_falls_back_to_the_source_mean_border_color():
    plan = _plan(source_size=(768, 766), padding="0%,10%,100%,10%", outpaint_fill="solid")

    assert plan.mode == "solid"
    assert plan.fill_color == (40, 90, 160)


@pytest.mark.fast
def test_explicit_solid_with_the_green_lora_defaults_to_green():
    plan = _plan(
        source_size=(768, 766),
        padding="0%,10%,100%,10%",
        outpaint_fill="solid",
        requested_lora_paths=[GREEN_BORDER_LORA_REQUEST],
    )

    assert plan.fill_color == FLUX2_GREEN_BORDER_FILL_COLOR


@pytest.mark.fast
def test_non_solid_fills_carry_no_fill_color():
    for mode in ("edge", "neutral", "blur"):
        plan = _plan(source_size=(768, 766), padding="0%,10%,100%,10%", outpaint_fill=mode)
        assert plan.fill_color is None, mode


@pytest.mark.fast
def test_guard_warns_loudly_but_proceeds_for_explicit_unsafe_edge_fill(capsys):
    plan = _plan(source_size=(768, 766), padding="0%,10%,100%,10%", outpaint_fill="edge")

    flux2_edit_generate._guard_unsafe_edge_fill(fill_plan=plan)

    warning = capsys.readouterr().err
    assert "--outpaint-fill edge" in warning
    assert "bottom padding of 766px (100% of the source height)" in warning
    assert "--outpaint-fill neutral" in warning
    assert "fal/flux-2-klein-4B-outpaint-lora" in warning


@pytest.mark.fast
def test_guard_fails_closed_when_auto_reaches_unsafe_edge_fill():
    plan = _plan(source_size=(768, 766), padding="0%,10%,100%,10%")
    unreachable = flux2_edit_generate.OutpaintFillPlan(
        requested="auto",
        mode="edge",
        fill_color=(255, 255, 255),
        reason=plan.reason,
        max_side=plan.max_side,
        max_side_padding_px=plan.max_side_padding_px,
        max_side_ratio=plan.max_side_ratio,
        max_side_reach_px=plan.max_side_reach_px,
        max_side_overreach=plan.max_side_overreach,
        uses_green_border_lora=False,
    )

    with pytest.raises(ValueError) as error:
        flux2_edit_generate._guard_unsafe_edge_fill(fill_plan=unreachable)

    message = str(error.value)
    assert "--outpaint-fill auto resolved to edge" in message
    assert "--outpaint-fill neutral" in message


@pytest.mark.fast
def test_guard_is_silent_inside_the_edge_fill_limit(capsys):
    plan = _plan(source_size=(432, 240), padding="5%,10%,5%,10%")

    flux2_edit_generate._guard_unsafe_edge_fill(fill_plan=plan)

    assert capsys.readouterr().err == ""


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
