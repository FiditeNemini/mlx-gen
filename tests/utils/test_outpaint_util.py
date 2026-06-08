from pathlib import Path

import pytest
from PIL import Image, ImageChops

from mflux.utils.outpaint_util import OutpaintUtil


def test_outpaint_canvas_expands_and_rounds_to_model_multiple(tmp_path: Path):
    source_path = tmp_path / "source.png"
    canvas_path = tmp_path / "canvas.png"
    Image.new("RGB", (10, 7), color=(20, 40, 60)).save(source_path)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="1,50%,2,3",
        output_path=canvas_path,
        dimension_multiple=8,
    )

    assert canvas.target_width == 24
    assert canvas.target_height == 16
    assert canvas.paste_left == 3
    assert canvas.paste_top == 1

    expanded = Image.open(canvas_path).convert("RGB")
    source_region = expanded.crop((3, 1, 13, 8))
    assert ImageChops.difference(source_region, Image.open(source_path).convert("RGB")).getbbox() is None


def test_outpaint_canvas_uses_edge_extension_by_default(tmp_path: Path):
    source_path = tmp_path / "source.png"
    canvas_path = tmp_path / "canvas.png"
    source = Image.new("RGB", (8, 8), color=(0, 0, 0))
    for y in range(source.height):
        source.putpixel((0, y), (255, 0, 0))
        source.putpixel((source.width - 1, y), (0, 0, 255))
    source.save(source_path)

    OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="0,8,0,8",
        output_path=canvas_path,
        dimension_multiple=1,
    )

    canvas = Image.open(canvas_path).convert("RGB")
    left_pixel = canvas.getpixel((2, 4))
    right_pixel = canvas.getpixel((21, 4))
    assert left_pixel[0] > left_pixel[2]
    assert right_pixel[2] > right_pixel[0]


def test_outpaint_composite_restores_source_pixels(tmp_path: Path):
    source_path = tmp_path / "source.png"
    canvas_path = tmp_path / "canvas.png"
    source = Image.new("RGB", (10, 7), color=(20, 40, 60))
    source.putpixel((4, 3), (255, 0, 0))
    source.save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="2,2,2,2",
        output_path=canvas_path,
        dimension_multiple=8,
    )
    generated = Image.new("RGB", (canvas.target_width, canvas.target_height), color=(0, 255, 0))

    composited = OutpaintUtil.composite_source_region(
        generated_image=generated,
        canvas=canvas,
        feather_px=0,
    )

    source_region = composited.crop((2, 2, 12, 9))
    assert ImageChops.difference(source_region, source).getbbox() is None
    assert composited.getpixel((0, 0)) == (0, 255, 0)


def test_default_outpaint_composite_skips_restore_when_generated_region_diverges(tmp_path: Path):
    source_path = tmp_path / "source.png"
    canvas_path = tmp_path / "canvas.png"
    source = Image.new("RGB", (80, 64), color=(20, 40, 60))
    source.save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="8,8,8,8",
        output_path=canvas_path,
        dimension_multiple=1,
    )
    generated = Image.new("RGB", (canvas.target_width, canvas.target_height), color=(0, 255, 0))

    composited = OutpaintUtil.composite_source_region(generated_image=generated, canvas=canvas)

    assert composited.outpaint_preservation_applied is False
    assert composited.getpixel((canvas.paste_left + 20, canvas.paste_top + 20)) == (0, 255, 0)


def test_default_outpaint_composite_restores_when_generated_region_matches(tmp_path: Path):
    source_path = tmp_path / "source.png"
    canvas_path = tmp_path / "canvas.png"
    source = Image.new("RGB", (80, 64), color=(20, 40, 60))
    source.save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="8,8,8,8",
        output_path=canvas_path,
        dimension_multiple=1,
    )
    generated = Image.open(canvas_path).convert("RGB")

    composited = OutpaintUtil.composite_source_region(generated_image=generated, canvas=canvas)

    assert composited.outpaint_preservation_applied is True
    assert composited.getpixel((canvas.paste_left + 20, canvas.paste_top + 20)) == (20, 40, 60)


def test_source_mask_preserves_detailed_edges_more_than_smooth_edges():
    source = Image.new("RGB", (96, 96), color="white")
    for y in range(source.height):
        source.putpixel((24, y), (0, 0, 0))

    mask = OutpaintUtil._source_mask(source=source, feather_px=24)

    assert mask.getpixel((24, 48)) > 200
    assert mask.getpixel((95, 48)) < 80


def test_default_source_mask_blends_smooth_borders_without_erasing_details():
    source = Image.new("RGB", (128, 96), color="white")
    for y in range(20, 76):
        source.putpixel((64, y), (0, 0, 0))

    mask = OutpaintUtil._source_mask(source=source, feather_px=None)

    assert mask.getpixel((64, 48)) > 200
    assert mask.getpixel((0, 48)) < 40
    assert mask.getpixel((127, 48)) < 40


def test_outpaint_rejects_noop_and_negative_padding(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (10, 7), color="white").save(source_path)

    with pytest.raises(ValueError, match="must add pixels"):
        OutpaintUtil.create_expanded_canvas(
            source_path=source_path,
            padding_value="0,0,0,0",
            output_path=tmp_path / "noop.png",
        )

    with pytest.raises(ValueError, match="zero or positive"):
        OutpaintUtil.create_expanded_canvas(
            source_path=source_path,
            padding_value="-1,0,0,0",
            output_path=tmp_path / "negative.png",
        )


def test_outpaint_attaches_metadata(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (10, 7), color="white").save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="1,2,3,4",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=8,
    )

    class FakeGeneratedImage:
        extra_metadata = {"existing": "kept"}
        source_image_width = None
        source_image_height = None

    image = FakeGeneratedImage()
    OutpaintUtil.attach_metadata(generated_image=image, canvas=canvas, padding_value="1,2,3,4")

    assert image.source_image_width == 10
    assert image.source_image_height == 7
    assert image.extra_metadata["existing"] == "kept"
    assert image.extra_metadata["outpaint_padding"] == "1,2,3,4"
    assert image.extra_metadata["outpaint_target_width"] == 16
    assert image.extra_metadata["outpaint_source_paste_left"] == 4


def test_reframe_attaches_metadata(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (10, 7), color="white").save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="1,2,3,4",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=8,
        option_name="--reframe-padding",
    )

    class FakeGeneratedImage:
        extra_metadata = {"existing": "kept"}
        source_image_width = None
        source_image_height = None

    image = FakeGeneratedImage()
    OutpaintUtil.attach_reframe_metadata(generated_image=image, canvas=canvas, padding_value="1,2,3,4")

    assert image.source_image_width == 10
    assert image.source_image_height == 7
    assert image.extra_metadata["existing"] == "kept"
    assert image.extra_metadata["reframe_padding"] == "1,2,3,4"
    assert image.extra_metadata["reframe_target_width"] == 16
    assert image.extra_metadata["reframe_source_paste_left"] == 4
    assert image.extra_metadata["reframe_mode"] == "expanded-conditioning-canvas"
