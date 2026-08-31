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


def _detailed_source(width: int, height: int) -> Image.Image:
    source = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            source.putpixel((x, y), ((x * 7) % 256, (y * 3) % 256, ((x ^ y) * 5) % 256))
    return source


def _band_lateral_std(band: Image.Image, *, vertical_expansion: bool) -> float:
    """Amplitude of the streaks: spread of the per-lateral-position mean value."""
    pixels = band.convert("L")
    width, height = pixels.size
    if vertical_expansion:
        profiles = [
            sum(pixels.getpixel((x, y)) for y in range(height)) / height  # noqa: PERF101
            for x in range(width)
        ]
    else:
        profiles = [sum(pixels.getpixel((x, y)) for x in range(width)) / width for y in range(height)]
    mean = sum(profiles) / len(profiles)
    return (sum((value - mean) ** 2 for value in profiles) / len(profiles)) ** 0.5


@pytest.mark.fast
def test_edge_strip_stays_within_stretch_bound_at_large_padding():
    strips = OutpaintUtil._edge_strip_extents(
        source_width=768,
        source_height=766,
        left=77,
        right=84,
        top=0,
        bottom=770,
    )
    strip_left, strip_right, strip_top, strip_bottom = strips

    # Shallow sides keep the historical 32 px strip.
    assert strip_left == 32
    assert strip_right == 32
    assert strip_top == 32
    # The 770 px side grows instead of stretching 24x.
    assert 770 / strip_bottom <= 12
    assert 770 / 32 > 24  # what the hard 32 px cap used to produce


@pytest.mark.fast
def test_edge_strip_never_samples_more_than_the_source_side():
    strip_left, _, _, strip_bottom = OutpaintUtil._edge_strip_extents(
        source_width=64,
        source_height=48,
        left=4096,
        right=0,
        top=0,
        bottom=4096,
    )

    assert strip_left == 64
    assert strip_bottom == 48


@pytest.mark.fast
def test_edge_fill_is_unchanged_inside_the_validated_padding_envelope(tmp_path: Path):
    source_path = tmp_path / "source.png"
    source = _detailed_source(96, 64)
    source.save(source_path)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="24,96,24,96",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=8,
    )

    # No side reaches past what the base strip covers, so no fade is applied at all.
    assert OutpaintUtil._edge_strip_extents(
        source_width=96, source_height=64, left=96, right=96, top=24, bottom=24
    ) == (12, 12, 8, 8)
    assert (
        OutpaintUtil._edge_fade_mask(
            target_width=canvas.target_width,
            target_height=canvas.target_height,
            source_width=96,
            source_height=64,
            paste_left=canvas.paste_left,
            paste_top=canvas.paste_top,
        )
        is None
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("padding_value", "expected_pads"),
    [
        # Every padding recorded across the 128 outpaint/reframe validation runs,
        # all of them on 432x240 sources. None may change the sampled strip or pick
        # up a fade, or those canvases stop matching what was validated.
        ("5%,80%,5%,60%", (12, 349, 20, 259)),
        ("25%,80%,25%,60%", (60, 349, 68, 259)),
        ("5%,35%,5%,35%", (12, 153, 20, 151)),
        ("5%,40%,5%,30%", (12, 175, 20, 129)),
        ("25%,50%,25%,50%", (60, 216, 68, 216)),
        ("0,25%,0,25%", (0, 116, 0, 108)),
    ],
)
def test_validated_envelope_keeps_the_historical_edge_fill(tmp_path: Path, padding_value, expected_pads):
    source_path = tmp_path / "source.png"
    _detailed_source(432, 240).save(source_path)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value=padding_value,
        output_path=tmp_path / "canvas.png",
    )

    top, right, bottom, left = expected_pads
    assert (canvas.paste_top, canvas.paste_left) == (top, left)
    assert canvas.target_width == 432 + left + right
    assert canvas.target_height == 240 + top + bottom
    # Historical strips: min(32, side // 8) on every side, and no neutral fade.
    assert OutpaintUtil._edge_strip_extents(
        source_width=432, source_height=240, left=left, right=right, top=top, bottom=bottom
    ) == (32, 32, 30, 30)
    assert (
        OutpaintUtil._edge_fade_mask(
            target_width=canvas.target_width,
            target_height=canvas.target_height,
            source_width=432,
            source_height=240,
            paste_left=left,
            paste_top=top,
        )
        is None
    )
    assert OutpaintUtil._edge_blur_radius(max(expected_pads)) == min(8, max(2, max(expected_pads) // 24))


@pytest.mark.fast
def test_edge_fill_fades_deep_padding_toward_the_border_colour(tmp_path: Path):
    source_path = tmp_path / "source.png"
    source = _detailed_source(128, 128)
    source.save(source_path)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="0,0,900,0",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=8,
    )

    expanded = Image.open(canvas.canvas_path).convert("RGB")
    bottom = max(0, canvas.target_height - canvas.paste_top - canvas.source_height)
    band = expanded.crop((0, canvas.paste_top + canvas.source_height, canvas.source_width, canvas.target_height))
    near = band.crop((0, 0, band.width, bottom // 4))
    far = band.crop((0, bottom // 2, band.width, bottom))

    near_streaks = _band_lateral_std(near, vertical_expansion=True)
    far_streaks = _band_lateral_std(far, vertical_expansion=True)
    assert far_streaks < near_streaks / 2


@pytest.mark.fast
def test_neutral_fill_matches_the_border_colour_and_is_flat(tmp_path: Path):
    source_path = tmp_path / "source.png"
    source = Image.new("RGB", (128, 96), color=(40, 90, 140))
    source.paste((200, 30, 30), (0, 0, 16, 96))
    source.save(source_path)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="0,0,0,128",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=8,
        fill_mode="neutral",
    )

    expanded = Image.open(canvas.canvas_path).convert("RGB")
    expected = OutpaintUtil._border_colors(source)["left"]
    band = expanded.crop((0, 24, 64, 72))
    extrema = band.getextrema()
    for channel, (low, high) in enumerate(extrema):
        assert high - low <= 1
        assert abs(low - expected[channel]) <= 2


@pytest.mark.fast
def test_neutral_fill_has_a_softer_seam_than_edge_fill(tmp_path: Path):
    source_path = tmp_path / "source.png"
    _detailed_source(128, 128).save(source_path)

    seams = {}
    for fill_mode in ("edge", "neutral"):
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_path,
            padding_value="0,0,512,0",
            output_path=tmp_path / f"canvas_{fill_mode}.png",
            dimension_multiple=8,
            fill_mode=fill_mode,
        )
        expanded = Image.open(canvas.canvas_path).convert("L")
        y = canvas.paste_top + canvas.source_height
        seams[fill_mode] = (
            sum(abs(expanded.getpixel((x, y)) - expanded.getpixel((x, y - 1))) for x in range(canvas.source_width))
            / canvas.source_width
        )

    assert seams["neutral"] < seams["edge"]


@pytest.mark.fast
def test_unknown_fill_mode_is_rejected(tmp_path: Path):
    source_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), color="white").save(source_path)

    with pytest.raises(ValueError, match="fill_mode must be"):
        OutpaintUtil.create_expanded_canvas(
            source_path=source_path,
            padding_value="4,4,4,4",
            output_path=tmp_path / "canvas.png",
            fill_mode="mirror",
        )


@pytest.mark.fast
@pytest.mark.parametrize("fill_mode", ["edge", "neutral", "blur", "solid"])
def test_source_paste_geometry_is_independent_of_fill_mode(tmp_path: Path, fill_mode: str):
    source_path = tmp_path / "source.png"
    _detailed_source(160, 96).save(source_path)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="10%,300%,200%,25",
        output_path=tmp_path / f"canvas_{fill_mode}.png",
        dimension_multiple=16,
        fill_mode=fill_mode,
    )

    assert canvas.paste_left == 25
    assert canvas.paste_top == 9
    assert canvas.target_width == 672
    assert canvas.target_height == 304
    expanded = Image.open(canvas.canvas_path).convert("RGB")
    pasted = expanded.crop((25, 9, 25 + 160, 9 + 96))
    assert ImageChops.difference(pasted, Image.open(source_path).convert("RGB")).getbbox() is None


@pytest.mark.fast
def test_negative_restore_threshold_and_preserve_source_false_agree(tmp_path: Path):
    source_path = tmp_path / "source.png"
    source = Image.new("RGB", (80, 64), color=(20, 40, 60))
    source.save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="8,8,8,8",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=1,
    )
    # The generated region matches the source, so the adaptive path *would* paste.
    generated = Image.open(canvas.canvas_path).convert("RGB")

    legacy = OutpaintUtil.composite_source_region(
        generated_image=generated,
        canvas=canvas,
        feather_px=None,
        restore_threshold=-1.0,
    )
    explicit = OutpaintUtil.composite_source_region(
        generated_image=generated,
        canvas=canvas,
        feather_px=None,
        preserve_source=False,
    )

    assert legacy.outpaint_preservation_applied is False
    assert legacy.outpaint_preservation_mode == "never"
    assert explicit.outpaint_preservation_applied is False
    assert ImageChops.difference(legacy, explicit).getbbox() is None
    assert ImageChops.difference(legacy, generated).getbbox() is None


@pytest.mark.fast
def test_preserve_source_true_pastes_even_past_the_restore_threshold(tmp_path: Path):
    source_path = tmp_path / "source.png"
    # Below 64 px the default mask is fully opaque, so the paste is exact.
    source = Image.new("RGB", (48, 48), color=(20, 40, 60))
    source.save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="8,8,8,8",
        output_path=tmp_path / "canvas.png",
        dimension_multiple=1,
    )
    generated = Image.new("RGB", (canvas.target_width, canvas.target_height), color=(0, 255, 0))

    adaptive = OutpaintUtil.composite_source_region(generated_image=generated, canvas=canvas)
    forced = OutpaintUtil.composite_source_region(
        generated_image=generated,
        canvas=canvas,
        restore_threshold=-1.0,
        preserve_source=True,
    )

    assert adaptive.outpaint_preservation_applied is False
    assert forced.outpaint_preservation_applied is True
    assert forced.outpaint_preservation_mode == "always"
    assert forced.getpixel((canvas.paste_left + 20, canvas.paste_top + 20)) == (20, 40, 60)
    assert forced.getpixel((0, 0)) == (0, 255, 0)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("preserve_source", "feather_px", "restore_threshold", "expected"),
    [
        (None, None, 12.0, "adaptive"),
        (None, None, -1.0, "never"),
        (None, 0, -1.0, "always"),
        (None, 8, 12.0, "always"),
        (True, None, -1.0, "always"),
        (False, 8, 12.0, "never"),
    ],
)
def test_preservation_mode_resolution(preserve_source, feather_px, restore_threshold, expected):
    assert (
        OutpaintUtil._resolve_preservation_mode(
            preserve_source=preserve_source,
            feather_px=feather_px,
            restore_threshold=restore_threshold,
        )
        == expected
    )


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
