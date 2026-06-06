from pathlib import Path

from PIL import Image

from mflux.utils.dimension_resolver import (
    CANVAS_POLICY_EXACT_RESIZE,
    CANVAS_POLICY_SOURCE_ASPECT,
    DimensionResolver,
)
from mflux.utils.scale_factor import ScaleFactor


def _image(path: Path, size: tuple[int, int]) -> Path:
    Image.new("RGB", size).save(path)
    return path


def test_source_aspect_auto_preserves_source_size(tmp_path: Path):
    image_path = _image(tmp_path / "wide.png", (432, 240))

    resolved = DimensionResolver.resolve_image_canvas(
        width=ScaleFactor(1),
        height=ScaleFactor(1),
        reference_image_path=image_path,
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    assert (resolved.width, resolved.height) == (432, 240)
    assert (resolved.source_width, resolved.source_height) == (432, 240)
    assert (resolved.requested_width, resolved.requested_height) == (432, 240)


def test_source_aspect_uses_width_as_target_when_height_is_auto(tmp_path: Path):
    image_path = _image(tmp_path / "wide.png", (432, 240))

    resolved = DimensionResolver.resolve_image_canvas(
        width=864,
        height=ScaleFactor(1),
        reference_image_path=image_path,
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    assert (resolved.width, resolved.height) == (864, 480)


def test_source_aspect_mismatched_explicit_target_does_not_return_square(tmp_path: Path):
    image_path = _image(tmp_path / "wide.png", (432, 240))

    resolved = DimensionResolver.resolve_image_canvas(
        width=512,
        height=512,
        reference_image_path=image_path,
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    assert resolved.width != resolved.height
    assert resolved.width % 16 == 0
    assert resolved.height % 16 == 0
    assert (resolved.width * resolved.height) / (512 * 512) > 0.95
    assert abs((resolved.width / resolved.height) - (432 / 240)) < 0.02


def test_exact_resize_keeps_requested_dimensions(tmp_path: Path):
    image_path = _image(tmp_path / "wide.png", (432, 240))

    resolved = DimensionResolver.resolve_image_canvas(
        width=512,
        height=512,
        reference_image_path=image_path,
        canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
    )

    assert (resolved.width, resolved.height) == (512, 512)
    assert (resolved.source_width, resolved.source_height) == (432, 240)


def test_source_aspect_prioritizes_requested_size_over_microscopic_ratio_gain(tmp_path: Path):
    image_path = _image(tmp_path / "odd.png", (1234, 777))

    resolved = DimensionResolver.resolve_image_canvas(
        width=512,
        height=512,
        reference_image_path=image_path,
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    requested_area = 512 * 512
    resolved_area = resolved.width * resolved.height
    assert resolved_area / requested_area > 0.8
    assert abs((resolved.width / resolved.height) - (1234 / 777)) < 0.04
