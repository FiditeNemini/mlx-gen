from pathlib import Path

import PIL.Image
import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE, CANVAS_POLICY_SOURCE_ASPECT


@pytest.mark.fast
def test_partial_width_none_resolves_from_reference_image(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    PIL.Image.new("RGB", (1200, 800)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=None,
        height=512,
        image_path=image_path,
    )

    assert config.width == 1200
    assert config.height == 512


@pytest.mark.fast
def test_partial_height_none_resolves_from_defaults_without_reference_image():
    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=640,
        height=None,
        image_path=None,
    )

    assert config.width == 640
    assert config.height == ui_defaults.HEIGHT


@pytest.mark.fast
def test_both_dimensions_none_resolve_from_reference_image(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    PIL.Image.new("RGB", (1200, 800)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=None,
        height=None,
        image_path=image_path,
    )

    assert config.width == 1200
    assert config.height == 800


@pytest.mark.fast
def test_both_dimensions_none_resolve_to_defaults_without_reference_image():
    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=None,
        height=None,
        image_path=None,
    )

    assert config.width == ui_defaults.WIDTH
    assert config.height == ui_defaults.HEIGHT


@pytest.mark.fast
def test_i2i_source_aspect_policy_resolves_mismatched_target_from_reference(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    PIL.Image.new("RGB", (432, 240)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=512,
        height=512,
        image_path=image_path,
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        preserve_image_aspect_ratio=True,
    )

    assert config.width != config.height
    requested_area = 512 * 512
    resolved_area = config.width * config.height
    assert resolved_area / requested_area == pytest.approx(1.0, abs=0.05)
    assert config.width / config.height == pytest.approx(432 / 240, abs=0.02)
    assert config.canvas_policy == CANVAS_POLICY_SOURCE_ASPECT
    assert config.requested_width == 512
    assert config.requested_height == 512
    assert config.source_image_width == 432
    assert config.source_image_height == 240


@pytest.mark.fast
def test_exact_resize_policy_keeps_mismatched_target(tmp_path: Path):
    image_path = tmp_path / "reference.png"
    PIL.Image.new("RGB", (432, 240)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=512,
        height=512,
        image_path=image_path,
        canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
    )

    assert config.width == 512
    assert config.height == 512
    assert config.canvas_policy == CANVAS_POLICY_EXACT_RESIZE
    assert config.source_image_width == 432
    assert config.source_image_height == 240


@pytest.mark.fast
def test_latent_img2img_strength_uses_standard_denoising_strength_semantics(tmp_path: Path):
    image_path = tmp_path / "input.png"
    PIL.Image.new("RGB", (512, 512)).save(image_path)

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        width=512,
        height=512,
        image_path=image_path,
        image_strength=0.4,
        num_inference_steps=50,
    )

    assert config.init_time_step == 30
    assert list(config.time_steps) == list(range(30, 50))


@pytest.mark.fast
def test_dimensions_that_floor_to_zero_are_rejected():
    with pytest.raises(ValueError, match="at least 16px"):
        Config(
            model_config=ModelConfig.flux2_klein_4b(),
            width=8,
            height=8,
        )
