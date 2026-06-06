from types import SimpleNamespace

from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE, CANVAS_POLICY_SOURCE_ASPECT
from mflux.utils.scale_factor import ScaleFactor


def test_qwen_edit_default_dimensions_preserve_reference_image(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (512, 384)).save(image_path)

    config, vl_width, vl_height, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.qwen_image_edit()),
        image_paths=[str(image_path)],
        num_inference_steps=4,
        height=None,
        width=None,
        guidance=2.5,
        image_path=image_path,
        scheduler="linear",
    )

    assert (config.width, config.height) == (1184, 896)
    assert (vae_width, vae_height) == (1184, 896)
    assert (vl_width, vl_height) == (1184, 896)


def test_qwen_edit_source_aspect_explicit_dimensions_are_size_target(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (512, 384)).save(image_path)

    config, _, _, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.qwen_image_edit()),
        image_paths=[str(image_path)],
        num_inference_steps=4,
        height=256,
        width=320,
        guidance=2.5,
        image_path=image_path,
        scheduler="linear",
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    assert (config.width, config.height) == (336, 256)
    assert (vae_width, vae_height) == (336, 256)
    assert (config.requested_width, config.requested_height) == (320, 256)
    assert abs((config.width / config.height) - (512 / 384)) < 0.03


def test_qwen_edit_exact_resize_policy_keeps_explicit_dimensions(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (512, 384)).save(image_path)

    config, _, _, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.qwen_image_edit()),
        image_paths=[str(image_path)],
        num_inference_steps=4,
        height=256,
        width=320,
        guidance=2.5,
        image_path=image_path,
        scheduler="linear",
        canvas_policy=CANVAS_POLICY_EXACT_RESIZE,
    )

    assert (config.width, config.height) == (320, 256)
    assert (vae_width, vae_height) == (320, 256)


def test_qwen_edit_plus_multi_reference_defaults_to_source_reference_canvas(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (512, 384)).save(source)
    Image.new("RGB", (640, 320)).save(target)

    config, vl_width, vl_height, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.from_name("qwen-image-edit-2511")),
        image_paths=[str(source), str(target)],
        num_inference_steps=4,
        height=None,
        width=None,
        guidance=4.0,
        image_path=None,
        scheduler="linear",
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    assert (config.width, config.height) == (1184, 896)
    assert (vae_width, vae_height) == (1184, 896)
    assert (config.requested_width, config.requested_height) == (1184, 896)
    assert (vl_width, vl_height) == (448, 320)
    assert config.image_path == source


def test_qwen_edit_plus_auto_scale_uses_diffusers_area_default(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (512, 384)).save(image_path)

    config, _, _, vae_width, vae_height = QwenImageEdit._compute_dimensions(
        SimpleNamespace(model_config=ModelConfig.from_name("qwen-image-edit-2511")),
        image_paths=[str(image_path)],
        num_inference_steps=40,
        height=ScaleFactor.parse("1x"),
        width=ScaleFactor.parse("1x"),
        guidance=4.0,
        image_path=image_path,
        scheduler="linear",
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
    )

    assert (config.width, config.height) == (1184, 896)
    assert (vae_width, vae_height) == (1184, 896)
