import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
from mflux.models.fibo.variants.edit import util as fibo_edit_util_module
from mflux.models.fibo.variants.edit.util import (
    FIBO_EDIT_DIMENSION_MULTIPLE,
    FIBO_EDIT_PROMPTIFIER_MODEL_ID,
    FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT,
    FiboEditUtil,
)
from mflux.utils.dimension_resolver import CANVAS_POLICY_SOURCE_ASPECT


def test_ensure_edit_instruction_uses_existing_value():
    prompt = json.dumps({"short_description": "owl", "edit_instruction": "make it white"})
    updated = FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction="ignored")
    updated_dict = json.loads(updated)
    assert updated_dict["edit_instruction"] == "make it white"


def test_ensure_edit_instruction_injects_value_when_missing():
    prompt = json.dumps({"short_description": "owl"})
    updated = FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction="add glasses")
    updated_dict = json.loads(updated)
    assert updated_dict["edit_instruction"] == "add glasses"


def test_ensure_edit_instruction_accepts_dict_prompt():
    updated = FiboEditUtil.ensure_edit_instruction(
        {"short_description": "owl", "edit_instruction": "add glasses"}
    )
    updated_dict = json.loads(updated)
    assert updated_dict["edit_instruction"] == "add glasses"


def test_ensure_edit_instruction_requires_value_when_missing():
    prompt = json.dumps({"short_description": "owl"})
    with pytest.raises(ValueError, match="edit_instruction"):
        FiboEditUtil.ensure_edit_instruction(prompt, edit_instruction=None)


def test_load_edit_image_raises_for_mask_size_mismatch(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(image_path)
    Image.new("L", (32, 32), 255).save(mask_path)

    with pytest.raises(ValueError, match="Mask and image must have the same size"):
        FiboEditUtil.load_edit_image(image_path=image_path, width=64, height=64, mask_path=mask_path)


def test_get_json_prompt_for_edit_returns_existing_json():
    args = SimpleNamespace(prompt='{"short_description":"owl","edit_instruction":"make it white"}', prompt_file=None)

    prompt = FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)

    assert json.loads(prompt)["edit_instruction"] == "make it white"


def test_get_json_prompt_for_edit_rejects_json_without_edit_instruction(monkeypatch):
    class _UnexpectedVLM:
        def __init__(self, quantize):
            raise AssertionError("valid JSON without edit_instruction must not fall back to VLM")

    monkeypatch.setattr(fibo_edit_util_module, "FiboVLM", _UnexpectedVLM)
    args = SimpleNamespace(prompt='{"short_description":"owl"}', prompt_file=None, image_path="input.png", mask_path=None)

    with pytest.raises(ValueError, match="edit_instruction"):
        FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)


def test_get_json_prompt_for_edit_requires_prompt_input():
    args = SimpleNamespace(prompt=None, prompt_file=None, image_path="input.png", mask_path=None)

    with pytest.raises(ValueError, match="requires an edit instruction via --prompt/--prompt-file"):
        FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)


def test_get_json_prompt_for_edit_uses_default_when_missing():
    args = SimpleNamespace(prompt=None, prompt_file=None, image_path="input.png", mask_path=None)

    prompt = FiboEditUtil.get_json_prompt_for_edit(
        args,
        quantize=None,
        default_json_prompt_if_missing=FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT,
    )

    assert json.loads(prompt)["edit_instruction"] == json.loads(FIBO_EDIT_RMBG_DEFAULT_JSON_PROMPT)["edit_instruction"]


def test_get_json_prompt_for_edit_requires_image_for_natural_language_prompt():
    args = SimpleNamespace(prompt="turn the owl white", prompt_file=None, image_path=None, mask_path=None)

    with pytest.raises(ValueError, match="requires --image-path"):
        FiboEditUtil.get_json_prompt_for_edit(args, quantize=None)


def test_get_json_prompt_for_edit_routes_plain_image_to_vlm(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(image_path)

    captured = {}

    class _FakeVLM:
        def __init__(self, model_id, quantize):
            captured["model_id"] = model_id
            captured["quantize"] = quantize

        def edit(self, image, edit_instruction, use_mask, seed):
            captured["image"] = image
            captured["edit_instruction"] = edit_instruction
            captured["use_mask"] = use_mask
            captured["seed"] = seed
            return json.dumps({"short_description": "owl", "edit_instruction": "make it white"})

    monkeypatch.setattr(fibo_edit_util_module, "FiboVLM", _FakeVLM)
    args = SimpleNamespace(prompt="turn the owl white", prompt_file=None, image_path=image_path, mask_path=None)

    prompt = FiboEditUtil.get_json_prompt_for_edit(args, quantize=8)

    assert json.loads(prompt)["edit_instruction"] == "make it white"
    assert captured["model_id"] == FIBO_EDIT_PROMPTIFIER_MODEL_ID
    assert captured["quantize"] == 8
    assert captured["edit_instruction"] == "turn the owl white"
    assert captured["use_mask"] is False
    assert captured["seed"] == 42
    assert captured["image"].getpixel((0, 0)) == (10, 20, 30)
    assert "Preparing FIBO edit JSON prompt" in capsys.readouterr().out


def test_get_json_prompt_for_edit_rejects_masked_local_vlm_conversion(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), (10, 20, 30)).save(image_path)
    Image.new("L", (64, 64), 255).save(mask_path)
    args = SimpleNamespace(prompt="turn the owl white", prompt_file=None, image_path=image_path, mask_path=mask_path)

    with pytest.raises(ValueError, match="Masked FIBO edit requires a JSON prompt"):
        FiboEditUtil.get_json_prompt_for_edit(args, quantize=8)


def test_build_rgba_composite_image_returns_rgba(tmp_path):
    src = tmp_path / "in.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(src)
    matte = Image.new("L", (8, 6), 200)

    rgba = FiboEditUtil.build_rgba_composite_image(src, matte)

    assert rgba.mode == "RGBA"
    assert rgba.size == (32, 24)


def test_resolve_preferred_canvas_size_matches_fibo_edit_auto_resize(tmp_path):
    image_path = tmp_path / "widescreen.png"
    Image.new("RGB", (1600, 900), (10, 20, 30)).save(image_path)

    width, height = FiboEditUtil.resolve_preferred_canvas_size(
        image_path=image_path,
        width=None,
        height=None,
    )

    assert (width, height) == (1360, 768)


def test_resolve_preferred_canvas_size_respects_explicit_dimensions(tmp_path):
    image_path = tmp_path / "widescreen.png"
    Image.new("RGB", (1600, 900), (10, 20, 30)).save(image_path)

    width, height = FiboEditUtil.resolve_preferred_canvas_size(
        image_path=image_path,
        width=672,
        height=384,
    )

    assert (width, height) == (672, 384)


def test_resolve_preferred_canvas_size_preserves_single_explicit_dimension(tmp_path):
    image_path = tmp_path / "widescreen.png"
    Image.new("RGB", (1600, 900), (10, 20, 30)).save(image_path)

    width, height = FiboEditUtil.resolve_preferred_canvas_size(
        image_path=image_path,
        width=512,
        height=None,
    )

    assert (width, height) == (512, None)


def test_fibo_edit_config_uses_16px_source_aspect_canvas(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (511, 299), (10, 20, 30)).save(image_path)

    config = Config(
        width=433,
        height=241,
        image_path=image_path,
        model_config=ModelConfig.fibo_edit(),
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        preserve_image_aspect_ratio=True,
        dimension_multiple=FIBO_EDIT_DIMENSION_MULTIPLE,
    )

    assert config.width % 16 == 0
    assert config.height % 16 == 0


def test_fibo_edit_config_infers_missing_dimension_from_source_aspect(tmp_path):
    image_path = tmp_path / "widescreen.png"
    Image.new("RGB", (1600, 900), (10, 20, 30)).save(image_path)

    width, height = FiboEditUtil.resolve_preferred_canvas_size(
        image_path=image_path,
        width=512,
        height=None,
    )
    config = Config(
        width=width,
        height=height,
        image_path=image_path,
        model_config=ModelConfig.fibo_edit(),
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        preserve_image_aspect_ratio=True,
        dimension_multiple=FIBO_EDIT_DIMENSION_MULTIPLE,
    )

    assert (config.width, config.height) == (512, 288)


def test_fibo_edit_auto_canvas_preserves_upstream_16px_preferred_size(tmp_path):
    image_path = tmp_path / "portrait.png"
    Image.new("RGB", (880, 1184), (10, 20, 30)).save(image_path)

    width, height = FiboEditUtil.resolve_preferred_canvas_size(
        image_path=image_path,
        width=None,
        height=None,
    )
    config = Config(
        width=width,
        height=height,
        image_path=image_path,
        model_config=ModelConfig.fibo_edit(),
        canvas_policy=CANVAS_POLICY_SOURCE_ASPECT,
        preserve_image_aspect_ratio=True,
        dimension_multiple=FIBO_EDIT_DIMENSION_MULTIPLE,
    )

    assert (width, height) == (880, 1184)
    assert (config.width, config.height) == (880, 1184)


def test_fibo_edit_auto_canvas_can_use_exact_upstream_preferred_size(tmp_path):
    image_path = tmp_path / "widescreen.png"
    Image.new("RGB", (432, 240), (10, 20, 30)).save(image_path)

    width, height = FiboEditUtil.resolve_preferred_canvas_size(
        image_path=image_path,
        width=None,
        height=None,
    )
    config = Config(
        width=width,
        height=height,
        image_path=image_path,
        model_config=ModelConfig.fibo_edit(),
        preserve_image_aspect_ratio=False,
        dimension_multiple=FIBO_EDIT_DIMENSION_MULTIPLE,
    )

    assert (width, height) == (1360, 768)
    assert (config.width, config.height) == (1360, 768)


def test_fibo_latent_creator_respects_requested_dtype():
    latents = FiboLatentCreator.create_noise(seed=1, height=64, width=64, dtype=mx.bfloat16)

    assert latents.dtype == mx.bfloat16


def test_fibo_edit_conditioning_image_respects_requested_dtype():
    class _FakeVAE:
        def __init__(self):
            self.seen_dtype = None

        def encode(self, image):
            self.seen_dtype = image.dtype
            return mx.zeros((1, 48, 4, 4), dtype=mx.float32)

    vae = _FakeVAE()
    image = Image.new("RGB", (64, 64), (10, 20, 30))

    latents = FiboEditUtil.encode_conditioning_image(
        vae=vae,
        image=image,
        height=64,
        width=64,
        dtype=mx.bfloat16,
    )

    assert vae.seen_dtype == mx.bfloat16
    assert latents.dtype == mx.bfloat16
    np.testing.assert_array_equal(np.array(latents.shape), np.array((1, 16, 48)))
