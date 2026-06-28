from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.utils.box_values import AbsoluteBoxValues
from mflux.utils.outpaint_util import OutpaintCanvas


class _FakeBatchNorm:
    def __init__(self):
        self.running_mean = mx.zeros((128,), dtype=mx.float32)
        self.running_var = mx.ones((128,), dtype=mx.float32)
        self.eps = 1e-5


class _FakeFlux2VAE:
    def __init__(self):
        self.bn = _FakeBatchNorm()


def test_flux2_reference_conditioning_uses_per_image_native_dimensions_and_crop(monkeypatch, tmp_path):
    tall = tmp_path / "tall.png"
    wide = tmp_path / "wide.png"
    Image.new("RGB", (640, 1280), color="white").save(tall)
    Image.new("RGB", (1536, 768), color="white").save(wide)

    captured: list[tuple[Path | str, int, int, str]] = []

    def fake_encode_image(*, vae, image_path, height, width, tiling_config, resize_mode, **kwargs):
        del vae, tiling_config, kwargs
        captured.append((image_path, width, height, resize_mode))
        return mx.zeros((1, 32, max(2, height // 8), max(2, width // 8)), dtype=mx.float32)

    monkeypatch.setattr(
        "mflux.models.flux2.variants.edit.flux2_klein_edit_helpers.LatentCreator.encode_image",
        staticmethod(fake_encode_image),
    )

    image_latents, image_latent_ids = _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
        vae=_FakeFlux2VAE(),
        tiling_config=None,
        image_paths=[tall, wide],
        height=1024,
        width=1024,
        batch_size=1,
    )

    assert captured == [
        (tall, 640, 1280, "crop"),
        (wide, 1440, 720, "crop"),
    ]
    assert image_latents.shape[0] == 1
    assert image_latent_ids.shape[0] == 1
    assert image_latents.shape[1] == image_latent_ids.shape[1]
    assert image_latent_ids[0, 0, 0].item() == 10
    assert image_latent_ids[0, -1, 0].item() == 20


def test_flux2_outpaint_mask_preserves_full_source_window():
    canvas = OutpaintCanvas(
        canvas_path=Path("canvas.png"),
        source_path=Path("source.png"),
        source_width=160,
        source_height=96,
        target_width=320,
        target_height=192,
        paste_left=48,
        paste_top=48,
        padding=AbsoluteBoxValues(top=48, right=112, bottom=48, left=48),
    )

    mask = _Flux2KleinEditHelpers.prepare_outpaint_edit_mask(
        canvas=canvas,
        height=192,
        width=320,
    )
    grid = np.array(mask).reshape(1, 12, 20, 1)[0, :, :, 0]

    assert grid[6, 8] == 0.0
    assert grid[0, 0] == 1.0
    assert grid[5, 4] == pytest.approx(0.5019608, abs=1e-6)
    assert grid[4, 4] > grid[5, 4]
    assert grid[5, 12] == 1.0


def test_flux2_distilled_runtime_rejects_guidance_above_one():
    txt2img = Flux2Klein.__new__(Flux2Klein)
    txt2img.model_config = SimpleNamespace(
        model_name="AbstractFramework/flux.2-klein-9b-8bit",
        base_model="black-forest-labs/FLUX.2-klein-9B",
    )
    edit = Flux2KleinEdit.__new__(Flux2KleinEdit)
    edit.model_config = txt2img.model_config

    with pytest.raises(ValueError, match="base models"):
        txt2img._validate_guidance(1.5)

    with pytest.raises(ValueError, match="base models"):
        edit._validate_guidance(1.5)


def test_flux2_base_runtime_accepts_guidance_above_one():
    txt2img = Flux2Klein.__new__(Flux2Klein)
    txt2img.model_config = SimpleNamespace(
        model_name="AbstractFramework/flux.2-klein-base-4b-8bit",
        base_model="black-forest-labs/FLUX.2-klein-base-4B",
    )
    edit = Flux2KleinEdit.__new__(Flux2KleinEdit)
    edit.model_config = txt2img.model_config

    txt2img._validate_guidance(4.0)
    edit._validate_guidance(4.0)
