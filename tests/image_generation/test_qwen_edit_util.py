import mlx.core as mx
from PIL import Image

from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit import qwen_edit_util as qwen_edit_util_module
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil


def test_image_conditioning_latents_use_vae_target_dimensions(monkeypatch):
    captured = {}

    def fake_encode_image(vae, image_path, height, width, tiling_config):
        captured["encode_height"] = height
        captured["encode_width"] = width
        return mx.zeros((1, height // 8, width // 8, 16))

    def fake_pack_latents(latents, height, width, num_channels_latents):
        captured["pack_height"] = height
        captured["pack_width"] = width
        captured["num_channels_latents"] = num_channels_latents
        return mx.zeros((1, (height // 16) * (width // 16), num_channels_latents * 4))

    monkeypatch.setattr(qwen_edit_util_module.LatentCreator, "encode_image", fake_encode_image)
    monkeypatch.setattr(qwen_edit_util_module.QwenLatentCreator, "pack_latents", fake_pack_latents)

    _, image_ids, cond_image_grid, num_images = QwenEditUtil.create_image_conditioning_latents(
        vae=object(),
        height=1024,
        width=768,
        image_paths="input.png",
        tiling_config=None,
    )

    assert captured == {
        "encode_height": 1024,
        "encode_width": 768,
        "pack_height": 1024,
        "pack_width": 768,
        "num_channels_latents": 16,
    }
    assert image_ids.shape == (1, 64 * 48, 3)
    assert cond_image_grid == [(1, 64, 48)]
    assert num_images == 1


def test_image_conditioning_latents_use_per_reference_vae_size(tmp_path, monkeypatch):
    wide_path = tmp_path / "wide.png"
    square_path = tmp_path / "square.png"
    Image.new("RGB", (512, 256)).save(wide_path)
    Image.new("RGB", (512, 512)).save(square_path)

    encoded_sizes = []

    def fake_encode_image(vae, image_path, height, width, tiling_config):
        encoded_sizes.append((width, height))
        return mx.zeros((1, height // 8, width // 8, 16))

    def fake_pack_latents(latents, height, width, num_channels_latents):
        return mx.zeros((1, (height // 16) * (width // 16), num_channels_latents * 4))

    monkeypatch.setattr(qwen_edit_util_module.LatentCreator, "encode_image", fake_encode_image)
    monkeypatch.setattr(qwen_edit_util_module.QwenLatentCreator, "pack_latents", fake_pack_latents)

    _, image_ids, cond_image_grid, num_images = QwenEditUtil.create_image_conditioning_latents(
        vae=object(),
        height=None,
        width=None,
        image_paths=[str(wide_path), str(square_path)],
        tiling_config=None,
    )

    assert encoded_sizes == [(1440, 736), (1024, 1024)]
    assert cond_image_grid == [(1, 46, 90), (1, 64, 64)]
    assert image_ids.shape == (1, 46 * 90 + 64 * 64, 3)
    assert num_images == 2


def test_create_inpaint_mask_latents_packs_binary_mask(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask_image = Image.new("L", (32, 32), 0)
    for x in range(16, 32):
        for y in range(32):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask_path)

    packed_mask = QwenEditUtil.create_inpaint_mask_latents(str(mask_path), height=32, width=32)
    unpacked_mask = QwenLatentCreator.unpack_latents(packed_mask, height=32, width=32)

    assert packed_mask.shape == (1, 4, 64)
    assert unpacked_mask.shape == (1, 16, 4, 4)
    assert float(mx.min(unpacked_mask[:, :, :, :2]).item()) == 0.0
    assert float(mx.max(unpacked_mask[:, :, :, :2]).item()) == 0.0
    assert float(mx.min(unpacked_mask[:, :, :, 2:]).item()) == 1.0
    assert float(mx.max(unpacked_mask[:, :, :, 2:]).item()) == 1.0


def test_blend_inpaint_latents_preserves_unmasked_region():
    latents = mx.ones((1, 4, 64), dtype=mx.float32) * 9
    image_latents = mx.zeros((1, 4, 64), dtype=mx.float32)
    noise = mx.ones((1, 4, 64), dtype=mx.float32) * 2
    mask_latents = mx.concatenate(
        [
            mx.zeros((1, 2, 64), dtype=mx.float32),
            mx.ones((1, 2, 64), dtype=mx.float32),
        ],
        axis=1,
    )

    blended = QwenEditUtil.blend_inpaint_latents(
        latents=latents,
        image_latents=image_latents,
        initial_noise=noise,
        mask_latents=mask_latents,
        sigma=0.5,
    )

    assert bool(mx.allclose(blended[:, :2], mx.ones((1, 2, 64), dtype=mx.float32)).item())
    assert bool(mx.allclose(blended[:, 2:], mx.ones((1, 2, 64), dtype=mx.float32) * 9).item())


def test_canvas_reference_image_is_conditioned_at_generation_canvas(tmp_path, monkeypatch):
    # A 614x512 source with a 608x512 resolved canvas must be conditioned at
    # the canvas so reference and target RoPE grids match (the mismatch is the
    # compounding zoom-crop drift defect).
    source_path = tmp_path / "portrait.png"
    Image.new("RGB", (614, 512)).save(source_path)
    captured = []

    def fake_encode_image(vae, image_path, height, width, tiling_config):
        captured.append((height, width))
        return mx.zeros((1, height // 8, width // 8, 16))

    monkeypatch.setattr(qwen_edit_util_module.LatentCreator, "encode_image", fake_encode_image)

    _, _, cond_image_grid, _ = QwenEditUtil.create_image_conditioning_latents(
        vae=object(),
        height=None,
        width=None,
        image_paths=str(source_path),
        tiling_config=None,
        canvas_size=(608, 512),
        canvas_image_index=0,
    )

    assert captured == [(512, 608)]
    assert cond_image_grid == [(1, 32, 38)]


def test_canvas_size_applies_only_to_the_canvas_reference_image(tmp_path, monkeypatch):
    primary_path = tmp_path / "primary.png"
    style_path = tmp_path / "style.png"
    Image.new("RGB", (614, 512)).save(primary_path)
    Image.new("RGB", (512, 512)).save(style_path)
    captured = []

    def fake_encode_image(vae, image_path, height, width, tiling_config):
        captured.append((height, width))
        return mx.zeros((1, height // 8, width // 8, 16))

    monkeypatch.setattr(qwen_edit_util_module.LatentCreator, "encode_image", fake_encode_image)

    QwenEditUtil.create_image_conditioning_latents(
        vae=object(),
        height=None,
        width=None,
        image_paths=[str(primary_path), str(style_path)],
        tiling_config=None,
        canvas_size=(608, 512),
        canvas_image_index=0,
    )

    # Primary at the canvas; the style reference keeps its own 1MP dims.
    assert captured[0] == (512, 608)
    assert captured[1] == (1024, 1024)


def test_explicit_width_height_still_condition_every_image(tmp_path, monkeypatch):
    # The inpaint path passes explicit dims; canvas_size must not override it.
    source_path = tmp_path / "inpaint.png"
    Image.new("RGB", (614, 512)).save(source_path)
    captured = []

    def fake_encode_image(vae, image_path, height, width, tiling_config):
        captured.append((height, width))
        return mx.zeros((1, height // 8, width // 8, 16))

    monkeypatch.setattr(qwen_edit_util_module.LatentCreator, "encode_image", fake_encode_image)

    QwenEditUtil.create_image_conditioning_latents(
        vae=object(),
        height=512,
        width=608,
        image_paths=str(source_path),
        tiling_config=None,
        canvas_size=None,
        canvas_image_index=0,
    )

    assert captured == [(512, 608)]
