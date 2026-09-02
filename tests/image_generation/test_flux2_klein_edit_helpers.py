from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
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


def test_flux2_reference_conditioning_canvas_primary_and_native_secondaries(monkeypatch, tmp_path):
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

    # The primary (edited) image is conditioned at the generation canvas
    # with a plain resize so reference and target grids match and no source
    # sliver is cropped away per pass; secondary references keep their own
    # per-image native sizing.
    assert captured == [
        (tall, 1024, 1024, "resize"),
        (wide, 1440, 720, "crop"),
    ]
    assert image_latents.shape[0] == 1
    assert image_latent_ids.shape[0] == 1
    assert image_latents.shape[1] == image_latent_ids.shape[1]
    assert image_latent_ids[0, 0, 0].item() == 10
    assert image_latent_ids[0, -1, 0].item() == 20


def test_flux2_reference_conditioning_inpaint_references_keep_native_dims(monkeypatch, tmp_path):
    # Inpaint passes secondary content references only (the edited source is
    # conditioned separately at the canvas), so no image follows the canvas.
    ref = tmp_path / "ref.png"
    Image.new("RGB", (640, 1280), color="white").save(ref)
    captured: list[tuple[Path | str, int, int, str]] = []

    def fake_encode_image(*, vae, image_path, height, width, tiling_config, resize_mode, **kwargs):
        del vae, tiling_config, kwargs
        captured.append((image_path, width, height, resize_mode))
        return mx.zeros((1, 32, max(2, height // 8), max(2, width // 8)), dtype=mx.float32)

    monkeypatch.setattr(
        "mflux.models.flux2.variants.edit.flux2_klein_edit_helpers.LatentCreator.encode_image",
        staticmethod(fake_encode_image),
    )

    _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
        vae=_FakeFlux2VAE(),
        tiling_config=None,
        image_paths=[ref],
        height=1024,
        width=1024,
        batch_size=1,
        t_coord_start=20,
        canvas_image_index=None,
    )

    assert captured == [(ref, 640, 1280, "crop")]


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


def _outpaint_canvas(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    paste_left: int,
    paste_top: int,
    padding: tuple[int, int, int, int],
) -> OutpaintCanvas:
    return OutpaintCanvas(
        canvas_path=Path("canvas.png"),
        source_path=Path("source.png"),
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        paste_left=paste_left,
        paste_top=paste_top,
        padding=AbsoluteBoxValues(*padding),
    )


def _latent_grid(canvas: OutpaintCanvas, **kwargs) -> np.ndarray:
    mask = _Flux2KleinEditHelpers.prepare_outpaint_edit_mask(
        canvas=canvas,
        height=canvas.target_height,
        width=canvas.target_width,
        **kwargs,
    )
    return np.array(mask).reshape(canvas.target_height // 16, canvas.target_width // 16)


# The reported regression: source 768x766, --outpaint-padding "0%,10%,100%,10%",
# canvas 928x1536. Nothing is added above the source, so nothing above the source
# may be regenerated.
_ZERO_TOP_PADDING_CANVAS = dict(
    source_width=768,
    source_height=766,
    target_width=928,
    target_height=1536,
    paste_left=76,
    paste_top=0,
    padding=(0, 76, 766, 76),
)


@pytest.mark.fast
def test_flux2_outpaint_preserve_box_skips_inset_on_unpadded_sides():
    canvas = _outpaint_canvas(**_ZERO_TOP_PADDING_CANVAS)

    # left/top/right/bottom counts of NEW canvas pixels; right is 76 requested plus the
    # 8px the 16-multiple round-up added, top is genuinely nothing.
    assert _Flux2KleinEditHelpers.outpaint_generated_gaps(canvas) == (76, 0, 84, 770)

    left, top, right, bottom = _Flux2KleinEditHelpers.outpaint_preserve_box(canvas=canvas)
    assert top == canvas.paste_top  # no padding above -> no transition band -> no inset
    assert (left, right, bottom) == (76 + 24, 76 + 768 - 24, 766 - 24)


@pytest.mark.fast
def test_flux2_outpaint_mask_keeps_unpadded_top_edge_fully_locked():
    canvas = _outpaint_canvas(**_ZERO_TOP_PADDING_CANVAS)
    grid = _latent_grid(canvas)

    assert grid.shape == (96, 58)
    # Latent rows 0-1 cover canvas rows 0-31, i.e. the top of the subject. Before the
    # per-side insets these were 1.0 / 0.502 (fully and half editable) across the whole
    # source width; they must now be hard-locked.
    assert grid[0:2, 8:44].max() == 0.0
    # The padded sides still get their band: the left seam sits at canvas x=76, and the
    # 24px inset pushes the first locked cell out to x=100 (latent col 7).
    assert grid[20, 6] == pytest.approx(0.282353, abs=1e-6)
    assert grid[20, 7] == 0.0
    # The padded bottom keeps its band too (source ends at canvas y=766, latent row 47).
    assert grid[46, 29] == pytest.approx(0.615686, abs=1e-6)
    assert grid[47, 29] == 1.0


@pytest.mark.fast
def test_flux2_outpaint_preserve_box_caps_inset_at_the_generated_gap():
    # padding right/bottom are 0; only the 16-multiple round-up adds pixels there
    # (848-76-768 = 4 on the right, 768-766 = 2 at the bottom). A side that gains 4px
    # must not surrender 24px of real source to a transition band.
    canvas = _outpaint_canvas(
        source_width=768,
        source_height=766,
        target_width=848,
        target_height=768,
        paste_left=76,
        paste_top=0,
        padding=(0, 0, 0, 76),
    )

    assert _Flux2KleinEditHelpers.outpaint_generated_gaps(canvas) == (76, 0, 4, 2)
    assert _Flux2KleinEditHelpers.outpaint_preserve_box(canvas=canvas) == (100, 0, 840, 764)


@pytest.mark.fast
def test_flux2_outpaint_preserve_box_never_collapses_on_a_tiny_source():
    # source_width // 2 - 1 == 0, so every inset clamps away and the box stays the full
    # source window rather than inverting.
    canvas = _outpaint_canvas(
        source_width=2,
        source_height=2,
        target_width=64,
        target_height=64,
        paste_left=16,
        paste_top=16,
        padding=(16, 46, 46, 16),
    )

    assert _Flux2KleinEditHelpers.outpaint_preserve_box(canvas=canvas) == (16, 16, 18, 18)


@pytest.mark.fast
@pytest.mark.parametrize(("source_size", "target_size"), [(128, 256), (1024, 2048)])
def test_flux2_outpaint_transition_band_is_scale_invariant(source_size, target_size):
    # transition_px is deliberately absolute, not a fraction of the canvas: the latent grid
    # is always canvas/16, so 24 canvas px is 1.5 latent cells at every resolution. Both a
    # 256x256 and a 2048x2048 canvas must therefore show the identical seam ramp, shifted
    # by exactly the same 1.5 cells.
    pad = (target_size - source_size) // 2
    canvas = _outpaint_canvas(
        source_width=source_size,
        source_height=source_size,
        target_width=target_size,
        target_height=target_size,
        paste_left=pad,
        paste_top=pad,
        padding=(pad, pad, pad, pad),
    )
    latent_cells = target_size // 16
    seam = pad // 16  # the source's left edge in latent cells
    row = latent_cells // 2

    no_band = _latent_grid(canvas, transition_px=0)
    banded = _latent_grid(canvas, transition_px=24)

    assert no_band.shape == (latent_cells, latent_cells)
    # With no band and 16-aligned geometry the mask is exactly the padded cell count.
    source_cells = source_size // 16
    assert no_band.sum() == pytest.approx(latent_cells**2 - source_cells**2)
    assert list(no_band[row, seam - 2 : seam + 2]) == pytest.approx([1.0, 0.874510, 0.125490, 0.0], abs=1e-6)
    # The band moves the seam 1.5 cells outward and the ramp shape is unchanged.
    assert list(banded[row, seam : seam + 3]) == pytest.approx([1.0, 0.501961, 0.0], abs=1e-6)
    assert banded.sum() > no_band.sum()


@pytest.mark.fast
def test_flux2_outpaint_mask_latent_layout_is_row_major_not_transposed():
    # Regression pin against a (height, width) transposition. The locked region here is
    # deliberately 4 latent cells wide by 2 tall and off-centre, so a transposed flatten
    # or a (latent_height, latent_width) swap in the PIL resize cannot reproduce it.
    canvas = _outpaint_canvas(
        source_width=128,
        source_height=96,
        target_width=320,
        target_height=256,
        paste_left=48,
        paste_top=64,
        padding=(64, 144, 96, 48),
    )

    mask = _Flux2KleinEditHelpers.prepare_outpaint_edit_mask(canvas=canvas, height=256, width=320)
    latent_height, latent_width = 256 // 16, 320 // 16
    assert mask.shape == (1, latent_height * latent_width, 1)

    grid = np.array(mask).reshape(latent_height, latent_width)
    rows, cols = np.where(grid == 0.0)
    assert (rows.min(), rows.max()) == (6, 7)
    assert (cols.min(), cols.max()) == (5, 8)
    # A transposed read lands outside the locked block.
    assert grid[5, 6] == pytest.approx(0.501961, abs=1e-6)

    # The mask is flattened with the same row-major order Flux2LatentCreator.pack_latents
    # uses for the latents it multiplies, so token i is latent cell (i // width, i % width).
    coordinates = mx.array(
        np.arange(latent_height * latent_width, dtype=np.float32).reshape(1, 1, latent_height, latent_width)
    )
    packed = np.array(Flux2LatentCreator.pack_latents(coordinates)).reshape(-1)
    flat = grid.reshape(-1)
    for index in np.where(flat == 0.0)[0]:
        row, col = divmod(int(packed[index]), latent_width)
        assert grid[row, col] == 0.0


def _traceable_reference_tokens(latent_height: int, latent_width: int) -> tuple[mx.array, mx.array]:
    # One token per latent cell, carrying its own flat index, with the ids
    # prepare_reference_image_conditioning would build for the canvas reference.
    tokens = latent_height * latent_width
    latents = mx.array(np.arange(tokens, dtype=np.float32).reshape(1, tokens, 1))
    ids = Flux2LatentCreator.prepare_grid_ids(
        mx.zeros((1, 128, latent_height, latent_width), dtype=mx.float32),
        t_coord=10,
    )
    return latents, ids


@pytest.mark.fast
@pytest.mark.parametrize(
    ("source_size", "target_size", "paste", "padding"),
    [
        # The reported A/B pair: same source, same 128px gap, opposite axes.
        ((640, 448), (640, 576), (0, 0), (0, 0, 115, 0)),
        ((640, 448), (768, 448), (0, 0), (0, 115, 0, 0)),
        # Both sides padded, and the recorded validation envelope.
        ((384, 384), (624, 384), (115, 0), (0, 115, 0, 115)),
        ((432, 240), (1040, 272), (259, 12), (12, 345, 12, 259)),
        # Two-deep-axis expansion: a deep bottom and a deep left open a free corner that shares
        # neither a row nor a column with the source, and every latent cell in it is pure filler.
        ((432, 240), (688, 400), (254, 0), (0, 2, 148, 254)),
    ],
)
def test_flux2_outpaint_reference_conditioning_carries_no_pure_filler_token(source_size, target_size, paste, padding):
    # The reported failure is the padded region coming back as the conditioning canvas.
    # The canvas reference tokens sit at the same (h, w) rope coordinates as the generation
    # latents, one t index apart, and stay clean at every step - so a token for a latent cell
    # holding nothing but synthetic fill is a noise-free copy of the answer the model is meant
    # to invent, at exactly the position it has to invent it. None may reach the transformer.
    canvas = _outpaint_canvas(
        source_width=source_size[0],
        source_height=source_size[1],
        target_width=target_size[0],
        target_height=target_size[1],
        paste_left=paste[0],
        paste_top=paste[1],
        padding=padding,
    )
    latent_height, latent_width = target_size[1] // 16, target_size[0] // 16
    image_latents, image_latent_ids = _traceable_reference_tokens(latent_height, latent_width)

    kept, kept_ids = _Flux2KleinEditHelpers.outpaint_reference_conditioning(
        image_latents=image_latents,
        image_latent_ids=image_latent_ids,
        canvas=canvas,
        height=target_size[1],
        width=target_size[0],
    )

    cells = np.array(kept).reshape(-1).astype(int)
    rows, cols = cells // latent_width, cells % latent_width
    # Every surviving cell's 16px footprint must overlap the pasted source rectangle.
    assert (rows * 16 < canvas.paste_top + canvas.source_height).all()
    assert ((rows + 1) * 16 > canvas.paste_top).all()
    assert (cols * 16 < canvas.paste_left + canvas.source_width).all()
    assert ((cols + 1) * 16 > canvas.paste_left).all()
    # Nothing that touches the source is dropped, so the seam still conditions the run.
    expected_rows = (canvas.paste_top + canvas.source_height - 1) // 16 - canvas.paste_top // 16 + 1
    expected_cols = (canvas.paste_left + canvas.source_width - 1) // 16 - canvas.paste_left // 16 + 1
    assert kept.shape[1] == expected_rows * expected_cols
    # The ids travel with the tokens they belong to, still at the reference t index.
    assert np.array_equal(np.array(kept_ids)[0, :, 1], rows)
    assert np.array_equal(np.array(kept_ids)[0, :, 2], cols)
    assert set(np.array(kept_ids)[0, :, 0].tolist()) == {10}


@pytest.mark.fast
def test_flux2_outpaint_reference_conditioning_keeps_secondary_references_intact():
    canvas = _outpaint_canvas(
        source_width=640,
        source_height=448,
        target_width=768,
        target_height=448,
        paste_left=0,
        paste_top=0,
        padding=(0, 115, 0, 0),
    )
    latent_height, latent_width = 28, 48
    canvas_latents, canvas_ids = _traceable_reference_tokens(latent_height, latent_width)
    extra_latents = mx.full((1, 7, 1), -1.0, dtype=mx.float32)
    extra_ids = mx.full((1, 7, 4), 20, dtype=mx.int32)

    kept, kept_ids = _Flux2KleinEditHelpers.outpaint_reference_conditioning(
        image_latents=mx.concatenate([canvas_latents, extra_latents], axis=1),
        image_latent_ids=mx.concatenate([canvas_ids, extra_ids], axis=1),
        canvas=canvas,
        height=448,
        width=768,
    )

    assert kept.shape[1] == 28 * 40 + 7
    assert np.array_equal(np.array(kept[:, -7:, :]), np.array(extra_latents))
    assert np.array_equal(np.array(kept_ids[:, -7:, :]), np.array(extra_ids))


@pytest.mark.fast
def test_flux2_outpaint_reference_conditioning_is_a_no_op_without_pure_filler_cells():
    # Only the 16-multiple round-up adds pixels, and it adds less than one latent cell,
    # so every cell still holds source and the reference stream must pass through unchanged.
    canvas = _outpaint_canvas(
        source_width=636,
        source_height=444,
        target_width=640,
        target_height=448,
        paste_left=0,
        paste_top=0,
        padding=(0, 0, 0, 0),
    )
    image_latents, image_latent_ids = _traceable_reference_tokens(28, 40)

    kept, kept_ids = _Flux2KleinEditHelpers.outpaint_reference_conditioning(
        image_latents=image_latents,
        image_latent_ids=image_latent_ids,
        canvas=canvas,
        height=448,
        width=640,
    )

    assert kept is image_latents
    assert kept_ids is image_latent_ids


@pytest.mark.fast
def test_flux2_outpaint_route_never_conditions_on_a_pure_filler_canvas_cell(monkeypatch, tmp_path):
    # End-to-end wiring pin for the reported failure, with no weights: run the outpaint route
    # over a stub transformer and assert that what actually reaches the model carries no
    # reference token for a latent cell made entirely of conditioning-canvas filler.
    from mlx import nn

    from mflux.callbacks.callback_registry import CallbackRegistry
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.edit.flux2_klein_outpaint import Flux2KleinOutpaint
    from mflux.utils.outpaint_util import OutpaintUtil

    source_path = tmp_path / "source.png"
    Image.new("RGB", (640, 448), color="white").save(source_path)
    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_path,
        padding_value="0,115,0,0",
        output_path=tmp_path / "canvas.png",
    )
    assert (canvas.target_width, canvas.target_height) == (768, 448)

    def fake_encode_image(*, vae, image_path, height, width, tiling_config=None, resize_mode="resize", **kwargs):
        del vae, image_path, tiling_config, resize_mode, kwargs
        return mx.zeros((1, 32, height // 8, width // 8), dtype=mx.float32)

    monkeypatch.setattr(
        "mflux.models.flux2.variants.edit.flux2_klein_edit_helpers.LatentCreator.encode_image",
        staticmethod(fake_encode_image),
    )
    monkeypatch.setattr(
        Flux2KleinEdit,
        "_encode_prompt_pair",
        lambda self, *, prompt, negative_prompt, guidance: (
            mx.zeros((1, 4, 7680), dtype=mx.float32),
            mx.zeros((1, 4, 4), dtype=mx.int32),
            None,
            None,
        ),
    )

    seen: dict = {}

    def capturing_predict(**kwargs):
        seen.update(kwargs)
        return mx.zeros_like(kwargs["latents"])

    class _FakeVAE(_FakeFlux2VAE):
        def decode_packed_latents(self, packed_latents):
            del packed_latents
            return mx.zeros((1, 3, canvas.target_height, canvas.target_width), dtype=mx.float32)

    model = Flux2KleinOutpaint.__new__(Flux2KleinOutpaint)
    nn.Module.__init__(model)
    model.model_config = ModelConfig.flux2_klein_4b()
    model.vae = _FakeVAE()
    model.transformer = SimpleNamespace()
    model.tiling_config = None
    model.prompt_cache = {}
    model.callbacks = CallbackRegistry()
    model.compiled_predict_cache = SimpleNamespace(get_or_build=lambda **kwargs: capturing_predict)
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None

    model.generate_image(seed=99, prompt="", canvas=canvas, num_inference_steps=2)

    latent_height, latent_width = canvas.target_height // 16, canvas.target_width // 16
    generation_tokens = latent_height * latent_width
    reference_ids = np.array(seen["image_latent_ids"])[0]
    assert seen["latents"].shape[1] == generation_tokens
    # Every reference token must sit on a latent cell that holds real source pixels. Before
    # the fix the reference stream was the whole canvas grid, so latent columns 40-47 - the
    # 128px of pure edge-fill smear - were handed to the model clean at every step.
    assert reference_ids[:, 2].max() == (640 - 1) // 16
    assert len(reference_ids) == latent_height * (640 // 16)
    editable = np.array(
        _Flux2KleinEditHelpers.prepare_outpaint_edit_mask(
            canvas=canvas, height=canvas.target_height, width=canvas.target_width
        )
    ).reshape(latent_height, latent_width)
    filler_only = np.zeros((latent_height, latent_width), dtype=bool)
    filler_only[:, 640 // 16 :] = True
    assert (editable[filler_only] == 1.0).all()  # those cells are free in the latent lock
    assert not any(bool(filler_only[h, w]) for h, w in reference_ids[:, 1:3])  # and free in the conditioning


@pytest.mark.fast
def test_flux2_preserved_source_latents_match_the_post_step_noise_level():
    # scheduler.step moves latents from sigmas[t] to sigmas[t + 1], and the mask blend runs
    # after the step, so the preserved branch must be built at sigmas[t + 1] too.
    clean = mx.ones((1, 4, 2), dtype=mx.float32)
    noise = mx.zeros((1, 4, 2), dtype=mx.float32)
    sigmas = mx.array([1.0, 0.6, 0.25, 0.0], dtype=mx.float32)

    for timestep, expected in ((0, 0.6), (1, 0.25)):
        preserved = _Flux2KleinEditHelpers.preserved_source_latents(
            clean_latents=clean,
            noise_latents=noise,
            sigmas=sigmas,
            timestep=timestep,
        )
        assert float(preserved[0, 0, 0]) == pytest.approx(1.0 - expected)

    # Terminal step: sigmas[t + 1] is the trailing zero, so the clean latents come back as-is.
    terminal = _Flux2KleinEditHelpers.preserved_source_latents(
        clean_latents=clean,
        noise_latents=mx.full((1, 4, 2), 7.0, dtype=mx.float32),
        sigmas=sigmas,
        timestep=2,
    )
    assert float(terminal[0, 0, 0]) == 1.0


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


def _model_config(*, base: bool):
    return SimpleNamespace(
        model_name="AbstractFramework/flux.2-klein-base-4b-8bit" if base else "AbstractFramework/flux.2-klein-9b-8bit",
        base_model="black-forest-labs/FLUX.2-klein-base-4B" if base else "black-forest-labs/FLUX.2-klein-9B",
    )


def test_flux2_negative_prompt_is_accepted_on_base_weights_under_guidance():
    _Flux2KleinEditHelpers.validate_negative_prompt(
        model_config=_model_config(base=True), guidance=4.0, negative_prompt="blurry"
    )
    # An empty or absent negative is always fine: that is the recorded base behaviour.
    for negative in ("", None):
        _Flux2KleinEditHelpers.validate_negative_prompt(
            model_config=_model_config(base=True), guidance=1.0, negative_prompt=negative
        )
        _Flux2KleinEditHelpers.validate_negative_prompt(
            model_config=_model_config(base=False), guidance=1.0, negative_prompt=negative
        )


def test_flux2_negative_prompt_is_rejected_on_distilled_weights():
    with pytest.raises(ValueError, match="distilled weights"):
        _Flux2KleinEditHelpers.validate_negative_prompt(
            model_config=_model_config(base=False), guidance=1.0, negative_prompt="blurry"
        )


def test_flux2_negative_prompt_on_base_weights_needs_guidance_above_one():
    with pytest.raises(ValueError, match="above 1.0"):
        _Flux2KleinEditHelpers.validate_negative_prompt(
            model_config=_model_config(base=True), guidance=1.0, negative_prompt="blurry"
        )
