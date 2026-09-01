"""Policy tests for the shared outpaint layer.

These run against `mflux.outpaint` directly - no argparse, no model weights - so the
conditioning-canvas contract is covered once for every route instead of once per backend CLI.
The end-to-end proof that the FLUX.2 command still produces the same canvas, notice and
metadata lives in `tests/cli/test_flux2_outpaint_fill.py`.
"""

from pathlib import Path

import pytest
from PIL import Image

from mflux.outpaint import (
    FLUX2_GREEN_BORDER_FILL_COLOR,
    OutpaintError,
    OutpaintFillPlan,
    OutpaintRequest,
    guard_outpaint_fill_plan,
    outpaint_contract,
    outpaint_contract_for_model,
    prepare_outpaint,
    resolve_outpaint_fill_plan,
)
from mflux.task_inference import get_model_capabilities
from mflux.utils.box_values import BoxValues
from mflux.utils.outpaint_util import OutpaintUtil

FLUX2_MODEL = "AbstractFramework/flux.2-klein-base-4b-8bit"
QWEN_MODEL = "AbstractFramework/qwen-image-edit-2511-8bit"

GREEN_BORDER_LORA_REQUEST = "fal/flux-2-klein-4B-outpaint-lora"
# What LoraResolution.resolve() turns the repo request into: a Hugging Face snapshot file. The
# repo identity survives only as the "models--org--repo" cache directory name.
GREEN_BORDER_LORA_RESOLVED = (
    "/Users/x/.cache/mflux/lora/models--fal--flux-2-klein-4B-outpaint-lora/snapshots/"
    "abc123/LyNiaZ53Tudg0J6sT8Xbx_pytorch_lora_weights_comfy_converted.safetensors"
)
GREEN_BORDER_LORA_RESOLVED_LEGACY_BASENAME = "/Users/x/.cache/mflux/lora/flux-outpaint-lora.safetensors"

# Every model whose capability rows the outpaint runtime table has to cover. One row per family
# and per outpaint-capable variant; a family added without an _OUTPAINT_RUNTIME entry fails
# test_every_outpaint_capable_route_has_a_runtime_contract below.
CAPABILITY_SURVEY_MODELS = (
    "flux2-klein-4b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    FLUX2_MODEL,
    "qwen-image",
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
    QWEN_MODEL,
    "z-image-turbo",
    "bonsai",
    "fibo",
    "ernie-image-turbo",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
)


@pytest.fixture(scope="module")
def flux2():
    return outpaint_contract_for_model(model=FLUX2_MODEL)


@pytest.fixture(scope="module")
def qwen():
    return outpaint_contract_for_model(model=QWEN_MODEL)


def _plan(contract, *, source_size: tuple[int, int], padding: str, **overrides):
    source = Image.new("RGB", source_size, (40, 90, 160))
    box = BoxValues.parse(padding).normalize_to_dimensions(width=source.width, height=source.height)
    request = OutpaintRequest(padding=padding, **overrides)
    return resolve_outpaint_fill_plan(contract=contract, request=request, source=source, padding=box)


@pytest.mark.fast
def test_auto_selects_edge_fill_for_small_padding(flux2):
    plan = _plan(flux2, source_size=(432, 240), padding="5%,10%,5%,10%")

    assert plan.mode == "edge"
    assert plan.requested == "auto"
    assert plan.uses_solid_fill_adapter is False
    assert plan.max_side == "right"
    assert plan.max_side_ratio == pytest.approx(0.1, abs=0.01)
    assert plan.edge_fill_within_reach is True
    assert "edge-fill reach" in plan.reason


@pytest.mark.fast
def test_auto_selects_blank_fill_for_large_padding_without_lora(flux2):
    # A 768x766 portrait grown by a full source height on the bottom edge runs far past what a
    # stretched border strip can carry, so `auto` must not choose edge fill there.
    plan = _plan(flux2, source_size=(768, 766), padding="0%,10%,100%,10%")

    # "neutral" is OutpaintUtil's flat per-side border-color canvas: blank, so there is nothing
    # for the model to continue, but without a hard chroma step at the source boundary.
    assert plan.mode == "neutral"
    assert plan.fill_color is None
    assert plan.max_side == "bottom"
    assert plan.max_side_ratio == pytest.approx(1.0, abs=0.01)
    assert plan.edge_fill_within_reach is False
    assert "blank canvas" in plan.reason


@pytest.mark.fast
def test_auto_threshold_boundary_is_inclusive(flux2):
    # `auto` switches at exactly the depth edge fill covers, so the boundary is read from
    # OutpaintUtil rather than restated here.
    reach = OutpaintUtil.edge_fill_reach(400)
    at_limit = _plan(flux2, source_size=(400, 400), padding=f"0,{reach},0,0")
    over_limit = _plan(flux2, source_size=(400, 400), padding=f"0,{reach + 1},0,0")

    assert at_limit.mode == "edge"
    assert over_limit.mode == "neutral"


@pytest.mark.fast
def test_auto_keeps_the_published_validation_profile_on_edge_fill(flux2):
    # The published profile runs 80% single-side padding on a 432x240 source at a 10.9x stretch
    # and is validated with edge fill. Selecting by stretch rather than by padding ratio is what
    # keeps that command on the fill mode its recorded artifacts were produced with.
    plan = _plan(flux2, source_size=(432, 240), padding="5%,80%,5%,60%")

    assert plan.mode == "edge"
    assert plan.edge_fill_within_reach is True


@pytest.mark.fast
@pytest.mark.parametrize(
    ("lora_field", "lora_value"),
    [
        ("requested_lora_paths", (GREEN_BORDER_LORA_REQUEST,)),
        ("requested_lora_paths", (f"{GREEN_BORDER_LORA_REQUEST}:pytorch_lora_weights_comfy_converted.safetensors",)),
        ("lora_paths", (GREEN_BORDER_LORA_RESOLVED,)),
        ("lora_paths", (GREEN_BORDER_LORA_RESOLVED_LEGACY_BASENAME,)),
    ],
)
@pytest.mark.parametrize("padding", ["5%,10%,5%,10%", "0%,10%,100%,10%"])
def test_green_border_lora_selects_green_in_requested_and_resolved_forms(flux2, lora_field, lora_value, padding):
    plan = _plan(flux2, source_size=(768, 766), padding=padding, **{lora_field: lora_value})

    assert plan.uses_solid_fill_adapter is True
    assert plan.mode == "solid"
    assert plan.fill_color == FLUX2_GREEN_BORDER_FILL_COLOR == (0, 255, 0)
    assert "green" in plan.reason


@pytest.mark.fast
def test_unrelated_lora_does_not_select_green(flux2):
    plan = _plan(
        flux2,
        source_size=(432, 240),
        padding="5%,10%,5%,10%",
        requested_lora_paths=("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",),
        lora_paths=("/Users/x/.cache/mflux/lora/multi_angles.safetensors",),
    )

    assert plan.uses_solid_fill_adapter is False
    assert plan.mode == "edge"


@pytest.mark.fast
@pytest.mark.parametrize("padding", ["5%,10%,5%,10%", "0%,10%,100%,10%"])
@pytest.mark.parametrize("requested", ["edge", "neutral", "solid", "blur"])
@pytest.mark.parametrize("with_green_lora", [False, True])
def test_explicit_fill_overrides_auto_in_every_case(flux2, padding, requested, with_green_lora):
    lora = {"requested_lora_paths": (GREEN_BORDER_LORA_REQUEST,)} if with_green_lora else {}
    plan = _plan(flux2, source_size=(768, 766), padding=padding, fill=requested, **lora)

    assert plan.mode == requested
    assert plan.requested == requested
    assert plan.is_explicit is True
    assert plan.reason == f"--outpaint-fill {requested} was passed explicitly."


@pytest.mark.fast
def test_explicit_fill_color_overrides_the_default_solid_color(flux2):
    plan = _plan(
        flux2,
        source_size=(768, 766),
        padding="0%,10%,100%,10%",
        fill="solid",
        fill_color=(12, 34, 56),
        requested_lora_paths=(GREEN_BORDER_LORA_REQUEST,),
    )

    assert plan.fill_color == (12, 34, 56)


@pytest.mark.fast
def test_explicit_solid_without_a_color_falls_back_to_the_source_mean_border_color(flux2):
    plan = _plan(flux2, source_size=(768, 766), padding="0%,10%,100%,10%", fill="solid")

    assert plan.mode == "solid"
    assert plan.fill_color == (40, 90, 160)


@pytest.mark.fast
def test_explicit_solid_with_the_green_lora_defaults_to_green(flux2):
    plan = _plan(
        flux2,
        source_size=(768, 766),
        padding="0%,10%,100%,10%",
        fill="solid",
        requested_lora_paths=(GREEN_BORDER_LORA_REQUEST,),
    )

    assert plan.fill_color == FLUX2_GREEN_BORDER_FILL_COLOR


@pytest.mark.fast
def test_non_solid_fills_carry_no_fill_color(flux2):
    for mode in ("edge", "neutral", "blur"):
        plan = _plan(flux2, source_size=(768, 766), padding="0%,10%,100%,10%", fill=mode)
        assert plan.fill_color is None, mode


@pytest.mark.fast
def test_guard_warns_loudly_but_proceeds_for_explicit_unsafe_edge_fill(flux2):
    plan = _plan(flux2, source_size=(768, 766), padding="0%,10%,100%,10%", fill="edge")

    warnings = guard_outpaint_fill_plan(contract=flux2, fill_plan=plan)

    assert len(warnings) == 1
    warning = warnings[0]
    assert "--outpaint-fill edge" in warning
    assert "bottom padding of 766px (100% of the source height)" in warning
    assert "--outpaint-fill neutral" in warning
    assert "fal/flux-2-klein-4B-outpaint-lora" in warning


@pytest.mark.fast
def test_guard_fails_closed_when_auto_reaches_unsafe_edge_fill(flux2):
    plan = _plan(flux2, source_size=(768, 766), padding="0%,10%,100%,10%")
    unreachable = OutpaintFillPlan(
        requested="auto",
        mode="edge",
        fill_color=(255, 255, 255),
        reason=plan.reason,
        max_side=plan.max_side,
        max_side_padding_px=plan.max_side_padding_px,
        max_side_ratio=plan.max_side_ratio,
        max_side_reach_px=plan.max_side_reach_px,
        max_side_overreach=plan.max_side_overreach,
        uses_solid_fill_adapter=False,
    )

    with pytest.raises(ValueError) as error:
        guard_outpaint_fill_plan(contract=flux2, fill_plan=unreachable)

    message = str(error.value)
    assert "--outpaint-fill auto resolved to edge" in message
    assert "--outpaint-fill neutral" in message


@pytest.mark.fast
def test_guard_is_silent_inside_the_edge_fill_limit(flux2):
    plan = _plan(flux2, source_size=(432, 240), padding="5%,10%,5%,10%")

    assert guard_outpaint_fill_plan(contract=flux2, fill_plan=plan) == ()


@pytest.mark.fast
def test_every_outpaint_capable_route_has_a_runtime_contract():
    # An outpaint-capable capability row with no _OUTPAINT_RUNTIME entry is what would let a new
    # model family silently inherit FLUX.2's latent-lock semantics. outpaint_contract() raises on
    # that pairing, so this sweep is the alarm.
    covered = set()
    for model in CAPABILITY_SURVEY_MODELS:
        for capability in get_model_capabilities(model=model).capabilities:
            if not capability.supports_outpaint:
                continue
            contract = outpaint_contract(capability=capability)
            assert contract.capability_id == capability.id
            assert contract.preservation == capability.outpaint_preservation
            assert contract.default_fill_mode in contract.fill_modes
            covered.add(capability.id)

    assert covered == {"flux2.outpaint", "qwen.outpaint"}


@pytest.mark.fast
def test_outpaint_contract_rejects_a_route_without_a_runtime_entry(flux2):
    from dataclasses import replace

    capability = next(
        capability
        for capability in get_model_capabilities(model=FLUX2_MODEL).capabilities
        if capability.id == "flux2.outpaint"
    )
    unlisted = replace(capability, id="newfamily.outpaint")

    with pytest.raises(OutpaintError) as error:
        outpaint_contract(capability=unlisted)

    assert "no runtime contract" in str(error.value)
    assert "_OUTPAINT_RUNTIME" in str(error.value)


@pytest.mark.fast
def test_outpaint_contract_rejects_a_route_that_does_not_support_outpaint():
    capability = next(
        capability
        for capability in get_model_capabilities(model=QWEN_MODEL).capabilities
        if capability.id == "qwen.edit"
    )

    with pytest.raises(OutpaintError) as error:
        outpaint_contract(capability=capability)

    assert "does not support outpaint" in str(error.value)


@pytest.mark.fast
@pytest.mark.parametrize("fill", [None, "auto", "edge"])
def test_qwen_resolves_every_accepted_request_to_its_fixed_edge_canvas(qwen, fill):
    plan = _plan(qwen, source_size=(432, 240), padding="5%,80%,5%,60%", fill=fill)

    assert plan.mode == "edge"
    assert plan.fill_color is None


@pytest.mark.fast
@pytest.mark.parametrize("fill", ["neutral", "solid", "blur"])
def test_qwen_rejects_a_fill_mode_it_does_not_publish(qwen, fill):
    with pytest.raises(OutpaintError) as error:
        _plan(qwen, source_size=(432, 240), padding="5%,80%,5%,60%", fill=fill)

    message = str(error.value)
    assert "qwen.outpaint" in message
    assert "fixed 'edge' conditioning canvas" in message
    assert repr(fill) in message


@pytest.mark.fast
def test_fixed_fill_route_warning_does_not_offer_an_option_it_has_not_got(qwen):
    plan = _plan(qwen, source_size=(768, 766), padding="0%,10%,100%,10%")

    warnings = guard_outpaint_fill_plan(contract=qwen, fill_plan=plan)

    assert len(warnings) == 1
    assert "qwen.outpaint has a fixed edge conditioning canvas" in warnings[0]
    assert "Use --outpaint-fill neutral" not in warnings[0]


@pytest.mark.fast
@pytest.mark.parametrize(
    "model",
    ["flux2-klein-4b", "AbstractFramework/flux.2-klein-4b-8bit", "AbstractFramework/flux.2-klein-9b-8bit"],
)
def test_distilled_klein_resolves_the_same_outpaint_contract_as_base_klein(model, flux2):
    # Distilled Klein runs the same latent-locked route; only the recommended adapter differs,
    # because the green-canvas A/B proof is a base row and there is no distilled measurement.
    contract = outpaint_contract_for_model(model=model)

    assert contract.capability_id == "flux2.outpaint"
    assert contract.preservation == flux2.preservation
    assert contract.preserve_source is False
    assert contract.restore_threshold == -1.0
    assert contract.conditioning == flux2.conditioning
    assert contract.fill_modes == flux2.fill_modes
    assert contract.recommended_lora is None


@pytest.mark.fast
def test_route_contracts_publish_the_two_preservation_strategies(flux2, qwen):
    assert flux2.preservation == "latent-locked-transition-band-no-postblend"
    assert flux2.preserve_source is False
    assert qwen.preservation == "adaptive-content-aware-source-blend"
    assert qwen.preserve_source is None
    assert qwen.restore_threshold == 12.0


@pytest.mark.fast
def test_prepare_outpaint_owns_and_cleans_up_its_workspace(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (80, 20, 10)).save(source)

    with prepare_outpaint(source_image=source, padding="2,4,2,4", model=FLUX2_MODEL) as session:
        canvas_path = session.canvas.canvas_path
        assert canvas_path.exists()
        assert (session.width, session.height) == (32, 16)
        assert session.canvas_policy == "exact-resize"

    assert not canvas_path.exists()


@pytest.mark.fast
def test_prepare_outpaint_requires_exactly_one_route_identifier(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (80, 20, 10)).save(source)

    with pytest.raises(OutpaintError) as error:
        prepare_outpaint(source_image=source, padding="2,4,2,4")

    assert "exactly one" in str(error.value)


class _FakeArtifact:
    def __init__(self):
        self.image = Image.new("RGB", (32, 16), (10, 20, 30))
        self.image_path = None
        self.image_paths = None
        self.extra_metadata: dict = {}
        self.saw_metadata_at_save: dict | None = None

    def save(self, path, overwrite=True, **kwargs):
        self.saw_metadata_at_save = dict(self.extra_metadata)
        Path(path).touch()
        return Path(path)


@pytest.mark.fast
def test_post_process_hook_runs_before_save(tmp_path):
    from mflux.python_runtime import LoadedGenerationModel
    from mflux.task_inference import GenerationPlan

    artifact = _FakeArtifact()
    order: list[str] = []

    class FakeModel:
        def generate_image(self, **kwargs):
            order.append("generate")
            return artifact

    loaded = LoadedGenerationModel(
        plan=GenerationPlan(
            public_task="image-to-image",
            mode="edit-reference",
            capability_id="flux2.outpaint",
            family="flux2",
            handler_id="flux2.edit",
            image_count=1,
        ),
        model_config=None,
        runtime_id="flux2.klein-outpaint",
        cache_key_base="base",
        cache_key="key",
        model=FakeModel(),
    )

    def post_process(generated):
        order.append("post_process")
        generated.extra_metadata["outpaint_fill"] = "edge"

    results = loaded.generate_outputs(
        seeds=[7],
        output=str(tmp_path / "out.png"),
        post_process=post_process,
        prompt="extend the scene",
    )

    assert order == ["generate", "post_process"]
    assert results[0].artifact is artifact
    # The hook's mutation has to be part of the artifact that reaches save(), not applied after.
    assert artifact.saw_metadata_at_save == {"outpaint_fill": "edge"}
