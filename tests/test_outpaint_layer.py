"""Policy tests for the shared outpaint layer.

These run against `mflux.outpaint` directly - no argparse, no model weights - so the
conditioning-canvas contract is covered once for every route instead of once per backend CLI.
The end-to-end proof that the FLUX.2 command still produces the same canvas, notice and
metadata lives in `tests/cli/test_flux2_outpaint_fill.py`.
"""

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from mflux.outpaint import (
    FLUX2_GREEN_BORDER_FILL_COLOR,
    OUTPAINT_TWO_DEEP_AXIS_RATIO,
    OutpaintError,
    OutpaintFillPlan,
    OutpaintRequest,
    guard_outpaint_fill_plan,
    outpaint_contract,
    outpaint_contract_for_model,
    prepare_outpaint,
    resolve_outpaint_fill_plan,
    resolve_outpaint_pass_plan,
    run_outpaint,
)
from mflux.task_inference import OUTPAINT_PASS_MODES, get_model_capabilities
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

# The geometry measured to duplicate the subject: a 432x240 source grown 148px down and 256px
# left, so a deep vertical side and a deep horizontal side open a free corner. Single-axis runs of
# the same depth on the same source, canvas and seeds are clean, which is what makes the corner -
# and not the depth, the canvas size or the fill - the trigger.
TWO_DEEP_AXIS_PADDING = "0,0,148,256"
TWO_DEEP_AXIS_SOURCE = (432, 240)
# The published axis-coverage bundle: every single side, both sides of each axis, all four sides,
# and an asymmetric four-side request, over three source aspect ratios. All 24 runs are good
# output, so none of them may warn.
AXIS_COVERAGE_PADDINGS = (
    "0,0,25%,0",
    "25%,0,0,0",
    "0,25%,0,0",
    "0,0,0,25%",
    "20%,0,20%,0",
    "0,20%,0,20%",
    "15%,15%,15%,15%",
    "10%,30%,20%,5%",
)
AXIS_COVERAGE_SOURCES = ((640, 448), (512, 512), (448, 640))
# The recorded validation envelope, on the source it was recorded against.
VALIDATED_PADDING = "5%,80%,5%,60%"


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


def _pass_plan(contract, *, source_size: tuple[int, int], padding: str, passes: str | None = None, **overrides):
    fill_plan = _plan(contract, source_size=source_size, padding=padding, passes=passes, **overrides)
    request = OutpaintRequest(padding=padding, passes=passes, **overrides)
    return resolve_outpaint_pass_plan(contract=contract, request=request, fill_plan=fill_plan)


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
    assert "edge-fill reach" in plan.reason


@pytest.mark.fast
def test_the_blank_canvas_reason_follows_how_far_the_route_fill_reaches(flux2, qwen):
    # What a blank canvas buys is route-dependent. The FLUX.2 route starts from pure noise and
    # drops the reference tokens of cells holding only fill, so the canvas conditions the seam ring
    # and nothing else; the Qwen route keeps every canvas token, so there the blank canvas is what
    # makes the model generate rather than continue the border.
    latent_locked = _plan(flux2, source_size=(768, 766), padding="0%,10%,100%,10%")
    blended = _plan(
        qwen_like := replace(flux2, fill_reaches_padding=True), source_size=(768, 766), padding="0%,10%,100%,10%"
    )

    assert latent_locked.mode == blended.mode == "neutral"
    assert flux2.fill_reaches_padding is False and qwen.fill_reaches_padding is True
    assert qwen_like.fill_reaches_padding is True
    assert latent_locked.reason.endswith(
        "the fill conditions the seam around the source rather than the padded region, and a blank "
        "seam does not hand the model a stretched border strip to continue."
    )
    assert "generate new subject matter" not in latent_locked.reason

    assert qwen.preservation == "adaptive-content-aware-source-blend"
    assert blended.reason.endswith(
        "a blank canvas makes the model generate new subject matter instead of smearing the source border."
    )


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
def test_guard_warns_about_subject_duplication_for_a_single_pass_over_a_deep_corner(flux2):
    plan = _plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING)

    # With no pass plan the guard reads the geometry alone; with an explicit single pass it
    # names the option that forced it.
    warnings = guard_outpaint_fill_plan(contract=flux2, fill_plan=plan)
    single = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING, passes="1")
    explicit = guard_outpaint_fill_plan(contract=flux2, fill_plan=plan, pass_plan=single)

    assert (plan.canvas_width, plan.canvas_height) == (688, 400)
    assert len(warnings) == len(explicit) == 1
    for warning in (warnings[0], explicit[0]):
        assert "measured to duplicate the subject" in warning
        assert "bottom 148px (62% of the source height)" in warning
        assert "left 256px (59% of the source width)" in warning
        # The true corner, after the dimension round-up, not the requested 148.
        assert "256x160px corner of the 688x400 canvas" in warning
        # The remedies named must be the ones measured on this geometry: the two-pass split and
        # prompt content. Step count is measured NOT to help and must never be offered.
        assert "two single-axis passes" in warning
        assert "Describing the area being added rather than the subject" in warning
        assert "Raising --steps does not move it" in warning
        # The fill is not one of them on a latent-locked route, and the warning says so rather
        # than sending the caller at a knob that cannot reach the free corner.
        assert "--outpaint-fill" not in warning
        assert "The conditioning canvas is not the lever on this route" in warning
    assert "--outpaint-passes auto takes that route" in warnings[0]
    assert "because --outpaint-passes 1 was passed" in explicit[0]


@pytest.mark.fast
def test_guard_is_silent_when_the_deep_corner_is_split_into_passes(flux2):
    plan = _plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING)
    split = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING)

    assert split.is_split
    assert guard_outpaint_fill_plan(contract=flux2, fill_plan=plan, pass_plan=split) == ()


@pytest.mark.fast
def test_guard_does_not_claim_duplication_on_a_route_that_publishes_no_split_depth(flux2):
    unmeasured = replace(flux2, auto_split_corner_ratio=None)
    plan = _plan(unmeasured, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING)

    assert plan.expands_two_deep_axes is True
    assert plan.opens_deep_corner(contract=unmeasured) is False
    assert guard_outpaint_fill_plan(contract=unmeasured, fill_plan=plan) == ()
    assert _pass_plan(unmeasured, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING).count == 1


@pytest.mark.fast
def test_two_deep_axis_warning_reports_the_published_canvas_envelope(flux2):
    # The measured geometry sits inside the envelope, so the warning must not claim otherwise;
    # a two-deep-axis request that does run past it says so, reading the bound off the route.
    inside = _plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING)
    outside = _plan(flux2, source_size=(640, 448), padding="50%,0,0,50%")

    assert inside.canvas_pixels <= flux2.validated_max_canvas_pixels < outside.canvas_pixels
    assert "validated maximum" not in guard_outpaint_fill_plan(contract=flux2, fill_plan=inside)[0]
    assert (
        f"past the {flux2.validated_max_canvas_pixels} px flux2.outpaint publishes as its validated maximum"
        in guard_outpaint_fill_plan(contract=flux2, fill_plan=outside)[0]
    )


@pytest.mark.fast
@pytest.mark.parametrize("source_size", AXIS_COVERAGE_SOURCES)
@pytest.mark.parametrize("padding", AXIS_COVERAGE_PADDINGS)
def test_guard_is_silent_for_every_published_axis_coverage_run(flux2, source_size, padding):
    plan = _plan(flux2, source_size=source_size, padding=padding)

    assert plan.expands_two_deep_axes is False
    assert guard_outpaint_fill_plan(contract=flux2, fill_plan=plan) == ()


@pytest.mark.fast
def test_guard_is_silent_for_the_recorded_validation_envelope(flux2):
    plan = _plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=VALIDATED_PADDING)

    assert plan.expands_two_deep_axes is False
    assert guard_outpaint_fill_plan(contract=flux2, fill_plan=plan) == ()


@pytest.mark.fast
def test_two_deep_axis_threshold_separates_the_measured_runs(flux2):
    # Requests rank by the shallower of their two axis depths - the free corner is only as square
    # as its smaller side, and a request that touches one axis opens no corner at all. The
    # threshold has to sit above every clean run and below the duplicating one.
    clean = {
        (source_size, padding): _plan(flux2, source_size=source_size, padding=padding).free_corner_ratio
        for source_size in AXIS_COVERAGE_SOURCES
        for padding in AXIS_COVERAGE_PADDINGS
    }
    clean[(TWO_DEEP_AXIS_SOURCE, VALIDATED_PADDING)] = _plan(
        flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=VALIDATED_PADDING
    ).free_corner_ratio
    duplicating = _plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING).free_corner_ratio

    assert max(clean.values()) < OUTPAINT_TWO_DEEP_AXIS_RATIO < duplicating
    # The deepest clean run is the asymmetric four-side request, at 20%; the measured failure is at
    # 59%. 30% sits 1.5x above the first and 2.0x below the second.
    assert max(clean.values()) == pytest.approx(0.200, abs=0.005)
    assert duplicating == pytest.approx(0.593, abs=0.005)
    # Only the four-side paddings put any depth on both axes at all.
    assert {padding for (_, padding), ratio in clean.items() if ratio > 0.0} == {
        "15%,15%,15%,15%",
        "10%,30%,20%,5%",
        VALIDATED_PADDING,
    }


@pytest.mark.fast
def test_single_axis_expansion_never_reads_as_a_free_corner(flux2):
    # Same source, same canvas area, same depth as the duplicating request, one axis at a time.
    for padding in ("0,0,0,256", "0,0,148,0"):
        plan = _plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=padding)
        assert plan.free_corner_ratio == 0.0, padding
        assert guard_outpaint_fill_plan(contract=flux2, fill_plan=plan) == (), padding


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
    assert contract.preserve_source is None
    assert contract.restore_threshold == 24.0
    assert contract.conditioning == flux2.conditioning
    assert contract.fill_modes == flux2.fill_modes
    assert contract.recommended_lora is None


@pytest.mark.fast
def test_route_contracts_publish_one_preservation_strategy(flux2, qwen):
    # Both routes hold the source in latent space and then paste the original back adaptively;
    # they differ only in how the lock is applied, which is unpublished mechanics.
    assert flux2.preservation == qwen.preservation == "adaptive-content-aware-source-blend"
    assert flux2.preserve_source is None and qwen.preserve_source is None
    # 1.5x the largest drift a locked window measured (15.6), under half a recomposed one (62).
    assert flux2.restore_threshold == qwen.restore_threshold == 24.0
    assert flux2.source_lock_mask is False and qwen.source_lock_mask is True


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


# --- pass planning ---------------------------------------------------------------------------


@pytest.mark.fast
def test_auto_splits_the_duplicating_geometry_into_two_single_axis_passes(flux2):
    plan = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING)

    assert plan.requested == "auto"
    assert plan.count == 2
    # The deeper axis runs first: bottom is 62% of the height, left 59% of the width. The first
    # pass leaves the width alone; the second adds the full 256 px and lands on the one-pass
    # canvas exactly.
    assert plan.padding_values == ("0,0,148,0", "0,0,0,256")
    assert (plan.canvas_width, plan.canvas_height) == (688, 400)
    assert (plan.paste_left, plan.paste_top) == (256, 0)
    assert "past the 30% auto-split depth" in plan.reason
    assert "second copy of the subject" in plan.reason


@pytest.mark.fast
@pytest.mark.parametrize("source_size", AXIS_COVERAGE_SOURCES)
@pytest.mark.parametrize("padding", AXIS_COVERAGE_PADDINGS)
def test_auto_keeps_every_published_axis_coverage_run_on_one_pass(flux2, source_size, padding):
    plan = _pass_plan(flux2, source_size=source_size, padding=padding)

    assert plan.count == 1
    assert plan.padding_values == (
        _format(BoxValues.parse(padding).normalize_to_dimensions(width=source_size[0], height=source_size[1])),
    )


@pytest.mark.fast
def test_auto_keeps_the_recorded_validation_envelope_on_one_pass(flux2, qwen):
    for contract in (flux2, qwen):
        plan = _pass_plan(contract, source_size=TWO_DEEP_AXIS_SOURCE, padding=VALIDATED_PADDING)
        assert plan.count == 1
        assert (plan.canvas_width, plan.canvas_height) == (1040, 272)
        assert (plan.paste_left, plan.paste_top) == (259, 12)
        assert "within the 30% auto-split depth" in plan.reason


@pytest.mark.fast
def test_auto_split_is_a_strict_threshold_on_the_shallower_axis(flux2):
    # 30% on both axes of the 432x240 source is measured clean on every seed and stays one pass;
    # 40% is the first sweep row that grew a second hull and splits.
    at_threshold = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding="30%")
    past_threshold = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding="40%")

    assert at_threshold.count == 1
    assert past_threshold.count == 2
    assert past_threshold.padding_values == ("96,0,96,0", "0,180,0,172")
    assert (past_threshold.canvas_width, past_threshold.canvas_height) == (784, 432)
    assert (past_threshold.paste_left, past_threshold.paste_top) == (172, 96)


@pytest.mark.fast
def test_explicit_single_pass_runs_the_deep_corner_as_one_canvas(flux2):
    plan = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING, passes="1")

    assert plan.count == 1
    assert plan.padding_values == ("0,0,148,256",)
    assert plan.reason == "--outpaint-passes 1 was passed explicitly."


@pytest.mark.fast
def test_explicit_two_passes_split_a_shallow_two_axis_request(flux2):
    plan = _pass_plan(flux2, source_size=(512, 512), padding="15%,15%,15%,15%", passes="2")
    fill_plan = _plan(flux2, source_size=(512, 512), padding="15%,15%,15%,15%")

    assert plan.count == 2
    # A tie in depth runs horizontal first.
    assert plan.padding_values == ("0,76,0,76", "76,0,84,0")
    assert (plan.canvas_width, plan.canvas_height) == (fill_plan.canvas_width, fill_plan.canvas_height)
    assert guard_outpaint_fill_plan(contract=flux2, fill_plan=fill_plan, pass_plan=plan) == ()


@pytest.mark.fast
def test_explicit_two_passes_reject_a_single_axis_request(flux2):
    with pytest.raises(OutpaintError) as error:
        _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding="0,0,0,256", passes="2")

    assert "needs padding on both axes" in str(error.value)
    assert "pads one axis only" in str(error.value)


@pytest.mark.fast
@pytest.mark.parametrize("passes", ["3", "two", "0"])
def test_an_unknown_pass_count_is_rejected(flux2, passes):
    with pytest.raises(OutpaintError) as error:
        _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding=TWO_DEEP_AXIS_PADDING, passes=passes)

    assert "--outpaint-passes must be one of auto, 1, 2" in str(error.value)


@pytest.mark.fast
def test_split_second_pass_lands_on_the_one_pass_canvas_when_the_source_is_not_a_multiple_of_16(flux2):
    # A 300x250 source grown 300 left and 100 up. The one-pass canvas is 608x352 with the source at
    # (300, 100). The horizontal pass runs first and rounds its height up to 256, six rows of which
    # are generated then; the vertical pass has 96 rows left to add, so the source lands four rows
    # higher than requested - inside one dimension multiple - and the canvas is exactly 608x352.
    fill_plan = _plan(flux2, source_size=(300, 250), padding="100,0,0,300")
    plan = _pass_plan(flux2, source_size=(300, 250), padding="100,0,0,300")

    assert (fill_plan.canvas_width, fill_plan.canvas_height) == (608, 352)
    assert plan.count == 2
    assert plan.padding_values == ("0,0,0,300", "96,0,0,0")
    assert (plan.canvas_width, plan.canvas_height) == (608, 352)
    assert (plan.paste_left, plan.paste_top) == (300, 96)
    first_width, first_height = OutpaintUtil.expanded_canvas_size(
        source_width=300, source_height=250, padding=plan.paddings[0]
    )
    assert (first_width, first_height) == (608, 256)
    assert OutpaintUtil.expanded_canvas_size(
        source_width=first_width, source_height=first_height, padding=plan.paddings[1]
    ) == (608, 352)


@pytest.mark.fast
def test_split_second_pass_absorbs_the_round_up_sliver_on_its_trailing_side(flux2):
    # 432x240 grown 148 down and 256 left: the one-pass canvas is 400 tall, so the vertical pass
    # (which runs first here, being the deeper axis) adds the requested 148 plus the 12 px sliver
    # the one-pass canvas would also have generated.
    plan = _pass_plan(flux2, source_size=TWO_DEEP_AXIS_SOURCE, padding="0,0,148,256")
    first = OutpaintUtil.expanded_canvas_size(source_width=432, source_height=240, padding=plan.paddings[0])

    assert first == (432, 400)
    assert plan.padding_values[1] == "0,0,0,256"


@pytest.mark.fast
def test_split_pass_fills_resolve_against_each_pass_source(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", TWO_DEEP_AXIS_SOURCE, (80, 20, 10)).save(source)

    with prepare_outpaint(source_image=source, padding=TWO_DEEP_AXIS_PADDING, model=FLUX2_MODEL) as session:
        assert session.passes == 2
        first, second = session.pass_fill_plans
        assert (first.source_width, first.source_height) == (432, 240)
        assert (first.canvas_width, first.canvas_height) == (432, 400)
        # The second pass is planned against the first pass's canvas, which is what it runs on.
        assert (second.source_width, second.source_height) == (432, 400)
        assert (second.canvas_width, second.canvas_height) == (688, 400)
        assert session.fill_plan is first
        # The session reports the final geometry, not the first pass's.
        assert (session.width, session.height) == (688, 400)
        assert (session.canvas.target_width, session.canvas.target_height) == (432, 400)
        assert (session.geometry.paste_left, session.geometry.paste_top) == (256, 0)
        assert session.warnings == ()
        assert session.notice.startswith("Outpaint: 2 passes because")
        assert "Pass 1 pads 0,0,148,0 to 432x400; pass 2 pads 0,0,0,256 to 688x400." in session.notice
        assert "Outpaint: pass 2 fill=edge, canvas 688x400 from source 432x400" in session.notice


@pytest.mark.fast
def test_single_pass_session_notice_and_metadata_are_unchanged(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (12, 8), (80, 20, 10)).save(source)

    with prepare_outpaint(source_image=source, padding="2,4,2,4", model=FLUX2_MODEL) as session:
        assert session.passes == 1
        assert session.pass_fill_plans == (session.fill_plan,)
        assert session.geometry == replace(session.canvas, source_path=Path(source))
        assert session.notice.startswith("Outpaint: fill=")
        assert "passes" not in session.notice


class _FakeGenerated:
    def __init__(self, image: Image.Image):
        self.image = image
        self.image_path = None
        self.image_paths = None
        self.extra_metadata: dict = {}


class _FakeCanvasModel:
    """A flux2.outpaint stand-in: returns the conditioning canvas with the padded area repainted."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        canvas = kwargs["canvas"]
        image = Image.open(canvas.canvas_path).convert("RGB")
        painted = Image.new("RGB", image.size, (0, 200, 0))
        painted.paste(
            image.crop(
                (
                    canvas.paste_left,
                    canvas.paste_top,
                    canvas.paste_left + canvas.source_width,
                    canvas.paste_top + canvas.source_height,
                )
            ),
            (canvas.paste_left, canvas.paste_top),
        )
        return _FakeGenerated(painted)


class _FakePathsModel:
    """A qwen.outpaint stand-in: returns the conditioning image it was handed."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeGenerated(Image.open(kwargs["image_paths"][0]).convert("RGB"))


@pytest.mark.fast
def test_session_runs_every_pass_and_finalizes_against_the_original_source(tmp_path):
    # 64x32 grown 24 down (75%) and 40 right (62%): vertical first to 64x64, then 48 right (the
    # requested 40 plus the 8 px round-up) to 112x64.
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (200, 30, 30)).save(source)
    model = _FakeCanvasModel()

    with prepare_outpaint(source_image=source, padding="0,40,24,0", model=FLUX2_MODEL) as session:
        generated = session.generate(model, seed=7, prompt="extend")
        workspace = session.canvas.canvas_path.parent
        assert session.pass_plan.padding_values == ("0,0,24,0", "0,48,0,0")
        assert [call["canvas"].canvas_path.name for call in model.calls] == [
            "outpaint_canvas.png",
            "outpaint_pass2_canvas.png",
        ]
        # The FLUX.2 route locks the source itself; the shared layer writes it no mask.
        assert all(call["canvas"].lock_mask_path is None for call in model.calls)
        assert "mask_path" not in model.calls[0]
        assert model.calls[0]["seed"] == model.calls[1]["seed"] == 7
        second = model.calls[1]["canvas"]
        assert second.source_path == workspace / "outpaint_pass2_source.png"
        assert (second.source_width, second.source_height) == (64, 64)
        assert (second.target_width, second.target_height) == (112, 64)
        assert Image.open(second.source_path).size == (64, 64)
        assert session.pass_canvases == (session.canvas, second)

    assert generated.image.size == (112, 64)
    # The original source survives both passes untouched in its final position; the padded area
    # is what the fake painted.
    assert generated.image.getpixel((10, 10)) == (200, 30, 30)
    assert generated.image.getpixel((100, 50)) == (0, 200, 0)
    assert generated.image_path == source
    metadata = generated.extra_metadata
    assert metadata["outpaint_padding"] == "0,40,24,0"
    assert (metadata["outpaint_target_width"], metadata["outpaint_target_height"]) == (112, 64)
    assert (metadata["outpaint_source_paste_left"], metadata["outpaint_source_paste_top"]) == (0, 0)
    assert metadata["outpaint_passes"] == 2
    assert metadata["outpaint_passes_requested"] == "auto"
    assert metadata["outpaint_pass_paddings"] == ["0,0,24,0", "0,48,0,0"]
    assert metadata["outpaint_pass_fills"] == ["edge", "edge"]
    assert "second copy of the subject" in metadata["outpaint_pass_reason"]
    assert metadata["outpaint_fill"] == "edge"
    assert metadata["outpaint_preservation"] == "adaptive-content-aware-source-blend"
    assert metadata["outpaint_source_restore_applied"] is True
    assert metadata["outpaint_source_restore_difference"] == 0.0


@pytest.mark.fast
def test_session_hands_each_pass_its_own_geometry_on_the_conditioning_paths_route(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (200, 30, 30)).save(source)
    model = _FakePathsModel()

    with prepare_outpaint(source_image=source, padding="0,40,24,0", model=QWEN_MODEL) as session:
        generated = session.generate(model, seed=7, prompt="extend", negative_prompt="blurry")
        # The workspace goes with the session, so the masks are read while it is open.
        first_mask = Image.open(model.calls[0]["mask_path"]).convert("L")
        second_mask = Image.open(model.calls[1]["mask_path"]).convert("L")

    first, second = model.calls
    assert (first["width"], first["height"]) == (64, 64)
    assert (second["width"], second["height"]) == (112, 64)
    assert Path(first["image_paths"][0]).name == "outpaint_canvas.png"
    assert Path(second["image_paths"][0]).name == "outpaint_pass2_canvas.png"
    # The route locks the source through its mask input, so every pass carries a mask beside its
    # canvas: black over the source minus the transition band on the sides that gained pixels
    # (bottom on the first pass, right on the second), white over everything to paint.
    assert Path(first["mask_path"]).name == "outpaint_canvas_mask.png"
    assert Path(second["mask_path"]).name == "outpaint_pass2_canvas_mask.png"
    assert first_mask.size == (64, 64)
    assert first_mask.getpixel((0, 0)) == 0 and first_mask.getpixel((63, 7)) == 0
    assert first_mask.getpixel((32, 20)) == 255 and first_mask.getpixel((32, 50)) == 255
    assert second_mask.size == (112, 64)
    assert second_mask.getpixel((0, 0)) == 0 and second_mask.getpixel((39, 63)) == 0
    assert second_mask.getpixel((50, 30)) == 255 and second_mask.getpixel((100, 30)) == 255
    # The original source is the recorded image path on every pass, and every pass gets the
    # caller's keywords unchanged.
    assert first["image_path"] == second["image_path"] == source
    assert first["negative_prompt"] == second["negative_prompt"] == "blurry"
    assert first["canvas_policy"] == second["canvas_policy"] == "exact-resize"
    assert generated.image.size == (112, 64)
    metadata = generated.extra_metadata
    assert metadata["outpaint_passes"] == 2
    assert metadata["outpaint_preservation"] == "adaptive-content-aware-source-blend"
    assert metadata["outpaint_source_restore_applied"] is True
    # Every pass restores against its own source before the original is measured and restored,
    # so a split run keeps the one-pass guarantee instead of summing two passes' drift.
    assert metadata["outpaint_pass_source_restore_differences"] == [0.0, 0.0]
    assert metadata["outpaint_pass_source_restore_applied"] == [True, True]
    assert metadata["outpaint_source_restore_difference"] == 0.0


class _DriftingPathsModel(_FakePathsModel):
    """Returns the canvas with its source window shifted in tone by `delta` on every pass."""

    def __init__(self, delta: int):
        super().__init__()
        self.delta = delta

    def generate_image(self, **kwargs):
        generated = super().generate_image(**kwargs)
        image = generated.image
        box = (0, 0, kwargs["width"], kwargs["height"])
        generated.image = image.point(lambda value: max(0, min(255, value + self.delta))).crop(box)
        return generated


@pytest.mark.fast
def test_split_run_restores_per_pass_so_drift_does_not_add_up(tmp_path):
    # Each pass drifts 18 mean-abs against its own source: under the 24.0 threshold on its own,
    # over it if two passes were judged together against the original. The per-pass settle keeps
    # the restore applying, and the recorded difference stays what the model drew underneath.
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (120, 120, 120)).save(source)
    model = _DriftingPathsModel(delta=18)

    with prepare_outpaint(source_image=source, padding="0,40,24,0", model=QWEN_MODEL) as session:
        generated = session.generate(model, seed=7, prompt="extend")

    metadata = generated.extra_metadata
    assert metadata["outpaint_pass_source_restore_differences"] == pytest.approx([18.0, 18.0], abs=0.5)
    assert metadata["outpaint_pass_source_restore_applied"] == [True, True]
    assert metadata["outpaint_source_restore_applied"] is True
    # Measured before any restore: the second pass drew an 18-step shift on top of the first
    # pass's restored (exact) source, so the original sat 18 away underneath.
    assert metadata["outpaint_source_restore_difference"] == pytest.approx(18.0, abs=0.5)
    assert generated.image.getpixel((10, 10)) == (120, 120, 120)


@pytest.mark.fast
def test_single_pass_run_records_no_per_pass_results(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (120, 120, 120)).save(source)

    with prepare_outpaint(source_image=source, padding="0,40,0,0", model=QWEN_MODEL) as session:
        generated = session.generate(_FakePathsModel(), seed=7, prompt="extend")

    assert generated.extra_metadata["outpaint_passes"] == 1
    assert generated.extra_metadata["outpaint_pass_source_restore_differences"] == []
    assert generated.extra_metadata["outpaint_pass_source_restore_applied"] == []


@pytest.mark.fast
def test_session_rejects_generation_keywords_it_owns(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (200, 30, 30)).save(source)

    with prepare_outpaint(source_image=source, padding="0,40,24,0", model=FLUX2_MODEL) as session:
        with pytest.raises(OutpaintError) as error:
            session.generate(_FakeCanvasModel(), seed=7, prompt="extend", canvas=None)

    assert "Outpaint owns these generation keywords: canvas" in str(error.value)


@pytest.mark.fast
def test_run_outpaint_routes_every_pass_through_the_session(tmp_path):
    from mflux.models.common.config import ModelConfig
    from mflux.python_runtime import LoadedGenerationModel
    from mflux.task_inference import GenerationPlan

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (200, 30, 30)).save(source)
    model = _FakeCanvasModel()
    saved: list[Path] = []

    def save(self, path, overwrite=True, **kwargs):
        saved.append(Path(path))
        Path(path).touch()
        return Path(path)

    _FakeGenerated.save = save  # type: ignore[attr-defined]
    try:
        loaded = LoadedGenerationModel(
            plan=GenerationPlan(
                public_task="image-to-image",
                mode="edit-reference",
                capability_id="flux2.outpaint",
                family="flux2",
                handler_id="flux2.edit",
                image_count=1,
            ),
            model_config=ModelConfig.from_name("flux2-klein-base-4b"),
            runtime_id="flux2.klein-outpaint",
            cache_key_base="base",
            cache_key="key",
            model=model,
        )
        results = run_outpaint(
            loaded=loaded,
            source_image=source,
            padding="0,40,24,0",
            seeds=[3, 5],
            output=str(tmp_path / "out.png"),
            prompt="extend",
        )
    finally:
        del _FakeGenerated.save  # type: ignore[attr-defined]

    assert [call["seed"] for call in model.calls] == [3, 3, 5, 5]
    assert len(results) == 2
    assert len(saved) == 2
    for result in results:
        assert result.artifact.extra_metadata["outpaint_passes"] == 2
        assert result.artifact.image.size == (112, 64)


@pytest.mark.fast
def test_run_outpaint_forwards_an_explicit_pass_count(tmp_path):
    from mflux.models.common.config import ModelConfig
    from mflux.python_runtime import LoadedGenerationModel
    from mflux.task_inference import GenerationPlan

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (200, 30, 30)).save(source)
    model = _FakeCanvasModel()
    loaded = LoadedGenerationModel(
        plan=GenerationPlan(
            public_task="image-to-image",
            mode="edit-reference",
            capability_id="flux2.outpaint",
            family="flux2",
            handler_id="flux2.edit",
            image_count=1,
        ),
        model_config=ModelConfig.from_name("flux2-klein-base-4b"),
        runtime_id="flux2.klein-outpaint",
        cache_key_base="base",
        cache_key="key",
        model=model,
    )

    results = run_outpaint(loaded=loaded, source_image=source, padding="0,40,24,0", seeds=[3], passes=1, prompt="x")

    assert len(model.calls) == 1
    assert results[0].artifact.extra_metadata["outpaint_passes"] == 1
    assert results[0].artifact.extra_metadata["outpaint_passes_requested"] == "1"


@pytest.mark.fast
def test_outpaint_capable_rows_publish_the_pass_contract():
    for model in CAPABILITY_SURVEY_MODELS:
        for capability in get_model_capabilities(model=model).capabilities:
            if capability.supports_outpaint:
                assert capability.outpaint_pass_modes == OUTPAINT_PASS_MODES == ("auto", "1", "2")
                assert capability.outpaint_default_passes == "auto"
                assert capability.outpaint_auto_split_corner_ratio == OUTPAINT_TWO_DEEP_AXIS_RATIO
                assert outpaint_contract(capability=capability).auto_split_corner_ratio == OUTPAINT_TWO_DEEP_AXIS_RATIO
            else:
                assert capability.outpaint_pass_modes == ()
                assert capability.outpaint_default_passes is None
                assert capability.outpaint_auto_split_corner_ratio is None
            payload = capability.to_dict()
            assert payload["outpaint_pass_modes"] == list(capability.outpaint_pass_modes)
            assert payload["outpaint_default_passes"] == capability.outpaint_default_passes
            assert payload["outpaint_auto_split_corner_ratio"] == capability.outpaint_auto_split_corner_ratio


def _format(box) -> str:
    return f"{box.top},{box.right},{box.bottom},{box.left}"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("source_size", "target_size", "paste"),
    [
        ((432, 240), (688, 400), (0, 0)),
        ((432, 240), (688, 400), (256, 0)),
        ((432, 240), (1040, 272), (259, 12)),
        ((768, 766), (928, 1536), (76, 0)),
        ((12, 8), (32, 16), (4, 2)),
        ((300, 250), (608, 256), (300, 0)),
    ],
)
def test_source_lock_box_matches_the_flux2_preserve_box(source_size, target_size, paste):
    # Two routes, one transition-band rule: the Qwen mask the shared layer writes and the FLUX.2
    # latent lock must hold the same canvas box, or the two routes would preserve different
    # amounts of source for the same request.
    from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import _Flux2KleinEditHelpers
    from mflux.utils.outpaint_util import SOURCE_LOCK_TRANSITION_PX, OutpaintCanvas

    canvas = OutpaintCanvas(
        canvas_path=Path("canvas.png"),
        source_path=Path("source.png"),
        source_width=source_size[0],
        source_height=source_size[1],
        target_width=target_size[0],
        target_height=target_size[1],
        paste_left=paste[0],
        paste_top=paste[1],
        padding=BoxValues.parse("0").normalize_to_dimensions(width=source_size[0], height=source_size[1]),
    )

    assert OutpaintUtil.source_lock_box(canvas=canvas) == _Flux2KleinEditHelpers.outpaint_preserve_box(
        canvas=canvas, transition_px=SOURCE_LOCK_TRANSITION_PX
    )


@pytest.mark.fast
def test_route_contracts_declare_who_locks_the_source_through_a_mask(flux2, qwen):
    assert flux2.source_lock_mask is False
    assert qwen.source_lock_mask is True
