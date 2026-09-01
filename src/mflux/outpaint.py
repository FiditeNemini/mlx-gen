"""Model-agnostic outpaint and reframe orchestration.

One implementation of the conditioning-canvas contract, shared by every backend CLI and by
embedding Python applications. Nothing here imports argparse: the CLI adapter that turns a
parsed namespace into an `OutpaintRequest` lives in `mflux.cli.outpaint_cli`.

Route behaviour is declared, never sniffed. Everything a host can observe is read off the
route's published `GenerationCapability` row (fill modes, default mode, whether the fill
option exists at all, the auto stretch bound, the recommended adapter, the preservation
strategy). Only the mechanics the capability schema deliberately does not publish - which
keyword the backend's `generate_image` accepts the canvas through, the base canvas colour
under edge/blur, and the adapter markers that switch `auto` onto a solid canvas - live in
`_OUTPAINT_RUNTIME`, one entry per capability id, next to the code that consumes them. That
table is the only place a sixth model family has to be named.

Scope: this module covers outpaint *by expanded conditioning canvas* - the source is pasted
into a larger canvas, the model denoises the whole canvas, and the route's preservation
strategy decides what happens to the source region afterwards. A route that outpaints by
native masked fill instead (a real mask channel, no expanded canvas) does not fit this
contract and must not be given an `_OUTPAINT_RUNTIME` entry to force it: `outpaint_contract`
raising for an unlisted capability is the intended outcome there, not a gap to paper over.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import PIL.Image
import PIL.ImageStat

from mflux.task_inference import (
    OUTPAINT_FILL_AUTO,
    GenerationCapability,
    get_model_capabilities,
    resolve_generation_plan,
)
from mflux.utils.box_values import AbsoluteBoxValues, BoxValues
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.image_util import ImageUtil
from mflux.utils.outpaint_util import OutpaintCanvas, OutpaintUtil

# How a backend's generate_image() accepts the expanded canvas. Declared per capability id
# rather than inferred, so adding a family is a table row and not a branch in the policy.
CONDITIONING_CANVAS_OBJECT = "canvas-object"
CONDITIONING_IMAGE_PATHS = "conditioning-image-paths"

# Green-border outpaint adapters are trained to paint into a pure-green canvas, so that exact
# colour is part of their contract and must not be replaced by the generic neutral fill.
FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS = (
    "fal/flux-2-klein-4b-outpaint-lora",
    "ming3d/flux-2-klein-4b-outpaint-lora",
    "flux-outpaint-lora.safetensors",
)
FLUX2_GREEN_BORDER_FILL_COLOR = (0, 255, 0)
FLUX2_GREEN_BORDER_FILL_REASON = (
    "a green-border outpaint LoRA is loaded and that adapter is trained to paint into a pure-green canvas."
)
# Base canvas colour under the edge/blur backgrounds. Only visible where the edge extension does
# not reach, which is nowhere for a rectangular padding box; kept at the historical value so
# `edge` reproduces pre-0.30 conditioning canvases byte for byte.
EDGE_FILL_BASE_COLOR = (255, 255, 255)
# Default colour for an explicit `solid` with no colour, when the source is too small to sample
# a border ring. Mid-gray is the safe generic blank.
OUTPAINT_NEUTRAL_FALLBACK_COLOR = (128, 128, 128)


class OutpaintError(ValueError):
    """Raised when an outpaint request cannot be satisfied by the selected route.

    Subclasses ValueError so the existing backend CLIs keep converting it to
    `parser.error(...)` with no change to their except clauses.
    """


@dataclass(frozen=True)
class _OutpaintRuntime:
    """Route mechanics the capability schema deliberately does not publish."""

    conditioning: str
    preserve_source: bool | None
    restore_threshold: float
    base_fill_color: tuple[int, int, int] = EDGE_FILL_BASE_COLOR
    neutral_fallback_color: tuple[int, int, int] = OUTPAINT_NEUTRAL_FALLBACK_COLOR
    adapter_fill_color: tuple[int, int, int] | None = None
    adapter_markers: tuple[str, ...] = ()
    adapter_fill_reason: str = ""


_OUTPAINT_RUNTIME: dict[str, _OutpaintRuntime] = {
    "flux2.outpaint": _OutpaintRuntime(
        conditioning=CONDITIONING_CANVAS_OBJECT,
        # The base route locks the source in latent space behind a transition band, so a
        # pixel post-blend would only fight it. Both spellings are passed: preserve_source
        # is the contract, restore_threshold is the legacy sentinel the same routine still
        # accepts, and OutpaintUtil pins them to the same behaviour.
        preserve_source=False,
        restore_threshold=-1.0,
        adapter_fill_color=FLUX2_GREEN_BORDER_FILL_COLOR,
        adapter_markers=FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS,
        adapter_fill_reason=FLUX2_GREEN_BORDER_FILL_REASON,
    ),
    "qwen.outpaint": _OutpaintRuntime(
        conditioning=CONDITIONING_IMAGE_PATHS,
        # Expanded-canvas generation plus adaptive source restoration: paste the source back
        # only while the generated source window still matches it.
        preserve_source=None,
        restore_threshold=12.0,
    ),
}


@dataclass(frozen=True)
class OutpaintContract:
    """Everything the shared layer needs to run outpaint on one route.

    Published fields are copied off the route's `GenerationCapability` row so a host reading
    `mlxgen capabilities --json` and the shared layer can never disagree; unpublished
    mechanics come from `_OUTPAINT_RUNTIME`.
    """

    capability_id: str
    fill_modes: tuple[str, ...]
    default_fill_mode: str
    supports_fill_option: bool
    auto_edge_fill_max_stretch: float | None
    recommended_lora: str | None
    preservation: str
    dimension_multiple: int
    conditioning: str
    preserve_source: bool | None
    restore_threshold: float
    base_fill_color: tuple[int, int, int]
    neutral_fallback_color: tuple[int, int, int]
    adapter_fill_color: tuple[int, int, int] | None
    adapter_markers: tuple[str, ...]
    adapter_fill_reason: str

    @property
    def has_auto_policy(self) -> bool:
        """Whether `fill="auto"` selects a mode at run time on this route."""
        return self.supports_fill_option and self.auto_edge_fill_max_stretch is not None


def outpaint_contract(*, capability: GenerationCapability) -> OutpaintContract:
    """Build the run-time outpaint contract for one published capability row.

    Raises OutpaintError when the row does not support outpaint, or when it does but the
    shared layer has no runtime entry for it - that pairing is a packaging bug, not a user
    error, and failing here is what stops a new family from silently getting FLUX.2's
    latent-lock semantics.
    """
    if not capability.supports_outpaint:
        raise OutpaintError(f"Capability {capability.id!r} does not support outpaint.")
    runtime = _OUTPAINT_RUNTIME.get(capability.id)
    if runtime is None:
        raise OutpaintError(
            f"Capability {capability.id!r} advertises outpaint but mflux.outpaint has no runtime "
            "contract for it. Add an _OUTPAINT_RUNTIME entry for the route."
        )
    if capability.outpaint_default_fill_mode is None or capability.outpaint_preservation is None:
        raise OutpaintError(f"Capability {capability.id!r} publishes an incomplete outpaint fill contract.")
    return OutpaintContract(
        capability_id=capability.id,
        fill_modes=capability.outpaint_fill_modes,
        default_fill_mode=capability.outpaint_default_fill_mode,
        supports_fill_option=capability.supports_outpaint_fill,
        auto_edge_fill_max_stretch=capability.outpaint_auto_edge_fill_max_stretch,
        recommended_lora=capability.outpaint_recommended_lora,
        preservation=capability.outpaint_preservation,
        dimension_multiple=capability.dimension_multiple or 16,
        conditioning=runtime.conditioning,
        preserve_source=runtime.preserve_source,
        restore_threshold=runtime.restore_threshold,
        base_fill_color=runtime.base_fill_color,
        neutral_fallback_color=runtime.neutral_fallback_color,
        adapter_fill_color=runtime.adapter_fill_color,
        adapter_markers=runtime.adapter_markers,
        adapter_fill_reason=runtime.adapter_fill_reason,
    )


def outpaint_contract_for_model(
    *,
    model: str | None = None,
    model_config: Any = None,
    base_model: str | None = None,
) -> OutpaintContract:
    """Resolve the outpaint route for a model name or ModelConfig and return its contract.

    Loads no weights. Prefer `model_config=...` when the caller already resolved one: a local
    `--model-path` folder has no routable model name. Raises TaskInferenceError when the model
    has no outpaint route at all.
    """
    plan = resolve_generation_plan(
        model=model, model_config=model_config, base_model=base_model, image_count=1, has_outpaint=True
    )
    capabilities = get_model_capabilities(model=model, model_config=model_config, base_model=base_model)
    for capability in capabilities.capabilities:
        if capability.id == plan.capability_id:
            return outpaint_contract(capability=capability)
    raise OutpaintError(f"No capability row {plan.capability_id!r} for model {model or model_config!r}.")


@dataclass(frozen=True)
class OutpaintRequest:
    """A plain-value outpaint request. No argparse, no model, no image loading.

    `fill` is None on routes without a fill option, or one of the route's
    `outpaint_fill_modes`. `fill_color_explicit` is separate from `fill_color` on purpose:
    the CLI only warns that a colour is ignored when the user typed it, not when it was
    replayed out of prior metadata.
    """

    padding: str
    fill: str | None = None
    fill_color: tuple[int, int, int] | None = None
    fill_color_explicit: bool = False
    lora_paths: tuple[str, ...] = ()
    requested_lora_paths: tuple[str, ...] = ()
    option_name: str = "--outpaint-padding"


@dataclass(frozen=True)
class OutpaintFillPlan:
    """The resolved conditioning-canvas decision for one outpaint run."""

    # What the caller asked for ("auto" | "edge" | "neutral" | "solid" | "blur").
    requested: str
    # The concrete OutpaintUtil.create_expanded_canvas fill_mode that will run.
    mode: str
    # Only "solid" consumes a colour; edge/neutral/blur derive their canvas from the source.
    fill_color: tuple[int, int, int] | None
    # Human-readable justification, printed for `auto` runs.
    reason: str
    # The side that stretches edge fill hardest, and how far past its reach it runs.
    max_side: str
    max_side_padding_px: int
    max_side_ratio: float
    # Padding depth edge fill covers on that side, and padding / reach. Above 1.0 the strip
    # would stretch past the validated bound.
    max_side_reach_px: int
    max_side_overreach: float
    # Whether an adapter the route declares as canvas-coupled is loaded.
    uses_solid_fill_adapter: bool

    @property
    def edge_fill_within_reach(self) -> bool:
        return self.max_side_overreach <= 1.0

    @property
    def is_explicit(self) -> bool:
        return self.requested != OUTPAINT_FILL_AUTO


def resolve_outpaint_fill_plan(
    *,
    contract: OutpaintContract,
    request: OutpaintRequest,
    source: PIL.Image.Image,
    padding: AbsoluteBoxValues,
) -> OutpaintFillPlan:
    """Choose the conditioning canvas for one request on one route.

    Routes that publish `supports_outpaint_fill=False` never reach the `auto` policy: they
    resolve to their single published `outpaint_default_fill_mode` and reject any other
    request. That is what keeps a fixed-fill route's recorded validation runs bit-identical
    when the shared policy layer grows a new default.
    """
    requested = resolve_requested_fill_mode(contract=contract, fill=request.fill)
    max_side, max_side_padding_px, max_side_ratio, max_side_reach_px, max_side_overreach = _largest_relative_padding(
        padding=padding,
        width=source.width,
        height=source.height,
    )
    uses_adapter = _uses_solid_fill_adapter(contract=contract, request=request)

    def solid_default_color() -> tuple[int, int, int]:
        if uses_adapter and contract.adapter_fill_color is not None:
            return contract.adapter_fill_color
        return mean_border_color(source, fallback=contract.neutral_fallback_color)

    if requested != OUTPAINT_FILL_AUTO:
        mode = requested
        fill_color = (request.fill_color or solid_default_color()) if mode == "solid" else None
        reason = f"--outpaint-fill {requested} was passed explicitly."
    elif uses_adapter and contract.adapter_fill_color is not None:
        mode = "solid"
        fill_color = request.fill_color or contract.adapter_fill_color
        reason = contract.adapter_fill_reason
    elif max_side_overreach <= 1.0:
        mode = "edge"
        fill_color = None
        reason = (
            f"the deepest padding is {max_side} {max_side_padding_px}px "
            f"({max_side_ratio:.0%} of the source {_axis_label(max_side)}), within the "
            f"{max_side_reach_px}px edge-fill reach, so continuing the source border texture is "
            "the better conditioning canvas."
        )
    else:
        # OutpaintUtil supports a "neutral" canvas: a flat per-side border colour taken from the
        # source. That is the better blank canvas than a fixed mid-gray -- equally textureless, so
        # there is nothing for the model to continue, but sitting in the source's own colour
        # neighbourhood so the boundary is not a hard chroma step the model redraws as a seam.
        mode = "neutral"
        fill_color = None
        reason = (
            f"the deepest padding is {max_side} {max_side_padding_px}px "
            f"({max_side_ratio:.0%} of the source {_axis_label(max_side)}), "
            f"{max_side_overreach:.1f}x the {max_side_reach_px}px edge-fill reach; a blank canvas "
            "makes the model generate new subject matter instead of smearing the source border."
        )

    if mode not in contract.fill_modes and mode != OUTPAINT_FILL_AUTO:
        raise OutpaintError(
            f"Fill mode {mode!r} is not supported by {contract.capability_id}; "
            f"supported modes are {', '.join(contract.fill_modes)}."
        )
    return OutpaintFillPlan(
        requested=requested,
        mode=mode,
        fill_color=fill_color,
        reason=reason,
        max_side=max_side,
        max_side_padding_px=max_side_padding_px,
        max_side_ratio=max_side_ratio,
        max_side_reach_px=max_side_reach_px,
        max_side_overreach=max_side_overreach,
        uses_solid_fill_adapter=uses_adapter,
    )


def resolve_requested_fill_mode(*, contract: OutpaintContract, fill: str | None) -> str:
    """The fill mode one request names on one route, or raise OutpaintError. Loads nothing.

    Pure policy over the route's published fill contract, with no source image and no canvas, so
    a caller that only holds a model name can ask the question before dispatch. `mlxgen generate`
    declares `--outpaint-fill` once for every model, so this is what keeps an unsupported request
    on the route's own terms instead of reaching a backend parser that never declared the option.
    """
    requested = fill or contract.default_fill_mode
    if contract.supports_fill_option:
        if requested not in contract.fill_modes:
            raise OutpaintError(
                f"Fill mode {requested!r} is not supported by {contract.capability_id}; "
                f"supported modes are {', '.join(contract.fill_modes)}."
            )
        return requested
    if requested not in {OUTPAINT_FILL_AUTO, contract.default_fill_mode}:
        raise OutpaintError(
            f"{contract.capability_id} has a fixed {contract.default_fill_mode!r} conditioning canvas "
            f"and does not accept an outpaint fill mode ({requested!r} was requested)."
        )
    return contract.default_fill_mode


def check_outpaint_fill_options(
    *,
    contract: OutpaintContract,
    fill: str | None = None,
    fill_color_requested: bool = False,
) -> None:
    """Raise OutpaintError when a route cannot honour the requested fill options. Loads nothing."""
    resolve_requested_fill_mode(contract=contract, fill=fill)
    if fill_color_requested and not contract.supports_fill_option:
        raise OutpaintError(
            f"{contract.capability_id} has a fixed {contract.default_fill_mode!r} conditioning canvas "
            "and does not accept an outpaint fill color."
        )


def guard_outpaint_fill_plan(*, contract: OutpaintContract, fill_plan: OutpaintFillPlan) -> tuple[str, ...]:
    """Fail closed on a known-bad canvas; return the warnings the caller should surface.

    Explicit user intent wins, but never silently. An `auto` run that reaches this state is
    unreachable today and raises rather than running the measured-bad configuration.
    """
    if fill_plan.mode != "edge" or fill_plan.edge_fill_within_reach:
        return ()
    if fill_plan.is_explicit:
        return (_unsafe_edge_fill_warning(fill_plan, contract=contract),)
    raise OutpaintError(
        f"--outpaint-fill auto resolved to edge with {fill_plan.max_side} padding of "
        f"{fill_plan.max_side_padding_px}px, {fill_plan.max_side_overreach:.1f}x the "
        f"{fill_plan.max_side_reach_px}px edge-fill reach. Edge fill smears at that depth. Pass "
        "--outpaint-fill neutral for a blank conditioning canvas, or --outpaint-fill edge to "
        "force the edge canvas anyway."
    )


def _unsafe_edge_fill_warning(fill_plan: OutpaintFillPlan, *, contract: OutpaintContract) -> str:
    # The remediation half is route-dependent: a route that publishes no fill option cannot be
    # told to pick a different canvas, so pointing it at --outpaint-fill would be a lie.
    if contract.supports_fill_option:
        remedy = (
            "Proceeding as requested. Use --outpaint-fill neutral for a blank conditioning canvas, or add "
            f"--lora-paths {contract.recommended_lora} for the validated green-canvas route."
            if contract.recommended_lora
            else "Proceeding as requested. Use --outpaint-fill neutral for a blank conditioning canvas."
        )
    else:
        remedy = (
            f"Proceeding: {contract.capability_id} has a fixed {contract.default_fill_mode} conditioning canvas. "
            "Reduce the padding, or use a route that publishes supports_outpaint_fill."
        )
    return (
        f"Warning: --outpaint-fill edge with {fill_plan.max_side} padding of "
        f"{fill_plan.max_side_padding_px}px ({fill_plan.max_side_ratio:.0%} of the source "
        f"{_axis_label(fill_plan.max_side)}) runs {fill_plan.max_side_overreach:.1f}x the "
        f"{fill_plan.max_side_reach_px}px edge-fill reach. Edge fill stretches a border strip "
        f"across the padded region and produces directional smear past its reach. {remedy}"
    )


def _largest_relative_padding(
    *, padding: AbsoluteBoxValues, width: int, height: int
) -> tuple[str, int, float, int, float]:
    # Each side is measured against the source dimension it grows along: top/bottom against the
    # height, left/right against the width. The side that matters is the one running furthest past
    # what edge fill can cover, so sides rank by overreach (padding / reach), not by raw ratio.
    # Ties resolve in top,right,bottom,left order so the reported side is stable.
    sides = (
        ("top", padding.top, height),
        ("right", padding.right, width),
        ("bottom", padding.bottom, height),
        ("left", padding.left, width),
    )
    best_side, best_pixels, best_ratio, best_reach, best_overreach = "top", 0, 0.0, 0, 0.0
    for name, pixels, base in sides:
        reach = OutpaintUtil.edge_fill_reach(base)
        overreach = (pixels / reach) if reach > 0 else 0.0
        if overreach > best_overreach:
            ratio = (pixels / base) if base > 0 else 0.0
            best_side, best_pixels, best_ratio = name, pixels, ratio
            best_reach, best_overreach = reach, overreach
    if best_reach == 0:
        best_reach = OutpaintUtil.edge_fill_reach(height)
    return best_side, best_pixels, best_ratio, best_reach, best_overreach


def _axis_label(side: str) -> str:
    return "height" if side in {"top", "bottom"} else "width"


def mean_border_color(
    source: PIL.Image.Image,
    *,
    fallback: tuple[int, int, int] = OUTPAINT_NEUTRAL_FALLBACK_COLOR,
) -> tuple[int, int, int]:
    """Mean RGB of the source's outer ring - the flat colour a blank canvas should use.

    A fixed mid-gray is equally flat but plants a hard chroma step along the source boundary,
    and that step is itself a strong edge the model tends to redraw as a visible seam.
    """
    rgb = source.convert("RGB")
    if rgb.width < 2 or rgb.height < 2:
        return fallback
    ring = max(1, min(16, min(rgb.width, rgb.height) // 32))
    strips = (
        rgb.crop((0, 0, rgb.width, ring)),
        rgb.crop((0, rgb.height - ring, rgb.width, rgb.height)),
        rgb.crop((0, 0, ring, rgb.height)),
        rgb.crop((rgb.width - ring, 0, rgb.width, rgb.height)),
    )
    totals = [0.0, 0.0, 0.0]
    sampled_pixels = 0
    for strip in strips:
        pixels = strip.width * strip.height
        if pixels == 0:
            continue
        means = PIL.ImageStat.Stat(strip).mean[:3]
        for channel, mean in enumerate(means):
            totals[channel] += mean * pixels
        sampled_pixels += pixels
    if sampled_pixels == 0:
        return fallback
    return tuple(max(0, min(255, round(total / sampled_pixels))) for total in totals)  # type: ignore[return-value]


def _uses_solid_fill_adapter(*, contract: OutpaintContract, request: OutpaintRequest) -> bool:
    # Match the pre-resolution request as well as the resolved path. LoraResolution rewrites
    # "fal/flux-2-klein-4B-outpaint-lora" into a concrete cache file, so the repo-form marker only
    # ever matched by accident, through whichever basename that repo happened to resolve to;
    # a sibling file in the same repo (the comfy-converted weights) silently did not match.
    if not contract.adapter_markers:
        return False
    specs = [*request.requested_lora_paths, *request.lora_paths]
    return any(_spec_matches_marker(str(spec), contract.adapter_markers) for spec in specs)


def _spec_matches_marker(spec: str, markers: tuple[str, ...]) -> bool:
    normalized = spec.lower()
    # Hugging Face cache directories spell "org/repo" as "models--org--repo"; normalizing the
    # double dash back to a slash lets the repo-form markers match a resolved snapshot path too.
    candidates = (normalized, normalized.replace("--", "/"))
    return any(marker in candidate for candidate in candidates for marker in markers)


@dataclass(kw_only=True)
class _ExpandedCanvasSession:
    """Shared plumbing for the two expanded-canvas workflows.

    Outpaint and reframe genuinely share the canvas builder, the derived generation geometry,
    and the conditioning-image plumbing. They do not share a preservation strategy or a fill
    policy, so those live only on OutpaintSession.
    """

    source_image: str | Path
    padding: str
    canvas: OutpaintCanvas
    _workspace: TemporaryDirectory | None = field(default=None, repr=False)

    @property
    def width(self) -> int:
        """Generation width. Callers must not also pass their own --width."""
        return self.canvas.target_width

    @property
    def height(self) -> int:
        return self.canvas.target_height

    @property
    def canvas_policy(self) -> str:
        return CANVAS_POLICY_EXACT_RESIZE

    @property
    def conditioning_image_paths(self) -> list[Path]:
        """The image paths the model should be conditioned on - the canvas, not the source."""
        return [self.canvas.canvas_path]

    def close(self) -> None:
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


@dataclass(kw_only=True)
class OutpaintSession(_ExpandedCanvasSession):
    """A prepared outpaint run: resolved policy, built canvas, and the generation contract.

    The whole outpaint-specific pipeline is three calls - build (`prepare_outpaint`),
    `generate(...)`, `finalize(...)` - and `generate` already calls `finalize`. A caller that
    owns its own seed loop (the CLI) calls `generate`; a caller that wants the runtime
    wrapper's multi-seed, progress and save behaviour calls `run_outpaint`.
    """

    contract: OutpaintContract
    fill_plan: OutpaintFillPlan
    warnings: tuple[str, ...] = ()
    fill_color_explicit: bool = False

    @property
    def notice(self) -> str:
        """The one-to-three line stderr summary of the resolved conditioning canvas."""
        fill_plan, canvas = self.fill_plan, self.canvas
        color = ""
        if fill_plan.mode == "solid":
            color = f" color={fill_plan.fill_color[0]},{fill_plan.fill_color[1]},{fill_plan.fill_color[2]}"
        lines = [
            f"Outpaint: fill={fill_plan.mode}{color}, canvas "
            f"{canvas.target_width}x{canvas.target_height} from source "
            f"{canvas.source_width}x{canvas.source_height}, padding top={canvas.padding.top} "
            f"right={canvas.padding.right} bottom={canvas.padding.bottom} left={canvas.padding.left}."
        ]
        if not fill_plan.is_explicit:
            lines.append(f"Outpaint: --outpaint-fill auto selected {fill_plan.mode} because {fill_plan.reason}")
        if fill_plan.mode != "solid" and self.fill_color_explicit:
            lines.append(
                f"Outpaint: --outpaint-fill-color only applies to --outpaint-fill solid "
                f"and is ignored by {fill_plan.mode}."
            )
        return "\n".join(lines)

    def conditioning_kwargs(self) -> dict[str, Any]:
        """The generate_image(...) keywords that carry the canvas on this route."""
        if self.contract.conditioning == CONDITIONING_CANVAS_OBJECT:
            return {"canvas": self.canvas}
        if self.contract.conditioning == CONDITIONING_IMAGE_PATHS:
            return {
                "image_path": self.source_image,
                "image_paths": [str(path) for path in self.conditioning_image_paths],
                "width": self.width,
                "height": self.height,
                "canvas_policy": self.canvas_policy,
            }
        raise OutpaintError(f"Unknown outpaint conditioning mode {self.contract.conditioning!r}.")

    def generate(self, model: Any, /, **generate_kwargs: Any):
        """Run one seed on `model` and finalize the artifact.

        The canvas keywords come from the route contract; everything else (seed, prompt,
        guidance, steps, scheduler, negative_prompt) is the caller's.
        """
        overlap = sorted(set(generate_kwargs) & set(self.conditioning_kwargs()))
        if overlap:
            raise OutpaintError(f"Outpaint owns these generation keywords: {', '.join(overlap)}.")
        generated = model.generate_image(**self.conditioning_kwargs(), **generate_kwargs)
        self.finalize(generated)
        return generated

    def finalize(self, generated: Any) -> None:
        """Composite the source back where the route asks for it, then attach metadata.

        Mutates `generated` in place: it replaces `generated.image`, repoints the recorded
        source paths at the original image rather than the canvas, and records both the
        padding geometry and the resolved fill so `-C metadata.json` replays the resolved
        canvas instead of re-running `auto`.
        """
        generated.image = OutpaintUtil.composite_source_region(
            generated_image=generated.image,
            canvas=self.canvas,
            feather_px=None,
            preserve_source=self.contract.preserve_source,
            restore_threshold=self.contract.restore_threshold,
        )
        generated.image_path = self.source_image
        generated.image_paths = [self.source_image]
        OutpaintUtil.attach_metadata(
            generated_image=generated,
            canvas=self.canvas,
            padding_value=self.padding,
            preservation=self.contract.preservation,
        )
        extra_metadata = dict(getattr(generated, "extra_metadata", None) or {})
        extra_metadata.update(
            {
                "outpaint_fill": self.fill_plan.mode,
                "outpaint_fill_color": list(self.fill_plan.fill_color)
                if self.fill_plan.fill_color is not None
                else None,
                "outpaint_fill_requested": self.fill_plan.requested,
                "outpaint_fill_reason": self.fill_plan.reason,
                "outpaint_max_side_padding": self.fill_plan.max_side,
                "outpaint_max_side_padding_px": self.fill_plan.max_side_padding_px,
                "outpaint_max_side_padding_ratio": round(self.fill_plan.max_side_ratio, 4),
                "outpaint_edge_fill_reach_px": self.fill_plan.max_side_reach_px,
                "outpaint_edge_fill_overreach": round(self.fill_plan.max_side_overreach, 4),
            }
        )
        generated.extra_metadata = extra_metadata


@dataclass(kw_only=True)
class ReframeSession(_ExpandedCanvasSession):
    """A prepared generative reframe run.

    Reframe shares the canvas builder and the derived geometry with outpaint and nothing
    else: there is no fill policy (the historical edge-extended canvas is the contract) and
    no source preservation (reframe is allowed to recompose the source).
    """

    def conditioning_kwargs(self) -> dict[str, Any]:
        return {
            "image_paths": [str(path) for path in self.conditioning_image_paths],
            "width": self.width,
            "height": self.height,
            "canvas_policy": self.canvas_policy,
        }

    def generate(self, model: Any, /, **generate_kwargs: Any):
        generated = model.generate_image(**self.conditioning_kwargs(), **generate_kwargs)
        self.finalize(generated)
        return generated

    def finalize(self, generated: Any) -> None:
        generated.image_path = self.source_image
        generated.image_paths = [self.source_image]
        OutpaintUtil.attach_reframe_metadata(
            generated_image=generated,
            canvas=self.canvas,
            padding_value=self.padding,
        )


def prepare_outpaint(
    *,
    source_image: str | Path,
    padding: str,
    contract: OutpaintContract | None = None,
    capability: GenerationCapability | None = None,
    model: str | None = None,
    model_config: Any = None,
    base_model: str | None = None,
    fill: str | None = None,
    fill_color: tuple[int, int, int] | None = None,
    fill_color_explicit: bool = False,
    lora_paths: Sequence[str] = (),
    requested_lora_paths: Sequence[str] = (),
    workspace: str | Path | None = None,
    canvas_name: str = "outpaint_canvas.png",
    option_name: str = "--outpaint-padding",
) -> OutpaintSession:
    """Resolve the outpaint policy and build the conditioning canvas. Loads no weights.

    Identify the route with exactly one of `contract`, `capability`, or `model`. When
    `workspace` is omitted the session owns a temporary directory and must be closed - use it
    as a context manager, or let `run_outpaint` own it.

    Raises OutpaintError for an unsupported fill mode, a padding value that adds no pixels,
    or an `auto` decision that would run a canvas measured to smear.
    """
    resolved = _resolve_contract(
        contract=contract, capability=capability, model=model, model_config=model_config, base_model=base_model
    )
    request = OutpaintRequest(
        padding=padding,
        fill=fill,
        fill_color=fill_color,
        fill_color_explicit=fill_color_explicit,
        lora_paths=tuple(str(path) for path in lora_paths),
        requested_lora_paths=tuple(str(path) for path in requested_lora_paths),
        option_name=option_name,
    )
    owned_workspace, workspace_path = _resolve_workspace(workspace)
    try:
        source = ImageUtil.load_image(source_image)
        box = BoxValues.parse(padding).normalize_to_dimensions(width=source.width, height=source.height)
        fill_plan = resolve_outpaint_fill_plan(contract=resolved, request=request, source=source, padding=box)
        warnings = guard_outpaint_fill_plan(contract=resolved, fill_plan=fill_plan)
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_image,
            padding_value=padding,
            output_path=workspace_path / canvas_name,
            dimension_multiple=resolved.dimension_multiple,
            fill_mode=fill_plan.mode,
            fill_color=fill_plan.fill_color or resolved.base_fill_color,
            option_name=option_name,
        )
    except BaseException:
        if owned_workspace is not None:
            owned_workspace.cleanup()
        raise
    return OutpaintSession(
        source_image=source_image,
        padding=padding,
        canvas=canvas,
        _workspace=owned_workspace,
        contract=resolved,
        fill_plan=fill_plan,
        warnings=warnings,
        fill_color_explicit=fill_color_explicit,
    )


def prepare_reframe(
    *,
    source_image: str | Path,
    padding: str,
    dimension_multiple: int = 16,
    workspace: str | Path | None = None,
    canvas_name: str = "reframe_canvas.png",
    option_name: str = "--reframe-padding",
) -> ReframeSession:
    """Build the reframe conditioning canvas. Loads no weights.

    Reframe keeps the historical edge-extended canvas: it has no fill contract, so it needs
    no capability row.
    """
    owned_workspace, workspace_path = _resolve_workspace(workspace)
    try:
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_image,
            padding_value=padding,
            output_path=workspace_path / canvas_name,
            dimension_multiple=dimension_multiple,
            option_name=option_name,
        )
    except BaseException:
        if owned_workspace is not None:
            owned_workspace.cleanup()
        raise
    return ReframeSession(
        source_image=source_image,
        padding=padding,
        canvas=canvas,
        _workspace=owned_workspace,
    )


def _resolve_contract(
    *,
    contract: OutpaintContract | None,
    capability: GenerationCapability | None,
    model: str | None,
    model_config: Any,
    base_model: str | None,
) -> OutpaintContract:
    named = [
        name
        for name, value in (
            ("contract", contract),
            ("capability", capability),
            ("model", model),
            ("model_config", model_config),
        )
        if value is not None
    ]
    if len(named) != 1:
        raise OutpaintError(
            "Pass exactly one of contract=..., capability=..., model=..., or model_config=... to identify the route."
        )
    if contract is not None:
        return contract
    if capability is not None:
        return outpaint_contract(capability=capability)
    return outpaint_contract_for_model(model=model, model_config=model_config, base_model=base_model)


def _resolve_workspace(workspace: str | Path | None) -> tuple[TemporaryDirectory | None, Path]:
    if workspace is not None:
        return None, Path(workspace)
    owned = TemporaryDirectory(prefix="mlxgen-outpaint-")
    return owned, Path(owned.name)


def run_outpaint(
    *,
    loaded: Any,
    source_image: str | Path,
    padding: str,
    seeds: Sequence[int] = (0,),
    output: str | Path | None = None,
    overwrite: bool = False,
    progress_callback: Any = None,
    save_kwargs: dict[str, Any] | None = None,
    fill: str | None = None,
    fill_color: tuple[int, int, int] | None = None,
    lora_paths: Sequence[str] = (),
    requested_lora_paths: Sequence[str] = (),
    **generate_kwargs: Any,
) -> list[Any]:
    """Outpaint one source image with a loaded runtime, for one or more seeds.

    `loaded` is a `LoadedGenerationModel` from `load_generation_model(..., has_outpaint=True)`:
    its plan names the route, so the fill policy, preservation strategy and canvas keywords
    are all read from that route's capability row. Multi-seed execution, progress events and
    save semantics are the runtime wrapper's; this only adds the outpaint pipeline.
    """
    with prepare_outpaint(
        source_image=source_image,
        padding=padding,
        capability=_capability_for_loaded(loaded),
        fill=fill,
        fill_color=fill_color,
        lora_paths=lora_paths,
        requested_lora_paths=requested_lora_paths,
    ) as session:
        return loaded.generate_outputs(
            seeds=seeds,
            output=output,
            overwrite=overwrite,
            progress_callback=progress_callback,
            save_kwargs=save_kwargs,
            post_process=session.finalize,
            **session.conditioning_kwargs(),
            **generate_kwargs,
        )


def _capability_for_loaded(loaded: Any) -> GenerationCapability:
    capabilities = get_model_capabilities(
        model=loaded.model_config.model_name,
        model_config=loaded.model_config,
    )
    for capability in capabilities.capabilities:
        if capability.id == loaded.plan.capability_id:
            return capability
    raise OutpaintError(f"No capability row {loaded.plan.capability_id!r} for the loaded runtime.")
