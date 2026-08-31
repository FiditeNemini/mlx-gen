import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import PIL.Image
import PIL.ImageStat

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.output_paths import resolve_output_path
from mflux.cli.parser.parsers import OUTPAINT_FILL_AUTO, CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.models.flux2.variants.edit.flux2_klein_inpaint import Flux2KleinInpaint
from mflux.models.flux2.variants.edit.flux2_klein_outpaint import Flux2KleinOutpaint
from mflux.utils.box_values import AbsoluteBoxValues, BoxValues
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from mflux.utils.outpaint_util import OutpaintCanvas, OutpaintUtil
from mflux.utils.prompt_util import PromptUtil

LEGACY_NOTICE = (
    "Warning: mflux-generate-flux2-edit is a legacy compatibility command. "
    "Use `mlxgen generate --model <model> --image <path> ...` for new integrations."
)

FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS = (
    "fal/flux-2-klein-4b-outpaint-lora",
    "ming3d/flux-2-klein-4b-outpaint-lora",
    "flux-outpaint-lora.safetensors",
)

# The green-border adapters above are trained to paint into a pure-green canvas, so that exact
# color is part of their contract and must not be replaced by the generic neutral fill.
FLUX2_GREEN_BORDER_FILL_COLOR = (0, 255, 0)

# Base canvas color under the edge/blur backgrounds. Only visible where the edge extension does
# not reach, which is nowhere for a rectangular padding box; kept at the historical value so
# `--outpaint-fill edge` reproduces pre-0.30 conditioning canvases byte for byte.
FLUX2_EDGE_FILL_BASE_COLOR = (255, 255, 255)

# Edge fill stretches a source border strip across the padded region, so it is a texture
# *continuation*, not a generator. What decides whether it reads as continued texture or as
# directional smear is the stretch factor, not the padding expressed as a fraction of the source:
# the published `5%,80%,5%,60%` profile runs 80% single-side padding at a 10.9x stretch and is
# validated, while a 768x766 portrait at `0%,10%,100%,10%` runs 24x and returns the conditioning
# canvas back as a field of vertical streaks.
#
# `OutpaintUtil.edge_fill_reach()` is that bound expressed in pixels, per source side, and it is
# the same quantity the fill itself uses to decide when to cross-fade to the neutral background.
# Selecting the fill mode by reach rather than by a separate ratio keeps one rule in one place and
# keeps every published profile on the fill mode it was validated with.

# Default color for an explicit `--outpaint-fill solid` with no `--outpaint-fill-color`, when the
# source is too small to sample a border ring. Mid-gray is the safe generic blank.
FLUX2_OUTPAINT_NEUTRAL_FALLBACK_COLOR = (128, 128, 128)


@dataclass(frozen=True)
class OutpaintFillPlan:
    # What the user asked for ("auto" | "edge" | "solid" | "blur").
    requested: str
    # The concrete OutpaintUtil.create_expanded_canvas fill_mode that will run.
    mode: str
    # Only "solid" consumes a color; edge/neutral/blur derive their canvas from the source.
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
    uses_green_border_lora: bool

    @property
    def edge_fill_within_reach(self) -> bool:
        return self.max_side_overreach <= 1.0

    @property
    def is_explicit(self) -> bool:
        return self.requested != OUTPAINT_FILL_AUTO


def _print_legacy_notice() -> None:
    print(LEGACY_NOTICE, file=sys.stderr)


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(
        description=(
            "Legacy compatibility command for FLUX.2 Klein image conditioning and edit workflows. "
            "Prefer `mlxgen generate --model <model> --image <path> ...` for new integrations."
        ),
        epilog=("Preferred migration target: mlxgen generate --model <flux2-model> --image <path> ..."),
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_mask_path_argument(
        help_text=(
            "Optional mask image path for localized FLUX.2 Klein masked edit. White pixels are repainted "
            "and black pixels are preserved."
        ),
    )
    parser.add_argument(
        "--reframe-padding",
        default=None,
        help=(
            "Generative reframe request: expand one source image by CSS-style "
            "top,right,bottom,left padding before edit generation."
        ),
    )
    parser.add_argument(
        "--outpaint-padding",
        "--image-outpaint-padding",
        dest="outpaint_padding",
        default=None,
        help=(
            "Expand one source image by CSS-style top,right,bottom,left padding and use an adaptive "
            "source blend when the generated source window still matches the original image."
        ),
    )
    parser.add_outpaint_fill_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    args = parser.parse_args()
    _print_legacy_notice()

    if getattr(args, "negative_prompt", ""):
        parser.error(
            "--negative-prompt is not supported for FLUX.2. Omit it for FLUX.2 routes. "
            "For new integrations, call `mlxgen generate --model <flux2-model> --image <path> ...` "
            "instead of `mflux-generate-flux2-edit`."
        )
    source_image_paths = [Path(p) for p in args.image_paths]
    _validate_canvas_args(parser=parser, args=args, source_image_paths=source_image_paths)

    model_name = args.model or "flux2-klein-4b"
    model_config = ModelConfig.from_name(model_name=model_name, base_model=args.base_model)

    is_base_model = _is_flux2_base_model(model_config)
    uses_masked_edit = args.mask_path is not None
    if args.guidance is None:
        uses_source_locked_denoise = args.outpaint_padding is not None or uses_masked_edit
        args.guidance = 4.0 if uses_source_locked_denoise and is_base_model else 1.0
    model_name_lower = model_config.model_name.lower()
    base_model_lower = (model_config.base_model or "").lower()
    is_flux2 = any(
        identifier in model_name_lower or identifier in base_model_lower for identifier in ("flux.2", "flux2")
    )
    if args.reframe_padding is not None and is_base_model:
        parser.error("--reframe-padding requires a validated non-base FLUX.2 Klein model.")
    if args.outpaint_padding is not None and not is_base_model:
        parser.error(
            "--outpaint-padding requires a FLUX.2 Klein base model because strict outpaint "
            "needs source-locked denoising. Use a base model such as "
            "black-forest-labs/FLUX.2-klein-base-9B or "
            "AbstractFramework/flux.2-klein-base-9b-8bit."
        )
    if args.guidance != 1.0 and not is_flux2:
        parser.error("--guidance is only supported for FLUX.2 models. Use --guidance 1.0.")
    if args.guidance != 1.0 and not is_base_model and args.outpaint_padding is None:
        parser.error("--guidance is only supported for FLUX.2 base models. Use --guidance 1.0.")

    CallbackManager.apply_runtime_memory_options(args)

    uses_strict_outpaint = args.outpaint_padding is not None and is_base_model
    model_kwargs = {
        "model_config": model_config,
        "quantize": args.quantize,
        "model_path": args.model_path,
        "lora_paths": args.lora_paths,
        "lora_scales": args.lora_scales,
    }
    if uses_masked_edit:
        model = Flux2KleinInpaint(**model_kwargs)
    elif uses_strict_outpaint:
        model = Flux2KleinOutpaint(**model_kwargs)
    else:
        model = Flux2KleinEdit(**model_kwargs)

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

    try:
        with TemporaryDirectory(prefix="mlxgen-outpaint-") as temporary_directory:
            try:
                image_paths, outpaint_canvas, reframe_canvas, outpaint_fill_plan = _resolve_image_paths(
                    args=args,
                    source_image_paths=source_image_paths,
                    temporary_directory=Path(temporary_directory),
                )
            except ValueError as exc:
                parser.error(str(exc))
            if outpaint_canvas is not None and outpaint_fill_plan is not None:
                print(outpaint_notice(fill_plan=outpaint_fill_plan, canvas=outpaint_canvas), file=sys.stderr)

            try:
                for seed in args.seed:
                    events = CliRuntimeEventStream(
                        enabled=bool(args.json_events),
                        command="mlxgen generate",
                        model=model_config.model_name,
                        seed=seed,
                    )
                    output_path = resolve_output_path(args.output, overwrite=args.replace, seed=seed)
                    events.set_output_path(output_path)
                    unsubscribe = events.subscribe_model(model, map_complete_to_generated=True)
                    try:
                        if uses_masked_edit:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                image_path=image_paths[0],
                                mask_path=args.mask_path,
                                reference_image_paths=image_paths[1:] or None,
                                width=args.width,
                                height=args.height,
                                guidance=args.guidance,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                                canvas_policy=args.canvas_policy,
                            )
                        elif uses_strict_outpaint:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                canvas=outpaint_canvas,
                                guidance=args.guidance,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                            )
                        else:
                            image = model.generate_image(
                                seed=seed,
                                prompt=PromptUtil.read_prompt(args),
                                width=args.width,
                                height=args.height,
                                guidance=args.guidance,
                                image_paths=image_paths,
                                num_inference_steps=args.steps,
                                scheduler="flow_match_euler_discrete",
                                canvas_policy=args.canvas_policy,
                            )
                        if outpaint_canvas is not None:
                            image.image = OutpaintUtil.composite_source_region(
                                generated_image=image.image,
                                canvas=outpaint_canvas,
                                feather_px=None,
                                restore_threshold=-1.0 if uses_strict_outpaint else 12.0,
                            )
                            image.image_path = source_image_paths[0]
                            image.image_paths = source_image_paths
                            OutpaintUtil.attach_metadata(
                                generated_image=image,
                                canvas=outpaint_canvas,
                                padding_value=args.outpaint_padding,
                                preservation=(
                                    "latent-locked-transition-band-no-postblend"
                                    if uses_strict_outpaint
                                    else "adaptive-content-aware-source-blend"
                                ),
                            )
                            _attach_fill_metadata(generated_image=image, fill_plan=outpaint_fill_plan)
                        if reframe_canvas is not None:
                            image.image_path = source_image_paths[0]
                            image.image_paths = source_image_paths
                            OutpaintUtil.attach_reframe_metadata(
                                generated_image=image,
                                canvas=reframe_canvas,
                                padding_value=args.reframe_padding,
                            )
                        events.emit_save()
                        image.save(
                            path=output_path,
                            export_json_metadata=args.metadata,
                            overwrite=True,
                            embed_metadata=args.embed_metadata,
                        )
                        events.emit_complete()
                    except Exception as exc:
                        events.emit_failed(error=exc)
                        raise
                    finally:
                        if unsubscribe is not None:
                            unsubscribe()
            except (StopImageGenerationException, PromptFileReadError) as exc:
                cli_print(str(exc), json_events=bool(args.json_events))
    finally:
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))


def _resolve_image_paths(
    *,
    args,
    source_image_paths: list[Path],
    temporary_directory: Path,
) -> tuple[list[Path], OutpaintCanvas | None, OutpaintCanvas | None, OutpaintFillPlan | None]:
    if args.outpaint_padding is None and args.reframe_padding is None:
        return source_image_paths, None, None, None
    padding_value = args.outpaint_padding or args.reframe_padding
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    canvas_name = "outpaint_canvas.png" if args.outpaint_padding is not None else "reframe_canvas.png"
    if len(source_image_paths) != 1:
        raise ValueError(f"{option_name} requires exactly one --image-paths value.")

    if args.outpaint_padding is None:
        # Generative reframe keeps the historical edge-extended canvas; --outpaint-fill is an
        # --outpaint-padding option and is rejected earlier when combined with --reframe-padding.
        canvas = OutpaintUtil.create_expanded_canvas(
            source_path=source_image_paths[0],
            padding_value=padding_value,
            output_path=temporary_directory / canvas_name,
            option_name=option_name,
        )
        args.width = canvas.target_width
        args.height = canvas.target_height
        args.canvas_policy = CANVAS_POLICY_EXACT_RESIZE
        return [canvas.canvas_path], None, canvas, None

    source = ImageUtil.load_image(source_image_paths[0])
    padding = BoxValues.parse(padding_value).normalize_to_dimensions(width=source.width, height=source.height)
    fill_plan = resolve_outpaint_fill_plan(args=args, source=source, padding=padding)
    _guard_unsafe_edge_fill(fill_plan=fill_plan)

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_image_paths[0],
        padding_value=padding_value,
        output_path=temporary_directory / canvas_name,
        option_name=option_name,
        fill_mode=fill_plan.mode,
        fill_color=fill_plan.fill_color or FLUX2_EDGE_FILL_BASE_COLOR,
    )
    args.width = canvas.target_width
    args.height = canvas.target_height
    args.canvas_policy = CANVAS_POLICY_EXACT_RESIZE
    return [canvas.canvas_path], canvas, None, fill_plan


def resolve_outpaint_fill_plan(*, args, source: PIL.Image.Image, padding: AbsoluteBoxValues) -> OutpaintFillPlan:
    requested = getattr(args, "outpaint_fill", None) or OUTPAINT_FILL_AUTO
    explicit_color = getattr(args, "outpaint_fill_color", None)
    uses_green_border_lora = _uses_green_border_outpaint_lora(args)
    max_side, max_side_padding_px, max_side_ratio, max_side_reach_px, max_side_overreach = _largest_relative_padding(
        padding=padding,
        width=source.width,
        height=source.height,
    )

    def _solid_default_color() -> tuple[int, int, int]:
        if uses_green_border_lora:
            return FLUX2_GREEN_BORDER_FILL_COLOR
        return _mean_border_color(source)

    if requested != OUTPAINT_FILL_AUTO:
        mode = requested
        fill_color = (explicit_color or _solid_default_color()) if mode == "solid" else None
        reason = f"--outpaint-fill {requested} was passed explicitly."
    elif uses_green_border_lora:
        mode = "solid"
        fill_color = explicit_color or FLUX2_GREEN_BORDER_FILL_COLOR
        reason = "a green-border outpaint LoRA is loaded and that adapter is trained to paint into a pure-green canvas."
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
        # OutpaintUtil supports a "neutral" canvas: a flat per-side border color taken from the
        # source. That is the better blank canvas than a fixed mid-gray -- equally textureless, so
        # there is nothing for the model to continue, but sitting in the source's own color
        # neighbourhood so the boundary is not a hard chroma step the model redraws as a seam.
        mode = "neutral"
        fill_color = None
        reason = (
            f"the deepest padding is {max_side} {max_side_padding_px}px "
            f"({max_side_ratio:.0%} of the source {_axis_label(max_side)}), "
            f"{max_side_overreach:.1f}x the {max_side_reach_px}px edge-fill reach; a blank canvas "
            "makes the model generate new subject matter instead of smearing the source border."
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
        uses_green_border_lora=uses_green_border_lora,
    )


def _guard_unsafe_edge_fill(*, fill_plan: OutpaintFillPlan) -> None:
    if fill_plan.mode != "edge":
        return
    if fill_plan.edge_fill_within_reach:
        return
    if fill_plan.is_explicit:
        # Explicit user intent wins, but never silently: this configuration is the one measured
        # to return the conditioning canvas back as directional smear.
        print(_unsafe_edge_fill_warning(fill_plan), file=sys.stderr)
        return
    # Unreachable through `auto`, which switches to a blank canvas past the reach. Fail closed
    # rather than run the known-bad configuration if that ever stops being true.
    raise ValueError(
        f"--outpaint-fill auto resolved to edge with {fill_plan.max_side} padding of "
        f"{fill_plan.max_side_padding_px}px, {fill_plan.max_side_overreach:.1f}x the "
        f"{fill_plan.max_side_reach_px}px edge-fill reach. Edge fill smears at that depth. Pass "
        "--outpaint-fill neutral for a blank conditioning canvas, or --outpaint-fill edge to "
        "force the edge canvas anyway."
    )


def _unsafe_edge_fill_warning(fill_plan: OutpaintFillPlan) -> str:
    return (
        f"Warning: --outpaint-fill edge with {fill_plan.max_side} padding of "
        f"{fill_plan.max_side_padding_px}px ({fill_plan.max_side_ratio:.0%} of the source "
        f"{_axis_label(fill_plan.max_side)}) runs {fill_plan.max_side_overreach:.1f}x the "
        f"{fill_plan.max_side_reach_px}px edge-fill reach. Edge fill stretches a border strip "
        "across the padded region and produces directional smear past its reach. Proceeding as "
        "requested. Use --outpaint-fill neutral for a blank conditioning canvas, or add "
        "--lora-paths fal/flux-2-klein-4B-outpaint-lora for the validated green-canvas route."
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


def _mean_border_color(source: PIL.Image.Image) -> tuple[int, int, int]:
    # Neutral blank canvas for large padding. A mid-gray is the safe generic choice because it is
    # flat -- there is no texture for the model to continue, which is exactly what edge fill gets
    # wrong -- but a fixed gray also plants a hard chroma step along the source boundary, and that
    # step is itself a strong edge the model tends to redraw as a visible seam. The mean of the
    # source's own border ring is just as flat while sitting in the source's color neighbourhood,
    # so it keeps the "nothing to continue" property without the seam. Mid-gray remains the
    # fallback when the source is too small to sample a ring.
    rgb = source.convert("RGB")
    if rgb.width < 2 or rgb.height < 2:
        return FLUX2_OUTPAINT_NEUTRAL_FALLBACK_COLOR
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
        return FLUX2_OUTPAINT_NEUTRAL_FALLBACK_COLOR
    return tuple(max(0, min(255, round(total / sampled_pixels))) for total in totals)  # type: ignore[return-value]


def _uses_green_border_outpaint_lora(args) -> bool:
    # Match the pre-resolution request as well as the resolved path. LoraResolution rewrites
    # "fal/flux-2-klein-4B-outpaint-lora" into a concrete cache file, so the repo-form marker only
    # ever matched by accident, through whichever basename that repo happened to resolve to;
    # a sibling file in the same repo (the comfy-converted weights) silently did not match.
    specs = [
        *(getattr(args, "requested_lora_paths", None) or []),
        *(getattr(args, "lora_paths", None) or []),
    ]
    return any(_spec_matches_green_border_marker(str(spec)) for spec in specs)


def _spec_matches_green_border_marker(spec: str) -> bool:
    normalized = spec.lower()
    # Hugging Face cache directories spell "org/repo" as "models--org--repo"; normalizing the
    # double dash back to a slash lets the repo-form markers match a resolved snapshot path too.
    candidates = (normalized, normalized.replace("--", "/"))
    return any(marker in candidate for candidate in candidates for marker in FLUX2_GREEN_BORDER_OUTPAINT_LORA_MARKERS)


def outpaint_notice(*, fill_plan: OutpaintFillPlan, canvas: OutpaintCanvas) -> str:
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
    if fill_plan.mode != "solid" and _option_was_provided(sys.argv[1:], "--outpaint-fill-color"):
        lines.append(
            f"Outpaint: --outpaint-fill-color only applies to --outpaint-fill solid and is ignored by {fill_plan.mode}."
        )
    return "\n".join(lines)


def _attach_fill_metadata(*, generated_image, fill_plan: OutpaintFillPlan | None) -> None:
    # Recorded next to outpaint_padding so `-C metadata.json` replays the RESOLVED canvas instead
    # of re-running `auto` against whatever source the replay points at.
    if fill_plan is None:
        return
    extra_metadata = dict(getattr(generated_image, "extra_metadata", None) or {})
    extra_metadata.update(
        {
            "outpaint_fill": fill_plan.mode,
            "outpaint_fill_color": list(fill_plan.fill_color) if fill_plan.fill_color is not None else None,
            "outpaint_fill_requested": fill_plan.requested,
            "outpaint_fill_reason": fill_plan.reason,
            "outpaint_max_side_padding": fill_plan.max_side,
            "outpaint_max_side_padding_px": fill_plan.max_side_padding_px,
            "outpaint_max_side_padding_ratio": round(fill_plan.max_side_ratio, 4),
            "outpaint_edge_fill_reach_px": fill_plan.max_side_reach_px,
            "outpaint_edge_fill_overreach": round(fill_plan.max_side_overreach, 4),
        }
    )
    generated_image.extra_metadata = extra_metadata


def _validate_canvas_args(*, parser: CommandLineParser, args, source_image_paths: list[Path]) -> None:
    fill_options_provided = _any_option_was_provided(sys.argv[1:], ("--outpaint-fill", "--outpaint-fill-color"))
    if fill_options_provided and args.outpaint_padding is None:
        parser.error(
            "--outpaint-fill and --outpaint-fill-color configure the --outpaint-padding conditioning "
            "canvas. Pass --outpaint-padding, or drop these options."
        )
    if args.mask_path is not None:
        if args.outpaint_padding is not None or args.reframe_padding is not None:
            parser.error("--mask-path cannot be combined with --reframe-padding or --outpaint-padding.")
    if args.outpaint_padding is None and args.reframe_padding is None:
        return
    if args.outpaint_padding is not None and args.reframe_padding is not None:
        parser.error("--reframe-padding and --outpaint-padding are different workflows and cannot be used together.")
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    if len(source_image_paths) != 1:
        parser.error(f"{option_name} requires exactly one --image-paths value.")
    if _any_option_was_provided(sys.argv[1:], ("--width", "--height")):
        parser.error(f"{option_name} computes --width and --height from the source image; do not pass either option.")
    if _option_was_provided(sys.argv[1:], "--canvas-policy"):
        parser.error(f"{option_name} uses --canvas-policy exact-resize; do not pass --canvas-policy.")


def _any_option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    return any(_option_was_provided(argv, option_name) for option_name in option_names)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


def _is_flux2_base_model(model_config: ModelConfig) -> bool:
    model_name_lower = model_config.model_name.lower()
    base_model_lower = (model_config.base_model or "").lower()
    return "klein-base" in model_name_lower or "klein-base" in base_model_lower


if __name__ == "__main__":
    main()
