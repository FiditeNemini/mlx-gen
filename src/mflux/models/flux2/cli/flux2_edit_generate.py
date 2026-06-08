import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.variants import Flux2KleinEdit
from mflux.utils.dimension_resolver import CANVAS_POLICY_EXACT_RESIZE
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.outpaint_util import OutpaintCanvas, OutpaintUtil
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Flux2 Klein Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
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
            "Expand one source image by CSS-style top,right,bottom,left padding and restore "
            "the source region with a feathered mask after generation."
        ),
    )
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_output_arguments()
    args = parser.parse_args()

    if getattr(args, "negative_prompt", ""):
        parser.error("--negative-prompt is not supported for FLUX.2. Focus on describing what you want.")
    source_image_paths = [Path(p) for p in args.image_paths]
    _validate_canvas_args(parser=parser, args=args, source_image_paths=source_image_paths)

    model_name = args.model or "flux2-klein-4b"
    model_config = ModelConfig.from_name(model_name=model_name, base_model=args.base_model)

    if args.guidance is None:
        args.guidance = 1.0
    model_name_lower = model_config.model_name.lower()
    base_model_lower = (model_config.base_model or "").lower()
    is_flux2 = any(identifier in model_name_lower or identifier in base_model_lower for identifier in ("flux.2", "flux2"))
    if args.guidance != 1.0 and not is_flux2:
        parser.error("--guidance is only supported for FLUX.2 models. Use --guidance 1.0.")

    model = Flux2KleinEdit(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=Flux2LatentCreator,
    )

    try:
        with TemporaryDirectory(prefix="mlxgen-outpaint-") as temporary_directory:
            try:
                image_paths, outpaint_canvas, reframe_canvas = _resolve_image_paths(
                    args=args,
                    source_image_paths=source_image_paths,
                    temporary_directory=Path(temporary_directory),
                )
            except ValueError as exc:
                parser.error(str(exc))

            try:
                for seed in args.seed:
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
                        )
                        image.image_path = source_image_paths[0]
                        image.image_paths = source_image_paths
                        OutpaintUtil.attach_metadata(
                            generated_image=image,
                            canvas=outpaint_canvas,
                            padding_value=args.outpaint_padding,
                        )
                    if reframe_canvas is not None:
                        image.image_path = source_image_paths[0]
                        image.image_paths = source_image_paths
                        OutpaintUtil.attach_reframe_metadata(
                            generated_image=image,
                            canvas=reframe_canvas,
                            padding_value=args.reframe_padding,
                        )
                    image.save(
                        path=args.output.format(seed=seed),
                        export_json_metadata=args.metadata,
                        overwrite=args.replace,
                    )
            except (StopImageGenerationException, PromptFileReadError) as exc:
                print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _resolve_image_paths(
    *,
    args,
    source_image_paths: list[Path],
    temporary_directory: Path,
) -> tuple[list[Path], OutpaintCanvas | None, OutpaintCanvas | None]:
    if args.outpaint_padding is None and args.reframe_padding is None:
        return source_image_paths, None, None
    padding_value = args.outpaint_padding or args.reframe_padding
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    canvas_name = "outpaint_canvas.png" if args.outpaint_padding is not None else "reframe_canvas.png"
    if len(source_image_paths) != 1:
        raise ValueError(f"{option_name} requires exactly one --image-paths value.")

    canvas = OutpaintUtil.create_expanded_canvas(
        source_path=source_image_paths[0],
        padding_value=padding_value,
        output_path=temporary_directory / canvas_name,
        option_name=option_name,
    )
    args.width = canvas.target_width
    args.height = canvas.target_height
    args.canvas_policy = CANVAS_POLICY_EXACT_RESIZE
    if args.outpaint_padding is not None:
        return [canvas.canvas_path], canvas, None
    return [canvas.canvas_path], None, canvas


def _validate_canvas_args(*, parser: CommandLineParser, args, source_image_paths: list[Path]) -> None:
    if args.outpaint_padding is None and args.reframe_padding is None:
        return
    if args.outpaint_padding is not None and args.reframe_padding is not None:
        parser.error("--reframe-padding and --outpaint-padding are different workflows and cannot be used together.")
    option_name = "--outpaint-padding" if args.outpaint_padding is not None else "--reframe-padding"
    if len(source_image_paths) != 1:
        parser.error(f"{option_name} requires exactly one --image-paths value.")
    if _any_option_was_provided(sys.argv[1:], ("--width", "--height")):
        parser.error(
            f"{option_name} computes --width and --height from the source image; "
            "do not pass either option."
        )
    if _option_was_provided(sys.argv[1:], "--canvas-policy"):
        parser.error(f"{option_name} uses --canvas-policy exact-resize; do not pass --canvas-policy.")


def _any_option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    return any(_option_was_provided(argv, option_name) for option_name in option_names)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


if __name__ == "__main__":
    main()
