import sys
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit as _QwenImageEditImplementation
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit with image conditioning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to one or more init images. For single image editing, provide one path. For multiple image editing, provide multiple paths.")  # fmt: off
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    try:
        model_config = ModelConfig.from_name(args.model or "qwen-image-edit", base_model=args.base_model)
    except ModelConfigError:
        if args.model_path is None:
            raise
        model_config = ModelConfig.from_name(args.base_model or "qwen-image-edit")
    image_paths = [str(p) for p in args.image_paths]
    if len(image_paths) > 1 and not _QwenImageEditImplementation._is_edit_plus_model_config(
        model_config=model_config,
        image_paths=image_paths,
    ):
        parser.error(
            "Multiple Qwen edit reference images require an Edit-Plus model, such as "
            "qwen-image-edit-2509 or qwen-image-edit-2511."
        )
    if not _option_was_provided(sys.argv[1:], "--scheduler"):
        args.scheduler = "flow_match_euler_discrete"
    if args.guidance is None:
        if _QwenImageEditImplementation._is_edit_plus_model_config(model_config=model_config, image_paths=image_paths):
            args.guidance = 4.0
        else:
            args.guidance = 4.0
    if not _option_was_provided(sys.argv[1:], "--steps") and _QwenImageEditImplementation._is_edit_plus_model_config(
        model_config=model_config,
        image_paths=image_paths,
    ):
        args.steps = 40

    qwen = QwenImageEdit(
        quantize=args.quantize,
        model_config=model_config,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=QwenLatentCreator,
    )

    try:
        for seed in args.seed:
            # 4. Generate an image for each seed value
            image = qwen.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                negative_prompt=_read_negative_prompt(args),
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                image_path=image_paths[0],  # Use first image for metadata
                image_paths=image_paths,
                num_inference_steps=args.steps,
                scheduler=args.scheduler,
                canvas_policy=args.canvas_policy,
            )

            # 5. Save the image
            output_path = Path(args.output.format(seed=seed))
            image.save(path=output_path, export_json_metadata=args.metadata, overwrite=args.replace)

    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


def _read_negative_prompt(args) -> str | None:
    if _any_option_was_provided(sys.argv[1:], ("--negative-prompt", "--negative")):
        return PromptUtil.read_negative_prompt(args)
    return None


def _any_option_was_provided(argv: list[str], option_names: tuple[str, ...]) -> bool:
    return any(_option_was_provided(argv, option_name) for option_name in option_names)


def _option_was_provided(argv: list[str], option_name: str) -> bool:
    for token in argv:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


if __name__ == "__main__":
    main()
