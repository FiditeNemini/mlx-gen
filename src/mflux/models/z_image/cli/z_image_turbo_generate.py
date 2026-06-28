from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.variants.z_image import ZImage
from mflux.task_inference import TaskInferenceError, resolve_generation_plan
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Z-Image Turbo based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_mask_path_argument(
        help_text=(
            "Optional mask image path for native Z-Image Turbo inpaint. White pixels are repainted and black pixels "
            "are preserved."
        ),
    )
    parser.add_output_arguments()
    args = parser.parse_args()

    if args.mask_path is not None and args.image_path is None:
        parser.error("--mask-path requires --image-path.")
    if args.mask_path is not None and args.image_strength is not None:
        parser.error(
            "--image-strength cannot be combined with --mask-path; native Z-Image Turbo inpaint is a separate route."
        )

    model_config = ModelConfig.from_name(args.model or "z-image-turbo", base_model=args.base_model)
    if args.mask_path is not None:
        try:
            resolve_generation_plan(
                model=args.model or "z-image-turbo",
                model_config=model_config,
                image_count=1,
                has_mask=True,
            )
        except TaskInferenceError as exc:
            parser.error(str(exc))

    CallbackManager.apply_runtime_memory_options(args)

    # 1. Load the model
    model = ZImage(
        model_config=model_config,
        quantize=args.quantize,
        model_path=args.model_path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register callbacks
    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=model,
        latent_creator=ZImageLatentCreator,
    )

    try:
        for seed in args.seed:
            # 3. Generate an image for each seed value
            image = model.generate_image(
                seed=seed,
                prompt=PromptUtil.read_prompt(args),
                width=args.width,
                height=args.height,
                guidance=args.guidance,
                image_path=args.image_path,
                mask_path=args.mask_path,
                num_inference_steps=args.steps,
                image_strength=args.image_strength,
                scheduler=args.scheduler,
                negative_prompt=args.negative_prompt,
                canvas_policy=args.canvas_policy,
            )
            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata, overwrite=args.replace)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
