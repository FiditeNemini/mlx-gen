import sys

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.config import ModelConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.variants.controlnet.qwen_image_controlnet import QwenImageControlNet
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.task_inference import TaskInferenceError, resolve_generation_plan
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image using Qwen Image model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # 0. Set model-specific defaults if not provided by user
    if "--scheduler" not in sys.argv:
        args.scheduler = "flow_match_euler_discrete"
    if args.guidance is None:
        args.guidance = ui_defaults.GUIDANCE_SCALE
    if args.controlnet_model is not None and args.controlnet_image_path is None:
        parser.error("--controlnet-model requires --controlnet-image-path.")

    # 1. Load the model
    model_config = ModelConfig.from_name(model_name=args.model or "qwen-image", base_model=args.base_model)
    if args.controlnet_image_path is not None:
        if args.image_path is not None:
            parser.error("--controlnet-image-path cannot be combined with --image-path or latent image-to-image mode.")
        if args.controlnet_model is None:
            parser.error("--controlnet-model is required when --controlnet-image-path is used.")
        try:
            resolve_generation_plan(
                model=args.model,
                model_config=model_config,
                image_count=0,
                has_control_image=True,
            )
        except TaskInferenceError as exc:
            parser.error(str(exc))
        qwen = QwenImageControlNet(
            controlnet_model=args.controlnet_model,
            model_config=model_config,
            quantize=args.quantize,
            model_path=args.model_path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )
    else:
        qwen = QwenImage(
            model_config=model_config,
            quantize=args.quantize,
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
            # 3. Generate an image for each seed value
            if args.controlnet_image_path is not None:
                image = qwen.generate_image(
                    seed=seed,
                    prompt=PromptUtil.read_prompt(args),
                    negative_prompt=PromptUtil.read_negative_prompt(args),
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    scheduler=args.scheduler,
                    controlnet_image_path=args.controlnet_image_path,
                    controlnet_strength=args.controlnet_strength,
                    num_inference_steps=args.steps,
                )
            else:
                image = qwen.generate_image(
                    seed=seed,
                    prompt=PromptUtil.read_prompt(args),
                    negative_prompt=PromptUtil.read_negative_prompt(args),
                    width=args.width,
                    height=args.height,
                    guidance=args.guidance,
                    scheduler=args.scheduler,
                    image_path=args.image_path,
                    num_inference_steps=args.steps,
                    image_strength=args.image_strength,
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
