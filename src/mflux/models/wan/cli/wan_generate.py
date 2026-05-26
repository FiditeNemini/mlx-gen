import argparse
import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import boolean_flag_value
from mflux.models.common.config import ModelConfig
from mflux.models.wan.variants import Wan2_2_TI2V, WanProgressEvent
from mflux.utils.exceptions import ModelConfigError, PromptFileReadError
from mflux.utils.prompt_util import PromptUtil

WAN_DEFAULT_WIDTH = Wan2_2_TI2V.RECOMMENDED_WIDTH
WAN_DEFAULT_HEIGHT = Wan2_2_TI2V.RECOMMENDED_HEIGHT
WAN_DEFAULT_FRAMES = Wan2_2_TI2V.RECOMMENDED_FRAMES
WAN_DEFAULT_FPS = Wan2_2_TI2V.RECOMMENDED_FPS


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _apply_metadata_defaults(args)
    _validate_args(parser, args)
    _apply_seed_defaults(args)

    model_config, model_path = _resolve_model(args.model)
    model = Wan2_2_TI2V(
        model_config=model_config,
        quantize=args.quantize,
        model_path=model_path,
    )

    try:
        for seed in args.seed:
            progress = _WanCliProgress(enabled=args.progress)
            try:
                video = model.generate_video(
                    seed=seed,
                    prompt=PromptUtil.read_prompt(args),
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames,
                    fps=args.fps,
                    guidance=args.guidance,
                    num_inference_steps=args.steps,
                    negative_prompt=args.negative_prompt,
                    image_path=args.image_path,
                    max_sequence_length=args.max_sequence_length,
                    progress_callback=progress if args.progress else None,
                )
            finally:
                progress.close()
            video.save(
                path=args.output.format(seed=seed),
                export_json_metadata=args.metadata,
                overwrite=args.replace,
            )
    except (PromptFileReadError, FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as exc:
        print(exc)
        raise SystemExit(1) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen-generate-wan",
        description="Generate a video using Wan2.2 TI2V.",
    )
    parser.add_argument("--model", "-m", required=True, help="Wan model alias, Hugging Face repo, or local path.")
    parser.add_argument("--image-path", default=None, help="Input image for Wan first-frame image-to-video.")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", type=str, help="Text prompt for video generation.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a text file containing the prompt.")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt used when guidance > 1.")
    parser.add_argument(
        "--width",
        type=int,
        default=WAN_DEFAULT_WIDTH,
        help="Video width. Adjusted down to a multiple of 32.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=WAN_DEFAULT_HEIGHT,
        help="Video height. Adjusted down to a multiple of 32.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=WAN_DEFAULT_FRAMES,
        help="Number of frames. Adjusted to 4n + 1.",
    )
    parser.add_argument("--fps", type=int, default=WAN_DEFAULT_FPS, help="Output video frame rate.")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", "-s", type=int, default=None, nargs="+", help="One or more random seeds.")
    parser.add_argument("--auto-seeds", type=int, default=-1, help="Generate N random seeds.")
    parser.add_argument("--quantize", "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512, help="UMT5 prompt token length.")
    parser.add_argument("--metadata", action="store_true", help="Export video metadata as JSON.")
    parser.add_argument("--output", type=str, default="video.mp4", help='Output path. Default is "video.mp4".')
    parser.add_argument(
        "--progress",
        type=boolean_flag_value,
        nargs="?",
        const=True,
        default=True,
        help="Show video frame progress. Default is true.",
    )
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.add_argument(
        "--replace",
        type=boolean_flag_value,
        nargs="?",
        const=True,
        default=True,
        help="Replace the target output when it already exists. Default is true.",
    )
    parser.add_argument("--no-replace", action="store_false", dest="replace")
    parser.add_argument("--config-from-metadata", "-C", type=Path, default=None)
    parser.add_argument("--battery-percentage-stop-limit", "-B", type=int, default=ui_defaults.BATTERY_PERCENTAGE_STOP_LIMIT)
    parser.add_argument("--low-ram", action="store_true")
    parser.add_argument("--mlx-cache-limit-gb", type=float, default=None)
    return parser


def _apply_metadata_defaults(args: argparse.Namespace) -> None:
    if args.config_from_metadata is None:
        return
    metadata = json.loads(args.config_from_metadata.read_text())
    if args.prompt is None and args.prompt_file is None:
        args.prompt = metadata.get("prompt")
    if args.negative_prompt == "":
        args.negative_prompt = metadata.get("negative_prompt") or ""
    if args.seed is None and metadata.get("seed") is not None:
        args.seed = [int(metadata["seed"])]
    if args.quantize is None:
        args.quantize = metadata.get("quantize")
    if args.image_path is None and metadata.get("image_path") is not None:
        args.image_path = metadata.get("image_path")
    for name in ("width", "height", "frames", "fps", "steps", "guidance"):
        value = metadata.get(name)
        if value is not None and getattr(args, name) == _parser().get_default(name):
            setattr(args, name, value)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt-file is required, or provide prompt in --config-from-metadata.")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero.")
    if args.steps <= 0:
        parser.error("--steps must be greater than zero.")
    if args.max_sequence_length <= 0:
        parser.error("--max-sequence-length must be greater than zero.")
    if args.image_path is not None and not Path(args.image_path).exists():
        parser.error(f"--image-path does not exist: {args.image_path}")


def _apply_seed_defaults(args: argparse.Namespace) -> None:
    if args.seed is None and args.auto_seeds > 0:
        args.seed = random.sample(range(int(1e7) + 1), args.auto_seeds)
    if args.seed is None:
        args.seed = [int(time.time())]
    if len(args.seed) > 1:
        output_path = Path(args.output)
        args.output = str(output_path.with_stem(output_path.stem + "_seed_{seed}"))


def _resolve_model(model: str) -> tuple[ModelConfig, str | None]:
    try:
        model_config = ModelConfig.from_name(model)
    except ModelConfigError:
        return ModelConfig.wan2_2_ti2v_5b(), model
    model_path = model if model_config.base_model is not None else None
    return model_config, model_path


class _WanCliProgress:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._bar: tqdm | None = None
        self._last_frame = 0

    def __call__(self, event: WanProgressEvent) -> None:
        if not self.enabled:
            return
        if self._bar is None:
            self._bar = tqdm(total=event.total_frames, desc="Generating video", unit="frame")
        delta = max(0, event.frame - self._last_frame)
        if delta:
            self._bar.update(delta)
            self._last_frame = event.frame
        self._bar.set_postfix_str(f"{event.phase} step {event.step}/{event.total_steps}")
        if event.phase == "complete":
            self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


if __name__ == "__main__":
    main()
