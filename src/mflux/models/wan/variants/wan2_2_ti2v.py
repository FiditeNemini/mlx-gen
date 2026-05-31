import gc
import html
import io
import re
import shutil
import time
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.wan.latent_creator import WanTimestepPolicy
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.scheduler import WanUniPCMultistepScheduler
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights import WanWeightDefinition
from mflux.utils.exceptions import ModelConfigError
from mflux.utils.generated_video import GeneratedVideo
from mflux.utils.image_util import ImageUtil
from mflux.utils.video_util import VideoUtil


@dataclass(frozen=True)
class WanProgressEvent:
    phase: str
    frame: int
    total_frames: int
    step: int
    total_steps: int

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, max(0.0, self.frame / self.total_frames))


_GUIDANCE_2_UNSET = object()


class Wan2_2_TI2V(nn.Module):
    RECOMMENDED_WIDTH = 1280
    RECOMMENDED_HEIGHT = 704
    RECOMMENDED_AREA = RECOMMENDED_WIDTH * RECOMMENDED_HEIGHT
    RECOMMENDED_FRAMES = 121
    RECOMMENDED_STEPS = 50
    RECOMMENDED_FPS = 24

    transformer: WanTransformer
    transformer_2: WanTransformer | None
    vae: Wan2_2_VAE

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        model_config = self._resolve_model_config(model_path=model_path, model_config=model_config)
        WanInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def generate_video(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 50,
        height: int = RECOMMENDED_HEIGHT,
        width: int = RECOMMENDED_WIDTH,
        num_frames: int = RECOMMENDED_FRAMES,
        fps: int = RECOMMENDED_FPS,
        guidance: float | None = None,
        guidance_2: float | None | object = _GUIDANCE_2_UNSET,
        negative_prompt: str | None = "",
        image_path: Path | str | None = None,
        max_sequence_length: int = 512,
        progress_callback: Callable[[WanProgressEvent], None] | None = None,
    ) -> GeneratedVideo:
        start_time = time.time()
        if (
            guidance_2 is not _GUIDANCE_2_UNSET
            and guidance_2 is not None
            and self._wan_config("boundary_ratio", None) is None
        ):
            raise ValueError("guidance_2 is only supported for Wan models with two-transformer boundary routing.")
        height, width = self._validated_spatial_size(height=height, width=width)
        num_frames = self._validated_frame_count(num_frames)
        is_image_to_video = image_path is not None
        if image_path is not None and not self._supports_image_to_video():
            raise ValueError(f"{self.model_config.model_name} does not support image-to-video input.")
        guidance, guidance_2 = self._resolve_guidance_pair(guidance=guidance, guidance_2=guidance_2)
        negative_prompt = self._default_negative_prompt() if not negative_prompt else negative_prompt
        self._validate_runtime_contract(is_image_to_video=is_image_to_video)
        self._emit_progress(
            progress_callback,
            phase="start",
            frame=0,
            total_frames=num_frames,
            step=0,
            total_steps=num_inference_steps,
        )
        batch_size = 1

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=guidance > 1.0,
            max_sequence_length=max_sequence_length,
        )

        scheduler = WanUniPCMultistepScheduler(flow_shift=float(self._wan_config("flow_shift", 5.0)))
        scheduler.set_timesteps(num_inference_steps)
        boundary_timestep = self._boundary_timestep(scheduler)
        latents = self.prepare_latents(
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        first_frame_mask = None
        condition = None
        if is_image_to_video:
            if self._uses_expanded_timesteps():
                first_frame_mask = WanTimestepPolicy.first_frame_mask(latent_shape=latents.shape)
                condition = self._encode_first_frame_condition(
                    image_path=image_path,
                    height=height,
                    width=width,
                )
            else:
                condition = self._encode_video_condition(
                    image_path=image_path,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    batch_size=batch_size,
                )

        total_steps = len(scheduler.timesteps)
        for step_index, timestep in enumerate(scheduler.timesteps.tolist()):
            current_transformer, current_guidance = self._select_transformer_and_guidance(
                timestep=timestep,
                boundary_timestep=boundary_timestep,
                guidance=guidance,
                guidance_2=guidance_2,
            )
            if first_frame_mask is not None and condition is not None:
                latent_model_input = WanTimestepPolicy.apply_first_frame_condition(
                    latents=latents,
                    condition=condition,
                    first_frame_mask=first_frame_mask,
                ).astype(ModelConfig.precision)
                expanded_timestep = WanTimestepPolicy.expand_from_mask(
                    mask=first_frame_mask,
                    batch_size=batch_size,
                    timestep=timestep,
                    patch_size=current_transformer.patch_size,
                )
            elif is_image_to_video and condition is not None:
                latent_model_input = mx.concatenate([latents, condition], axis=1).astype(ModelConfig.precision)
                expanded_timestep = self._batch_timestep(batch_size=batch_size, timestep=timestep)
            else:
                latent_model_input = latents.astype(ModelConfig.precision)
                if self._uses_expanded_timesteps():
                    expanded_timestep = WanTimestepPolicy.expand_for_text_to_video(
                        latent_shape=latents.shape,
                        timestep=timestep,
                        patch_size=current_transformer.patch_size,
                    )
                else:
                    expanded_timestep = self._batch_timestep(batch_size=batch_size, timestep=timestep)
            noise_pred = current_transformer(
                hidden_states=latent_model_input,
                timestep=expanded_timestep,
                encoder_hidden_states=prompt_embeds,
            )
            if negative_prompt_embeds is not None:
                noise_uncond = current_transformer(
                    hidden_states=latent_model_input,
                    timestep=expanded_timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                )
                noise_pred = noise_uncond + current_guidance * (noise_pred - noise_uncond)

            latents = scheduler.step(noise_pred.astype(mx.float32), timestep, latents, return_dict=False)[0]
            mx.eval(latents)
            self._emit_progress(
                progress_callback,
                phase="denoise",
                frame=self._progress_frame_for_step(
                    step_index=step_index,
                    total_steps=total_steps,
                    total_frames=num_frames,
                ),
                total_frames=num_frames,
                step=step_index + 1,
                total_steps=total_steps,
            )

        if first_frame_mask is not None and condition is not None:
            latents = WanTimestepPolicy.apply_first_frame_condition(
                latents=latents,
                condition=condition,
                first_frame_mask=first_frame_mask,
            )
        del prompt_embeds, negative_prompt_embeds, scheduler
        del latent_model_input, expanded_timestep, noise_pred
        if "noise_uncond" in locals():
            del noise_uncond
        gc.collect()
        mx.clear_cache()
        decoded = self.vae.decode_normalized_latents(latents.astype(ModelConfig.precision))
        mx.eval(decoded)
        mx.clear_cache()
        video = VideoUtil.to_video(
            decoded_latents=decoded,
            fps=fps,
            model_config=self.model_config,
            seed=seed,
            prompt=prompt,
            steps=num_inference_steps,
            guidance=guidance,
            guidance_2=guidance_2,
            quantization=self.bits,
            generation_time=time.time() - start_time,
            task="image-to-video" if is_image_to_video else "text-to-video",
            image_path=image_path,
            negative_prompt=negative_prompt,
        )
        self._emit_progress(
            progress_callback,
            phase="complete",
            frame=num_frames,
            total_frames=num_frames,
            step=total_steps,
            total_steps=total_steps,
        )
        return video

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str | None,
        do_classifier_free_guidance: bool,
        max_sequence_length: int = 512,
    ) -> tuple[mx.array, mx.array | None]:
        prompts = [prompt]
        if not do_classifier_free_guidance:
            return self._get_t5_prompt_embeds(prompts, max_sequence_length=max_sequence_length), None
        prompts.append(negative_prompt or "")
        embeds = self._get_t5_prompt_embeds(prompts, max_sequence_length=max_sequence_length)
        return embeds[0:1], embeds[1:2]

    def prepare_latents(
        self,
        seed: int,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> mx.array:
        mx.random.seed(seed)
        latent_frames = (num_frames - 1) // self.vae.temporal_scale + 1
        shape = (
            batch_size,
            self.vae.z_dim,
            latent_frames,
            height // self.vae.spatial_scale,
            width // self.vae.spatial_scale,
        )
        return mx.random.normal(shape, dtype=mx.float32)

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=getattr(self, "weight_definition", WanWeightDefinition.for_config(self.model_config)),
        )
        self._copy_runtime_assets(base_path)

    def _get_t5_prompt_embeds(self, prompts: list[str], max_sequence_length: int) -> mx.array:
        try:
            import torch
            from transformers import UMT5EncoderModel
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError("Wan prompt encoding requires torch and transformers.") from exc

        text_encoder_path = self.root_path / "text_encoder"
        if not text_encoder_path.exists():
            raise FileNotFoundError(
                f"Wan text encoder files were not found in {text_encoder_path}. "
                f"Run `mlxgen download --model {self.model_config.model_name}` first."
            )

        cleaned = [self._prompt_clean(prompt) for prompt in prompts]
        tokenizer = self.tokenizers["wan"].tokenizer
        text_inputs = tokenizer(
            cleaned,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        transformers_verbosity = transformers_logging.get_verbosity()
        try:
            transformers_logging.set_verbosity_error()
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                text_encoder = UMT5EncoderModel.from_pretrained(
                    text_encoder_path,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
        finally:
            transformers_logging.set_verbosity(transformers_verbosity)
        text_encoder.eval()
        if hasattr(text_encoder, "shared") and hasattr(text_encoder, "encoder"):
            text_encoder.encoder.embed_tokens = text_encoder.shared

        with torch.no_grad():
            output = text_encoder(
                text_inputs.input_ids,
                text_inputs.attention_mask,
            ).last_hidden_state
        seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1).long()
        padded = torch.stack(
            [
                torch.cat(
                    [
                        hidden[:seq_len_int],
                        hidden.new_zeros(max_sequence_length - seq_len_int, hidden.size(1)),
                    ]
                )
                for hidden, seq_len in zip(output, seq_lens)
                for seq_len_int in [int(seq_len.item())]
            ],
            dim=0,
        )
        embeds = mx.array(padded.float().cpu().numpy()).astype(ModelConfig.precision)
        del text_encoder
        gc.collect()
        return embeds

    def _encode_first_frame_condition(self, image_path: Path | str | None, height: int, width: int) -> mx.array:
        if image_path is None:
            raise ValueError("Wan image-to-video requires image_path.")
        image = ImageUtil.scale_to_dimensions(ImageUtil.load_image(image_path), target_width=width, target_height=height)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[None, ...]
        image_mx = mx.array(image_np)
        image_mx = mx.transpose(image_mx, (0, 3, 1, 2))
        image_mx = ImageUtil._normalize(image_mx)
        condition = self.vae.encode_normalized(image_mx.astype(ModelConfig.precision))
        mx.eval(condition)
        return condition.astype(mx.float32)

    def _encode_video_condition(
        self,
        image_path: Path | str | None,
        height: int,
        width: int,
        num_frames: int,
        batch_size: int,
    ) -> mx.array:
        if image_path is None:
            raise ValueError("Wan image-to-video requires image_path.")
        image = ImageUtil.scale_to_dimensions(ImageUtil.load_image(image_path), target_width=width, target_height=height)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_mx = mx.array(image_np[None, ...])
        image_mx = mx.transpose(image_mx, (0, 3, 1, 2))
        image_mx = ImageUtil._normalize(image_mx)
        first_frame = image_mx[:, :, None, :, :]
        zero_frames = mx.zeros((batch_size, first_frame.shape[1], num_frames - 1, height, width), dtype=first_frame.dtype)
        video_condition = mx.concatenate([first_frame, zero_frames], axis=2).astype(ModelConfig.precision)
        latent_condition = self.vae.encode_normalized(video_condition).astype(mx.float32)
        latent_frames = latent_condition.shape[2]
        latent_height = latent_condition.shape[3]
        latent_width = latent_condition.shape[4]
        mask_np = np.ones((batch_size, 1, num_frames, latent_height, latent_width), dtype=np.float32)
        mask_np[:, :, 1:] = 0
        mask = mx.array(mask_np)
        first_frame_mask = mx.repeat(mask[:, :, 0:1], self.vae.temporal_scale, axis=2)
        mask = mx.concatenate([first_frame_mask, mask[:, :, 1:]], axis=2)
        mask = mx.reshape(mask, (batch_size, -1, self.vae.temporal_scale, latent_height, latent_width))
        mask = mx.transpose(mask, (0, 2, 1, 3, 4))
        condition = mx.concatenate([mask[:, :, :latent_frames], latent_condition], axis=1)
        mx.eval(condition)
        return condition.astype(mx.float32)

    def _copy_runtime_assets(self, base_path: str) -> None:
        target = Path(base_path)
        for subdir in ("text_encoder", "scheduler"):
            source = self.root_path / subdir
            if source.exists():
                shutil.copytree(source, target / subdir, dirs_exist_ok=True)
        model_index = self.root_path / "model_index.json"
        if model_index.exists():
            shutil.copy2(model_index, target / "model_index.json")

    def _validated_spatial_size(self, height: int, width: int) -> tuple[int, int]:
        multiple_h = self.vae.spatial_scale * self.transformer.patch_size[1]
        multiple_w = self.vae.spatial_scale * self.transformer.patch_size[2]
        calc_height = height // multiple_h * multiple_h
        calc_width = width // multiple_w * multiple_w
        if calc_height <= 0 or calc_width <= 0:
            raise ValueError(f"Wan height and width must be at least ({multiple_h}, {multiple_w})px.")
        if (height, width) != (calc_height, calc_width):
            print(
                "`height` and `width` must be multiples of "
                f"({multiple_h}, {multiple_w}) for Wan patchification. "
                f"Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
        return calc_height, calc_width

    def _validated_frame_count(self, num_frames: int) -> int:
        if num_frames < 1:
            raise ValueError("Wan num_frames must be at least 1.")
        if num_frames % self.vae.temporal_scale != 1:
            adjusted = num_frames // self.vae.temporal_scale * self.vae.temporal_scale + 1
            print(
                f"`frames - 1` must be divisible by {self.vae.temporal_scale}. "
                f"Adjusting {num_frames} -> {adjusted}."
            )
            num_frames = adjusted
        return max(num_frames, 1)

    def _validate_runtime_contract(self, *, is_image_to_video: bool) -> None:
        task = self._wan_config("task", "text-image-to-video")
        if task == "image-to-video" and not is_image_to_video:
            raise ValueError(f"{self.model_config.model_name} requires image-to-video input.")

        expected_config = self.model_config.transformer_overrides
        expected_vae_config = expected_config.get("vae_config", {})
        expected_transformer_channels = int(expected_config.get("in_channels", self.transformer.in_channels))
        expected_output_channels = int(expected_config.get("out_channels", self.transformer.out_channels))
        expected_vae_channels = int(expected_vae_config.get("z_dim", self.vae.z_dim))
        expected_transformer_2 = bool(expected_config.get("has_transformer_2", False))

        if int(self.transformer.in_channels) != expected_transformer_channels:
            self._raise_runtime_contract_mismatch(
                "transformer.in_channels",
                actual=int(self.transformer.in_channels),
                expected=expected_transformer_channels,
            )
        if int(self.transformer.out_channels) != expected_output_channels:
            self._raise_runtime_contract_mismatch(
                "transformer.out_channels",
                actual=int(self.transformer.out_channels),
                expected=expected_output_channels,
            )
        if int(self.vae.z_dim) != expected_vae_channels:
            self._raise_runtime_contract_mismatch(
                "vae.z_dim",
                actual=int(self.vae.z_dim),
                expected=expected_vae_channels,
            )
        if (self.transformer_2 is not None) != expected_transformer_2:
            self._raise_runtime_contract_mismatch(
                "transformer_2",
                actual="present" if self.transformer_2 is not None else "absent",
                expected="present" if expected_transformer_2 else "absent",
            )

        transformer_channels = int(self.transformer.in_channels)
        expected_channels = int(self.vae.z_dim)
        if is_image_to_video and not self._uses_expanded_timesteps():
            expected_channels += 20
        if transformer_channels != expected_channels:
            raise ValueError(
                "Wan runtime config mismatch: "
                f"{self.model_config.model_name} transformer expects {transformer_channels} input channels, "
                f"but the selected VAE/input path provides {expected_channels}. "
                "This usually means the model weights were paired with the wrong Wan config; refusing to continue."
            )

    @staticmethod
    def _resolve_model_config(model_path: str | None, model_config: ModelConfig | None) -> ModelConfig:
        if model_config is not None:
            return model_config
        if model_path is None:
            return ModelConfig.wan2_2_ti2v_5b()
        try:
            return ModelConfig.from_name(model_path)
        except ModelConfigError as exc:
            raise ValueError(
                f"Cannot infer a supported Wan model config from {model_path}. "
                "Pass model_config explicitly; MLX-Gen will not fall back to another Wan architecture."
            ) from exc

    def _raise_runtime_contract_mismatch(self, key: str, actual, expected) -> None:
        raise ValueError(
            "Wan runtime config mismatch: "
            f"{self.model_config.model_name} has {key}={actual!r}, but the selected config expects {expected!r}. "
            "Pass the exact Wan model/config that matches these weights; MLX-Gen will not fall back silently."
        )

    @staticmethod
    def _progress_frame_for_step(step_index: int, total_steps: int, total_frames: int) -> int:
        if total_steps <= 0 or total_frames <= 1:
            return 0
        return min(total_frames - 1, int(((step_index + 1) * (total_frames - 1)) / total_steps))

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[WanProgressEvent], None] | None,
        *,
        phase: str,
        frame: int,
        total_frames: int,
        step: int,
        total_steps: int,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            WanProgressEvent(
                phase=phase,
                frame=frame,
                total_frames=total_frames,
                step=step,
                total_steps=total_steps,
            )
        )

    def _select_transformer_and_guidance(
        self,
        *,
        timestep: int,
        boundary_timestep: float | None,
        guidance: float,
        guidance_2: float | None,
    ) -> tuple[WanTransformer, float]:
        if boundary_timestep is None or timestep >= boundary_timestep:
            return self.transformer, guidance
        if self.transformer_2 is None:
            raise ValueError("Wan model config requested low-noise routing but transformer_2 is missing.")
        if guidance_2 is None:
            raise ValueError("Wan low-noise routing requires guidance_2.")
        return self.transformer_2, guidance_2

    def _boundary_timestep(self, scheduler: WanUniPCMultistepScheduler) -> float | None:
        boundary_ratio = self._wan_config("boundary_ratio", None)
        if boundary_ratio is None:
            return None
        return float(boundary_ratio) * scheduler.num_train_timesteps

    def _uses_expanded_timesteps(self) -> bool:
        return bool(self._wan_config("expand_timesteps", True))

    def _supports_image_to_video(self) -> bool:
        return bool(self._wan_config("supports_image_to_video", True))

    def _default_guidance(self) -> float:
        return float(self._wan_config("default_guidance", 5.0))

    def _default_guidance_2(self) -> float | None:
        value = self._wan_config("default_guidance_2", None)
        return None if value is None else float(value)

    def _resolve_guidance_pair(
        self, guidance: float | None, guidance_2: float | None | object
    ) -> tuple[float, float | None]:
        guidance_was_default = guidance is None
        resolved_guidance = self._default_guidance() if guidance is None else float(guidance)
        if guidance_2 is _GUIDANCE_2_UNSET:
            default_guidance_2 = self._default_guidance_2()
            if default_guidance_2 is not None:
                resolved_guidance_2 = default_guidance_2 if guidance_was_default else resolved_guidance
            else:
                resolved_guidance_2 = None
        else:
            if guidance_2 is None:
                resolved_guidance_2 = (
                    resolved_guidance if self._wan_config("boundary_ratio", None) is not None else None
                )
            else:
                resolved_guidance_2 = float(guidance_2)
        return resolved_guidance, resolved_guidance_2

    def _default_negative_prompt(self) -> str:
        return str(self._wan_config("default_negative_prompt", ""))

    def _wan_config(self, key: str, default):
        return self.model_config.transformer_overrides.get(key, default)

    @staticmethod
    def _batch_timestep(batch_size: int, timestep: int) -> mx.array:
        return mx.full((batch_size,), timestep, dtype=mx.float32)

    @staticmethod
    def _prompt_clean(text: str) -> str:
        try:
            import ftfy
        except ImportError:
            ftfy = None
        if ftfy is not None:
            text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return re.sub(r"\s+", " ", text).strip()
