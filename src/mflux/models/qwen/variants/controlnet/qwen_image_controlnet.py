import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_controlnet import QwenTransformerControlNet
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.variants.controlnet.qwen_controlnet_util import QwenControlNetUtil
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class QwenImageControlNet(nn.Module):
    vae: QwenVAE
    transformer: QwenTransformer
    transformer_controlnet: QwenTransformerControlNet
    text_encoder: QwenTextEncoder

    def __init__(
        self,
        *,
        controlnet_model: str,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig = ModelConfig.qwen_image(),
    ):
        super().__init__()
        QwenImageInitializer.init_controlnet(
            model=self,
            controlnet_model=controlnet_model,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        *,
        seed: int,
        prompt: str,
        controlnet_image_path: str,
        controlnet_strength: float = 0.85,
        num_inference_steps: int = 4,
        height: int | None = None,
        width: int | None = None,
        guidance: float = 4.0,
        scheduler: str = "flow_match_euler_discrete",
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            controlnet_strength=controlnet_strength,
        )
        controlnet_condition = QwenControlNetUtil.create_controlnet_condition(
            vae=self.vae,
            controlnet_image_path=controlnet_image_path,
            height=config.height,
            width=config.width,
            tiling_config=self.tiling_config,
        )
        latents = QwenLatentCreator.create_noise(seed=seed, width=config.width, height=config.height)
        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_cache=self.prompt_cache,
            qwen_tokenizer=self.tokenizers["qwen"],
            qwen_text_encoder=self.text_encoder,
        )
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                latents = config.scheduler.scale_model_input(latents, t)
                controlnet_block_samples = self.transformer_controlnet(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    controlnet_cond=controlnet_condition,
                    conditioning_scale=controlnet_strength,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                )
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_mask,
                    controlnet_block_samples=controlnet_block_samples,
                )
                noise_negative = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    controlnet_block_samples=controlnet_block_samples,
                )
                guided_noise = QwenImageControlNet.compute_guided_noise(noise, noise_negative, config.guidance)
                latents = config.scheduler.step(noise=guided_noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)
        latents = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            controlnet_image_path=controlnet_image_path,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
            extra_metadata={
                **LoRALoader.extra_metadata_for_model(self),
                "controlnet_model": self.controlnet_model,
            },
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=QwenWeightDefinition,
        )

    @staticmethod
    def compute_guided_noise(
        noise: mx.array,
        noise_negative: mx.array,
        guidance: float,
    ) -> mx.array:
        combined = noise_negative + guidance * (noise - noise_negative)
        cond_norm = mx.sqrt(mx.sum(noise * noise, axis=-1, keepdims=True) + 1e-12)
        noise_norm = mx.sqrt(mx.sum(combined * combined, axis=-1, keepdims=True) + 1e-12)
        return combined * (cond_norm / noise_norm)
