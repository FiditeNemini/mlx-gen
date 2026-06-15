import mlx.core as mx

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator


class QwenControlNetUtil:
    @staticmethod
    def create_controlnet_condition(
        *,
        vae,
        controlnet_image_path: str,
        height: int,
        width: int,
        tiling_config=None,
    ) -> mx.array:
        encoded = LatentCreator.encode_image(
            vae=vae,
            image_path=controlnet_image_path,
            height=height,
            width=width,
            tiling_config=tiling_config,
        )
        return QwenLatentCreator.pack_latents(
            latents=encoded,
            height=height,
            width=width,
            num_channels_latents=16,
        )
