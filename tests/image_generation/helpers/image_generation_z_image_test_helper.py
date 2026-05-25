import os
from pathlib import Path

import pytest

from mflux.models.common.resolution.lora_resolution import LoraResolution
from mflux.models.z_image import ZImageTurbo
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorZImageTestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        prompt: str,
        steps: int,
        seed: int,
        height: int,
        width: int,
        quantize: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        mismatch_threshold: float | None = None,
    ):
        reference_image_path = ImageGeneratorZImageTestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorZImageTestHelper.resolve_path(output_image_path)

        if lora_paths:
            for lora_path in lora_paths:
                ImageGeneratorZImageTestHelper.require_cached_lora(lora_path)

        try:
            model = ZImageTurbo(
                quantize=quantize,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )

            image = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
            )

            image.save(output_image_path)

            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
            )
        finally:
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)

    @staticmethod
    def require_cached_lora(path: str) -> None:
        try:
            LoraResolution.resolve(path)
        except FileNotFoundError as exc:
            pytest.skip(f"LoRA is not cached. Run the command from the error first: {exc}")

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
