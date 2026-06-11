from PIL import Image

from mflux.lora_validation_registry import QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationResult, LoRAFileReport
from mflux.utils.generated_image import GeneratedImage


def test_generated_image_records_lora_application_metadata():
    report = LoRAFileReport(
        requested_path="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:adapter.safetensors",
        resolved_path="/tmp/adapter.safetensors",
        scale=0.9,
        role="transformer",
        total_key_count=1680,
        matched_key_count=1680,
        unmatched_key_count=0,
        applied_target_count=560,
    )
    result = LoRAApplicationResult(
        resolved_paths=["/tmp/adapter.safetensors"],
        resolved_scales=[0.9],
        reports=(report,),
    )
    image = GeneratedImage(
        image=Image.new("RGB", (8, 8), color="white"),
        model_config=ModelConfig.from_name("AbstractFramework/qwen-image-edit-2511-8bit"),
        seed=42,
        prompt="test",
        steps=4,
        guidance=4.0,
        precision=ModelConfig.from_name("AbstractFramework/qwen-image-edit-2511-8bit").precision,
        quantization=8,
        generation_time=1.23,
        lora_paths=["/tmp/adapter.safetensors"],
        lora_scales=[0.9],
        extra_metadata={
            **result.extra_metadata(),
            "lora_validation_profile": QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID,
        },
    )

    metadata = image._get_metadata()

    assert metadata["lora_applied_file_count"] == 1
    assert metadata["lora_applied_target_count"] == 560
    assert metadata["lora_validation_profile"] == QWEN2511_Q8_SINGLE_EDIT_MULTI_ANGLE_PROFILE_ID
    assert metadata["lora_application_reports"][0]["matched_key_count"] == 1680
    assert metadata["lora_application_reports"][0]["requested_path"].startswith("fal/Qwen-Image-Edit-2511")
