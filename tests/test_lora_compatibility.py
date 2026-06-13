import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError


def test_lora_model_card_base_model_is_read_from_frontmatter(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "license: apache-2.0",
                "base_model: Qwen/Qwen-Image-Edit-2511",
                "---",
                "# Adapter",
            ]
        ),
        encoding="utf-8",
    )

    assert LoRACompatibility._base_models_from_card(readme) == ("Qwen/Qwen-Image-Edit-2511",)


def test_lora_model_card_base_model_list_is_read_from_frontmatter(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "---",
                "base_model:",
                "- black-forest-labs/FLUX.2-klein-4B",
                "- black-forest-labs/FLUX.2-klein-9B",
                "---",
                "# Adapter",
            ]
        ),
        encoding="utf-8",
    )

    assert LoRACompatibility._base_models_from_card(readme) == (
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
    )


def test_flux2_dev_lora_is_rejected_for_flux2_klein(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("black-forest-labs/FLUX.2-dev",)),
    )

    with pytest.raises(LoRAApplicationError, match="targets black-forest-labs/FLUX.2-dev"):
        LoRACompatibility.validate_for_model_config(
            model_config=ModelConfig.flux2_klein_4b(),
            selected_model="flux2-klein-4b",
            lora_paths=["lovis93/Flux-2-Multi-Angles-LoRA-v2:flux-multi-angles-v2-72poses-comfy.safetensors"],
        )


def test_qwen_2511_lora_is_accepted_for_qwen_2511(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("Qwen/Qwen-Image-Edit-2511",)),
    )

    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.from_name("AbstractFramework/qwen-image-edit-2511-8bit"),
        selected_model="AbstractFramework/qwen-image-edit-2511-8bit",
        lora_paths=["fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:multiple-angles.safetensors"],
    )


def test_qwen_2511_lora_is_rejected_for_qwen_2509(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("Qwen/Qwen-Image-Edit-2511",)),
    )

    with pytest.raises(LoRAApplicationError, match="not compatible"):
        LoRACompatibility.validate_for_model_config(
            model_config=ModelConfig.from_name("qwen-image-edit-2509"),
            selected_model="qwen-image-edit-2509",
            lora_paths=["fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:multiple-angles.safetensors"],
        )


def test_ernie_image_lora_is_accepted_for_ernie_turbo(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("baidu/ERNIE-Image",)),
    )

    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.ernie_image_turbo(),
        selected_model="AbstractFramework/ernie-image-turbo-8bit",
        lora_paths=["reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors"],
    )


def test_wan_ti2v_lora_is_accepted_for_ti2v(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("Wan-AI/Wan2.2-TI2V-5B-Diffusers",)),
    )

    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.wan2_2_ti2v_5b(),
        selected_model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        lora_paths=["wouterverweirder/wan_2_2_5B_woven_fabric_02-lora:wan_2_2_5B_woven_fabric_02.safetensors"],
    )


def test_wan_ti2v_lora_is_accepted_for_prepared_q8_package(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("Wan-AI/Wan2.2-TI2V-5B-Diffusers",)),
    )

    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.from_name("AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"),
        selected_model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
        lora_paths=["AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets:HSToric_color_Wan22_5b_LoRA.safetensors"],
    )


def test_wan_ti2v_lora_is_rejected_for_a14b(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(lambda repo_id: ("Wan-AI/Wan2.2-TI2V-5B-Diffusers",)),
    )

    with pytest.raises(LoRAApplicationError, match="not compatible"):
        LoRACompatibility.validate_for_model_config(
            model_config=ModelConfig.wan2_2_t2v_a14b(),
            selected_model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            lora_paths=["wouterverweirder/wan_2_2_5B_woven_fabric_02-lora:wan_2_2_5B_woven_fabric_02.safetensors"],
        )


def test_wan_lightning_i2v_subpath_is_accepted_for_i2v_even_when_repo_lists_multiple_base_models(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(
            lambda repo_id: (
                "Wan-AI/Wan2.2-T2V-A14B",
                "Wan-AI/Wan2.2-I2V-A14B",
                "Wan-AI/Wan2.2-TI2V-5B",
            )
        ),
    )

    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.from_name("AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"),
        selected_model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
        lora_paths=[
            "lightx2v/Wan2.2-Lightning:Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
        ],
    )


def test_wan_lightning_t2v_subpath_is_rejected_for_i2v_even_when_repo_lists_i2v_too(monkeypatch):
    monkeypatch.setattr(
        LoRACompatibility,
        "_cached_base_models",
        staticmethod(
            lambda repo_id: (
                "Wan-AI/Wan2.2-T2V-A14B",
                "Wan-AI/Wan2.2-I2V-A14B",
                "Wan-AI/Wan2.2-TI2V-5B",
            )
        ),
    )

    with pytest.raises(LoRAApplicationError, match="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"):
        LoRACompatibility.validate_for_model_config(
            model_config=ModelConfig.from_name("AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"),
            selected_model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
            lora_paths=[
                "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
            ],
        )


def test_local_lora_without_card_metadata_is_left_to_loader_shape_checks(tmp_path):
    lora_path = tmp_path / "adapter.safetensors"
    lora_path.touch()

    LoRACompatibility.validate_for_model_config(
        model_config=ModelConfig.flux2_klein_4b(),
        selected_model="flux2-klein-4b",
        lora_paths=[str(lora_path)],
    )
