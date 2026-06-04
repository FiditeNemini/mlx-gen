import pytest

import mlxgen
from mflux.task_inference import (
    MODE_EDIT_REFERENCE,
    MODE_FIRST_FRAME_I2V,
    MODE_LATENT_IMG2IMG,
    MODE_MULTI_REFERENCE,
    MODE_TEXT_ONLY,
    TaskInferenceError,
)


def test_public_resolver_uses_image_presence_for_image_models():
    assert mlxgen.infer_task(model="flux2-klein-4b") == "text-to-image"
    assert mlxgen.infer_task(model="flux2-klein-4b", image_count=1) == "image-to-image"
    assert mlxgen.infer_task(model="flux2-klein-4b", image_count=2) == "image-to-image"


def test_generation_plan_exposes_internal_mode_for_flux2():
    assert mlxgen.resolve_generation_plan(model="flux2-klein-4b").mode == MODE_TEXT_ONLY
    assert mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1).mode == MODE_EDIT_REFERENCE
    assert (
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_image_strength=True).mode
        == MODE_LATENT_IMG2IMG
    )
    assert mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=2).mode == MODE_MULTI_REFERENCE


def test_public_resolver_uses_wan_model_capability():
    assert mlxgen.infer_task(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers") == "text-to-video"
    assert mlxgen.infer_task(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", image_count=1) == "image-to-video"
    assert mlxgen.infer_task(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers", image_count=1) == "image-to-video"
    assert (
        mlxgen.resolve_generation_plan(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", image_count=1).mode
        == MODE_FIRST_FRAME_I2V
    )


def test_public_resolver_rejects_wan_fixed_task_contradictions():
    with pytest.raises(TaskInferenceError, match="text-to-video model does not accept input images"):
        mlxgen.infer_task(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", image_count=1)

    with pytest.raises(TaskInferenceError, match="image-to-video model requires --image"):
        mlxgen.infer_task(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")


def test_public_resolver_rejects_generic_wan_names_without_specific_config():
    with pytest.raises(TaskInferenceError, match="Cannot infer a supported Wan model config"):
        mlxgen.infer_task(model="models/my-wan-video-folder", image_count=1)


def test_public_resolver_rejects_explicit_task_image_contradictions():
    with pytest.raises(TaskInferenceError, match="image-to-image requires --image"):
        mlxgen.infer_task(model="Qwen/Qwen-Image", task="image-to-image")

    with pytest.raises(TaskInferenceError, match="text-to-image cannot be combined with --image"):
        mlxgen.infer_task(model="Qwen/Qwen-Image", task="text-to-image", image_count=1)


def test_task_edit_is_compatibility_alias_for_image_to_image_mode():
    plan = mlxgen.resolve_generation_plan(model="flux2-klein-4b", task="edit", image_count=1)

    assert plan.public_task == "image-to-image"
    assert plan.mode == MODE_EDIT_REFERENCE


def test_edit_only_models_require_an_image_in_auto_mode():
    with pytest.raises(TaskInferenceError, match="image-to-image requires --image"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit")

    with pytest.raises(TaskInferenceError, match="image-to-image requires --image"):
        mlxgen.resolve_generation_plan(model="fibo-edit")


def test_image_strength_is_rejected_for_edit_reference_mode():
    with pytest.raises(TaskInferenceError, match="image-strength is only supported for latent"):
        mlxgen.resolve_generation_plan(
            model="flux2-klein-4b",
            image_count=1,
            i2i_mode="edit",
            has_image_strength=True,
        )

    with pytest.raises(TaskInferenceError, match="image-strength is only supported for latent"):
        mlxgen.resolve_generation_plan(
            model="qwen-image-edit",
            image_count=1,
            has_image_strength=True,
        )


def test_mask_and_outpaint_options_are_checked_against_capabilities():
    plan = mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_mask=True)
    assert plan.mode == MODE_EDIT_REFERENCE

    plan = mlxgen.resolve_generation_plan(model="fibo", image_count=1, has_mask=True)
    assert plan.mode == MODE_EDIT_REFERENCE

    with pytest.raises(TaskInferenceError, match="mask-path is only supported"):
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_mask=True)

    with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_outpaint=True)


def test_model_capabilities_are_publicly_inspectable():
    capabilities = mlxgen.get_model_capabilities(model="flux2-klein-4b")

    assert capabilities.schema_version == 1
    assert capabilities.family == "flux2"
    assert {capability.mode for capability in capabilities.capabilities} >= {
        MODE_TEXT_ONLY,
        MODE_LATENT_IMG2IMG,
        MODE_EDIT_REFERENCE,
        MODE_MULTI_REFERENCE,
    }
