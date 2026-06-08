from pathlib import Path

import pytest

import mlxgen
from mflux.task_inference import (
    CANVAS_POLICY_EXACT_RESIZE,
    CANVAS_POLICY_SOURCE_ASPECT,
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


def test_generation_plan_requires_strength_for_qwen_base_latent_i2i():
    with pytest.raises(TaskInferenceError, match="image-strength is required"):
        mlxgen.resolve_generation_plan(model="qwen-image", image_count=1)

    latent = mlxgen.resolve_generation_plan(model="qwen-image", image_count=1, has_image_strength=True)
    assert latent.mode == MODE_LATENT_IMG2IMG
    assert latent.model_override is None

    with pytest.raises(TaskInferenceError, match="does not support edit-reference"):
        mlxgen.resolve_generation_plan(model="qwen-image", image_count=1, i2i_mode="edit")


def test_base_fibo_no_longer_advertises_unvalidated_latent_i2i():
    capabilities = mlxgen.get_model_capabilities(model="fibo")
    assert MODE_LATENT_IMG2IMG not in {capability.mode for capability in capabilities.capabilities}

    with pytest.raises(TaskInferenceError, match="supports text-to-image only"):
        mlxgen.resolve_generation_plan(model="fibo", image_count=1)

    with pytest.raises(TaskInferenceError, match="supports text-to-image only"):
        mlxgen.resolve_generation_plan(model="fibo", image_count=1, i2i_mode="latent")


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


def test_qwen_edit_versions_expose_distinct_reference_modes():
    regular_capabilities = mlxgen.get_model_capabilities(model="qwen-image-edit")
    edit_2509_capabilities = mlxgen.get_model_capabilities(model="qwen-image-edit-2509")
    edit_2511_capabilities = mlxgen.get_model_capabilities(model="qwen-image-edit-2511")
    regular_modes = {
        capability.mode for capability in regular_capabilities.capabilities
    }
    edit_2509_modes = {
        capability.mode for capability in edit_2509_capabilities.capabilities
    }
    edit_2511_modes = {
        capability.mode for capability in edit_2511_capabilities.capabilities
    }

    assert regular_capabilities.label == "Qwen Image Edit"
    assert edit_2509_capabilities.label == "Qwen Image Edit 2509"
    assert edit_2511_capabilities.label == "Qwen Image Edit 2511"
    assert regular_modes == {MODE_EDIT_REFERENCE}
    assert edit_2509_modes == {MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}
    assert edit_2511_modes == {MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}

    with pytest.raises(TaskInferenceError, match="does not support multi-reference"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit", image_count=2)
    assert (
        mlxgen.resolve_generation_plan(model="qwen-image-edit-2509", image_count=2).mode == MODE_MULTI_REFERENCE
    )


def test_qwen_prepared_edit_versions_inherit_distinct_reference_modes():
    regular_modes = {
        capability.mode
        for capability in mlxgen.get_model_capabilities(model="AbstractFramework/qwen-image-edit-8bit").capabilities
    }
    edit_2509_modes = {
        capability.mode
        for capability in mlxgen.get_model_capabilities(
            model="AbstractFramework/qwen-image-edit-2509-8bit"
        ).capabilities
    }

    assert regular_modes == {MODE_EDIT_REFERENCE}
    assert edit_2509_modes == {MODE_EDIT_REFERENCE, MODE_MULTI_REFERENCE}


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
    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_mask=True)

    with pytest.raises(TaskInferenceError, match="mask-path is only supported"):
        mlxgen.resolve_generation_plan(model="fibo", image_count=1, has_mask=True)

    with pytest.raises(TaskInferenceError, match="mask-path is only supported"):
        mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_mask=True)

    flux2_outpaint = mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_outpaint=True)
    qwen_outpaint = mlxgen.resolve_generation_plan(
        model="qwen-image-edit-2511",
        image_count=1,
        has_outpaint=True,
    )
    assert flux2_outpaint.capability_id == "flux2.edit"
    assert qwen_outpaint.capability_id == "qwen.edit"

    with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
        mlxgen.resolve_generation_plan(model="z-image-turbo", image_count=1, has_outpaint=True)

    with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit", image_count=1, has_outpaint=True)

    with pytest.raises(TaskInferenceError, match="outpaint-padding is only supported"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit-2509", image_count=1, has_outpaint=True)


def test_reframe_option_is_limited_to_validated_edit_capabilities():
    flux2 = mlxgen.resolve_generation_plan(model="flux2-klein-4b", image_count=1, has_reframe=True)
    qwen = mlxgen.resolve_generation_plan(model="qwen-image-edit-2511", image_count=1, has_reframe=True)

    assert flux2.mode == MODE_EDIT_REFERENCE
    assert flux2.capability_id == "flux2.edit"
    assert qwen.mode == MODE_EDIT_REFERENCE
    assert qwen.capability_id == "qwen.edit"

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="qwen-image", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="qwen-image-edit-2509", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="z-image-turbo", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="reframe-padding is only supported"):
        mlxgen.resolve_generation_plan(model="ernie-image-turbo", image_count=1, has_reframe=True)

    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_reframe=True)


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
    latent = next(capability for capability in capabilities.capabilities if capability.mode == MODE_LATENT_IMG2IMG)
    assert latent.default_canvas_policy == CANVAS_POLICY_SOURCE_ASPECT
    assert latent.canvas_policies == (CANVAS_POLICY_SOURCE_ASPECT, CANVAS_POLICY_EXACT_RESIZE)
    assert latent.primary_image_index == 0
    assert latent.dimension_multiple == 16
    edit = next(capability for capability in capabilities.capabilities if capability.mode == MODE_EDIT_REFERENCE)
    assert edit.supports_reframe is True
    assert edit.supports_outpaint is True


def test_fibo_edit_exposes_no_public_generation_capabilities():
    capabilities = mlxgen.get_model_capabilities(model="fibo-edit")
    assert capabilities.capabilities == ()

    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1)

    with pytest.raises(TaskInferenceError, match="does not expose unified generation capabilities"):
        mlxgen.resolve_generation_plan(model="fibo-edit", image_count=1, has_image_strength=True)

    assert mlxgen.get_model_capabilities(model="fibo-edit-rmbg").capabilities == ()


def test_model_validation_is_separate_from_route_capabilities():
    capabilities = mlxgen.get_model_capabilities(model="briaai/Fibo-Edit")
    assert capabilities.capabilities == ()

    validation = mlxgen.get_model_validation("briaai/Fibo-Edit")
    assert validation.status == "FAIL"
    assert {record.mode for record in validation.records} == {MODE_EDIT_REFERENCE}

    alias_validation = mlxgen.get_model_validation("fibo-edit")
    assert alias_validation.status == "FAIL"
    assert {record.model for record in alias_validation.records} == {"briaai/Fibo-Edit"}


def test_validation_registry_reports_variant_specific_statuses():
    qwen2509_q8 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2509-8bit")
    assert qwen2509_q8.status == "PASS"
    assert {record.step for record in qwen2509_q8.records} == {"B", "C", "D", "E"}
    assert {record.model for record in qwen2509_q8.records} == {"AbstractFramework/qwen-image-edit-2509-8bit"}
    for record in qwen2509_q8.records:
        assert record.artifact_path is not None
        assert Path(record.artifact_path).exists()
        if record.mode == MODE_MULTI_REFERENCE:
            assert len(record.source_images) >= 2
        for source_image in record.source_images:
            assert Path(source_image).exists()

    qwen2509_q4 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2509-4bit")
    assert qwen2509_q4.status == "PARTIAL"
    assert {record.model for record in qwen2509_q4.records} == {"AbstractFramework/qwen-image-edit-2509-4bit"}
    assert next(record for record in qwen2509_q4.records if record.step == "E").status == "PARTIAL"

    qwen2511_q8 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2511-8bit")
    assert qwen2511_q8.status == "PASS"
    assert {record.step for record in qwen2511_q8.records} == {"B", "C", "E"}
    assert {record.model for record in qwen2511_q8.records} == {"AbstractFramework/qwen-image-edit-2511-8bit"}
    assert next(record for record in qwen2511_q8.records if record.step == "E").status == "PASS"

    qwen2511_q4 = mlxgen.get_model_validation("AbstractFramework/qwen-image-edit-2511-4bit")
    assert qwen2511_q4.status == "PASS"
    assert {record.step for record in qwen2511_q4.records} == {"B", "C", "E"}
    assert {record.model for record in qwen2511_q4.records} == {"AbstractFramework/qwen-image-edit-2511-4bit"}
    assert next(record for record in qwen2511_q4.records if record.step == "E").status == "PASS"


def test_multi_reference_validation_records_list_reference_inputs():
    profile = mlxgen.get_validation_profile()

    for record in profile.records:
        if record.mode != MODE_MULTI_REFERENCE:
            continue
        assert len(record.source_images) >= 2
        for source_image in record.source_images:
            assert Path(source_image).exists()
