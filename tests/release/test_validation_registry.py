import json
from pathlib import Path

from mflux.release.validation_registry import (
    BERNINI_R_1_3B_PROFILE_ID,
    BERNINI_R_1_3B_REPORT,
    MASKED_EDIT_MATRIX_PROFILE_ID,
    STATUS_FAIL,
    default_validation_profile_id_for_model,
    get_model_validation,
)


def test_default_profile_prefers_exact_row_evidence_over_base_model_fallback():
    # The 8bit package row must not default to the source checkpoint's masked-matrix records
    # (which only match through the base_model fallback); its own exact evidence wins.
    profile_id = default_validation_profile_id_for_model("AbstractFramework/qwen-image-8bit")
    validation = get_model_validation("AbstractFramework/qwen-image-8bit", profile_id=profile_id)
    assert profile_id != MASKED_EDIT_MATRIX_PROFILE_ID
    assert validation.records
    assert {record.model for record in validation.records} == {"AbstractFramework/qwen-image-8bit"}


def test_default_profile_uses_exact_masked_matrix_rows():
    for model in [
        "Qwen/Qwen-Image",
        "AbstractFramework/qwen-image-4bit",
        "AbstractFramework/z-image-8bit",
    ]:
        assert default_validation_profile_id_for_model(model) == MASKED_EDIT_MATRIX_PROFILE_ID, model


def test_base_model_fallback_still_serves_source_repacks_without_exact_rows():
    # No profile holds exact rows for the never-published bf16 repack; it inherits the source
    # checkpoint's records through the documented base-model fallback.
    profile_id = default_validation_profile_id_for_model("AbstractFramework/qwen-image-bf16")
    validation = get_model_validation("AbstractFramework/qwen-image-bf16", profile_id=profile_id)
    assert validation.records
    assert {record.model for record in validation.records} == {"Qwen/Qwen-Image"}


def test_bernini_alias_defaults_to_experimental_failed_video_profile_bound_to_report():
    profile_id = default_validation_profile_id_for_model("bernini-r-1.3b")
    validation = get_model_validation("bernini-r-1.3b", profile_id=profile_id)

    assert profile_id == BERNINI_R_1_3B_PROFILE_ID
    assert validation.status == STATUS_FAIL
    assert len(validation.records) == 3
    assert {record.model for record in validation.records} == {"ByteDance/Bernini-R-1.3B-Diffusers"}
    assert {record.public_task for record in validation.records} == {"text-to-video", "video-to-video"}
    assert {record.mode for record in validation.records} == {
        "reference-video",
        "reference-video-edit",
        "latent-video",
    }
    assert all(record.artifact_path and record.artifact_path.endswith(".mp4") for record in validation.records)
    assert all(Path(record.artifact_path).is_file() for record in validation.records if record.artifact_path)
    assert all(
        record.evidence_type == "model_backed_video_and_recorded_adversarial_visual_failure"
        for record in validation.records
    )
    assert all(record.status == STATUS_FAIL for record in validation.records)
    report = json.loads(Path(BERNINI_R_1_3B_REPORT).read_text())
    assert report["schema_version"] == 3
    assert report["machine_contract_passed"] is True
    assert report["visual_review_complete"] is True
    assert report["visual_quality_passed"] is False
    assert report["passed"] is False
