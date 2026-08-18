"""`mlxgen capabilities` must describe restoration routes, not just generation routes.

Restoration was previously invisible to capability inspection: asking for a SeedVR2 or SwiftVR
handle failed to infer a backend, and `--family` offered no restoration choice. A consumer had to
match handle strings to learn whether a model accepted images or could scale. These tests pin the
emitted payload so that stays machine-readable.

Nothing here loads model weights.
"""

import pytest

from mflux.task_inference import CAPABILITIES_SCHEMA_VERSION, get_model_capabilities

SEEDVR2_HANDLES = ["seedvr2", "seedvr2-3b", "seedvr2-7b", "seedvr2-7b-sharp"]
SWIFTVR_HANDLES = ["swiftvr", "swiftvr-5b"]


def payload(handle: str) -> dict:
    return get_model_capabilities(model=handle).to_dict()


class TestRestorationIsInspectable:
    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES + SWIFTVR_HANDLES)
    def test_handle_emits_a_restoration_array(self, handle):
        assert payload(handle)["restoration"], f"{handle} reports no restoration route"

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES + SWIFTVR_HANDLES)
    def test_generation_array_is_empty_for_restoration_models(self, handle):
        # An empty `capabilities` array means "not routable through mlxgen generate",
        # which is different from unsupported. The restoration array carries the truth.
        assert payload(handle)["capabilities"] == []

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES + SWIFTVR_HANDLES)
    def test_schema_version_is_current(self, handle):
        assert payload(handle)["schema_version"] == CAPABILITIES_SCHEMA_VERSION


class TestAcceptedMediaIsMachineReadable:
    """The question 'can this model restore an image?' must be answerable without string matching."""

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
    def test_seedvr2_accepts_images_and_video(self, handle):
        media = {tuple(row["accepted_media"]) for row in payload(handle)["restoration"]}
        assert ("image",) in media
        assert ("video",) in media

    @pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
    def test_swiftvr_accepts_video_only(self, handle):
        rows = payload(handle)["restoration"]
        media = {tuple(row["accepted_media"]) for row in rows}
        assert media == {("video",)}

    @pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
    def test_swiftvr_declares_zero_images(self, handle):
        # This is the field a consumer checks instead of parsing an error message.
        for row in payload(handle)["restoration"]:
            assert row["max_images"] == 0

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
    def test_seedvr2_image_row_admits_an_image(self, handle):
        image_rows = [r for r in payload(handle)["restoration"] if r["accepted_media"] == ["image"]]
        assert image_rows
        assert image_rows[0]["max_images"] >= 1


class TestRouteLimitsAreDeclared:
    @pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
    def test_swiftvr_is_source_resolution_only(self, handle):
        for row in payload(handle)["restoration"]:
            assert row["supports_scaling"] is False
            assert row["scale_factors"] == ["1x"]

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
    def test_seedvr2_scales(self, handle):
        for row in payload(handle)["restoration"]:
            assert row["supports_scaling"] is True

    @pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
    def test_swiftvr_declares_bf16_only_and_no_quantization(self, handle):
        for row in payload(handle)["restoration"]:
            assert row["supports_quantization"] is False
            assert row["weight_precision"] == "bf16"

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
    def test_seedvr2_supports_quantization(self, handle):
        for row in payload(handle)["restoration"]:
            assert row["supports_quantization"] is True

    @pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
    def test_swiftvr_declares_the_four_a_plus_one_clip_rule(self, handle):
        video = [r for r in payload(handle)["restoration"] if r["accepted_media"] == ["video"]]
        assert video
        # A clip length must satisfy t % 4 == 1; the contract states it rather than only
        # enforcing it at run time.
        assert video[0]["frame_multiple"] == 4
        assert video[0]["frame_remainder"] == 1


class TestCanonicalIdentity:
    """Route ids follow the existing handler_id convention so consumers can key on them."""

    @pytest.mark.parametrize(
        "handle,expected_family",
        [(h, "seedvr2") for h in SEEDVR2_HANDLES] + [(h, "swiftvr") for h in SWIFTVR_HANDLES],
    )
    def test_family_and_row_ids_agree(self, handle, expected_family):
        data = payload(handle)
        assert data["family"] == expected_family
        for row in data["restoration"]:
            assert row["id"].startswith(f"{expected_family}.")
            assert row["handler_id"].startswith(f"{expected_family}.")
            assert row["command"] == "mlxgen upscale"
