"""The restoration capability record is the single source of truth for both families.

`mlxgen upscale` hosts two families with genuinely different reach: SeedVR2 restores images and
video and can scale, SwiftVR restores video at the source resolution only. Before this contract
existed, callers had to string-match a model handle to learn any of that, and SwiftVR's refusals
lived as hardcoded parser errors. These tests pin the declaration to the real classes so the two
cannot drift apart.

Nothing here loads model weights.
"""

import pytest

from mflux.models.common.restore_capabilities import (
    INPUT_IMAGE,
    INPUT_KINDS,
    INPUT_VIDEO,
    RESTORE_CAPABILITIES_SCHEMA_VERSION,
    RESTORE_FAMILIES,
    RestoreCapabilityError,
    get_restore_capabilities,
    is_restore_family_handle,
    require_capability,
)
from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
from mflux.models.swiftvr.variants.upscale.swiftvr import SwiftVR

ROUTE_CLASSES = {"seedvr2": SeedVR2, "swiftvr": SwiftVR}

SEEDVR2_HANDLES = ["seedvr2", "seedvr2-3b", "seedvr2-7b", "seedvr2-7b-sharp"]
SWIFTVR_HANDLES = ["swiftvr", "swiftvr-5b"]


class TestDeclarationMatchesTheCode:
    """A declared route must exist on the class that serves it.

    This is what replaces an ABC: the record names routes as strings, so the guarantee has to be
    asserted against the real classes rather than enforced by inheritance.
    """

    @pytest.mark.parametrize("family", sorted(RESTORE_FAMILIES))
    def test_every_declared_route_method_exists(self, family):
        capabilities = get_restore_capabilities(family=family)
        route_class = ROUTE_CLASSES[family]
        for capability in capabilities.capabilities:
            assert hasattr(route_class, capability.route_method), (
                f"{family} declares route_method {capability.route_method!r} for "
                f"{capability.input_kind}, but {route_class.__name__} has no such attribute"
            )

    @pytest.mark.parametrize("family", sorted(RESTORE_FAMILIES))
    def test_declared_input_kinds_are_known_and_unique(self, family):
        capabilities = get_restore_capabilities(family=family)
        kinds = [capability.input_kind for capability in capabilities.capabilities]
        assert kinds, f"{family} declares no restoration capability"
        assert set(kinds) <= set(INPUT_KINDS)
        assert len(kinds) == len(set(kinds)), f"{family} declares a duplicate input kind: {kinds}"


class TestFamilyReach:
    """The reach each family actually has, stated once."""

    def test_seedvr2_restores_images_and_video(self):
        capabilities = get_restore_capabilities(family="seedvr2")
        kinds = {capability.input_kind for capability in capabilities.capabilities}
        assert kinds == {INPUT_IMAGE, INPUT_VIDEO}

    def test_swiftvr_restores_video_only(self):
        capabilities = get_restore_capabilities(family="swiftvr")
        kinds = {capability.input_kind for capability in capabilities.capabilities}
        assert kinds == {INPUT_VIDEO}

    def test_both_families_share_the_video_route_name(self):
        # The point of the abstraction: pick a route by input kind, not by family.
        seedvr2_video = require_capability(get_restore_capabilities(family="seedvr2"), INPUT_VIDEO)
        swiftvr_video = require_capability(get_restore_capabilities(family="swiftvr"), INPUT_VIDEO)
        assert seedvr2_video.route_method == swiftvr_video.route_method == "restore_video_to_path"

    def test_seedvr2_image_route_is_the_write_to_disk_pairing(self):
        image = require_capability(get_restore_capabilities(family="seedvr2"), INPUT_IMAGE)
        assert image.route_method == "restore_image_to_path"
        assert hasattr(SeedVR2, "restore_image_to_path")

    def test_only_seedvr2_scales(self):
        for capability in get_restore_capabilities(family="seedvr2").capabilities:
            assert capability.scale_mode == "scalable"
        for capability in get_restore_capabilities(family="swiftvr").capabilities:
            assert capability.scale_mode == "source-only"


class TestRequireCapabilityFailsClosed:
    """ADR 0002: an unavailable route refuses with actionable text, never silently substitutes."""

    def test_asking_swiftvr_for_images_raises(self):
        capabilities = get_restore_capabilities(family="swiftvr")
        with pytest.raises(RestoreCapabilityError) as error:
            require_capability(capabilities, INPUT_IMAGE)
        message = str(error.value)
        assert "swiftvr" in message.lower()
        # Must name the alternative rather than just failing.
        assert "seedvr2" in message.lower()

    def test_seedvr2_image_and_video_both_resolve(self):
        capabilities = get_restore_capabilities(family="seedvr2")
        assert require_capability(capabilities, INPUT_IMAGE) is not None
        assert require_capability(capabilities, INPUT_VIDEO) is not None

    def test_unknown_input_kind_raises(self):
        capabilities = get_restore_capabilities(family="seedvr2")
        with pytest.raises((RestoreCapabilityError, ValueError, KeyError)):
            require_capability(capabilities, "audio")


class TestHandleResolution:
    """Every handle that worked before must still resolve to the same family."""

    @pytest.mark.parametrize("handle", SEEDVR2_HANDLES)
    def test_seedvr2_handles_resolve_to_seedvr2(self, handle):
        assert get_restore_capabilities(model=handle).family == "seedvr2"
        assert is_restore_family_handle(handle, None) is True

    @pytest.mark.parametrize("handle", SWIFTVR_HANDLES)
    def test_swiftvr_handles_resolve_to_swiftvr(self, handle):
        assert get_restore_capabilities(model=handle).family == "swiftvr"
        assert is_restore_family_handle(handle, None) is True

    @pytest.mark.parametrize("handle", ["qwen-image", "flux2-klein-4b", "wan2.2-ti2v-5b", "z-image-turbo"])
    def test_generation_handles_are_not_restoration_handles(self, handle):
        assert is_restore_family_handle(handle, None) is False

    def test_unknown_handle_fails_closed(self):
        with pytest.raises((RestoreCapabilityError, ValueError)):
            get_restore_capabilities(model="definitely-not-a-restoration-model")


class TestSwiftVRRefusalsAreData:
    """SwiftVR's refusals moved from hardcoded parser errors into the record."""

    def test_image_path_refusal_is_declared_and_actionable(self):
        refusals = get_restore_capabilities(family="swiftvr").unsupported_options
        assert "image-path" in refusals or any("image" in key for key in refusals)
        message = next(value for key, value in refusals.items() if "image" in key)
        assert "seedvr2" in message.lower(), "a refusal must name the route that can do the job"

    def test_quantize_refusal_carries_a_value_placeholder(self):
        refusals = get_restore_capabilities(family="swiftvr")
        quantize = next((value for key, value in refusals.unsupported_options.items() if "quantize" in key), None)
        assert quantize is not None
        assert "{value}" in quantize, "the CLI interpolates the requested bit width into this message"

    def test_seedvr2_declares_no_family_level_refusals(self):
        # SeedVR2 supports both media kinds, scaling and quantization; nothing to refuse up front.
        assert dict(get_restore_capabilities(family="seedvr2").unsupported_options) == {}


class TestFrameContract:
    """SwiftVR's 4a+1 clip rule and frame ceiling are declared, not just enforced downstream."""

    def test_swiftvr_frame_ceiling_is_declared(self):
        video = require_capability(get_restore_capabilities(family="swiftvr"), INPUT_VIDEO)
        assert video.max_source_frames is None or video.max_source_frames > 0

    def test_allows_source_frame_count_accepts_one_frame_clip(self):
        video = require_capability(get_restore_capabilities(family="swiftvr"), INPUT_VIDEO)
        assert video.allows_source_frame_count(1) is True

    def test_allows_source_frame_count_rejects_beyond_the_ceiling(self):
        video = require_capability(get_restore_capabilities(family="swiftvr"), INPUT_VIDEO)
        if video.max_source_frames is not None:
            assert video.allows_source_frame_count(video.max_source_frames) is True
            assert video.allows_source_frame_count(video.max_source_frames + 1) is False


def test_schema_version_is_declared():
    assert RESTORE_CAPABILITIES_SCHEMA_VERSION >= 1
    for family in RESTORE_FAMILIES:
        assert get_restore_capabilities(family=family).schema_version == RESTORE_CAPABILITIES_SCHEMA_VERSION
