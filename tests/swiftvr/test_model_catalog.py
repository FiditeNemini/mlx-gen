"""How a SwiftVR handle reaches the SwiftVR catalog entry, and what happens when it does not.

Two resolvers see a ``--model`` value and they answer different questions. The shared
catalog resolver (``ModelConfig.from_name``) is deliberately fuzzy - it matches aliases as
substrings so that ``mlxgen download --model H-oliday/SwiftVR`` and a repo id with a
different case both land on the right entry. The route's own resolver
(``resolve_swiftvr_model``) is strict, and it is the one that must fail closed: it decides
what gets loaded.

Both are pinned here, including the fuzziness, so that a future reader does not "fix" the
catalog resolver into strictness and break every repo-id download.
"""

import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.swiftvr.cli.swiftvr_restore import (
    SWIFTVR_ALIASES,
    SWIFTVR_REPO_IDS,
    resolve_swiftvr_model,
)

SWIFTVR_ENTRY_ALIASES = {"swiftvr", "swiftvr-5b"}


class TestCatalogEntry:
    def test_the_entry_is_reachable_by_its_accessor(self):
        assert ModelConfig.swiftvr() is AVAILABLE_MODELS["swiftvr"]

    @pytest.mark.parametrize("handle", ["swiftvr", "swiftvr-5b", "SwiftVR", "SWIFTVR-5B"])
    def test_every_declared_alias_resolves_to_the_swiftvr_entry(self, handle):
        assert set(ModelConfig.from_name(handle).aliases) == SWIFTVR_ENTRY_ALIASES

    @pytest.mark.parametrize("handle", ["H-oliday/SwiftVR", "h-oliday/swiftvr"])
    def test_the_repo_id_resolves_to_the_swiftvr_entry(self, handle):
        config = ModelConfig.from_name(handle)
        assert set(config.aliases) == SWIFTVR_ENTRY_ALIASES
        # The catalog resolver keeps the caller's spelling as the model name so the
        # download command fetches the repository the user actually named.
        assert config.model_name == handle

    def test_the_declared_aliases_are_exactly_what_the_cli_accepts(self):
        """A new catalog alias that the route's resolver does not know would resolve for
        `download` and then be refused by `upscale`."""
        assert set(ModelConfig.swiftvr().aliases) == SWIFTVR_ALIASES
        assert SWIFTVR_REPO_IDS == {ModelConfig.swiftvr().model_name.lower()}

    def test_an_unrelated_handle_does_not_reach_the_swiftvr_entry(self):
        for handle in ("seedvr2-3b", "wan2.2-ti2v-5b", "dev"):
            assert set(ModelConfig.from_name(handle).aliases) != SWIFTVR_ENTRY_ALIASES

    def test_a_handle_no_family_claims_raises(self):
        from mflux.utils.exceptions import ModelConfigError

        with pytest.raises(ModelConfigError):
            ModelConfig.from_name("definitely-not-a-model")

    def test_the_catalog_resolver_is_substring_based_by_design(self):
        """Pinned deliberately: this fuzziness is what lets an arbitrary SwiftVR repo id
        resolve, and it is also why the route has its own strict resolver below."""
        assert set(ModelConfig.from_name("swiftvr-9b").aliases) == SWIFTVR_ENTRY_ALIASES

    def test_the_entry_declares_no_base_model(self):
        """A base_model naming Wan would reroute prepare, download and family inference."""
        assert ModelConfig.swiftvr().base_model is None

    def test_the_entry_declares_no_sampler_and_no_text_encoder(self):
        config = ModelConfig.swiftvr()
        assert config.supports_guidance is False
        assert config.requires_sigma_shift is False
        assert config.text_encoder_overrides == {}
        assert config.max_sequence_length == 512

    def test_the_entry_declares_its_task(self):
        overrides = ModelConfig.swiftvr().transformer_overrides
        assert overrides["task"] == "video-to-video"
        assert overrides["supports_video_to_video"] is True
        assert overrides["supports_image_to_video"] is False

    def test_the_entry_shares_the_wan_transformer_shape_verbatim(self):
        from mflux.models.common.config.model_config import WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE

        overrides = ModelConfig.swiftvr().transformer_overrides
        for key, value in WAN_2_2_TI2V_5B_TRANSFORMER_SHAPE.items():
            assert overrides[key] == value

    def test_the_padded_canvas_multiple_is_not_a_catalog_knob(self):
        """It is fixed by ReAE's 16x compression and the patch embed, and lives once in
        SwiftVRUtil; a catalog copy would be a second source nothing reads."""
        assert "spatial_pad_multiple" not in ModelConfig.swiftvr().transformer_overrides


class TestRouteResolver:
    @pytest.mark.parametrize("handle", ["swiftvr", "swiftvr-5b", "SwiftVR", " swiftvr ", "SWIFTVR"])
    def test_an_alias_resolves_without_a_path(self, handle):
        config, path = resolve_swiftvr_model(handle, None)
        assert config is ModelConfig.swiftvr()
        assert path is None

    @pytest.mark.parametrize("handle", ["H-oliday/SwiftVR", "h-oliday/swiftvr"])
    def test_the_repo_id_resolves_to_itself_as_the_checkpoint_source(self, handle):
        config, path = resolve_swiftvr_model(handle, None)
        assert config is ModelConfig.swiftvr()
        assert path == handle

    def test_no_handle_at_all_falls_back_to_the_catalog_entry(self):
        config, path = resolve_swiftvr_model(None, None)
        assert config is ModelConfig.swiftvr()
        assert path is None

    def test_an_explicit_path_wins_over_the_repo_id(self, tmp_path):
        config, path = resolve_swiftvr_model("H-oliday/SwiftVR", str(tmp_path))
        assert config is ModelConfig.swiftvr()
        assert path == str(tmp_path)

    def test_a_local_directory_resolves_to_itself(self, tmp_path):
        config, path = resolve_swiftvr_model(str(tmp_path), None)
        assert config is ModelConfig.swiftvr()
        assert path == str(tmp_path)

    def test_an_unknown_repo_id_fails_closed_and_names_the_alternatives(self):
        with pytest.raises(ValueError, match="Unsupported SwiftVR model handle") as exc:
            resolve_swiftvr_model("H-oliday/NotSwiftVR", None)
        assert "swiftvr-5b" in str(exc.value)

    @pytest.mark.parametrize("handle", ["swiftvr-9b", "swift-vr", "/no/such/directory", ""])
    def test_an_unknown_bare_handle_fails_closed(self, handle):
        """The strict half of the pair: from_name would happily accept swiftvr-9b."""
        with pytest.raises(ValueError, match="could not resolve"):
            resolve_swiftvr_model(handle, None)


class TestDownloadPreflight:
    """The 20 GB fetch declares its size, and the declaration is actually consumed.

    ``expected_download_bytes`` and ``download_headroom_bytes`` were declared on the
    catalog entry and read by nothing: the factored-source preflight only fires for models
    that split across two repositories, which SwiftVR does not. A declared-but-unread key
    is worse than no key, because the allow-list that is meant to catch dropped settings
    certifies it as live.
    """

    def test_the_entry_declares_the_download_size_and_headroom(self):
        overrides = ModelConfig.swiftvr().transformer_overrides
        assert overrides["expected_download_bytes"] == 20_167_236_128
        assert overrides["download_headroom_bytes"] == 2 * 1024**3

    def test_the_declared_size_is_read_back(self):
        from mflux.cli.mlx_gen import _expected_download_bytes

        assert _expected_download_bytes(ModelConfig.swiftvr()) == 20_167_236_128

    def test_a_model_without_a_declared_size_reports_none(self):
        from mflux.cli.mlx_gen import _expected_download_bytes

        assert _expected_download_bytes(ModelConfig.seedvr2_3b()) is None
        assert _expected_download_bytes(None) is None

    def test_a_download_that_cannot_fit_is_refused_before_it_starts(self):
        import argparse

        from mflux.cli.mlx_gen import _preflight_download_space

        parser = argparse.ArgumentParser(prog="mlxgen download")
        with pytest.raises(SystemExit) as exc:
            _preflight_download_space(
                parser,
                ModelConfig.swiftvr(),
                [("Nonexistent/SwiftVR-Fixture", ["*.safetensors"], None, 10**15)],
            )
        assert exc.value.code == 2

    def test_a_source_with_no_declared_size_is_not_guessed_at(self):
        import argparse

        from mflux.cli.mlx_gen import _preflight_download_space

        parser = argparse.ArgumentParser(prog="mlxgen download")
        assert (
            _preflight_download_space(
                parser,
                ModelConfig.swiftvr(),
                [("Nonexistent/SwiftVR-Fixture", ["*.safetensors"], None, None)],
            )
            is None
        )
