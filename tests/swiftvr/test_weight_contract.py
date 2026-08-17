"""The weight contract: nothing in the checkpoint may be dropped in silence.

Two layers of the loading stack fail open by design. ``WeightMapper.apply_mapping``
skips any source key it cannot place, and ``Module.update(..., strict=False)`` leaves an
unmatched parameter at its random initialisation. Between them, a checkpoint that has
drifted - a renamed tensor, an EMA copy, a MemBlock ``skip.weight`` that this build has
no slot for - loads cleanly and restores video with a quietly different autoencoder.

The guards that close that hole are ``SwiftVRWeightMapping.assert_reae_source_coverage``
(source side) and ``SwiftVRInitializer._assert_weight_coverage`` (model side). These
tests exercise both, in both directions, and use the published ``reae.safetensors`` when
it is on disk. The synthetic ``.safetensors`` files below are inputs to the validators,
never stand-ins for the model: each one exists to prove a specific rejection happens.
"""

from pathlib import Path

import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.swiftvr.model.swiftvr_reae.reae import ReAE
from mflux.models.swiftvr.swiftvr_initializer import SwiftVRInitializer
from mflux.models.swiftvr.weights.swiftvr_weight_definition import (
    PROMPT_EMBEDDING_KEY,
    PROMPT_EMBEDDING_UNUSED_KEYS,
    TRANSFORMER_COMPUTED_PARAMETER_KEYS,
    SwiftVRWeightDefinition,
)
from mflux.models.swiftvr.weights.swiftvr_weight_mapping import (
    REAE_TENSOR_COUNT,
    TRANSFORMER_TENSOR_COUNT,
    SwiftVRWeightMapping,
)
from tests.swiftvr.parity.parity_support import REAE_CHECKPOINT, SWIFTVR_SNAPSHOT

REAE_MAPPING = SwiftVRWeightMapping.get_reae_mapping()
MAPPING_SOURCES = {target.from_pattern[0] for target in REAE_MAPPING}
MAPPING_TARGETS = {target.to_pattern for target in REAE_MAPPING}


def _published_reae_keys() -> list[str]:
    from safetensors import safe_open

    if not REAE_CHECKPOINT.is_file():
        pytest.skip(f"reae.safetensors is not downloaded ({REAE_CHECKPOINT})")
    with safe_open(str(REAE_CHECKPOINT), framework="numpy") as handle:
        return list(handle.keys())


def _catalog_with(**override_changes) -> ModelConfig:
    """The real SwiftVR entry with its transformer_overrides edited.

    ModelConfig is a plain class, not a dataclass, so the entry is rebuilt from the real
    one's fields rather than copied field by field into a fake.
    """
    base = ModelConfig.swiftvr()
    overrides = {**base.transformer_overrides, **override_changes}
    for key, value in list(overrides.items()):
        if value is _REMOVE:
            del overrides[key]
    return ModelConfig(
        priority=base.priority,
        aliases=list(base.aliases),
        model_name=base.model_name,
        base_model=base.base_model,
        controlnet_model=base.controlnet_model,
        custom_transformer_model=base.custom_transformer_model,
        num_train_steps=base.num_train_steps,
        max_sequence_length=base.max_sequence_length,
        supports_guidance=base.supports_guidance,
        requires_sigma_shift=base.requires_sigma_shift,
        transformer_overrides=overrides,
        text_encoder_overrides=dict(base.text_encoder_overrides or {}),
    )


_REMOVE = object()


class TestReAEMappingShape:
    def test_the_mapping_has_one_target_per_checkpoint_tensor(self):
        assert len(REAE_MAPPING) == REAE_TENSOR_COUNT == 128
        assert len(MAPPING_SOURCES) == REAE_TENSOR_COUNT
        assert len(MAPPING_TARGETS) == REAE_TENSOR_COUNT

    def test_every_target_is_a_real_parameter_of_the_built_module(self):
        """Model-side coverage without the checkpoint: a target with no slot would be
        dropped by Module.update(strict=False) and never noticed."""
        parameters = {key for key, _ in tree_flatten(ReAE().parameters())}
        assert MAPPING_TARGETS == parameters

    def test_every_target_is_its_source_with_one_layers_segment_inserted(self):
        for target in REAE_MAPPING:
            stack, index, rest = target.from_pattern[0].split(".", 2)
            assert target.to_pattern == f"{stack}.layers.{index}.{rest}"
            assert stack in ("encoder", "decoder")

    def test_convolution_weights_are_transposed_and_biases_are_not(self):
        """One transpose at load time; a second one anywhere would be undetectable."""
        for target in REAE_MAPPING:
            if target.to_pattern.endswith(".bias"):
                assert target.transform is None
            else:
                assert target.transform is not None

    def test_there_is_no_single_blob_mapping(self):
        with pytest.raises(NotImplementedError, match="get_transformer_mapping"):
            SwiftVRWeightMapping.get_mapping()

    def test_the_transformer_mapping_is_wan_s_own(self):
        from mflux.models.wan.weights.wan_weight_mapping import WanWeightMapping

        mine = SwiftVRWeightMapping.get_transformer_mapping(num_layers=30)
        wan = WanWeightMapping.get_transformer_mapping(num_layers=30)
        assert [target.to_pattern for target in mine] == [target.to_pattern for target in wan]


class TestReAESourceCoverage:
    def test_the_published_checkpoint_is_covered_exactly(self):
        keys = _published_reae_keys()
        assert len(keys) == REAE_TENSOR_COUNT
        SwiftVRWeightMapping.assert_reae_source_coverage(keys)
        assert set(keys) == MAPPING_SOURCES

    def test_the_mapping_sources_are_accepted(self):
        """Runs with or without the checkpoint: the mapping must at least accept itself."""
        SwiftVRWeightMapping.assert_reae_source_coverage(MAPPING_SOURCES)

    def test_an_unknown_checkpoint_key_raises_and_is_named(self):
        """A MemBlock skip.weight, an EMA copy, a new head: all silently dropped without
        this guard, leaving a plausible-looking but different autoencoder."""
        keys = MAPPING_SOURCES | {"encoder.4.skip.weight"}
        with pytest.raises(ValueError, match="does not match the SwiftVR mapping") as exc:
            SwiftVRWeightMapping.assert_reae_source_coverage(keys)
        assert "encoder.4.skip.weight" in str(exc.value)
        assert "1 unmapped checkpoint key" in str(exc.value)

    def test_a_missing_checkpoint_key_raises_and_is_named(self):
        keys = set(MAPPING_SOURCES)
        keys.remove("encoder.0.weight")
        with pytest.raises(ValueError, match="does not match the SwiftVR mapping") as exc:
            SwiftVRWeightMapping.assert_reae_source_coverage(keys)
        assert "encoder.0.weight" in str(exc.value)
        assert "absent from the checkpoint" in str(exc.value)

    def test_an_empty_checkpoint_raises(self):
        with pytest.raises(ValueError, match="does not match the SwiftVR mapping"):
            SwiftVRWeightMapping.assert_reae_source_coverage([])


def _write_checkpoint(root: Path, *, reae_keys, prompt_keys, transformer_tensors: int) -> Path:
    """A minimal on-disk layout that exercises assert_source_key_coverage.

    Every tensor is a 1-element float32 array: the validators read names and counts from
    safetensors headers and never touch the data. This is a fixture for the guard, not a
    checkpoint - nothing here is ever loaded into a model.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "transformer").mkdir(exist_ok=True)
    tiny = np.zeros((1,), dtype=np.float32)
    save_file({key: tiny for key in reae_keys}, str(root / "reae.safetensors"))
    save_file({key: tiny for key in prompt_keys}, str(root / "prompt_embedding.safetensors"))
    save_file(
        {f"tensor.{index}": tiny for index in range(transformer_tensors)},
        str(root / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    return root


class TestSourceKeyCoverageAcrossTheRepository:
    def test_a_complete_layout_passes(self, tmp_path):
        root = _write_checkpoint(
            tmp_path / "ok",
            reae_keys=MAPPING_SOURCES,
            prompt_keys={PROMPT_EMBEDDING_KEY, *PROMPT_EMBEDDING_UNUSED_KEYS},
            transformer_tensors=TRANSFORMER_TENSOR_COUNT,
        )
        SwiftVRWeightDefinition.assert_source_key_coverage(root)

    def test_the_two_knowingly_unused_prompt_tensors_are_tolerated(self, tmp_path):
        root = _write_checkpoint(
            tmp_path / "minimal-prompt",
            reae_keys=MAPPING_SOURCES,
            prompt_keys={PROMPT_EMBEDDING_KEY},
            transformer_tensors=TRANSFORMER_TENSOR_COUNT,
        )
        SwiftVRWeightDefinition.assert_source_key_coverage(root)

    def test_a_missing_prompt_embedding_raises(self, tmp_path):
        root = _write_checkpoint(
            tmp_path / "no-prompt",
            reae_keys=MAPPING_SOURCES,
            prompt_keys={"input_ids"},
            transformer_tensors=TRANSFORMER_TENSOR_COUNT,
        )
        with pytest.raises(ValueError, match="cannot synthesize one"):
            SwiftVRWeightDefinition.assert_source_key_coverage(root)

    def test_an_unexpected_prompt_tensor_raises(self, tmp_path):
        root = _write_checkpoint(
            tmp_path / "extra-prompt",
            reae_keys=MAPPING_SOURCES,
            prompt_keys={PROMPT_EMBEDDING_KEY, "negative_prompt_emb"},
            transformer_tensors=TRANSFORMER_TENSOR_COUNT,
        )
        with pytest.raises(ValueError, match="unexpected tensor") as exc:
            SwiftVRWeightDefinition.assert_source_key_coverage(root)
        assert "negative_prompt_emb" in str(exc.value)

    @pytest.mark.parametrize("count", [824, 826, 0])
    def test_a_transformer_that_is_no_longer_tensor_identical_to_wan_raises(self, tmp_path, count):
        """The DiT is mapped by Wan's mapping precisely because it is stock Wan; a changed
        tensor count means that assumption no longer holds."""
        root = _write_checkpoint(
            tmp_path / f"transformer-{count}",
            reae_keys=MAPPING_SOURCES,
            prompt_keys={PROMPT_EMBEDDING_KEY},
            transformer_tensors=count,
        )
        with pytest.raises(ValueError, match="no longer tensor-identical to stock Wan"):
            SwiftVRWeightDefinition.assert_source_key_coverage(root)

    def test_an_unknown_reae_tensor_raises_through_the_definition(self, tmp_path):
        root = _write_checkpoint(
            tmp_path / "extra-reae",
            reae_keys=MAPPING_SOURCES | {"encoder.18.weight"},
            prompt_keys={PROMPT_EMBEDDING_KEY},
            transformer_tensors=TRANSFORMER_TENSOR_COUNT,
        )
        with pytest.raises(ValueError, match="does not match the SwiftVR mapping"):
            SwiftVRWeightDefinition.assert_source_key_coverage(root)


class TestComponentDefinitions:
    def test_the_three_components_are_the_four_published_files(self):
        components = {component.name: component for component in SwiftVRWeightDefinition.get_components()}
        assert set(components) == {"transformer", "reae", "prompt_embedding"}
        assert components["reae"].weight_files == ["reae.safetensors"]
        assert components["prompt_embedding"].weight_files == ["prompt_embedding.safetensors"]
        assert components["transformer"].hf_subdir == "transformer"
        assert SwiftVRWeightDefinition.get_download_patterns() == [
            "transformer/config.json",
            "transformer/*.safetensors",
            "reae.safetensors",
            "prompt_embedding.safetensors",
        ]

    def test_reae_and_the_prompt_embedding_are_never_quantized(self):
        """ReAE is 164 MB against a bf16 DiT, and the upstream author names the
        autoencoder as the ceiling on output quality."""
        components = {component.name: component for component in SwiftVRWeightDefinition.get_components()}
        assert components["reae"].skip_quantization is True
        assert components["prompt_embedding"].skip_quantization is True
        assert components["transformer"].skip_quantization is False

    def test_swiftvr_has_no_tokenizer(self):
        assert SwiftVRWeightDefinition.get_tokenizers() == []

    def test_the_quantization_predicate_is_wan_s_verbatim(self):
        from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition

        for path in ("proj_out", "condition_embedder.time_proj", "blocks.0.attn1.to_q", "blocks.0.ffn.net.0"):
            for bits in (None, 4, 8):
                assert SwiftVRWeightDefinition.quantization_predicate(
                    path, _Quantizable(), bits
                ) == WanWeightDefinition.quantization_predicate(path, _Quantizable(), bits)

    def test_a_prepared_layout_is_refused_rather_than_guessed(self):
        with pytest.raises(NotImplementedError, match="ADR 0001"):
            SwiftVRWeightDefinition.for_saving(ModelConfig.swiftvr())

    def test_the_rope_tables_are_the_only_computed_parameters(self):
        """827 module parameters against 825 checkpoint tensors; excluding by name keeps
        a genuinely missing weight failing."""
        assert TRANSFORMER_COMPUTED_PARAMETER_KEYS == ("rope.freqs_cos", "rope.freqs_sin")


class _Quantizable:
    """Stands in for a module with a ``to_quantized`` method, which is all the predicate reads."""

    def to_quantized(self):  # pragma: no cover - never called, only probed with hasattr
        raise NotImplementedError


class TestResidentWeightEstimate:
    def test_the_estimate_counts_loaded_tensors_at_runtime_precision(self):
        if not (SWIFTVR_SNAPSHOT / "reae.safetensors").is_file():
            pytest.skip(f"SwiftVR snapshot is not downloaded ({SWIFTVR_SNAPSHOT})")
        estimate = SwiftVRWeightDefinition.estimate_resident_weight_bytes(SWIFTVR_SNAPSHOT)
        # bf16 DiT (~5.0G params) plus the 40.95M-parameter ReAE, not the F32 file sizes.
        assert 9.0e9 < estimate < 11.0e9

    def test_the_estimate_ignores_the_unused_prompt_tensors(self, tmp_path):
        root = _write_checkpoint(
            tmp_path / "estimate",
            reae_keys=MAPPING_SOURCES,
            prompt_keys={PROMPT_EMBEDDING_KEY, *PROMPT_EMBEDDING_UNUSED_KEYS},
            transformer_tensors=4,
        )
        item_size = ModelConfig.precision.size
        # 128 ReAE + 4 transformer + 1 prompt tensor survive the prefix filter, one element each.
        assert SwiftVRWeightDefinition.estimate_resident_weight_bytes(root) == 133 * item_size


class TestCatalogProjection:
    def test_every_override_key_is_either_a_constructor_argument_or_a_named_runtime_key(self):
        from mflux.models.swiftvr.swiftvr_initializer import RUNTIME_OVERRIDE_KEYS, TRANSFORMER_CONSTRUCTOR_KEYS

        overrides = set(ModelConfig.swiftvr().transformer_overrides)
        assert overrides - TRANSFORMER_CONSTRUCTOR_KEYS - RUNTIME_OVERRIDE_KEYS == set()
        assert RUNTIME_OVERRIDE_KEYS - overrides == set(), "the allow-list names a key the catalog does not carry"

    def test_an_unclassified_override_raises_instead_of_vanishing(self):
        with pytest.raises(ValueError, match="unclassified keys"):
            SwiftVRInitializer.transformer_kwargs(_catalog_with(swiftvr_new_knob=1))

    def test_the_transformer_kwargs_are_the_wan_2_2_ti2v_5b_shape(self):
        kwargs = SwiftVRInitializer.transformer_kwargs(ModelConfig.swiftvr())
        assert kwargs == {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "in_channels": 48,
            "out_channels": 48,
            "text_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 14336,
            "num_layers": 30,
            "cross_attn_norm": True,
            "eps": 1e-06,
            "added_kv_proj_dim": None,
            "rope_max_seq_len": 1024,
        }

    def test_the_reae_kwargs_build_the_published_topology(self):
        model = ReAE(**SwiftVRInitializer.reae_kwargs(ModelConfig.swiftvr()))
        assert {key for key, _ in tree_flatten(model.parameters())} == MAPPING_TARGETS

    def test_the_window_settings_come_from_the_catalog(self):
        window_hw, shift = SwiftVRInitializer.window_settings(ModelConfig.swiftvr())
        assert window_hw == (16, 16)
        assert shift is True

    def test_a_malformed_window_size_raises(self):
        with pytest.raises(ValueError, match="must hold two values"):
            SwiftVRInitializer.window_settings(_catalog_with(swiftvr_window_size=[16]))


class TestRuntimeSettings:
    """Run defaults come from the catalog and nowhere else, so there is one source."""

    def test_the_defaults_are_read_from_the_catalog_entry(self):
        settings = SwiftVRInitializer.runtime_settings(ModelConfig.swiftvr())
        overrides = ModelConfig.swiftvr().transformer_overrides
        assert settings.clip_len == overrides["default_clip_len"] == 24
        assert settings.dit_overlap == overrides["default_dit_overlap"] == 0
        assert settings.inference_timestep == overrides["swiftvr_inference_timestep"] == 1000.0

    def test_an_edited_catalog_entry_actually_changes_the_run_defaults(self):
        """The point of the resolver: a catalog edit must reach the runtime, not vanish."""
        settings = SwiftVRInitializer.runtime_settings(_catalog_with(default_clip_len=8, default_dit_overlap=2))
        assert settings.clip_len == 8
        assert settings.dit_overlap == 2

    @pytest.mark.parametrize("key", ["default_clip_len", "default_dit_overlap", "swiftvr_inference_timestep"])
    def test_a_missing_default_raises_rather_than_falling_back_to_code(self, key):
        with pytest.raises(ValueError, match="missing transformer_overrides") as exc:
            SwiftVRInitializer.runtime_settings(_catalog_with(**{key: _REMOVE}))
        assert key in str(exc.value)

    @pytest.mark.parametrize("clip_len", [0, -4, 6, 25])
    def test_a_clip_len_the_protocol_cannot_honour_raises(self, clip_len):
        with pytest.raises(ValueError, match="positive multiple of 4"):
            SwiftVRInitializer.runtime_settings(_catalog_with(default_clip_len=clip_len))

    def test_a_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="zero or positive"):
            SwiftVRInitializer.runtime_settings(_catalog_with(default_dit_overlap=-1))
