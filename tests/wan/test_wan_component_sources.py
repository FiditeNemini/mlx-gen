import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.lora_compatibility import LoRACompatibility
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.wan_initializer import WanInitializer
from mflux.models.wan.weights.wan_weight_definition import WanWeightDefinition


class TestWanComponentSources:
    REVISION = "ff4c5d4d2d31365c2ffeb30e9753065ee18f58ce"

    @staticmethod
    def _config(
        *,
        component_base_model: str | None = "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        base_model: str | None = None,
        transformer_model: str | None = "ByteDance/Bernini-R-1.3B-Diffusers",
        expected_revision: str | None = REVISION,
    ) -> ModelConfig:
        overrides = {
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 30,
            "num_attention_heads": 12,
            "attention_head_dim": 128,
            "ffn_dim": 8960,
            "patch_size": [1, 2, 2],
            "has_transformer_2": False,
            "flow_shift": 3.0,
            "default_solver": "unipc",
            "vae_variant": "wan21",
            "vae_config": {
                "base_dim": 96,
                "z_dim": 16,
                "in_channels": 3,
                "out_channels": 3,
                "patch_size": 1,
                "scale_factor_spatial": 8,
                "scale_factor_temporal": 4,
                "is_residual": False,
            },
        }
        if component_base_model is not None:
            overrides["component_base_model"] = component_base_model
        if expected_revision is not None:
            overrides["expected_transformer_revision"] = expected_revision
        return ModelConfig(
            priority=99,
            aliases=["bernini-test"],
            model_name="ByteDance/Bernini-R-1.3B-Diffusers",
            base_model=base_model,
            controlnet_model=None,
            custom_transformer_model=transformer_model,
            num_train_steps=1000,
            max_sequence_length=512,
            supports_guidance=True,
            requires_sigma_shift=False,
            transformer_overrides=overrides,
            text_encoder_overrides={
                "model_type": "umt5",
                "d_model": 4096,
                "d_ff": 10240,
                "num_layers": 24,
                "num_heads": 64,
                "vocab_size": 256384,
            },
        )

    @staticmethod
    def _source_roots(tmp_path: Path, *, revision: str = REVISION) -> tuple[Path, Path]:
        base_root = tmp_path / "base"
        transformer_root = tmp_path / "models--ByteDance--Bernini" / "snapshots" / revision
        for directory in (
            base_root / "tokenizer",
            base_root / "text_encoder",
            base_root / "vae",
            transformer_root / "transformer",
        ):
            directory.mkdir(parents=True, exist_ok=True)

        (base_root / "tokenizer" / "tokenizer.json").write_text(json.dumps({"model": {"vocab": ["a", "b"]}}))
        (base_root / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "T5Tokenizer"}))
        (base_root / "tokenizer" / "spiece.model").write_bytes(b"sentencepiece")
        (base_root / "text_encoder" / "model.safetensors").write_bytes(b"weights")
        (base_root / "text_encoder" / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "umt5",
                    "d_model": 4096,
                    "d_ff": 10240,
                    "num_layers": 24,
                    "num_heads": 64,
                    "vocab_size": 256384,
                }
            )
        )
        (base_root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        (base_root / "vae" / "config.json").write_text(
            json.dumps(
                {
                    "base_dim": 96,
                    "z_dim": 16,
                    "in_channels": 3,
                    "out_channels": 3,
                    "patch_size": 1,
                    "scale_factor_spatial": 8,
                    "scale_factor_temporal": 4,
                    "is_residual": False,
                }
            )
        )
        (transformer_root / "config.json").write_text("{}")
        (transformer_root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
        (transformer_root / "transformer" / "config.json").write_text(
            json.dumps(
                {
                    "in_channels": 16,
                    "out_channels": 16,
                    "num_layers": 30,
                    "num_attention_heads": 12,
                    "attention_head_dim": 128,
                    "ffn_dim": 8960,
                    "patch_size": [1, 2, 2],
                }
            )
        )
        return base_root, transformer_root

    def test_factored_patterns_do_not_fetch_repeated_components(self):
        definition = WanWeightDefinition.for_config(self._config())

        base_patterns = definition.get_base_download_patterns()
        transformer_patterns = definition.get_transformer_download_patterns()

        assert "transformer/*.safetensors" not in base_patterns
        assert "text_encoder/*.safetensors" in base_patterns
        assert "vae/*.safetensors" in base_patterns
        assert "text_encoder/*.safetensors" not in transformer_patterns
        assert "vae/*.safetensors" not in transformer_patterns
        assert transformer_patterns == [
            "config.json",
            "transformer/*.safetensors",
            "transformer/*.json",
        ]

    def test_factored_sources_resolve_exact_patterns_roots_and_provenance(self, monkeypatch, tmp_path):
        config = self._config()
        definition = WanWeightDefinition.for_config(config)
        base_root, transformer_root = self._source_roots(tmp_path)
        calls = []

        def resolve(*, path, patterns, revision=None):
            calls.append((path, list(patterns), revision))
            if path == config.transformer_overrides["component_base_model"]:
                return base_root
            if path == config.custom_transformer_model:
                return transformer_root
            raise AssertionError(f"Unexpected source resolution: {path}")

        monkeypatch.setattr(PathResolution, "resolve", resolve)

        sources = WanInitializer._resolve_component_sources(
            model_config=config,
            model_path=None,
            weight_definition=definition,
        )

        assert calls == [
            (config.transformer_overrides["component_base_model"], definition.get_base_download_patterns(), None),
            (
                config.custom_transformer_model,
                definition.get_transformer_download_patterns(),
                self.REVISION,
            ),
        ]
        assert sources.factored is True
        assert sources.root_path == base_root
        assert sources.component_roots["tokenizer"] == base_root
        assert sources.component_roots["text_encoder"] == base_root
        assert sources.component_roots["vae"] == base_root
        assert sources.component_roots["transformer"] == transformer_root
        assert sources.provenance["transformer"] == {
            "source": config.custom_transformer_model,
            "source_role": "transformer",
            "revision": self.REVISION,
        }

    def test_derived_config_uses_base_model_as_component_base(self, monkeypatch, tmp_path):
        derived_base = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        config = self._config(component_base_model=None, base_model=derived_base)
        definition = WanWeightDefinition.for_config(config)
        base_root, transformer_root = self._source_roots(tmp_path)
        calls = []

        def resolve(*, path, patterns, revision=None):
            calls.append(path)
            return base_root if path == derived_base else transformer_root

        monkeypatch.setattr(PathResolution, "resolve", resolve)

        sources = WanInitializer._resolve_component_sources(
            model_config=config,
            model_path=None,
            weight_definition=definition,
        )

        assert sources.factored is True
        assert calls == [derived_base, config.custom_transformer_model]

    def test_explicit_model_path_is_a_monolithic_override(self, monkeypatch, tmp_path):
        config = self._config()
        definition = WanWeightDefinition.for_config(config)
        monolithic_root = tmp_path / "explicit-model"
        monolithic_root.mkdir()
        calls = []

        def resolve(*, path, patterns, revision=None):
            calls.append((path, list(patterns), revision))
            return monolithic_root

        monkeypatch.setattr(PathResolution, "resolve", resolve)

        sources = WanInitializer._resolve_component_sources(
            model_config=config,
            model_path="/models/explicit-model",
            weight_definition=definition,
        )

        assert calls == [("/models/explicit-model", definition.get_download_patterns(), None)]
        assert sources.factored is False
        assert sources.root_path == monolithic_root
        assert set(sources.component_roots.values()) == {monolithic_root}

    def test_missing_transformer_source_does_not_fall_back_to_monolithic_model(self, monkeypatch, tmp_path):
        config = self._config()
        definition = WanWeightDefinition.for_config(config)
        base_root, _ = self._source_roots(tmp_path)
        calls = []

        def resolve(*, path, patterns, revision=None):
            calls.append(path)
            if path == config.transformer_overrides["component_base_model"]:
                return base_root
            raise FileNotFoundError("transformer source is not cached")

        monkeypatch.setattr(PathResolution, "resolve", resolve)

        with pytest.raises(FileNotFoundError, match="transformer source is not cached"):
            WanInitializer._resolve_component_sources(
                model_config=config,
                model_path=None,
                weight_definition=definition,
            )

        assert calls == [
            config.transformer_overrides["component_base_model"],
            config.custom_transformer_model,
        ]

    def test_incomplete_factor_root_fails_without_cache_substitution(self, monkeypatch, tmp_path):
        config = self._config()
        definition = WanWeightDefinition.for_config(config)
        incomplete_root = tmp_path / "incomplete"
        incomplete_root.mkdir()
        monkeypatch.setattr(PathResolution, "resolve", lambda **kwargs: incomplete_root)

        with pytest.raises(FileNotFoundError, match="will not substitute another cached model"):
            WanInitializer._resolve_component_sources(
                model_config=config,
                model_path=None,
                weight_definition=definition,
            )

    def test_cached_transformer_revision_mismatch_fails_closed(self, monkeypatch, tmp_path):
        config = self._config(expected_revision="expected-revision")
        definition = WanWeightDefinition.for_config(config)
        base_root, transformer_root = self._source_roots(tmp_path, revision="other-revision")

        def resolve(*, path, patterns, revision=None):
            return base_root if path == config.transformer_overrides["component_base_model"] else transformer_root

        monkeypatch.setattr(PathResolution, "resolve", resolve)

        with pytest.raises(ValueError, match="transformer revision mismatch"):
            WanInitializer._resolve_component_sources(
                model_config=config,
                model_path=None,
                weight_definition=definition,
            )

    def test_pinned_transformer_rejects_unverifiable_local_revision(self, monkeypatch, tmp_path):
        config = self._config()
        definition = WanWeightDefinition.for_config(config)
        base_root, transformer_root = self._source_roots(tmp_path)
        unverifiable_root = tmp_path / "local-transformer"
        transformer_root.rename(unverifiable_root)

        def resolve(*, path, patterns, revision=None):
            return base_root if path == config.transformer_overrides["component_base_model"] else unverifiable_root

        monkeypatch.setattr(PathResolution, "resolve", resolve)

        with pytest.raises(ValueError, match="unverifiable local path"):
            WanInitializer._resolve_component_sources(
                model_config=config,
                model_path=None,
                weight_definition=definition,
            )

    def test_factored_validation_rejects_transformer_and_vae_mismatches(self, tmp_path):
        config = self._config()
        base_root, transformer_root = self._source_roots(tmp_path)
        transformer_config_path = transformer_root / "transformer" / "config.json"
        transformer_config = json.loads(transformer_config_path.read_text())
        transformer_config["num_layers"] = 31
        transformer_config_path.write_text(json.dumps(transformer_config))

        with pytest.raises(ValueError, match=r"transformer\.num_layers=31"):
            WanInitializer._validate_factored_source_config(
                base_root=base_root,
                transformer_root=transformer_root,
                model_config=config,
            )

        transformer_config["num_layers"] = 30
        transformer_config_path.write_text(json.dumps(transformer_config))
        vae_config_path = base_root / "vae" / "config.json"
        vae_config = json.loads(vae_config_path.read_text())
        vae_config["z_dim"] = 48
        vae_config_path.write_text(json.dumps(vae_config))

        with pytest.raises(ValueError, match=r"vae\.z_dim=48"):
            WanInitializer._validate_factored_source_config(
                base_root=base_root,
                transformer_root=transformer_root,
                model_config=config,
            )

    def test_catalog_bernini_validation_rejects_renderer_and_attention_mismatches(self, tmp_path):
        config = ModelConfig.bernini_r_1_3b()
        base_root, transformer_root = self._source_roots(tmp_path)
        expected_renderer_config = config.transformer_overrides["expected_renderer_config"]
        (transformer_root / "config.json").write_text(json.dumps(expected_renderer_config))

        renderer_config = dict(expected_renderer_config)
        renderer_config["use_src_id_rotary_emb"] = False
        (transformer_root / "config.json").write_text(json.dumps(renderer_config))
        with pytest.raises(ValueError, match=r"renderer\.use_src_id_rotary_emb=False"):
            WanInitializer._validate_factored_source_config(
                base_root=base_root,
                transformer_root=transformer_root,
                model_config=config,
            )

        (transformer_root / "config.json").write_text(json.dumps(expected_renderer_config))
        transformer_config_path = transformer_root / "transformer" / "config.json"
        transformer_config = json.loads(transformer_config_path.read_text())
        transformer_config["attention_head_dim"] = 64
        transformer_config_path.write_text(json.dumps(transformer_config))
        with pytest.raises(ValueError, match=r"transformer\.attention_head_dim=64"):
            WanInitializer._validate_factored_source_config(
                base_root=base_root,
                transformer_root=transformer_root,
                model_config=config,
            )

    @pytest.mark.parametrize(
        ("component", "mutate", "message"),
        [
            (
                "text_encoder",
                lambda payload: payload.update({"d_model": 2048}),
                r"text_encoder\.d_model=2048",
            ),
            (
                "tokenizer",
                lambda payload: payload.update({"tokenizer_class": "Qwen2Tokenizer"}),
                r"tokenizer\.tokenizer_class",
            ),
        ],
    )
    def test_factored_validation_rejects_text_and_tokenizer_mismatches(
        self,
        tmp_path,
        component,
        mutate,
        message,
    ):
        config = self._config()
        base_root, transformer_root = self._source_roots(tmp_path)
        config_path = (
            base_root / "text_encoder" / "config.json"
            if component == "text_encoder"
            else base_root / "tokenizer" / "tokenizer_config.json"
        )
        payload = json.loads(config_path.read_text())
        mutate(payload)
        config_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match=message):
            WanInitializer._validate_factored_source_config(
                base_root=base_root,
                transformer_root=transformer_root,
                model_config=config,
            )

    def test_factored_validation_requires_code_native_unipc_semantics(self, tmp_path):
        config = self._config()
        config.transformer_overrides["default_solver"] = "euler"
        base_root, transformer_root = self._source_roots(tmp_path)

        with pytest.raises(ValueError, match=r"scheduler\.default_solver"):
            WanInitializer._validate_factored_source_config(
                base_root=base_root,
                transformer_root=transformer_root,
                model_config=config,
            )

    def test_initializer_keeps_base_root_for_prompt_encoder_and_records_sources(self, monkeypatch, tmp_path):
        config = self._config()
        base_root, transformer_root = self._source_roots(tmp_path)
        observed = {}

        def resolve(*, path, patterns, revision=None):
            return base_root if path == config.transformer_overrides["component_base_model"] else transformer_root

        monkeypatch.setattr(PathResolution, "resolve", resolve)
        monkeypatch.setattr(LoRACompatibility, "validate_for_model_config", lambda **kwargs: None)
        monkeypatch.setattr(
            WanInitializer,
            "_init_tokenizers",
            lambda model, model_path, weight_definition: observed.update(tokenizer_root=model_path),
        )
        monkeypatch.setattr(WanInitializer, "_init_models", lambda model, model_config: None)
        monkeypatch.setattr(
            WanInitializer,
            "_load_and_apply_weights",
            lambda model, root_path, quantize, weight_definition: observed.update(weight_root=root_path),
        )
        monkeypatch.setattr(WanInitializer, "_apply_lora", lambda model, **kwargs: None)
        monkeypatch.setattr(WanInitializer, "_apply_svi_loras", lambda model, **kwargs: None)
        model = SimpleNamespace()

        WanInitializer.init(model=model, model_config=config, quantize=4)

        assert model.root_path == base_root
        assert observed == {"tokenizer_root": str(base_root), "weight_root": base_root}
        assert model.factored_component_sources is True
        assert model.component_roots["transformer"] == transformer_root
        assert model.component_roots["vae"] == base_root
        assert model.component_source_provenance["transformer"]["revision"] == self.REVISION

    def test_weight_loading_uses_each_components_selected_root(self, monkeypatch, tmp_path):
        config = self._config()
        definition = WanWeightDefinition.for_config(config)
        base_root, transformer_root = self._source_roots(tmp_path)
        model = SimpleNamespace(
            transformer=object(),
            transformer_2=None,
            vae=object(),
            component_roots={"transformer": transformer_root, "vae": base_root},
        )
        observed = []

        def load_component(*, root_path, component):
            observed.append((component.name, root_path))
            return {}, None, None

        def apply_component(*, component, **kwargs):
            return None if component.skip_quantization else 4

        monkeypatch.setattr(WeightLoader, "_load_component", load_component)
        monkeypatch.setattr(WeightApplier, "apply_and_quantize_single", apply_component)

        WanInitializer._load_and_apply_weights(
            model=model,
            root_path=base_root,
            quantize=4,
            weight_definition=definition,
        )

        assert observed == [("transformer", transformer_root), ("vae", base_root)]
        assert model.bits == 4
