import json
from pathlib import Path

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.wan.model.wan_transformer import WanTransformer
from mflux.models.wan.model.wan_vae import Wan2_2_VAE
from mflux.models.wan.weights import WanWeightDefinition


class WanInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        weight_definition = WanWeightDefinition.for_config(model_config)
        root_path = PathResolution.resolve(path=path, patterns=weight_definition.get_download_patterns())
        WanInitializer._validate_source_config(root_path, model_config)
        WanInitializer._init_config(model, model_config, root_path, weight_definition)
        weights = WanInitializer._load_weights(str(root_path), weight_definition)
        WanInitializer._init_tokenizers(model, str(root_path), weight_definition)
        WanInitializer._init_models(model, model_config)
        WanInitializer._apply_weights(model, weights, quantize, weight_definition)

    @staticmethod
    def _init_config(model, model_config: ModelConfig, root_path: Path, weight_definition: WanWeightDefinition) -> None:
        model.model_config = model_config
        model.root_path = root_path
        model.weight_definition = weight_definition
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(model_path: str, weight_definition: WanWeightDefinition) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=weight_definition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str, weight_definition: WanWeightDefinition) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=weight_definition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        transformer_kwargs = WanInitializer._transformer_kwargs(model_config)
        model.transformer = WanTransformer(**transformer_kwargs)
        model.transformer_2 = (
            WanTransformer(**transformer_kwargs)
            if model_config.transformer_overrides.get("has_transformer_2", False)
            else None
        )
        model.vae = Wan2_2_VAE(**WanInitializer._vae_kwargs(model_config))

    @staticmethod
    def _apply_weights(
        model,
        weights: LoadedWeights,
        quantize: int | None,
        weight_definition: WanWeightDefinition,
    ) -> None:
        models = {
            "transformer": model.transformer,
            "vae": model.vae,
        }
        if model.transformer_2 is not None:
            models["transformer_2"] = model.transformer_2
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=weight_definition,
            models=models,
        )

    @staticmethod
    def _transformer_kwargs(model_config: ModelConfig) -> dict:
        allowed = {
            "patch_size",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "eps",
            "added_kv_proj_dim",
            "rope_max_seq_len",
        }
        kwargs = {key: value for key, value in model_config.transformer_overrides.items() if key in allowed}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return kwargs

    @staticmethod
    def _vae_kwargs(model_config: ModelConfig) -> dict:
        return dict(model_config.transformer_overrides.get("vae_config", {}))

    @staticmethod
    def _validate_source_config(root_path: Path | None, model_config: ModelConfig) -> None:
        if root_path is None:
            return

        overrides = model_config.transformer_overrides
        model_index = WanInitializer._read_json(root_path / "model_index.json")
        if model_index is not None:
            WanInitializer._validate_model_index(root_path, model_index, overrides)

        transformer_config = WanInitializer._read_json(root_path / "transformer" / "config.json")
        if transformer_config is not None:
            WanInitializer._validate_transformer_config(root_path, transformer_config, overrides, "transformer")

        transformer_2_config = WanInitializer._read_json(root_path / "transformer_2" / "config.json")
        has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        if transformer_2_config is not None and not has_transformer_2:
            raise ValueError(
                "Wan source/config mismatch: "
                f"{root_path} contains transformer_2/config.json, but {model_config.model_name} is configured "
                "as a single-transformer Wan model."
            )
        if transformer_2_config is not None:
            WanInitializer._validate_transformer_config(root_path, transformer_2_config, overrides, "transformer_2")

        vae_config = WanInitializer._read_json(root_path / "vae" / "config.json")
        if vae_config is not None:
            WanInitializer._validate_vae_config(root_path, vae_config, overrides)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        with path.open("rt") as json_file:
            return json.load(json_file)

    @staticmethod
    def _validate_model_index(root_path: Path, model_index: dict, overrides: dict) -> None:
        expected_has_transformer_2 = bool(overrides.get("has_transformer_2", False))
        actual_transformer_2 = model_index.get("transformer_2")
        actual_has_transformer_2 = isinstance(actual_transformer_2, list) and any(value is not None for value in actual_transformer_2)
        if actual_transformer_2 is not None and actual_has_transformer_2 != expected_has_transformer_2:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key="model_index.transformer_2",
                actual=actual_transformer_2,
                expected="present" if expected_has_transformer_2 else "absent",
            )

        for key in ("boundary_ratio", "expand_timesteps"):
            if key in model_index and key in overrides and model_index[key] != overrides[key]:
                WanInitializer._raise_source_mismatch(
                    root_path=root_path,
                    key=f"model_index.{key}",
                    actual=model_index[key],
                    expected=overrides[key],
                )

    @staticmethod
    def _validate_transformer_config(root_path: Path, transformer_config: dict, overrides: dict, component: str) -> None:
        for key in ("in_channels", "out_channels", "num_layers", "num_attention_heads", "ffn_dim"):
            WanInitializer._validate_config_key(root_path, transformer_config, overrides, f"{component}.{key}", key)
        WanInitializer._validate_config_key(root_path, transformer_config, overrides, f"{component}.patch_size", "patch_size")

    @staticmethod
    def _validate_vae_config(root_path: Path, vae_config: dict, overrides: dict) -> None:
        expected_vae_config = overrides.get("vae_config", {})
        for key in (
            "base_dim",
            "decoder_base_dim",
            "z_dim",
            "in_channels",
            "out_channels",
            "patch_size",
            "scale_factor_spatial",
            "scale_factor_temporal",
            "is_residual",
        ):
            WanInitializer._validate_config_key(root_path, vae_config, expected_vae_config, f"vae.{key}", key)

    @staticmethod
    def _validate_config_key(root_path: Path, source: dict, expected_source: dict, label: str, key: str) -> None:
        if key not in source or key not in expected_source:
            return
        if source[key] != expected_source[key]:
            WanInitializer._raise_source_mismatch(
                root_path=root_path,
                key=label,
                actual=source[key],
                expected=expected_source[key],
            )

    @staticmethod
    def _raise_source_mismatch(root_path: Path, key: str, actual, expected) -> None:
        raise ValueError(
            "Wan source/config mismatch: "
            f"{root_path} has {key}={actual!r}, but the selected Wan runtime expects {expected!r}. "
            "Pass the exact Wan model/config that matches this checkpoint; MLX-Gen will not fall back silently."
        )
