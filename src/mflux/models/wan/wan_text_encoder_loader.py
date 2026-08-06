from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any


class WanTextEncoderLoader:
    BERNINI_PRECISION_POLICY_ID = "bernini-umt5-wo-bf16-v1"

    _load_lock = RLock()
    _missing = object()

    @classmethod
    def load(cls, *, text_encoder_path: Path, torch_dtype: Any, bernini_compatibility: bool) -> Any:
        from transformers import UMT5EncoderModel

        with cls._precision_scope(UMT5EncoderModel, bernini_compatibility=bernini_compatibility):
            return UMT5EncoderModel.from_pretrained(
                text_encoder_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )

    @classmethod
    def precision_policy_id(cls, model_config) -> str | None:
        if model_config is not None and bool(
            model_config.transformer_overrides.get("supports_bernini_renderer", False)
        ):
            return cls.BERNINI_PRECISION_POLICY_ID
        return None

    @classmethod
    @contextmanager
    def _precision_scope(cls, model_class, *, bernini_compatibility: bool) -> Iterator[None]:
        with cls._load_lock:
            if not bernini_compatibility:
                yield
                return

            original = vars(model_class).get("_keep_in_fp32_modules", cls._missing)
            inherited = getattr(model_class, "_keep_in_fp32_modules", None)
            protected_modules = [] if inherited is None else list(inherited)
            model_class._keep_in_fp32_modules = [name for name in protected_modules if name != "wo"]
            try:
                yield
            finally:
                if original is cls._missing:
                    delattr(model_class, "_keep_in_fp32_modules")
                else:
                    model_class._keep_in_fp32_modules = original
