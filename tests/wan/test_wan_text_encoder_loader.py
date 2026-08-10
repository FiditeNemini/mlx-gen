from threading import Event, Thread
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers import UMT5EncoderModel

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.wan.variants.wan2_2_ti2v import Wan2_2_TI2V
from mflux.models.wan.wan_text_encoder_loader import WanTextEncoderLoader


def test_bernini_precision_scope_preserves_upstream_keep_set_and_exact_object():
    original = ["wo", "other-protected-module"]

    class FakeEncoder:
        _keep_in_fp32_modules = original

    with WanTextEncoderLoader._precision_scope(FakeEncoder, bernini_compatibility=True):
        assert FakeEncoder._keep_in_fp32_modules is original

    assert FakeEncoder._keep_in_fp32_modules is original


def test_bernini_precision_scope_restores_after_failure():
    original = ["wo"]

    class FakeEncoder:
        _keep_in_fp32_modules = original

    with pytest.raises(RuntimeError, match="load failed"):
        with WanTextEncoderLoader._precision_scope(FakeEncoder, bernini_compatibility=True):
            assert FakeEncoder._keep_in_fp32_modules is original
            raise RuntimeError("load failed")

    assert FakeEncoder._keep_in_fp32_modules is original


def test_ordinary_wan_precision_scope_is_inert():
    original = ["wo", "other-protected-module"]

    class FakeEncoder:
        _keep_in_fp32_modules = original

    with WanTextEncoderLoader._precision_scope(FakeEncoder, bernini_compatibility=False):
        assert FakeEncoder._keep_in_fp32_modules is original

    assert FakeEncoder._keep_in_fp32_modules is original
    assert WanTextEncoderLoader.precision_policy_id(ModelConfig.wan2_2_ti2v_5b()) is None


def test_bernini_precision_scope_restores_inherited_attribute_shape():
    original = ["wo", "base-protected-module"]

    class BaseEncoder:
        _keep_in_fp32_modules = original

    class DerivedEncoder(BaseEncoder):
        pass

    with WanTextEncoderLoader._precision_scope(DerivedEncoder, bernini_compatibility=True):
        assert DerivedEncoder._keep_in_fp32_modules is original
        assert "_keep_in_fp32_modules" not in vars(DerivedEncoder)

    assert "_keep_in_fp32_modules" not in vars(DerivedEncoder)
    assert DerivedEncoder._keep_in_fp32_modules is original


def test_all_wan_load_scopes_serialize_around_bernini_mutation():
    original = ["wo", "other-protected-module"]

    class FakeEncoder:
        _keep_in_fp32_modules = original

    bernini_entered = Event()
    ordinary_started = Event()
    ordinary_entered = Event()
    release_bernini = Event()
    observed = {}

    def bernini_load():
        with WanTextEncoderLoader._precision_scope(FakeEncoder, bernini_compatibility=True):
            observed["bernini"] = list(FakeEncoder._keep_in_fp32_modules)
            bernini_entered.set()
            assert release_bernini.wait(timeout=2)

    def ordinary_load():
        ordinary_started.set()
        with WanTextEncoderLoader._precision_scope(FakeEncoder, bernini_compatibility=False):
            observed["ordinary"] = list(FakeEncoder._keep_in_fp32_modules)
            ordinary_entered.set()

    bernini_thread = Thread(target=bernini_load)
    ordinary_thread = Thread(target=ordinary_load)
    bernini_thread.start()
    assert bernini_entered.wait(timeout=2)
    ordinary_thread.start()
    assert ordinary_started.wait(timeout=2)
    assert not ordinary_entered.wait(timeout=0.05)
    release_bernini.set()
    bernini_thread.join(timeout=2)
    ordinary_thread.join(timeout=2)

    assert not bernini_thread.is_alive()
    assert not ordinary_thread.is_alive()
    assert observed == {
        "bernini": ["wo", "other-protected-module"],
        "ordinary": ["wo", "other-protected-module"],
    }
    assert FakeEncoder._keep_in_fp32_modules is original


@pytest.mark.parametrize(
    ("bernini_compatibility", "expected_during_load"),
    [(False, ["wo"]), (True, ["wo"])],
)
def test_loader_applies_scoped_v5_policy_and_restores(
    monkeypatch, tmp_path, bernini_compatibility, expected_during_load
):
    original = UMT5EncoderModel._keep_in_fp32_modules
    loaded = object()
    observed = {}

    def fake_from_pretrained(cls, path, **kwargs):
        observed["protected"] = list(cls._keep_in_fp32_modules)
        observed["path"] = path
        observed["kwargs"] = kwargs
        return loaded

    monkeypatch.setattr(UMT5EncoderModel, "from_pretrained", classmethod(fake_from_pretrained))

    actual = WanTextEncoderLoader.load(
        text_encoder_path=tmp_path,
        torch_dtype=torch.bfloat16,
        bernini_compatibility=bernini_compatibility,
    )

    assert actual is loaded
    assert observed == {
        "protected": expected_during_load,
        "path": tmp_path,
        "kwargs": {"torch_dtype": torch.bfloat16, "local_files_only": True},
    }
    assert UMT5EncoderModel._keep_in_fp32_modules is original


def test_loader_restores_v5_policy_after_load_failure(monkeypatch, tmp_path):
    original = UMT5EncoderModel._keep_in_fp32_modules

    def failing_from_pretrained(cls, path, **kwargs):
        assert cls._keep_in_fp32_modules == ["wo"]
        raise RuntimeError("load failed")

    monkeypatch.setattr(UMT5EncoderModel, "from_pretrained", classmethod(failing_from_pretrained))

    with pytest.raises(RuntimeError, match="load failed"):
        WanTextEncoderLoader.load(
            text_encoder_path=tmp_path,
            torch_dtype=torch.bfloat16,
            bernini_compatibility=True,
        )

    assert UMT5EncoderModel._keep_in_fp32_modules is original


def test_bernini_prompt_load_retains_shared_embedding_tie(monkeypatch, tmp_path):
    (tmp_path / "text_encoder").mkdir()
    fake_encoder = SimpleNamespace(embed_tokens=object())

    class FakeTextEncoder:
        def __init__(self):
            self.shared = object()
            self.encoder = fake_encoder
            self.eval_called = False

        def eval(self):
            self.eval_called = True

        def __call__(self, input_ids, attention_mask):
            assert self.encoder.embed_tokens is self.shared
            hidden = torch.ones((*input_ids.shape, 4), dtype=torch.bfloat16)
            return SimpleNamespace(last_hidden_state=hidden)

    text_encoder = FakeTextEncoder()
    observed = {}

    def fake_load(**kwargs):
        observed.update(kwargs)
        return text_encoder

    monkeypatch.setattr(WanTextEncoderLoader, "load", staticmethod(fake_load))
    model = SimpleNamespace(
        root_path=tmp_path,
        model_config=ModelConfig.bernini_r_1_3b(),
        _resident_text_encoder=None,
        _keep_text_encoder_resident=False,
        _component_subdir_path=lambda component: tmp_path / component,
    )
    text_inputs = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.array([[1, 1]], dtype=np.int64),
    }

    embeds = Wan2_2_TI2V._load_t5_prompt_embeds(model, text_inputs=text_inputs, max_sequence_length=2)

    assert text_encoder.eval_called is True
    assert fake_encoder.embed_tokens is text_encoder.shared
    assert observed == {
        "text_encoder_path": tmp_path / "text_encoder",
        "torch_dtype": torch.bfloat16,
        "bernini_compatibility": True,
    }
    assert embeds.shape == (1, 2, 4)
    assert embeds.dtype == mx.bfloat16


def test_bernini_precision_policy_invalidates_only_bernini_prompt_cache_key():
    text_inputs = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.array([[1, 1]], dtype=np.int64),
    }

    def model(config):
        return SimpleNamespace(
            _prompt_embed_store=SimpleNamespace(enabled=True),
            _prompt_embed_fingerprint="same-encoder",
            root_path=None,
            model_config=config,
        )

    ordinary = Wan2_2_TI2V._prompt_embed_disk_key(
        model(ModelConfig.wan2_2_ti2v_5b()),
        text_inputs=text_inputs,
        max_sequence_length=2,
    )
    bernini = Wan2_2_TI2V._prompt_embed_disk_key(
        model(ModelConfig.bernini_r_1_3b()),
        text_inputs=text_inputs,
        max_sequence_length=2,
    )

    assert ordinary != bernini
    assert WanTextEncoderLoader.precision_policy_id(ModelConfig.bernini_r_1_3b()) == (
        WanTextEncoderLoader.BERNINI_PRECISION_POLICY_ID
    )
