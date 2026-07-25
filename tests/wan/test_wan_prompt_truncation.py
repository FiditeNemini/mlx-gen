from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.models.wan.variants import Wan2_2_TI2V
from tests.wan.test_wan_a14b_config import _fake_t2v_a14b_model, _patch_fake_wan_generation


class _FakeUMT5Tokenizer:
    # One token per whitespace word plus one EOS: enough to exercise the
    # uncapped-count logic without the real UMT5 sentencepiece assets.
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()) + 1))}


def _model_with_fake_tokenizer():
    model = Wan2_2_TI2V.__new__(Wan2_2_TI2V)
    model.tokenizers = {"wan": SimpleNamespace(tokenizer=_FakeUMT5Tokenizer())}
    return model


def test_wan_prompt_truncation_warns_and_reports_counts(capsys):
    model = _model_with_fake_tokenizer()

    report = model._check_prompt_truncation(
        prompt=" ".join(["word"] * 12),
        negative_prompt=None,
        max_sequence_length=8,
    )

    assert report == {"prompt_tokens": 13, "prompt_truncated": True}
    err = capsys.readouterr().err
    assert "Wan prompt truncated: 13 -> 8 UMT5 tokens" in err
    assert "last 5 tokens" in err


def test_wan_prompt_within_budget_is_silent(capsys):
    model = _model_with_fake_tokenizer()

    report = model._check_prompt_truncation(
        prompt="a calm lake at dawn",
        negative_prompt=None,
        max_sequence_length=8,
    )

    assert report == {"prompt_tokens": 6, "prompt_truncated": False}
    assert capsys.readouterr().err == ""


def test_wan_negative_prompt_truncation_reported_only_when_encoded(capsys):
    model = _model_with_fake_tokenizer()

    report = model._check_prompt_truncation(
        prompt="short",
        negative_prompt=" ".join(["blurry"] * 10),
        max_sequence_length=8,
    )

    assert report["prompt_truncated"] is False
    assert report["negative_prompt_tokens"] == 11
    assert report["negative_prompt_truncated"] is True
    assert "Wan negative prompt truncated: 11 -> 8 UMT5 tokens" in capsys.readouterr().err


def test_wan_encode_prompt_records_truncation_report(monkeypatch, capsys):
    model = _model_with_fake_tokenizer()
    monkeypatch.setattr(
        model, "_get_t5_prompt_embeds", lambda prompts, max_sequence_length: mx.zeros((len(prompts), 1, 8))
    )

    # CFG off: the negative prompt is never encoded, so it is not probed.
    model.encode_prompt(
        prompt=" ".join(["word"] * 12),
        negative_prompt=" ".join(["blurry"] * 20),
        do_classifier_free_guidance=False,
        max_sequence_length=8,
    )
    assert model._last_prompt_truncation == {"prompt_tokens": 13, "prompt_truncated": True}
    assert "negative" not in capsys.readouterr().err

    # CFG on: both prompts are probed.
    model.encode_prompt(
        prompt="short",
        negative_prompt=" ".join(["blurry"] * 20),
        do_classifier_free_guidance=True,
        max_sequence_length=8,
    )
    assert model._last_prompt_truncation == {
        "prompt_tokens": 2,
        "prompt_truncated": False,
        "negative_prompt_tokens": 21,
        "negative_prompt_truncated": True,
    }
    assert "Wan negative prompt truncated: 21 -> 8" in capsys.readouterr().err


def test_wan_generate_metadata_carries_prompt_truncation_report(monkeypatch):
    model = _fake_t2v_a14b_model()
    _patch_fake_wan_generation(monkeypatch, model, patch_to_video=False)
    observed = {}

    def encode_prompt_with_report(**kwargs):
        # Mirror the real encode_prompt contract: the report is refreshed on
        # every encode and generate_video copies it into extra_metadata.
        model._last_prompt_truncation = {"prompt_tokens": 547, "prompt_truncated": True}
        return mx.zeros((1, 1, 4096)), None

    def to_video(**kwargs):
        observed["to_video"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(model, "encode_prompt", encode_prompt_with_report)
    monkeypatch.setattr("mflux.models.wan.variants.wan2_2_ti2v.VideoUtil.to_video_from_frame_batches", to_video)

    model.generate_video(
        seed=1,
        prompt="a very long prompt",
        width=64,
        height=64,
        num_frames=1,
        num_inference_steps=2,
        guidance=1,
        guidance_2=1,
    )

    extras = observed["to_video"]["extra_metadata"]
    assert extras["prompt_tokens"] == 547
    assert extras["prompt_truncated"] is True


def test_wan_prompt_truncation_probe_uses_cleaned_text():
    # The probe must count the SAME text the capped encode tokenizes:
    # _prompt_clean collapses whitespace, so padding spaces cannot skew counts.
    model = _model_with_fake_tokenizer()

    report = model._check_prompt_truncation(
        prompt="word   word \n word",
        negative_prompt=None,
        max_sequence_length=8,
    )

    assert report == {"prompt_tokens": 4, "prompt_truncated": False}


@pytest.mark.parametrize("max_sequence_length", [1, 512])
def test_wan_prompt_truncation_boundary_exact_fit_is_not_truncated(max_sequence_length):
    model = _model_with_fake_tokenizer()

    # Exactly at the cap (words + EOS == cap) is a full fit, not truncation.
    report = model._check_prompt_truncation(
        prompt=" ".join(["word"] * (max_sequence_length - 1)) if max_sequence_length > 1 else "",
        negative_prompt=None,
        max_sequence_length=max_sequence_length,
    )

    assert report["prompt_truncated"] is False
