import mlx.core as mx
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.seedvr2.model.seedvr2_transformer.attention import MMAttention


@pytest.mark.fast
def test_seedvr2_3b_model_config_uses_mmrope3d_split():
    overrides = ModelConfig.seedvr2_3b().transformer_overrides or {}

    assert overrides["rope_freqs_for"] == "pixel"
    assert overrides["text_rope_freqs_for"] == "lang"


@pytest.mark.fast
def test_seedvr2_7b_model_config_uses_video_only_rope():
    overrides = ModelConfig.seedvr2_7b().transformer_overrides or {}

    assert overrides["rope_on_text"] is False
    assert "text_attention_mode" not in overrides


@pytest.mark.fast
def test_seedvr2_attention_rejects_unsupported_text_attention_mode():
    with pytest.raises(ValueError, match="Unsupported SeedVR2 text attention mode"):
        MMAttention(
            vid_dim=6,
            txt_dim=6,
            heads=1,
            head_dim=6,
            rope_dim=6,
            rope_on_text=False,
            text_attention_mode="global",
            window=(1, 1, 1),
        )


@pytest.mark.fast
def test_seedvr2_attention_window_pool_mode_runs_on_toy_shapes():
    attn = MMAttention(
        vid_dim=6,
        txt_dim=6,
        heads=1,
        head_dim=6,
        rope_dim=6,
        rope_on_text=True,
        text_attention_mode="window_pool",
        window=(1, 1, 1),
    )

    vid = mx.arange(12, dtype=mx.float32).reshape(1, 2, 6) / 10
    txt = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6) / 10
    vid_shape = mx.array([[2, 1, 1]], dtype=mx.int32)
    txt_shape = mx.array([[1]], dtype=mx.int32)

    vid_out, txt_out = attn(vid, txt, vid_shape, txt_shape)

    assert vid_out.shape == vid.shape
    assert txt_out.shape == txt.shape
