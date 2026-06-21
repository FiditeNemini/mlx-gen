import mlx.core as mx
import pytest

from mflux.models.seedvr2.model.seedvr2_vae.decoder.upsample_3d import Upsample3D


@pytest.mark.fast
def test_seedvr2_temporal_upsample_matches_official_2t_minus_1_contract():
    layer = Upsample3D(channels=1, temporal_up=True)
    x = mx.zeros((1, 1, 8, 2, 2), dtype=mx.float32)

    y = layer(x)

    assert y.shape == (1, 1, 15, 4, 4)


@pytest.mark.fast
def test_seedvr2_temporal_upsample_keeps_single_frame_single_frame():
    layer = Upsample3D(channels=1, temporal_up=True)
    x = mx.zeros((1, 1, 1, 2, 2), dtype=mx.float32)

    y = layer(x)

    assert y.shape == (1, 1, 1, 4, 4)
