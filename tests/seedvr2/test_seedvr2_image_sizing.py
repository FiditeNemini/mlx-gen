import numpy as np
import pytest
from PIL import Image

from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.utils.scale_factor import ScaleFactor


@pytest.mark.fast
def test_seedvr2_1x_keeps_exact_input_geometry():
    image = Image.new("RGB", (1451, 1600), (40, 30, 60))
    resized, true_h, true_w = SeedVR2Util._resize_and_soften(
        image=image,
        resolution=ScaleFactor(1),
        softness=0.0,
    )
    assert (true_w, true_h) == (1451, 1600)
    assert resized.size == (1451, 1600)


@pytest.mark.fast
def test_seedvr2_shortest_edge_resolution_keeps_exact_aspect():
    image = Image.new("RGB", (320, 192), (40, 30, 60))
    resized, true_h, true_w = SeedVR2Util._resize_and_soften(
        image=image,
        resolution=256,
        softness=0.0,
    )
    # 320 * (256/192) = 426.67 -> 427; no snapping to a network multiple.
    assert (true_w, true_h) == (427, 256)
    assert resized.size == (427, 256)


@pytest.mark.fast
def test_seedvr2_2x_scales_without_dimension_snap():
    image = Image.new("RGB", (301, 201), (40, 30, 60))
    resized, true_h, true_w = SeedVR2Util._resize_and_soften(
        image=image,
        resolution=ScaleFactor(2),
        softness=0.0,
    )
    assert (true_w, true_h) == (602, 402)


@pytest.mark.fast
def test_seedvr2_pad_to_multiple_reflects_content():
    arr = np.zeros((30, 20, 3), dtype=np.uint8)
    arr[:, 18, :] = 200  # bright column one in from the right edge
    arr[28, :, :] = 150  # bright row two up from the bottom edge
    padded = SeedVR2Util._pad_to_multiple(Image.fromarray(arr), factor=16)
    assert padded.size == (32, 32)
    padded_arr = np.asarray(padded)
    # np.pad reflect mirrors around the edge (col 20 <- col 18, row 31 <- row 27),
    # so the bright band reappears in the pad region; black fill would leave it 0.
    assert padded_arr[0, 20, 0] == 200
    assert padded_arr[30, 0, 0] == 150


@pytest.mark.fast
def test_seedvr2_pad_to_multiple_noop_when_divisible():
    image = Image.new("RGB", (32, 48), (10, 20, 30))
    padded = SeedVR2Util._pad_to_multiple(image, factor=16)
    assert padded.size == (32, 48)
