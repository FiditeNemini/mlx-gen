import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_util import Qwen3VLUtil


@pytest.mark.fast
def test_sample_top_p_keeps_sampling_on_mlx_and_respects_top_token_cutoff():
    logits = mx.array([0.0, 10.0, 1.0], dtype=mx.float32)

    token_id = Qwen3VLUtil.sample_top_p(logits, top_p=0.1)

    assert int(token_id.item()) == 1


@pytest.mark.fast
def test_replace_image_token_embeddings_preserves_text_and_inserts_images_in_order():
    inputs_embeds = mx.array(
        [
            [
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
                [40.0, 40.0],
            ]
        ],
        dtype=mx.float32,
    )
    image_positions = mx.array([[False, True, False, True]])
    image_embeds = mx.array([[1.0, 1.0], [2.0, 2.0]], dtype=mx.float32)

    result = Qwen3VLDecoder._replace_image_token_embeddings(
        inputs_embeds=inputs_embeds,
        image_positions=image_positions,
        image_embeds=image_embeds,
    )

    expected = np.array([[[10.0, 10.0], [1.0, 1.0], [30.0, 30.0], [2.0, 2.0]]], dtype=np.float32)
    np.testing.assert_array_equal(np.array(result), expected)
