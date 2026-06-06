import mlx.core as mx
import pytest

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder


def test_qwen_multimodal_rope_positions_match_image_grid():
    encoder = QwenEncoder(num_hidden_layers=0)
    image_token = encoder.image_token_id
    input_ids = mx.array([[10, 11, image_token, image_token, image_token, image_token, image_token, image_token, 12, 13]])
    attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
    image_grid_thw = mx.array([[1, 4, 6]], dtype=mx.int32)

    position_ids = encoder._compute_multimodal_position_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
    )

    assert position_ids.tolist() == [
        [[0, 1, 2, 2, 2, 2, 2, 2, 5, 6]],
        [[0, 1, 2, 2, 2, 3, 3, 3, 5, 6]],
        [[0, 1, 2, 3, 4, 2, 3, 4, 5, 6]],
    ]


def test_qwen_multimodal_rope_consumes_multiple_reference_grids():
    encoder = QwenEncoder(num_hidden_layers=0)
    image_token = encoder.image_token_id
    input_ids = mx.array([[7, image_token, 8, image_token, image_token, 9]])
    attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
    image_grid_thw = mx.array([[1, 2, 2], [1, 2, 4]], dtype=mx.int32)

    position_ids = encoder._compute_multimodal_position_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
    )

    assert position_ids.tolist() == [
        [[0, 1, 2, 3, 3, 5]],
        [[0, 1, 2, 3, 3, 5]],
        [[0, 1, 2, 3, 4, 5]],
    ]


def test_qwen_multimodal_rope_ignores_padding_positions():
    encoder = QwenEncoder(num_hidden_layers=0)
    image_token = encoder.image_token_id
    input_ids = mx.array([[10, image_token, 11, 0, 0]])
    attention_mask = mx.array([[1, 1, 1, 0, 0]], dtype=mx.int32)
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    position_ids = encoder._compute_multimodal_position_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
    )

    assert position_ids.tolist() == [
        [[0, 1, 2, 0, 0]],
        [[0, 1, 2, 0, 0]],
        [[0, 1, 2, 0, 0]],
    ]


def test_qwen_multimodal_rope_rejects_image_token_grid_mismatch():
    encoder = QwenEncoder(num_hidden_layers=0)
    image_token = encoder.image_token_id
    input_ids = mx.array([[10, image_token, image_token, 11]])
    attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    with pytest.raises(ValueError, match="image token group length"):
        encoder._compute_multimodal_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=2,
        )
