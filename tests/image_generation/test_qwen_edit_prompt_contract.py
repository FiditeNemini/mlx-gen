from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit


class _CapturingProcessor:
    def __init__(self):
        self.text = None
        self.image_count = None
        self.image_sizes = None

    def __call__(self, *, text, images, padding, return_tensors):
        self.text = text[0]
        self.image_count = len(images)
        self.image_sizes = [image.size for image in images]
        return {
            "input_ids": mx.array([[1, 2, 3]]),
            "attention_mask": mx.array([[1, 1, 1]]),
            "pixel_values": np.zeros((len(images), 1), dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 3]], dtype=np.int32),
        }


def test_qwen_edit_plus_prompt_places_one_picture_token_per_reference(tmp_path):
    image_1 = tmp_path / "source.png"
    image_2 = tmp_path / "style.png"
    Image.new("RGB", (512, 256)).save(image_1)
    Image.new("RGB", (256, 512)).save(image_2)
    processor = _CapturingProcessor()
    tokenizer = QwenVisionLanguageTokenizer(processor=processor, use_picture_prefix=False)

    tokenizer.tokenize_with_image(
        prompt="compose the references",
        image=[str(image_1), str(image_2)],
        use_picture_prefix=True,
    )

    assert processor.image_count == 2
    assert "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" in processor.text
    assert "Picture 2: <|vision_start|><|image_pad|><|vision_end|>" in processor.text
    assert processor.text.count("<|vision_start|><|image_pad|><|vision_end|>") == 2


def test_qwen_regular_edit_tokenizer_honors_single_image_vl_size(tmp_path):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (512, 256)).save(image_path)
    processor = _CapturingProcessor()
    tokenizer = QwenVisionLanguageTokenizer(processor=processor, use_picture_prefix=False)

    tokenizer.tokenize_with_image(
        prompt="make it a sketch",
        image=str(image_path),
        vl_width=1024,
        vl_height=512,
        use_picture_prefix=False,
    )

    assert processor.image_sizes == [(1024, 512)]


def test_qwen_initializer_rejects_all_zero_text_encoder_projection():
    weights = LoadedWeights(
        components={
            "text_encoder": {
                "encoder": {
                    "layers": [
                        {
                            "self_attn": {
                                "q_proj": {"weight": mx.zeros((2, 2))},
                            },
                        }
                    ],
                },
            },
        },
        meta_data=MetaData(),
    )

    with pytest.raises(ValueError, match="Qwen text encoder weights appear corrupt"):
        QwenImageInitializer._validate_text_encoder_weights(weights, "Qwen/Qwen-Image-Edit")


def test_qwen_initializer_accepts_nonzero_text_encoder_projection():
    weights = LoadedWeights(
        components={
            "text_encoder": {
                "encoder": {
                    "layers": [
                        {
                            "self_attn": {
                                "q_proj": {"weight": mx.array([[0, 0], [0, 1]])},
                            },
                        }
                    ],
                },
            },
        },
        meta_data=MetaData(),
    )

    QwenImageInitializer._validate_text_encoder_weights(weights, "Qwen/Qwen-Image-Edit")


def test_qwen_initializer_ignores_partial_weight_fixtures():
    weights = LoadedWeights(components={"text_encoder": {}}, meta_data=MetaData())

    QwenImageInitializer._validate_text_encoder_weights(weights, "test-fixture")


def test_qwen_edit_plus_prompt_is_enabled_by_model_config():
    edit_regular = SimpleNamespace(model_config=ModelConfig.qwen_image_edit())
    edit_2509 = SimpleNamespace(model_config=ModelConfig.qwen_image_edit_2509())
    edit_2511 = SimpleNamespace(model_config=ModelConfig.from_name("qwen-image-edit-2511"))
    base_text = SimpleNamespace(model_config=ModelConfig.qwen_image())

    assert not QwenImageEdit._should_use_edit_plus_prompt(edit_regular, ["source.png"])
    assert not QwenImageEdit._should_use_edit_plus_prompt(edit_regular, ["source.png", "style.png"])
    assert QwenImageEdit._should_use_edit_plus_prompt(edit_2509, ["source.png"])
    assert QwenImageEdit._should_use_edit_plus_prompt(edit_2509, ["source.png", "style.png"])
    assert QwenImageEdit._should_use_edit_plus_prompt(edit_2511, ["source.png"])
    assert not QwenImageEdit._should_use_edit_plus_prompt(base_text, ["source.png"])
    assert not QwenImageEdit._should_use_edit_plus_prompt(base_text, ["source.png", "style.png"])


def test_qwen_edit_true_cfg_requires_negative_prompt():
    assert not QwenImageEdit._should_use_true_cfg(guidance=4.0, negative_prompt=None)
    assert QwenImageEdit._should_use_true_cfg(guidance=4.0, negative_prompt="")
    assert not QwenImageEdit._should_use_true_cfg(guidance=1.0, negative_prompt="")


def test_qwen_edit_models_use_official_blank_negative_prompt_for_true_cfg():
    edit_regular = SimpleNamespace(model_config=ModelConfig.qwen_image_edit())
    edit_2509 = SimpleNamespace(model_config=ModelConfig.qwen_image_edit_2509())
    edit_2511 = SimpleNamespace(model_config=ModelConfig.from_name("qwen-image-edit-2511"))

    assert (
        QwenImageEdit._resolve_negative_prompt_for_model(
            edit_regular,
            guidance=4.0,
            negative_prompt=None,
            image_paths=["source.png"],
        )
        == " "
    )
    assert (
        QwenImageEdit._resolve_negative_prompt_for_model(
            edit_2509,
            guidance=4.0,
            negative_prompt=None,
            image_paths=["source.png"],
        )
        == " "
    )
    assert (
        QwenImageEdit._resolve_negative_prompt_for_model(
            edit_2511,
            guidance=4.0,
            negative_prompt=None,
            image_paths=["source.png"],
        )
        == " "
    )
    assert (
        QwenImageEdit._resolve_negative_prompt_for_model(
            edit_2511,
            guidance=4.0,
            negative_prompt="low quality",
            image_paths=["source.png"],
        )
        == "low quality"
    )
    assert (
        QwenImageEdit._resolve_negative_prompt_for_model(
            edit_2511,
            guidance=1.0,
            negative_prompt=None,
            image_paths=["source.png"],
        )
        is None
    )


def test_qwen_edit_plus_python_default_steps_match_model_card():
    edit_regular = SimpleNamespace(model_config=ModelConfig.qwen_image_edit())
    edit_2509 = SimpleNamespace(model_config=ModelConfig.qwen_image_edit_2509())
    edit_2511 = SimpleNamespace(model_config=ModelConfig.from_name("qwen-image-edit-2511"))

    assert QwenImageEdit._default_num_inference_steps(edit_regular, None, image_paths=["source.png"]) == 50
    assert QwenImageEdit._default_num_inference_steps(edit_2509, None, image_paths=["source.png"]) == 40
    assert QwenImageEdit._default_num_inference_steps(edit_2511, None, image_paths=["source.png"]) == 40
    assert QwenImageEdit._default_num_inference_steps(edit_2511, 12, image_paths=["source.png"]) == 12


def test_qwen_2511_config_enables_zero_condition_timestep():
    qwen_2509_config = ModelConfig.qwen_image_edit_2509()
    qwen_2511_config = ModelConfig.from_name("qwen-image-edit-2511")
    qwen_2509 = QwenTransformer(
        num_layers=0,
        zero_cond_t=bool(qwen_2509_config.transformer_overrides.get("zero_cond_t", False)),
    )
    qwen_2511 = QwenTransformer(
        num_layers=0,
        zero_cond_t=bool(qwen_2511_config.transformer_overrides.get("zero_cond_t", False)),
    )

    assert not qwen_2509.zero_cond_t
    assert qwen_2511.zero_cond_t


def test_qwen_zero_cond_modulate_index_marks_reference_tokens():
    index = QwenTransformer._compute_modulate_index(
        batch_size=1,
        img_shapes=[(1, 2, 3), (1, 1, 2)],
        sequence_length=8,
    )

    assert np.array(index).tolist() == [[0, 0, 0, 0, 0, 0, 1, 1]]


def test_qwen_zero_cond_modulation_selects_target_and_reference_embeddings():
    x = mx.zeros((1, 3, 1))
    mod_params = mx.array(
        [
            [10.0, 0.0, 1.0],
            [20.0, 0.0, 2.0],
        ]
    )
    index = mx.array([[0, 0, 1]], dtype=mx.int32)

    modulated, gate = QwenTransformerBlock._modulate(x, mod_params, index)

    assert np.array(modulated).reshape(-1).tolist() == [10.0, 10.0, 20.0]
    assert np.array(gate).reshape(-1).tolist() == [1.0, 1.0, 2.0]
