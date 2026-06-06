import mlx.core as mx
import numpy as np

from mflux.models.common.tokenizer.tokenizer_output import TokenizerOutput
from mflux.models.fibo.model.fibo_text_encoder.prompt_encoder import PromptEncoder


class _RawTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|begin_of_text|>":
            return 128000
        return -1


class _Tokenizer:
    tokenizer = _RawTokenizer()

    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, max_length):
        self.calls.append((prompt, max_length))
        rows = len(prompt)
        return TokenizerOutput(
            input_ids=mx.array([[10, 11, 12], [20, 21, 22]][:rows], dtype=mx.int32),
            attention_mask=mx.array([[1, 1, 1], [1, 1, 1]][:rows], dtype=mx.int32),
        )


class _UnexpectedTokenizer(_Tokenizer):
    def tokenize(self, prompt, max_length):
        raise AssertionError("all-empty FIBO prompts must bypass zero-length tokenizer output")


class _TextEncoder:
    def __init__(self):
        self.input_ids = None
        self.attention_mask = None

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        self.input_ids = np.array(input_ids)
        self.attention_mask = np.array(attention_mask)
        batch, seq = self.input_ids.shape
        return [
            mx.zeros((batch, seq, 2), dtype=mx.float32),
            mx.ones((batch, seq, 2), dtype=mx.float32),
            mx.ones((batch, seq, 2), dtype=mx.float32) * 2,
        ]


def test_fibo_prompt_encoder_uses_bot_token_for_all_empty_prompt_rows():
    text_encoder = _TextEncoder()

    PromptEncoder._get_prompt_embeds(
        prompt=["", ""],
        tokenizer=_UnexpectedTokenizer(),
        text_encoder=text_encoder,
        max_sequence_length=3000,
    )

    assert text_encoder.input_ids.tolist() == [[128000], [128000]]
    assert text_encoder.attention_mask.tolist() == [[1], [1]]


def test_fibo_prompt_encoder_uses_bot_token_for_mixed_empty_prompt_rows():
    tokenizer = _Tokenizer()
    text_encoder = _TextEncoder()

    PromptEncoder._get_prompt_embeds(
        prompt=["{\"edit_instruction\":\"make it blue\"}", ""],
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        max_sequence_length=3000,
    )

    assert tokenizer.calls == [(["{\"edit_instruction\":\"make it blue\"}", ""], 3000)]
    assert text_encoder.input_ids.tolist() == [[10, 11, 12], [128000, 128000, 128000]]
    assert text_encoder.attention_mask.tolist() == [[1, 1, 1], [1, 1, 1]]


def test_fibo_prompt_encoder_casts_outputs_to_requested_dtype():
    _, encoder_hidden_states, prompt_layers, prompt_attention_mask = PromptEncoder.encode_prompt(
        prompt='{"edit_instruction":"make it blue"}',
        negative_prompt="",
        tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(),
        guidance=5.0,
        dtype=mx.bfloat16,
    )

    assert encoder_hidden_states.dtype == mx.bfloat16
    assert prompt_attention_mask.dtype == mx.bfloat16
    assert all(layer.dtype == mx.bfloat16 for layer in prompt_layers)


def test_fibo_prompt_encoder_uses_requested_transformer_layer_count():
    _, _, prompt_layers, _ = PromptEncoder.encode_prompt(
        prompt='{"edit_instruction":"make it blue"}',
        negative_prompt="",
        tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(),
        guidance=5.0,
        total_transformer_layers=5,
    )

    assert len(prompt_layers) == 5
