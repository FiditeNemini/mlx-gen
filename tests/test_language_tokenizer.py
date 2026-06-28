from mflux.models.common.tokenizer.tokenizer import LanguageTokenizer


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(
        self,
        prompts,
        padding,
        max_length,
        truncation,
        add_special_tokens,
        return_length,
        return_overflowing_tokens,
        return_tensors,
    ):
        del padding, max_length, truncation, add_special_tokens, return_length, return_overflowing_tokens, return_tensors
        prompt_lengths = [[len(prompt), 0] for prompt in prompts]
        attention_masks = [[1, 1] for _prompt in prompts]
        return {
            "input_ids": prompt_lengths,
            "attention_mask": attention_masks,
        }


def test_language_tokenizer_keeps_empty_chat_template_prompts_as_real_tokens():
    tokenizer = LanguageTokenizer(
        tokenizer=_FakeTokenizer(),
        use_chat_template=True,
    )

    output = tokenizer.tokenize("")

    assert output.input_ids.shape == (1, 2)
    assert output.attention_mask.shape == (1, 2)
    assert output.input_ids.tolist()[0][0] > 0


def test_language_tokenizer_keeps_empty_template_prompts_as_real_tokens():
    tokenizer = LanguageTokenizer(
        tokenizer=_FakeTokenizer(),
        template="<sys>{}</sys>",
    )

    output = tokenizer.tokenize("")

    assert output.input_ids.shape == (1, 2)
    assert output.attention_mask.tolist() == [[1, 1]]
