import pytest

from mflux.models.common.config import Config, ModelConfig
from mflux.models.common.config.inference_defaults import default_inference_steps


@pytest.mark.fast
def test_default_inference_steps_follow_model_config_aliases():
    assert default_inference_steps(ModelConfig.dev()) == 25
    assert default_inference_steps(ModelConfig.z_image()) == 50
    assert default_inference_steps(ModelConfig.z_image_turbo()) == 9
    assert default_inference_steps(ModelConfig.fibo_edit_rmbg()) == 10
    assert default_inference_steps(ModelConfig.flux2_klein_base_4b()) == 50


@pytest.mark.fast
def test_default_inference_steps_follow_explicit_base_model_for_custom_paths():
    model_config = ModelConfig.from_name("/models/custom-qwen", base_model="qwen-image")

    assert default_inference_steps(model_config) == 20


@pytest.mark.fast
def test_config_uses_model_default_when_steps_are_unspecified():
    config = Config(model_config=ModelConfig.z_image(), height=1024, width=1024)

    assert config.num_inference_steps == 50
