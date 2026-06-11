import mlx.core as mx
import mlx.nn as nn
import pytest

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_saver import LoRABakeError, LoRASaver


def test_lora_saver_rejects_shape_mismatch():
    base = nn.Linear(4, 3, bias=False)
    lora = LoRALinear.from_linear(base, r=2, scale=1.0)
    lora.lora_A = mx.zeros((5, 5))
    lora.lora_B = mx.zeros((5, 5))

    with pytest.raises(LoRABakeError, match="shape mismatch"):
        LoRASaver.bake_and_strip_lora(lora)
