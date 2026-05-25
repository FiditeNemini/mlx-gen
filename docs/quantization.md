# Quantization

MLX-Gen prepares quantized model folders with the same mflux/MLX layout used for local inference. Use `mlxgen prepare --model ... --path ... --quantize ...` to create those folders. They are designed for MLX-Gen and are not Diffusers or Transformers `from_pretrained()` checkpoints.

## Qwen q4

Qwen Image and Qwen Image Edit use a mixed q4/q8 policy when prepared with `--quantize 4`. Fully q4 Qwen checkpoints can lose coherent generative behavior, so MLX-Gen keeps only the sensitive paths at higher precision:

- q4 for most Qwen transformer attention, feed-forward, and projection linears.
- q8 for Qwen `*.img_mod_linear` transformer modulation layers.
- q4 for group64-compatible Qwen text-encoder language linears.
- q8 for group64-compatible Qwen text-encoder visual linears.
- BF16 for the VAE, norms, embeddings, and linears that are not MLX group64-compatible.

This policy applies to Qwen q4 prepared folders only. It is used for Qwen Image and Qwen Image Edit variants, including 2509 and 2511 edit checkpoints.

## q8

The q8 path was not changed by the mixed-q4 work. Qwen q8 uses the standard MLX-Gen/mflux quantization flow: quantizable modules are saved at 8-bit where the model layout supports MLX quantization, while VAE weights and non-quantizable layers remain BF16.

Other model families use their existing model-specific quantization predicates.

## ERNIE Image Turbo

ERNIE Image Turbo supports BF16, q8, and q4 text-to-image generation. Use `mlxgen prepare` to create reusable q8 or q4 folders:

```sh
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-8bit --quantize 8
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-4bit --quantize 4
```

ERNIE q4 uses full MLX quantization for modules where MLX supports quantization. Current validation did not show the Qwen-style q4 quality failure, so ERNIE does not use a mixed q4/q8 policy.

- q4/q8 for quantizable ERNIE transformer modules.
- q4/q8 for quantizable ERNIE text-encoder modules.
- q4/q8 for quantizable ERNIE VAE attention modules.
- BF16 for norms, convolutions, and other non-quantizable parameters.

Local validation on Apple Silicon with 512x512, 8 steps, guidance 1:

| Layout | Folder Size | Peak RSS | Average Generation Time | Notes |
| --- | ---: | ---: | ---: | --- |
| BF16 source generation components | ~22.4 GiB | 23.5 GiB | 6.38 s | Fastest at 512px, largest memory footprint. |
| q8 prepared folder | 12 GiB | 12.9 GiB | 7.57 s | About half the memory footprint. |
| q4 prepared folder | 6.2 GiB | 7.2 GiB | 9.31 s | Smallest footprint; coherent output in validation. |

At 1024x1024 with 8 steps and guidance 1, q8 generated in 84.69 s with 12.9 GiB peak RSS, and q4 generated in 78.94 s with 7.2 GiB peak RSS. Speed is workload-sensitive, but q4 is the best memory-footprint option and remains usable for full-resolution ERNIE Turbo generation.

Prepared ERNIE q8/q4 folders contain the ordinary text-to-image generation components. ERNIE Prompt Enhancer remains an optional full-source-snapshot feature and is not bundled into prepared quantized folders.

## Compatibility

Saved MLX-Gen folders can be loaded by MLX-Gen and by compatible mflux code that understands the same saved-weight layout and quantization predicates. They are not directly readable by Diffusers or Transformers because the files contain MLX quantization tensors and the mflux/MLX component layout.
