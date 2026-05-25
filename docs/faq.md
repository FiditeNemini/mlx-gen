# FAQ

## Where Is The Save Command?

Use `mlxgen prepare`.

`prepare` is the public MLX-Gen workflow for creating a reusable local model folder. It can quantize the model with `--quantize` and writes a Hugging Face `README.md` model card into the prepared folder.

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

New MLX-Gen integrations should not call a separate save workflow; use `mlxgen prepare`.

## What Is The Difference Between Download And Prepare?

`mlxgen download` populates the local Hugging Face cache with source files. It does not create a separate model folder and does not write a model card.

`mlxgen prepare` creates a reusable local MLX-Gen folder at `--path`. It can quantize weights and writes a generated Hugging Face model card.

## Can Generate Prepare A Model Folder?

No. `mlxgen generate` is only for inference. It does not accept `--path` for model preparation.

To create a model folder, use:

```sh
mlxgen prepare --model black-forest-labs/FLUX.2-klein-4B --path models/flux.2-klein-4b-4bit --quantize 4
```

To choose the output image path during generation, use `--output`.

## Does Generation Download Missing Files?

No. Generation and ordinary Python model construction are cache-only by default. Missing artifacts raise `DownloadRequiredError` with the exact `mlxgen download` or `mlxgen prepare` command to run.

## Is `HF_HUB_ENABLE_HF_TRANSFER=1` Required?

No. It is optional acceleration for explicit Hugging Face downloads and prepare operations. `mlxgen download` and `mlxgen prepare` already authorize network access.

## Can Prepared Folders Load In Diffusers Or Transformers?

No. Prepared MLX-Gen folders use the MLX/mflux saved-weight layout and MLX quantization tensors. They are intended for MLX-Gen and compatible mflux code, not direct Diffusers or Transformers `from_pretrained()` loading.

## Can I Quantize ERNIE Image Turbo?

Yes. ERNIE Image Turbo supports q8 and q4 prepared folders:

```sh
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-8bit --quantize 8
mlxgen prepare --model baidu/ERNIE-Image-Turbo --path ./models/ernie-image-turbo-4bit --quantize 4
```

ERNIE q4 uses full MLX quantization for quantizable ERNIE modules. It does not currently require the Qwen mixed q4/q8 policy.

## Does ERNIE Image Turbo Support Image Input Or Prompt Enhancer?

Prompt Enhancer is supported for ERNIE Image Turbo when the full source snapshot is available:

```sh
mlxgen download --model baidu/ERNIE-Image-Turbo --all-files

mlxgen generate \
  --model baidu/ERNIE-Image-Turbo \
  --prompt "A ceramic mug" \
  --use-prompt-enhancer
```

ERNIE image-to-image/edit inputs are still rejected instead of silently ignoring them.

Prepared ERNIE q8/q4 folders do not bundle Prompt Enhancer files; use the full source snapshot path or the Hugging Face repo after `mlxgen download --all-files` when you need `--use-prompt-enhancer`.

## Why Do Some Imports Or Paths Still Say `mflux`?

MLX-Gen is currently built on the mflux codebase. Some internal modules and compatibility entry points still use `mflux.*` names while the public package and command surface evolve under `mlx-gen` and `mlxgen`.

## How Does This Relate To AbstractVision?

MLX-Gen is intended to be the Apple Silicon / MLX backend dependency for AbstractVision. AbstractVision remains a cross-platform orchestration layer, while MLX-Gen owns MLX model loading, quantized local formats, and Apple Silicon runtime behavior.
