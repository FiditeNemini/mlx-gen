import os

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mflux.task_inference import (
    GenerationCapability,
    GenerationPlan,
    ModelCapabilities,
    ResolvedTask,
    TaskInferenceError,
    get_model_capabilities,
    infer_task,
    normalize_i2i_mode,
    normalize_task,
    resolve_generation_plan,
    resolve_task,
)

__all__ = [
    "GenerationCapability",
    "GenerationPlan",
    "ModelCapabilities",
    "ResolvedTask",
    "TaskInferenceError",
    "get_model_capabilities",
    "infer_task",
    "normalize_i2i_mode",
    "normalize_task",
    "resolve_generation_plan",
    "resolve_task",
]
