import os

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mflux.release.validation_registry import (
    I2I_EDIT_5X4_PROFILE_ID,
    REFRAME_OUTPAINT_PROFILE_ID,
    ModelValidation,
    ValidationProfile,
    ValidationRecord,
    get_model_validation,
    get_validation_profile,
    list_validation_profiles,
)
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
    "I2I_EDIT_5X4_PROFILE_ID",
    "ModelValidation",
    "ModelCapabilities",
    "REFRAME_OUTPAINT_PROFILE_ID",
    "ResolvedTask",
    "TaskInferenceError",
    "ValidationProfile",
    "ValidationRecord",
    "get_model_capabilities",
    "get_model_validation",
    "get_validation_profile",
    "infer_task",
    "list_validation_profiles",
    "normalize_i2i_mode",
    "normalize_task",
    "resolve_generation_plan",
    "resolve_task",
]
