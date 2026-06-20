# Architecture Decision Records

This directory contains durable engineering decisions for MLX-Gen. ADRs are policy, not task
tracking; backlog items record execution state and evidence.

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR 0001](0001_runtime_smoke_validation_for_model_routes.md) | Accepted | New model routes need model-backed smoke proof before being described as working. |
| [ADR 0002](0002_no_silent_automatic_fallbacks.md) | Accepted | Ambiguous or unsupported requests fail closed; automatic fallback is explicit opt-in only. |
| [ADR 0003](0003_runtime_truth_vs_consumer_convenience.md) | Accepted | MLX-Gen owns exact runtime truth; higher-level integrations own convenience and curation. |
