# Bernini-R 1.3B official public parity bundle

This bundle assembles the current official public-case proof rows from the local Bernini 1.3B validation runs.
Each case directory contains the human-readable case README, prompt, expected result, actual result,
reproduce command, high-resolution input/official/mlx-gen contact sheets, and the generated mlx artifact
(video or image) with metadata when available.

Dispositioned rows (`r2v_case2_*`, `v2v_case3_*`) document oracle-proven 1.3B limits and the
tuned recovery recipes described in the parity matrix.

## Included rows

| Row | Task | Proof |
| --- | --- | --- |
| `t2i` | T2I Van Gogh corgi | [t2i/README.md](t2i/README.md) |
| `i2i` | I2I bicycle removal | [i2i/README.md](i2i/README.md) |
| `t2v` | T2V polar bear guitar | [t2v/README.md](t2v/README.md) |
| `v2v_case1` | V2V add snowman | [v2v_case1/README.md](v2v_case1/README.md) |
| `mv2v` | MV2V crouching beside dog | [mv2v/README.md](mv2v/README.md) |
| `r2v` | R2V statue on bench | [r2v/README.md](r2v/README.md) |
| `rv2v_case1` | RV2V garment replacement | [rv2v_case1/README.md](rv2v_case1/README.md) |
| `ads2v` | ADS2V insert video on computer | [ads2v/README.md](ads2v/README.md) |
| `r2v_case2_official` | R2V statue with cup | [r2v_case2_official/README.md](r2v_case2_official/README.md) |
| `r2v_case2_tuned` | R2V statue with cup | [r2v_case2_tuned/README.md](r2v_case2_tuned/README.md) |
| `v2v_case3_official` | V2V robot to robotic dog | [v2v_case3_official/README.md](v2v_case3_official/README.md) |
| `v2v_case3_mv2vprefix` | V2V robot to robotic dog (mv2v-prefix recovery) | [v2v_case3_mv2vprefix/README.md](v2v_case3_mv2vprefix/README.md) |

## Manifest

- [manifest.json](manifest.json)
- [bernini_proof_report.json](bernini_proof_report.json)

