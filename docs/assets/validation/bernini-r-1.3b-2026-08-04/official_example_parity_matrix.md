# Bernini-R 1.3B official example parity matrix

## Metadata

- Created: 2026-08-10
- Scope date: 2026-08-10
- Scope: official public `ByteDance/Bernini-R-1.3B-Diffusers` examples only
- Official source revision: `2d2b4591ac053ec25c6371b01a5a6746679e5793`

## What this matrix means

This matrix defines the stricter target the current Bernini work should be judged against:

- use the public `ByteDance/Bernini-R-1.3B-Diffusers` checkpoint;
- use the public official Bernini code and bundled public case files;
- use the official example inputs without local workaround assets;
- judge success qualitatively on the official task outcome, not on pixel identity.

This is narrower than "general-case Bernini is product-ready" and stricter than "some custom local
Bernini runs looked interesting." Accepted rows below means the current local MLX output was
manually judged qualitatively good against the public official example, not that the output is
pixel-identical.

## Official public example inventory

The public 1.3B model card, testcase tree, and helper scripts expose these example rows:

| Row | Upstream case | Upstream notes | Current mlx-gen status | Immediate next action |
| --- | --- | --- | --- | --- |
| T2I | `assets/testcases/t2i/t2i.json` | single GPU, `--num_frames 1` | Accepted | Keep as accepted current parity evidence; public CLI surfacing is optional follow-up. |
| I2I | `assets/testcases/i2i/i2i.json` | single GPU, `--num_frames 1` | Accepted with minor wall-texture caveat | Keep as accepted current parity evidence; public CLI surfacing is optional follow-up. |
| T2V | `assets/testcases/t2v/t2v.json` | official multi-GPU renderer-only text-to-video example | Accepted | Keep as accepted current parity evidence; public CLI surfacing is optional follow-up. |
| V2V case 1 | `assets/testcases/v2v/v2v_case1.json` | add a snowman | Open | Re-run or re-review against the current implementation; do not reuse the older bounded failed bundle as the final word. |
| V2V case 2 / MV2V | `assets/testcases/v2v/v2v_case2.json` | make the person crouch down | Open | Run a full current official-case parity pass; typed `mv2v` support exists internally but is not yet proven here. |
| V2V case 3 | `assets/testcases/v2v/v2v_case3.json` | replace humanoid robot with robotic dog | Open | Promote beyond the existing step-1 probe to a full current official-case run. |
| R2V case 1 | `assets/testcases/r2v/r2v.json` | five-reference statue-on-bench case | Accepted | Keep as accepted current parity evidence. |
| R2V case 2 | `assets/testcases/r2v/r2v_case2.json` | eight-reference statue-with-cup case | Open | Re-run or re-review as a distinct official row; the older bounded bundle is not enough. |
| RV2V case 1 | `assets/testcases/rv2v/rv2v_case1.json` | garment replacement | Accepted | Keep as accepted current parity evidence; shirt closure differs slightly from the official clip, but the garment transfer and body fit now read correctly. |
| ADS2V case 1 | `assets/testcases/rv2v/rv2v_case2.json` | two-video insertion at `121` frames, `24` fps, `1280` max image size | Open | Run a full current official-case pass; typed `ads2v` support exists internally but is not yet proven here. |

## Current gap, stated plainly

As of Monday, August 10, 2026:

- 5 official public rows are accepted on current local review:
  - `i2i`
  - `t2i`
  - `t2v`
  - `r2v`
  - `rv2v_case1`
- 5 official public rows are still open:
  - `v2v_case1`
  - `v2v_case2` / `mv2v`
  - `v2v_case3`
  - `r2v_case2`
  - `ads2v`

So the honest current state is no longer "only old failed bounded proofs exist." The current state
is:

- five official rows are qualitatively accepted;
- the rest still need current official-case proof.

## Accepted evidence on Monday, August 10, 2026

- `i2i`
  - MLX: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_i2i/i2i/mlx_sheet.png`
  - Official: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_i2i/i2i/official_sheet.png`
- `t2i`
  - MLX: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_t2i/t2i/mlx_sheet.png`
  - Official: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_t2i/t2i/official_sheet.png`
- `t2v`
  - MLX: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_t2v/t2v/mlx_sheet.png`
  - Official: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity/exact_noise_t2v/t2v/official_sheet.png`
- `r2v`
  - MLX: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_segmented_r2v_40step_launchd_round7/r2v/mlx_sheet.png`
  - Official: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_segmented_r2v_40step_launchd_round7/r2v/official_sheet.png`
- `rv2v_case1`
  - MLX: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1/rv2v_case1/mlx_sheet.png`
  - Official: `validation_outputs/bernini_r_1_3b_2026_08_10/official_parity_rv2v_case1_steps20_current_launchd_v1/rv2v_case1/official_sheet.png`
  - Note: the mlx-gen shirt stays more closed than the official output, so less undershirt is
    visible, but the replacement is stable and fits correctly on-body.

## Next row

- `v2v_case1`
  - next full official rerun target

## Known upstream ambiguities that must be resolved explicitly

### 1. Prompt enhancer ambiguity

The public 1.3B model card says `--use_pe` is "highly recommended" for best quality, but the
published example commands do not state which retained outputs were generated with it. Official
parity should therefore be tracked in two layers:

1. raw public case-file parity without prompt enhancement;
2. prompt-enhanced parity only if the raw case still misses and the upstream retained output is
   shown to depend on `--use_pe`.

### 2. Multiple official R2V rows, not one

The current pinned upstream revision publicly contains both `assets/testcases/r2v/r2v.json` and
`assets/testcases/r2v/r2v_case2.json`, and the helper script launches both. The parity target must
therefore treat them as two distinct official rows rather than treating one as a naming mistake.

### 3. Multi-GPU recipe versus semantic contract

The public docs show the video examples with `torchrun --nproc-per-node 8 --ulysses 8`, but the
same docs also state that the single-GPU and multi-GPU scripts take the same inputs. The
eight-GPU launcher is therefore an execution recipe, not by itself a different semantic contract.

## Immediate execution order

1. Re-run or re-review the unresolved current video rows:
   - `v2v_case1`
   - `v2v_case3`
   - `v2v_case2` / `mv2v`
2. Close the remaining distinct reference rows:
   - `r2v_case2`
   - `ads2v`
3. Only after the 1.3B public matrix is current, decide whether larger Bernini rows are justified.

## Why this matters

Without this matrix, "Bernini parity" can mean three incompatible things:

- current narrow renderer routes pass local custom diagnostics;
- currently surfaced official-like cases pass;
- all public 1.3B examples from the vendor pass.

This file fixes the target to the third meaning.

## Source references

- Public 1.3B model card: https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers
- Public 1.3B files tree: https://huggingface.co/ByteDance/Bernini-R-1.3B-Diffusers/tree/main
- Locked inventory snapshot: [official_example_inventory.json](official_example_inventory.json)
- Official testcases README: https://github.com/bytedance/Bernini/blob/main/assets/testcases/README.md
- Official repository README/examples: https://github.com/bytedance/Bernini
- Official Gradio task/defaults file: https://github.com/bytedance/Bernini/blob/main/gradio_demo.py
