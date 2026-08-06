# Bernini-R 1.3B MLX model-backed proof

- Created: `2026-08-04T06:24:12+0200`
- Official source revision: `2d2b4591ac053ec25c6371b01a5a6746679e5793`
- Machine-level contract pass: `True`
- Visual evidence review complete and hash-bound: `True`
- Required visual quality cases pass: `False`
- Overall pass: `False`
- Recorded visual inspection: see `visual_review.md` and `visual_review.json`.

| Case | Mode | Size | Frames | Steps | Wall | Physical peak | Contract | Visual |
|---|---|---:|---:|---:|---:|---:|---|---|
| r2v_eight_reference | r2v_apg | 320x192 | 17 | 20 | 248.95s | 9.116 GB | True | fail |
| rv2v_garment | rv2v | 176x320 | 17 | 20 | 451.62s | 9.215 GB | True | fail |
| rv2v_no_reference_control | v2v_apg | 176x320 | 17 | 20 | 335.42s | 9.448 GB | True | negative_result |
| rv2v_no_source_control | r2v_apg | 176x320 | 17 | 20 | 209.44s | 8.991 GB | True | negative_result |
| rv2v_reference_pinstripe_ab | rv2v | 176x320 | 17 | 20 | 279.13s | 9.094 GB | True | fail |
| rv2v_reference_black_ab | rv2v | 176x320 | 17 | 20 | 122.04s | 4.707 GB | True | fail |
| rv2v_reference_none_ab | v2v_apg | 176x320 | 17 | 20 | 221.62s | 9.030 GB | True | negative_result |
| v2v_snowman | v2v_apg | 320x176 | 17 | 20 | 212.61s | 9.215 GB | True | fail |
| r2v_848_condition_smoke | r2v_apg | 128x128 | 5 | 1 | 14.57s | 5.776 GB | True | structural_only |
| r2v_1280_condition_smoke | r2v_apg | 128x128 | 5 | 1 | 20.86s | 9.340 GB | True | structural_only |

## Artifacts

### r2v_eight_reference

- MP4: `<bundle-root>/cases/run_1/r2v_eight_reference/r2v_eight_reference_17f.mp4`
- SHA-256: `b13faa921a7db7a119324e80223c949fc79924a630673db5c061a61027e77ae8`
- Metadata: `<bundle-root>/cases/run_1/r2v_eight_reference/r2v_eight_reference_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/r2v_eight_reference/mlx_contact_sheet.png", "official_page_01": "<bundle-root>/cases/run_1/r2v_eight_reference/official_contact_sheet_page_01.png", "official_page_02": "<bundle-root>/cases/run_1/r2v_eight_reference/official_contact_sheet_page_02.png", "official_page_03": "<bundle-root>/cases/run_1/r2v_eight_reference/official_contact_sheet_page_03.png", "official_page_04": "<bundle-root>/cases/run_1/r2v_eight_reference/official_contact_sheet_page_04.png", "official_page_05": "<bundle-root>/cases/run_1/r2v_eight_reference/official_contact_sheet_page_05.png", "references": "<bundle-root>/cases/run_1/r2v_eight_reference/reference_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/r2v_eight_reference/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "eight_latent_shapes": true, "eight_source_ids": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### rv2v_garment

- MP4: `<bundle-root>/cases/run_1/rv2v_garment/rv2v_garment_17f.mp4`
- SHA-256: `e70f01590db4c14298ae7cdc7de07acd7c20ec000d39eb441723e98bb1ad63d7`
- Metadata: `<bundle-root>/cases/run_1/rv2v_garment/rv2v_garment_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/rv2v_garment/mlx_contact_sheet.png", "official_page_01": "<bundle-root>/cases/run_1/rv2v_garment/official_contact_sheet_page_01.png", "official_page_02": "<bundle-root>/cases/run_1/rv2v_garment/official_contact_sheet_page_02.png", "official_page_03": "<bundle-root>/cases/run_1/rv2v_garment/official_contact_sheet_page_03.png", "official_page_04": "<bundle-root>/cases/run_1/rv2v_garment/official_contact_sheet_page_04.png", "official_page_05": "<bundle-root>/cases/run_1/rv2v_garment/official_contact_sheet_page_05.png", "references": "<bundle-root>/cases/run_1/rv2v_garment/reference_contact_sheet.png", "source": "<bundle-root>/cases/run_1/rv2v_garment/source_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/rv2v_garment/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### rv2v_no_reference_control

- MP4: `<bundle-root>/cases/run_1/rv2v_no_reference_control/rv2v_no_reference_control_17f.mp4`
- SHA-256: `ff9204287cd9eddab8e2cac5cddcdc36c02ac62780df39c3f5c64b997104f4dd`
- Metadata: `<bundle-root>/cases/run_1/rv2v_no_reference_control/rv2v_no_reference_control_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/rv2v_no_reference_control/mlx_contact_sheet.png", "source": "<bundle-root>/cases/run_1/rv2v_no_reference_control/source_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/rv2v_no_reference_control/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### rv2v_no_source_control

- MP4: `<bundle-root>/cases/run_1/rv2v_no_source_control/rv2v_no_source_control_17f.mp4`
- SHA-256: `f55f201e8c496d8f2c64aea9b6721828dae56f5e3699c1b5d6fb32a4b42259de`
- Metadata: `<bundle-root>/cases/run_1/rv2v_no_source_control/rv2v_no_source_control_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/rv2v_no_source_control/mlx_contact_sheet.png", "references": "<bundle-root>/cases/run_1/rv2v_no_source_control/reference_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/rv2v_no_source_control/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### rv2v_reference_pinstripe_ab

- MP4: `<bundle-root>/cases/run_1/rv2v_reference_pinstripe_ab/rv2v_reference_pinstripe_ab_17f.mp4`
- SHA-256: `ac447a502bda2287ee87657431bcfb37cc374435f317c5fd55703083ec863e91`
- Metadata: `<bundle-root>/cases/run_1/rv2v_reference_pinstripe_ab/rv2v_reference_pinstripe_ab_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/rv2v_reference_pinstripe_ab/mlx_contact_sheet.png", "references": "<bundle-root>/cases/run_1/rv2v_reference_pinstripe_ab/reference_contact_sheet.png", "source": "<bundle-root>/cases/run_1/rv2v_reference_pinstripe_ab/source_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/rv2v_reference_pinstripe_ab/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### rv2v_reference_black_ab

- MP4: `<bundle-root>/cases/run_1/rv2v_reference_black_ab/rv2v_reference_black_ab_17f.mp4`
- SHA-256: `b10c410f234c713057aa244be2712c8d4244739bdff57b1975dcbadd7df654cc`
- Metadata: `<bundle-root>/cases/run_1/rv2v_reference_black_ab/rv2v_reference_black_ab_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/rv2v_reference_black_ab/mlx_contact_sheet.png", "references": "<bundle-root>/cases/run_1/rv2v_reference_black_ab/reference_contact_sheet.png", "source": "<bundle-root>/cases/run_1/rv2v_reference_black_ab/source_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/rv2v_reference_black_ab/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### rv2v_reference_none_ab

- MP4: `<bundle-root>/cases/run_1/rv2v_reference_none_ab/rv2v_reference_none_ab_17f.mp4`
- SHA-256: `16c68a54043501b617908af54fa67b460daedc62ca81ebd40277e8d6a9b746f2`
- Metadata: `<bundle-root>/cases/run_1/rv2v_reference_none_ab/rv2v_reference_none_ab_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/rv2v_reference_none_ab/mlx_contact_sheet.png", "source": "<bundle-root>/cases/run_1/rv2v_reference_none_ab/source_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/rv2v_reference_none_ab/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### v2v_snowman

- MP4: `<bundle-root>/cases/run_1/v2v_snowman/v2v_snowman_17f.mp4`
- SHA-256: `ecb8e77f5e89227991bb7f5e1ff3eeb29dfe49a55b0d187d72ce1a3c54fef249`
- Metadata: `<bundle-root>/cases/run_1/v2v_snowman/v2v_snowman_17f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/v2v_snowman/mlx_contact_sheet.png", "official_page_01": "<bundle-root>/cases/run_1/v2v_snowman/official_contact_sheet_page_01.png", "official_page_02": "<bundle-root>/cases/run_1/v2v_snowman/official_contact_sheet_page_02.png", "official_page_03": "<bundle-root>/cases/run_1/v2v_snowman/official_contact_sheet_page_03.png", "official_page_04": "<bundle-root>/cases/run_1/v2v_snowman/official_contact_sheet_page_04.png", "official_page_05": "<bundle-root>/cases/run_1/v2v_snowman/official_contact_sheet_page_05.png", "source": "<bundle-root>/cases/run_1/v2v_snowman/source_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/v2v_snowman/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### r2v_848_condition_smoke

- MP4: `<bundle-root>/cases/run_1/r2v_848_condition_smoke/r2v_848_condition_smoke_5f.mp4`
- SHA-256: `bb4122738c4a54e7b6aa97c3db864b0887a786ffa78b01e93c8b64f9a11121dd`
- Metadata: `<bundle-root>/cases/run_1/r2v_848_condition_smoke/r2v_848_condition_smoke_5f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/r2v_848_condition_smoke/mlx_contact_sheet.png", "references": "<bundle-root>/cases/run_1/r2v_848_condition_smoke/reference_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/r2v_848_condition_smoke/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

### r2v_1280_condition_smoke

- MP4: `<bundle-root>/cases/run_1/r2v_1280_condition_smoke/r2v_1280_condition_smoke_5f.mp4`
- SHA-256: `b442a7025a76bac872f7bffe350b6d9b66f28cd78e0a3db4b4872afef10dc5dd`
- Metadata: `<bundle-root>/cases/run_1/r2v_1280_condition_smoke/r2v_1280_condition_smoke_5f.metadata.json`
- Contact sheets: `{"mlx": "<bundle-root>/cases/run_1/r2v_1280_condition_smoke/mlx_contact_sheet.png", "references": "<bundle-root>/cases/run_1/r2v_1280_condition_smoke/reference_contact_sheet.png", "worst_transitions": "<bundle-root>/cases/run_1/r2v_1280_condition_smoke/mlx_worst_transitions_contact_sheet.png"}`
- Contract checks: `{"bf16_unquantized": true, "component_provenance": true, "factored_sources": true, "fps": true, "frames": true, "guidance_mode": true, "height": true, "max_condition_size": true, "nonzero_temporal_change": true, "output_exists": true, "physical_memory_sampled": true, "process_exit": true, "prompt_exact": true, "prompt_truncation_expected": true, "prompt_truncation_recorded": true, "reference_count": true, "reference_order": true, "renderer_checkpoint": true, "renderer_only": true, "required_sheets": true, "runtime_precision": true, "sheet_contract": true, "sheet_topology": true, "source_path": true, "stderr_log": true, "stdout_log": true, "steps": true, "video_health": true, "width": true}`

