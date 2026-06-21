# SeedVR2 Eiffel 70s-75s Validation Report

This bundle is the accepted public SeedVR2 video-restoration proof for the June 21 route.

## Scope

- source clip: `70.0s` to `75.0s`
- source length: `149` frames at `29.9700 fps`
- source size: `320x240`
- compared models: official `ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B`
- compared profiles:
  - native-resolution restore: `1x`, chunk `29/8`
  - enlarged restore: `2x`, chunk `29/8`

## Labels

- `Wavelet reconstruction`: restore detail, then reuse the source frame's low-frequency tone structure
- `LAB tone matching`: restore detail, then match the output back toward the source in LAB space
- `Raw model output`: no postprocess after VAE decode

The accepted public proof uses `Wavelet reconstruction`.

## Reproduction Commands

See [seedvr2_video_restore_command_log.md](seedvr2_video_restore_command_log.md).

## Accepted 1x Proof

- [3B 1x restored video](eiffel_70s_149f_3b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4)
- [7B 1x restored video](eiffel_70s_149f_7b_chunk29_overlap8_wavelet_1x_after_causal_slicing.mp4)
- [1x comparison video](eiffel_70s_149f_3b_vs_7b_wavelet_1x_after_causal_slicing_comparison.mp4)
- [1x contact sheet](eiffel_70s_149f_3b_vs_7b_wavelet_1x_after_causal_slicing_contact_sheet.jpg)
- [1x motion strip, mid clip](motion_strip_1x_after_causal_slicing_frames_64_72.jpg)
- [1x motion strip, tail](motion_strip_1x_after_causal_slicing_frames_141_148.jpg)
- [1x metrics JSON](eiffel_70s_149f_3b_vs_7b_wavelet_1x_after_causal_slicing_metrics.json)

Direct read:

- both outputs preserve `149` frames at `29.9700 fps`
- both outputs keep the late tail intact
- the `1x` native-resolution path is mechanically healthy for both models
- `3B` is crisper at `1x`
- `7B` is smoother and stays closer to the source motion envelope

## Accepted 2x Proof

- [3B 2x restored video](eiffel_70s_149f_3b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4)
- [7B 2x restored video](eiffel_70s_149f_7b_chunk29_overlap8_wavelet_2x_after_causal_slicing.mp4)
- [2x comparison video](eiffel_70s_149f_3b_vs_7b_wavelet_2x_after_causal_slicing_comparison.mp4)
- [2x contact sheet](eiffel_70s_149f_3b_vs_7b_wavelet_2x_after_causal_slicing_contact_sheet.jpg)
- [2x motion strip, moving crowd](motion_strip_2x_after_causal_slicing_frames_64_72.jpg)
- [2x motion strip, tail](motion_strip_2x_after_causal_slicing_frames_141_149.jpg)
- [2x crowd detail crop](detail_crop_crowd_2x_after_causal_slicing_64_72.jpg)
- [2x metrics JSON](eiffel_70s_149f_3b_vs_7b_wavelet_2x_after_causal_slicing_metrics.json)

Direct read:

- both outputs preserve `149` frames at `29.9700 fps`
- motion strips stay coherent through the tail
- `7B` is slightly cleaner and more stable than `3B` on this enlarged bounded proof

## Measured Stats

See [seedvr2_video_restore_stats_m5max.json](seedvr2_video_restore_stats_m5max.json).

Key values:

- `3B 1x 29/8`: `71.33s` generation, `14.55 GB` peak MLX, `27.40 GB` max RSS
- `7B 1x 29/8`: `107.44s` generation, `24.54 GB` peak MLX, `66.18 GB` max RSS
- `3B 2x 29/8`: `539.03s` generation, `34.40 GB` peak MLX, `27.40 GB` max RSS
- `7B 2x 29/8`: `454.46s` generation, `44.27 GB` peak MLX, `66.18 GB` max RSS

## Conclusion

- The SeedVR2 streamed video path is working correctly for both `3B` and `7B`.
- The bounded public proof now verifies frame count, FPS, sequencing, and late-tail integrity.
- `1x` is the right proof surface to show that both routes restore without cadence or tail failure.
- `2x` is the stronger proof surface to compare `3B` versus `7B` quality on this archival clip.
