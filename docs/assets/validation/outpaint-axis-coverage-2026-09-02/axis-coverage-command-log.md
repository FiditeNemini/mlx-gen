# Outpaint Axis Coverage Command Log

Three sources of different aspect ratio, eight padding configurations each: every single
side, both vertical sides together, both horizontal sides together, all four sides, and an
asymmetric four-side request.

- Model: `AbstractFramework/flux.2-klein-9b-8bit`, 16 steps, guidance 1, seed 99
- Prompt: **empty** — the hardest case, with no instruction for the model to work from
- Machine: Apple M5 Max, 40-core GPU, 128 GB unified memory

## Sources

```sh
mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit \
  --prompt "an empty arctic snow plain under a pale sky, aerial view" \
  --steps 8 --guidance 1.0 --seed 1 --width 640 --height 448 --output src_land.png --replace

mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit \
  --prompt "a weathered stone courtyard with moss between the slabs, overhead view" \
  --steps 8 --guidance 1.0 --seed 3 --width 512 --height 512 --output src_square.png --replace

mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit \
  --prompt "a dense pine forest canopy seen from above, morning mist" \
  --steps 8 --guidance 1.0 --seed 5 --width 448 --height 640 --output src_port.png --replace
```

## Padding configurations

Run for each source, with `--outpaint-padding` in `top,right,bottom,left` order:

| case | padding |
| --- | --- |
| bottom only | `0,0,25%,0` |
| top only | `25%,0,0,0` |
| top + bottom | `20%,0,20%,0` |
| right only | `0,25%,0,0` |
| left only | `0,0,0,25%` |
| left + right | `0,20%,0,20%` |
| all four sides | `15%,15%,15%,15%` |
| asymmetric four-side | `10%,30%,20%,5%` |

```sh
mlxgen generate --model AbstractFramework/flux.2-klein-9b-8bit \
  --image src_land.png --prompt "" \
  --steps 16 --guidance 1.0 --seed 99 \
  --outpaint-padding "15%,15%,15%,15%" \
  --metadata --replace --output land__allfour.png
```

Measurements are in `axis-coverage-measurements.txt`.
