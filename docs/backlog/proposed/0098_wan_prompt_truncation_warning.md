# Proposed: warn on silent UMT5 prompt truncation (Wan routes)

## Metadata

- Created: 2026-07-25
- Status: Proposed (storyboard cross-scene-consistency investigation, 2026-07-25)
- Completed: N/A
- Effort: S

## ADR status

- Governing ADRs: none directly; aligns with
  [ADR 0003](../../adr/0003_runtime_truth_vs_consumer_convenience.md) (the
  runtime should tell the truth about what it actually conditioned on).
- ADR impact: None.

## Context (investigation provenance)

Verified in the 2026-07-25 storyboard consistency investigation: Wan prompt
encoding tokenizes with `truncation=True` at `max_length=512`
(`_tokenize_prompts` in `src/mflux/models/wan/variants/wan2_2_ti2v.py`) and
nothing checks whether the raw prompt exceeded the budget. A prompt beyond
512 UMT5 tokens is silently right-truncated: the user's trailing action or
style text simply never conditions the video, with no signal anywhere (CLI,
metadata, events).

This matters more now that hosts are steered toward PREPENDING board-level
world/style anchors to every scene prompt (the highest-leverage consistency
fix from the same investigation — measured anchor+longest-scene-prompt was 74
tokens on the real film, so the headroom is large, but power users stacking
anchors, subject sheets, and long action text can reach the cap).

## What we want to do

1. Detect overflow at encode time (tokenize once without the cap, or check
   `attention_mask` saturation plus an uncapped length probe) and emit ONE
   clear warning per generate call naming the dropped token count.
2. Record `prompt_tokens` / `prompt_truncated` in the metadata sidecar so
   hosts can surface it.
3. Same for the negative prompt WHEN it is actually encoded (guidance > 1.0).

## Non-goals

- Raising the 512 budget (checkpoint contract).
- Auto-summarizing or rewriting prompts.

## Dependencies and related tasks

- BlackPixel companion track `proposed/storyboard_consistency_2026_07/`
  (anchor prepending is what consumes budget).

## Validation

- Focused test: >512-token prompt produces the warning + metadata fields;
  <=512 stays silent.

## Progress checklist

- [ ] Overflow detection at encode time
- [ ] CLI warning + metadata fields
- [ ] Negative-prompt case under CFG
- [ ] Focused tests
