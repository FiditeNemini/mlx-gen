# Upstream Bernini evidence attribution

The files below `cases/run_1/*/inputs/` named `reference_*`, `source.*`, or
`official_output.*` are copied from ByteDance's Bernini repository at revision
`2d2b4591ac053ec25c6371b01a5a6746679e5793` solely to make this engineering proof
self-contained. The upstream repository declares Apache License 2.0; the exact license text is
retained as `UPSTREAM_BERNINI_LICENSE.txt`. The repository contained no `NOTICE` file at that
revision.

The `mlx_*.png` sheets, generated MP4s without the `official_output` name, sidecars, logs,
reports, and review records are MLX-Gen validation artifacts, not ByteDance benchmark outputs.
Official comparison clips remain labeled `official_output` and are never presented as MLX output.
The upstream repository does not attest which checkpoint or inference recipe produced those clips;
they are qualitative targets, not Bernini-R 1.3B parity baselines.
