# tools/ — recovered original probe & diagnostic scripts

These are the ORIGINAL scripts that produced the paper's Results
(`results/tables/probe_distances*.csv`, `paper/PROBE_RESULTS.md`,
`paper/MULTI_SEED_VARIANCE.md`, `paper/C2_NORMALIZATION_DIAGNOSTIC.md`).

They were written in May 2026 in a Claude Code worktree
(`.claude/worktrees/affectionate-dewdney-ef7609/`) and never committed;
recovered verbatim on 2026-09-19. File mtimes in the worktree align with the
published CSV timestamps (e.g. `quick_unsup_probe.py` @ May 25 15:55 vs
`probe_distances_seed123.csv` @ May 25 15:59).

- `quick_unsup_probe.py` — the unsupervised W2 probe (paper §III.F/§IV)
- `diagnose_c2_scaling.py` — Diagnostic B (Z-normalization check)
- `diagnose_results.py`, `figure_audit.py`, `extract_methods_disclosures.py` — supporting diagnostics

These are preserved AS-IS for provenance. The maintained, tested
implementation of the same computation lives in the `uav_tda` package
(`uav_tda/probe.py`, `uav_tda/metrics.py`), validated to reproduce the
published AUCs within seed variance on clean-lineage diagrams.
