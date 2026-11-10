# Recovered May 2026 drafting documents (historical, read-only)

These four files were written 2026-05-21 to 2026-05-23 in the Claude worktree
`.claude/worktrees/affectionate-dewdney-ef7609` during the extended-abstract
pivot and were **never committed** to git. They were recovered verbatim on
2026-10-18 (same recovery path as the `tools/` probe scripts in PR #11).
They are kept as a historical record only; nothing here is current guidance.

| File | What it is | Current status |
|---|---|---|
| `HANDOFF_NOTES.md` | 2026-05-23 reframing memo written after the first production run; precursor of `CLAUDE/PROJECT_BRIEF.md` | Superseded by `CLAUDE/PROJECT_BRIEF.md` and `CLAUDE/STATUS.md` |
| `OUTLINE.md` | Pre-pivot 5-page outline built around the retired "TDA improves supervised accuracy" framing, with `{{...}}` fill tokens | Dead framing; do not reuse |
| `METHODS_DISCLOSURES.md` | Prose-ready Methods/Limitations/Future-work disclosure paragraphs (D1-D8 deviations, M1-M8 limitations) from the 2026-05-21 pipeline audit | Still useful: several deviations (C2-only medoid selection, essential-H0 truncation, p25 cap rationale, SVM 5k subsample) are not yet disclosed in the manuscript |
| `FIGURE_AUDIT.md` | `tools/figure_audit.py` output of 2026-05-21 showing every April 2026 PNG in `results/figures/` is too wide for single-column print | Still true; those PNGs are legacy monolith output and are not used by the paper |

All numbers inside these files are pre-Phase-3 (raw-sum probe, 3 seeds) and are
superseded by `paper/EXACT_RESULTS.md`, `paper/MANUSCRIPT_STATS.md`,
`paper/WINDOWED_RESULTS.md`, `paper/LATENCY_RESULTS.md`, `paper/ZNORM_RESULTS.md`.
