# DEPRECATED — legacy quarantine

Everything under `archive/` is **read-only history**. It is kept for provenance
(git blame / historical reference) only. Nothing here is maintained, tested, or
safe to run as-is. Do not import from it, do not run scripts from it, and do
not cite any number that came from it.

For the maintained implementation, use `uav_tda/` (the package) and its CLI
(`uav-tda ...`). For the frozen equivalence oracle / historical production
run, see `pipeline.py` at the repo root (also read-only, but authoritative —
see `CLAUDE/CLAUDE.md`).

## Contents

- **`poster_eda/`** — ECS-Summit-era exploratory data analysis and poster
  figure generation. **Contains pre-fix backups (`outputs_backup_with_leakage/`,
  `results_backup_with_leakage/`) with a KNOWN DATA-LEAKAGE flaw — never use
  any number or figure from this directory for results.** Superseded entirely
  by `uav_tda/` + `pipeline.py`.
- **`scripts_archive/`** — pre-`pipeline.py` numbered scripts (`01_data_scripts.py`
  … `08_comparative_analysis.py`, plus the `*W_wasserstein_*.py` series).
  Superseded by `pipeline.py` in May 2026; kept only for history.
- **`poster.jsx`** — a standalone poster artifact (React component), not part
  of the pipeline or the paper build.

## Where to look instead

- `uav_tda/` — the maintained package (all pipeline phases as `uav-tda`
  subcommands).
- `pipeline.py` — the frozen equivalence oracle; every `uav_tda` phase is
  tested against its debug-mode output. Historical record of the paper's
  production run. Do not modify.
