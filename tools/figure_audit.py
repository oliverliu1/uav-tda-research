"""Audit PNG figures in ``results/figures/`` for publication-quality issues.

Reads every PNG in the figures directory (read-only via PIL/Pillow), reports
pixel dimensions, saved DPI, physical size at 300 DPI, and a readability
estimate at AIAA single-column width (3.5 inches). Checks for the eight
expected figures the evaluate phase produces and flags any that are missing.

The readability check is mechanical, not visual: it estimates the print font
size assuming the figure used the matplotlib-default 10-pt body font, then
applies the linear shrink that occurs when a wider figure is placed in a
single-column slot. Figures designed for double-column or full-page
placement will flag here and that is expected.

Usage:
    python tools/figure_audit.py
    python tools/figure_audit.py --figures-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ==== SECTION 1: PATHS AND CONFIG ====

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "FIGURE_AUDIT.md"

# AIAA single-column print width in inches.
SINGLE_COLUMN_WIDTH_IN = 3.5

# Print-quality DPI standard for sizing checks.
PRINT_DPI = 300

# Estimated print-font thresholds (in points).
MIN_PRINT_PT = 8.0
BORDERLINE_PRINT_PT = 6.0

# Matplotlib's default font size; used to estimate post-shrink print size.
ASSUMED_SOURCE_PT = 10.0

# Eight figures the evaluate phase is expected to produce.
EXPECTED_FIGURES = (
    "methodology_flowchart.png",
    "persistence_barcode_examples.png",
    "headline_results.png",
    "supervised_comparison.png",
    "feature_importance_curated.png",
    "unsupervised_distance_distributions.png",
    "unsupervised_roc_curves.png",
    "unsupervised_pattern_heatmap.png",
)


# ==== SECTION 2: IMAGE INSPECTION ====

def inspect_image(path: Path) -> dict[str, Any]:
    """Read one PNG and return its metadata plus a readability assessment.

    Args:
        path: Read-only path to a PNG file.

    Returns:
        Dict with filename, size_kb, pixel dims, saved DPI (or None), inches at
        300 DPI, rendered inches at the saved DPI, single-column shrink factor,
        estimated print font, and verdict (OK / BORDERLINE / TOO_SMALL / UNKNOWN).
    """
    from PIL import Image

    with Image.open(path) as img:
        width_px, height_px = img.size
        dpi_info = img.info.get("dpi", (None, None))

    saved_dpi = float(dpi_info[0]) if dpi_info and dpi_info[0] is not None else None
    size_kb = path.stat().st_size / 1024.0
    inches_at_300 = (width_px / PRINT_DPI, height_px / PRINT_DPI)

    if saved_dpi is None or saved_dpi <= 0:
        rendered_w_in: float | None = None
        rendered_h_in: float | None = None
        scale_to_col: float | None = None
        est_print_pt: float | None = None
        verdict = "UNKNOWN"
    else:
        rendered_w_in = width_px / saved_dpi
        rendered_h_in = height_px / saved_dpi
        if rendered_w_in <= SINGLE_COLUMN_WIDTH_IN:
            scale_to_col = 1.0
        else:
            scale_to_col = SINGLE_COLUMN_WIDTH_IN / rendered_w_in
        est_print_pt = ASSUMED_SOURCE_PT * scale_to_col
        if est_print_pt >= MIN_PRINT_PT:
            verdict = "OK"
        elif est_print_pt >= BORDERLINE_PRINT_PT:
            verdict = "BORDERLINE"
        else:
            verdict = "TOO_SMALL"

    return {
        "filename": path.name,
        "size_kb": size_kb,
        "width_px": width_px,
        "height_px": height_px,
        "saved_dpi": saved_dpi,
        "inches_at_300_w": inches_at_300[0],
        "inches_at_300_h": inches_at_300[1],
        "rendered_w_in": rendered_w_in,
        "rendered_h_in": rendered_h_in,
        "scale_to_col": scale_to_col,
        "est_print_pt": est_print_pt,
        "verdict": verdict,
    }


def check_expected(figures_dir: Path) -> dict[str, Any]:
    """Bucket the expected-figure names into present (with path) vs missing."""
    present: dict[str, Path] = {}
    missing: list[str] = []
    for name in EXPECTED_FIGURES:
        path = figures_dir / name
        if path.exists():
            present[name] = path
        else:
            missing.append(name)
    return {"present": present, "missing": missing}


def list_extra_pngs(figures_dir: Path) -> list[Path]:
    """Return sorted PNGs in ``figures_dir`` that are not in EXPECTED_FIGURES."""
    expected_set = set(EXPECTED_FIGURES)
    if not figures_dir.exists():
        return []
    return sorted(
        p for p in figures_dir.glob("*.png") if p.name not in expected_set
    )


# ==== SECTION 3: RENDERING ====

def fmt_optional(value: Any, fmt: str = "{:.2f}") -> str:
    """Format ``value`` with ``fmt`` if not None; otherwise return em-dash."""
    if value is None:
        return "—"
    return fmt.format(value)


def render_image_row(r: dict[str, Any]) -> str:
    """Render one PNG's metadata as a single markdown table row."""
    pixels = f"{r['width_px']} x {r['height_px']}"
    in300 = f"{r['inches_at_300_w']:.2f} x {r['inches_at_300_h']:.2f}"
    if r["rendered_w_in"] is not None:
        in_saved = f"{r['rendered_w_in']:.2f} x {r['rendered_h_in']:.2f}"
    else:
        in_saved = "—"
    dpi = fmt_optional(r["saved_dpi"], "{:.0f}")
    scale = fmt_optional(r["scale_to_col"], "{:.2f}x")
    pt = fmt_optional(r["est_print_pt"], "{:.1f}")
    verdict = r["verdict"]
    if verdict != "OK":
        verdict = f"**{verdict}**"
    return (
        f"| `{r['filename']}` | {r['size_kb']:.1f} | {pixels} | {dpi} | "
        f"{in300} | {in_saved} | {scale} | {pt} | {verdict} |"
    )


def render_image_table(rows: list[dict[str, Any]]) -> str:
    """Render a markdown table summarising a list of inspected PNGs."""
    if not rows:
        return "_No images analysed._"
    header = (
        "| File | Size (KB) | Pixels | DPI | Inches @ 300 DPI | "
        "Inches @ saved DPI | 1-col shrink | Est. print pt | Verdict |"
    )
    sep = (
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |"
    )
    return "\n".join([header, sep, *[render_image_row(r) for r in rows]])


def render_expected_summary(check: dict[str, Any]) -> str:
    """Render the present/missing block for the expected-figure list."""
    n_present = len(check["present"])
    n_total = len(EXPECTED_FIGURES)
    lines = [
        "## Expected figures",
        "",
        f"**{n_present} of {n_total} expected figures present.**",
        "",
    ]
    if check["missing"]:
        lines.append("Missing:")
        for name in check["missing"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("_All expected figures present._")
    return "\n".join(lines)


def render_legend() -> str:
    """Render the verdict legend explaining each readability bucket."""
    lines = [
        "## Verdict legend",
        "",
        f"- **OK** -- estimated print font >= {MIN_PRINT_PT:.0f} pt (assuming "
        f"{ASSUMED_SOURCE_PT:.0f}-pt source). Acceptable for single-column print.",
        f"- **BORDERLINE** -- estimated print font in "
        f"[{BORDERLINE_PRINT_PT:.0f}, {MIN_PRINT_PT:.0f}) pt. Likely needs "
        f"source-font increase or figure-width reduction.",
        f"- **TOO_SMALL** -- estimated print font < {BORDERLINE_PRINT_PT:.0f} "
        f"pt. Figure is too wide for single-column print at its current source "
        f"font; redraw with larger source font or accept double-column placement.",
        "- **UNKNOWN** -- saved DPI not recorded in PNG metadata; cannot "
        "estimate physical size.",
        "",
        "Notes:",
        f"- *1-col shrink* is the linear scale factor when the figure is "
        f"placed at the {SINGLE_COLUMN_WIDTH_IN}-inch column width. "
        f"1.00x means no shrinking required.",
        f"- *Est. print pt* assumes the figure used the matplotlib default "
        f"{ASSUMED_SOURCE_PT:.0f}-pt body font. Figures with explicit fontsize "
        f"overrides may print larger or smaller than the estimate.",
        f"- *Inches @ 300 DPI* is the physical figure size at print-quality "
        f"resolution; this is independent of the saved DPI tag in the PNG.",
        "- Aspect-ratio and aesthetic quality are not assessed; this audit "
        "covers only mechanical print-size concerns.",
    ]
    return "\n".join(lines)


def render_report(
    figures_dir: Path,
    expected_check: dict[str, Any],
    expected_rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
) -> str:
    """Compose the full markdown report."""
    now = datetime.now()
    sections = [
        "# Figure Audit",
        "",
        f"_Auto-generated by `tools/figure_audit.py` on "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}._",
        "",
        (
            f"Scanned: `{figures_dir}`. "
            f"Single-column target: {SINGLE_COLUMN_WIDTH_IN} inches. "
            f"Minimum print font: {MIN_PRINT_PT:.0f} pt. "
            f"Assumed source font: {ASSUMED_SOURCE_PT:.0f} pt."
        ),
        "",
        render_expected_summary(expected_check),
        "",
        "## Expected figure inspection",
        "",
        render_image_table(expected_rows),
        "",
    ]
    if extra_rows:
        sections.extend([
            "## Other PNGs in the figures directory",
            "",
            render_image_table(extra_rows),
            "",
        ])
    sections.append(render_legend())
    return "\n".join(sections) + "\n"


# ==== SECTION 4: CLI ====

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(
        prog="figure_audit.py",
        description=(
            "Audit publication-quality of PNGs in results/figures/. "
            "Writes paper/FIGURE_AUDIT.md. Reads PNGs read-only."
        ),
    )
    p.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
        help="Path to figures directory (read-only).",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Markdown output path.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Inspect every PNG in --figures-dir and write the audit report."""
    args = build_parser().parse_args(argv)

    if not args.figures_dir.exists():
        print(f"figures dir does not exist: {args.figures_dir}", file=sys.stderr)
        return 2

    expected_check = check_expected(args.figures_dir)
    expected_rows = sorted(
        [inspect_image(p) for p in expected_check["present"].values()],
        key=lambda r: r["filename"],
    )
    extra_rows = [inspect_image(p) for p in list_extra_pngs(args.figures_dir)]

    markdown = render_report(
        args.figures_dir, expected_check, expected_rows, extra_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"  expected present: {len(expected_rows)}/{len(EXPECTED_FIGURES)}")
    if expected_check["missing"]:
        print(f"  expected missing: {len(expected_check['missing'])}")
    if extra_rows:
        print(f"  other PNGs analysed: {len(extra_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
