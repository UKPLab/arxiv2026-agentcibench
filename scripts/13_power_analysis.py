#!/usr/bin/env python3
"""
Formal power analysis justifying sample sizes for the AgentCIBench paper.

Computes for binary proportion outcomes (leakage rate, utility rate):
  - Minimum detectable effect (MDE) at 80% and 95% power for each study arm
  - Power curves for fixed n at a range of effect sizes
  - Cohen's h for all key observed comparisons
  - Bootstrap CI width analysis (precision achieved at actual n)

All calculations use the standard two-proportion z-test approximation:
  power = Phi(|h| * sqrt(n) - z_{alpha/2})
where h = 2*arcsin(sqrt(p2)) - 2*arcsin(sqrt(p1))  [Cohen's h, 1988]

Outputs to data/results/aggregated/:
  power_mde.csv, power_curves.csv, power_observed.csv, power_ci_widths.csv,
  power_analysis.tex  (LaTeX tables for paper appendix)

Usage:
  uv run python scripts/13_power_analysis.py
  uv run python scripts/13_power_analysis.py --out-dir data/results/aggregated
"""

import argparse
import csv
import os
from math import asin, sqrt, sin

import numpy as np

try:
    from scipy.stats import norm as _norm
    _norm_cdf = _norm.cdf
    _norm_ppf = _norm.ppf
except ImportError:
    import math
    # fallback: use erfc-based normal CDF
    def _norm_cdf(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))
    def _norm_ppf(p):
        # rational approximation (Abramowitz & Stegun 26.2.17)
        if p <= 0 or p >= 1:
            raise ValueError(f"p={p} out of range")
        if p < 0.5:
            return -_norm_ppf(1 - p)
        t = math.sqrt(-2 * math.log(1 - p))
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - (c[0] + c[1]*t + c[2]*t**2) / (1 + d[0]*t + d[1]*t**2 + d[2]*t**3)


# ---------------------------------------------------------------------------
# Core power analysis primitives (Cohen 1988)
# ---------------------------------------------------------------------------

def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions (signed)."""
    return 2 * asin(sqrt(p2)) - 2 * asin(sqrt(p1))


def power_from_h(h: float, n: int, alpha: float = 0.05) -> float:
    """Power of two-sided two-proportion z-test given Cohen's h and n per group."""
    z_alpha = _norm_ppf(1 - alpha / 2)
    return float(_norm_cdf(abs(h) * sqrt(n) - z_alpha))


def mde_h(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable Cohen's h for given n, alpha, power (one-sided direction)."""
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = _norm_ppf(power)
    return (z_alpha + z_beta) / sqrt(n)


def h_to_p2(h_val: float, p1: float) -> float:
    """Convert Cohen's h back to p2 given baseline p1 (positive direction)."""
    phi2 = 2 * asin(sqrt(p1)) + h_val
    return float(sin(phi2 / 2) ** 2)


def ci_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Normal-approximation 95% CI half-width for a Bernoulli proportion."""
    return z * sqrt(p * (1 - p) / n)


# ---------------------------------------------------------------------------
# Study arm constants (from paper data)
# ---------------------------------------------------------------------------

S2A_N = 117
S2A_BASELINE_LEAK = 0.65   # average leakage across 15 models

E2E_N = 50
E2E_BASELINE_LEAK = 0.65

SUBGROUPS = [
    ("TAO (task-ambiguity overshare)", 75),
    ("RMA (recipient misalignment)", 24),
    ("VCL (visual co-location)", 18),
]

# Key comparisons from actual CSV data
OBSERVED_COMPARISONS = [
    ("Best vs. worst model",       0.14, 0.98, S2A_N),
    ("Defense: baseline to best",  0.65, 0.40, S2A_N),
    ("Opus: S2A to E2E",           0.14, 0.43, E2E_N),
    ("Sonnet: S2A to E2E",         0.46, 0.80, E2E_N),
]

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Compute tables
# ---------------------------------------------------------------------------

def compute_mde_table() -> list[dict]:
    """MDE for each study arm at 80% and 95% power."""
    configs = [
        ("S2A main study",  S2A_N,  S2A_BASELINE_LEAK),
        ("E2E deployment",  E2E_N,  E2E_BASELINE_LEAK),
        ("TAO sub-group",   75,     S2A_BASELINE_LEAK),
        ("RMA sub-group",   24,     S2A_BASELINE_LEAK),
        ("VCL sub-group",   18,     S2A_BASELINE_LEAK),
    ]
    rows = []
    for label, n, p_base in configs:
        for pwr_target in [0.80, 0.95]:
            h_min = mde_h(n, alpha=ALPHA, power=pwr_target)
            p2 = h_to_p2(h_min, p_base)
            rows.append({
                "study_arm": label,
                "n": n,
                "baseline_p": round(p_base, 3),
                "power_target": pwr_target,
                "mde_cohen_h": round(h_min, 4),
                "mde_delta_pp": round(p2 - p_base, 4),
                "mde_p2": round(p2, 4),
            })
    return rows


def compute_power_table() -> list[dict]:
    """Power at actual n for a grid of absolute effect sizes."""
    rows = []
    deltas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.29, 0.30]
    for label, n, p_base in [("S2A n=117", S2A_N, S2A_BASELINE_LEAK),
                              ("E2E n=50",  E2E_N, E2E_BASELINE_LEAK)]:
        for delta in deltas:
            p2 = p_base + delta
            if not (0 < p2 < 1):
                continue
            h = cohen_h(p_base, p2)
            pwr = power_from_h(h, n, alpha=ALPHA)
            rows.append({
                "study_arm": label,
                "n": n,
                "baseline_p": round(p_base, 3),
                "delta_pp": round(delta, 3),
                "cohen_h": round(abs(h), 4),
                "power": round(pwr, 4),
            })
    return rows


def compute_observed_effects() -> list[dict]:
    """Cohen's h and power for each observed key comparison."""
    rows = []
    for label, p1, p2, n_ref in OBSERVED_COMPARISONS:
        h = cohen_h(p1, p2)
        pwr = power_from_h(h, n_ref, alpha=ALPHA)
        rows.append({
            "comparison": label,
            "p1": round(p1, 3),
            "p2": round(p2, 3),
            "delta_pp": round(p2 - p1, 3),
            "cohen_h": round(abs(h), 4),
            "n_used": n_ref,
            "power": round(pwr, 4),
        })
    return rows


def compute_ci_widths() -> list[dict]:
    """Bootstrap CI half-widths at actual sample sizes and representative p values."""
    cases = [
        ("S2A — median model",          S2A_N, 0.65),
        ("S2A — best model (Opus)",      S2A_N, 0.14),
        ("S2A — worst model (Gemini)",   S2A_N, 0.98),
        ("E2E — median",                 E2E_N, 0.65),
        ("E2E — Opus",                   E2E_N, 0.14),
        ("E2E — Sonnet",                 E2E_N, 0.22),
    ]
    rows = []
    for label, n, p in cases:
        hw = ci_halfwidth(p, n)
        rows.append({
            "context": label,
            "n": n,
            "p": round(p, 3),
            "ci_halfwidth": round(hw, 4),
            "ci_width_total": round(2 * hw, 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv_generic(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows → {path}")


def write_latex_tables(mde_rows: list[dict], obs_rows: list[dict],
                       ci_rows: list[dict], path: str) -> None:
    """Write LaTeX table snippet for the paper appendix."""
    lines = []

    # Table A: MDE
    lines += [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Minimum detectable effect (MDE) at $\alpha{=}0.05$ for each study arm. "
        r"$\Delta L$ is the absolute leakage-rate difference detectable at the stated power.}",
        r"\label{tab:power-mde}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Study arm} & $n$ & Target power & MDE $h$ & MDE $\Delta L$ (pp) \\",
        r"\midrule",
    ]
    for r in mde_rows:
        pwr_str = f"{int(r['power_target']*100)}\\%"
        lines.append(
            f"\\quad {r['study_arm']} & {r['n']} & {pwr_str} & "
            f"{r['mde_cohen_h']:.3f} & {r['mde_delta_pp']*100:+.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table B: Observed effects
    lines += [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Cohen's $h$ and retrospective power for key observed comparisons "
        r"($\alpha{=}0.05$, two-sided two-proportion $z$-test).}",
        r"\label{tab:power-observed}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Comparison} & $p_1$ & $p_2$ & $\Delta L$ (pp) & $h$ & Power \\",
        r"\midrule",
    ]
    for r in obs_rows:
        lines.append(
            f"\\quad {r['comparison']} & {r['p1']:.2f} & {r['p2']:.2f} & "
            f"{r['delta_pp']*100:+.0f} & {r['cohen_h']:.3f} & {r['power']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote LaTeX → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute power analysis tables for AgentCIBench paper."
    )
    parser.add_argument("--out-dir", default="data/results/aggregated",
                        help="Output directory for CSV and LaTeX files")
    args = parser.parse_args()

    print("Computing power analysis...")

    mde_rows = compute_mde_table()
    pwr_rows = compute_power_table()
    obs_rows = compute_observed_effects()
    ci_rows  = compute_ci_widths()

    # Print summary
    print("\n=== MDE Summary (baseline L=0.65) ===")
    for r in mde_rows:
        print(f"  {r['study_arm']:38s}  n={r['n']:3d}  "
              f"power={r['power_target']:.0%}  "
              f"MDE h={r['mde_cohen_h']:.3f}  "
              f"delta={r['mde_delta_pp']*100:+.1f}pp")

    print("\n=== Observed Effects ===")
    for r in obs_rows:
        print(f"  {r['comparison']:38s}  h={r['cohen_h']:.3f}  "
              f"power={r['power']:.4f}  (n={r['n_used']})")

    print("\n=== Bootstrap CI Widths (95%, normal approx) ===")
    for r in ci_rows:
        print(f"  {r['context']:40s}  n={r['n']}  p={r['p']}  "
              f"±{r['ci_halfwidth']:.3f}  (total {r['ci_width_total']:.3f})")

    # Write outputs
    write_csv_generic(mde_rows, os.path.join(args.out_dir, "power_mde.csv"))
    write_csv_generic(pwr_rows, os.path.join(args.out_dir, "power_curves.csv"))
    write_csv_generic(obs_rows, os.path.join(args.out_dir, "power_observed.csv"))
    write_csv_generic(ci_rows,  os.path.join(args.out_dir, "power_ci_widths.csv"))
    write_latex_tables(mde_rows, obs_rows, ci_rows,
                       os.path.join(args.out_dir, "power_analysis.tex"))

    print("\nDone.")


if __name__ == "__main__":
    main()
