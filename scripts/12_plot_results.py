#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper from aggregated CSVs.

Run after scripts/11_bootstrap_ci.py has written the CI files.

Usage:
  uv run python scripts/12_plot_results.py
  uv run python scripts/12_plot_results.py --agg-dir data/results/aggregated --out-dir figures/generated --fmt pdf

Outputs (in --out-dir):
  s2a_pareto.{fmt}         utility vs leak scatter with 95% CI, Pareto frontier
  s2a_per_mode_bars.{fmt}  grouped bars: leak rate per model × failure mode
  defenses_pareto.{fmt}    utility vs leak Pareto under 4 defense conditions
  defenses_per_mode_bars.{fmt}  leak reduction by defense × failure mode
  e2e_bars.{fmt}           E2E utility/leak per model with CI (preliminary)
  e2e_comparison.{fmt}     S2A vs E2E all-tasks vs E2E completed-tasks leak rate
  s2a_combined_panel.{fmt} Two-panel: regression+Pareto (left) | leak vs engaged (right)
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from adjustText import adjust_text

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "font.weight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#888888",
    "legend.borderpad": 0.5,
    "legend.handlelength": 1.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "axes.axisbelow": True,
    "axes.edgecolor": "#222222",
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.6,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})

BAR_EDGE_LW = 1.2   # linewidth for bar borders
LABEL_FONT_SIZE = 6.5
LABEL_FONT_WEIGHT = "bold"
SCATTER_LABEL_SIZE = 6.5


def _adjust(texts, xs, ys, ax):
    """Run adjustText with consistent settings across all scatter plots.
    Connector arrows are disabled — labels float without lines to their dots."""
    adjust_text(
        texts,
        x=np.array(xs), y=np.array(ys),
        ax=ax,
        expand=(1.5, 1.8),
        force_points=(0.5, 0.8),
        force_text=(0.7, 1.0),
        arrowprops=None,
    )


def lighten(hex_color, amount=0.58):
    """Blend hex_color toward white by `amount` (0=unchanged, 1=white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ── colour palette ────────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "anthropic": "#c05a18",
    "google": "#1a73e8",
    "openai": "#19a974",
    "deepseek": "#7b2d8b",
    "qwen": "#e8a020",
    "x-ai": "#e04040",
    "minimax": "#5b8c5a",
    "moonshotai": "#3d6b8c",
    "z-ai": "#888888",
}
DEFENSE_COLORS = {
    "none": "#aaaaaa",
    "recipient_typed": "#1a73e8",
    "restrictive": "#e8a020",
    "rubric_informed": "#19a974",
}
DEFENSE_MARKERS = {
    "none": "o",
    "recipient_typed": "s",
    "restrictive": "^",
    "rubric_informed": "D",
}
DEFENSE_LABELS = {
    "none": "No defense",
    "recipient_typed": "Recipient-typed",
    "restrictive": "Restrictive",
    "rubric_informed": "Rubric-informed",
}
MODE_COLORS = {
    "task_ambiguity_overshare": "#1a73e8",
    "recipient_misalignment": "#e8a020",
    "visual_co_location": "#19a974",
}
MODE_LABELS = {
    "task_ambiguity_overshare": "Task ambiguity",
    "recipient_misalignment": "Recipient mismatch",
    "visual_co_location": "Visual co-location",
}

# Fixed two-series colors used across several bar charts
UTILITY_COLOR = "#1a73e8"
LEAK_COLOR    = "#e04040"


def model_label(model_str):
    """Short display name from openrouter slug."""
    m = model_str.split("/")[-1]
    mapping = {
        "claude-opus-4.7": "Opus 4.7",
        "claude-sonnet-4.6": "Sonnet 4.6",
        "gpt-5.4": "GPT-5.4",
        "gpt-5.4-mini": "GPT-5.4-mini",
        "gpt-oss-120b": "GPT-OSS-120B",
        "gemini-3.1-pro-preview": "Gemini-3.1-Pro",
        "gemini-3-flash-preview": "Gemini-3-Flash",
        "gemma-4-26b-a4b-it": "Gemma-4-26B",
        "deepseek-v4-pro": "DeepSeek-v4-Pro",
        "qwen3.6-35b-a3b": "Qwen3.6-35B",
        "qwen3.6-max-preview": "Qwen3.6-Max",
        "grok-4.3": "Grok-4.3",
        "kimi-k2.6": "Kimi-K2.6",
        "minimax-m2.7": "MiniMax-M2.7",
        "glm-5.1": "GLM-5.1",
    }
    return mapping.get(m, m)


def model_family(model_str):
    """Provider family from openrouter slug."""
    parts = model_str.split("/")
    provider = parts[-2] if len(parts) >= 2 else "unknown"
    for fam in FAMILY_COLORS:
        if fam in provider:
            return fam
    return "z-ai"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pareto_frontier(xs, ys):
    """Points on the lower-left Pareto frontier (minimise both x and y)."""
    pts = sorted(zip(xs, ys))
    frontier = []
    min_y = float("inf")
    for x, y in pts:
        if y <= min_y:
            frontier.append((x, y))
            min_y = y
    return zip(*frontier) if frontier else ([], [])


def _save(fig, out_dir, stem, fmt):
    """Save figure in the requested format and always also as PNG."""
    for f in sorted({fmt, "png"}):
        path = os.path.join(out_dir, f"{stem}.{f}")
        fig.savefig(path)
        print(f"  saved {path}")


def _bar(ax, x, height, width, color, **kwargs):
    """Uniform bar style: light fill with solid border in the base color."""
    return ax.bar(
        x, height, width,
        color=lighten(color),
        edgecolor=color,
        linewidth=BAR_EDGE_LW,
        **kwargs,
    )


def _add_ygrid(ax):
    """Subtle horizontal grid lines for bar charts."""
    ax.yaxis.grid(True)


# ── Figure 1: S2A utility vs leak scatter with CI bars ──────────────────────

def plot_s2a_pareto(agg_dir, out_dir, fmt):
    rows = load_csv(os.path.join(agg_dir, "s2a_ci.csv"))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    xs, ys, texts = [], [], []
    for r in rows:
        x = float(r["utility_rate"])
        y = float(r["leak_rate"])
        x_lo = x - float(r["utility_lo"])
        x_hi = float(r["utility_hi"]) - x
        y_lo = y - float(r["leak_lo"])
        y_hi = float(r["leak_hi"]) - y

        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")

        ax.errorbar(
            x, y,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt="o",
            color=color,
            markersize=6,
            capsize=2,
            linewidth=0.8,
            markeredgewidth=0.8,
            markeredgecolor=color,
            markerfacecolor=lighten(color, 0.35),
            alpha=0.95,
        )
        texts.append(ax.text(x, y, model_label(r["model"]), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x)
        ys.append(y)

    px, py = pareto_frontier(xs, ys)
    px, py = list(px), list(py)
    if px:
        ax.plot(px, py, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=0)

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leak rate")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    _adjust(texts, xs, ys, ax)

    os.makedirs(out_dir, exist_ok=True)
    _save(fig, out_dir, "s2a_pareto", fmt)
    plt.close(fig)


# ── Figure 2: S2A per-mode grouped bars ─────────────────────────────────────

def plot_s2a_per_mode_bars(agg_dir, out_dir, fmt):
    rows = load_csv(os.path.join(agg_dir, "s2a_per_mode_ci.csv"))

    s2a = {r["model"]: float(r["leak_rate"]) for r in load_csv(os.path.join(agg_dir, "s2a_ci.csv"))}
    models = sorted(s2a.keys(), key=lambda m: s2a[m])
    short = [model_label(m) for m in models]
    modes = ["task_ambiguity_overshare", "recipient_misalignment", "visual_co_location"]

    data = {mode: [] for mode in modes}
    err_lo = {mode: [] for mode in modes}
    err_hi = {mode: [] for mode in modes}

    row_index = {(r["model"], r["failure_mode"]): r for r in rows}
    for m in models:
        for mode in modes:
            key = (m, mode)
            if key in row_index:
                r = row_index[key]
                l = float(r["leak_rate"])
                data[mode].append(l)
                err_lo[mode].append(l - float(r["leak_lo"]))
                err_hi[mode].append(float(r["leak_hi"]) - l)
            else:
                data[mode].append(0)
                err_lo[mode].append(0)
                err_hi[mode].append(0)

    n_models = len(models)
    x = np.arange(n_models)
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    _add_ygrid(ax)

    for i, mode in enumerate(modes):
        offset = (i - 1) * width
        _bar(ax, x + offset, data[mode], width, MODE_COLORS[mode],
             label=MODE_LABELS[mode])
        ax.errorbar(
            x + offset, data[mode],
            yerr=[err_lo[mode], err_hi[mode]],
            fmt="none", color="#444444", capsize=2, linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=9.5)
    ax.set_ylabel("Leak rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left")

    _save(fig, out_dir, "s2a_per_mode_bars", fmt)
    plt.close(fig)


# ── Figure 3: Defenses Pareto (utility vs leak, per model × defense) ─────────

def plot_defenses_pareto(agg_dir, out_dir, fmt):
    rows = load_csv(os.path.join(agg_dir, "defenses_per_model_ci.csv"))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    xs, ys, texts = [], [], []
    for r in rows:
        defense = r["defense"]
        model = r["model"]
        x = float(r["utility_rate"])
        y = float(r["leak_rate"])
        x_lo = x - float(r["utility_lo"])
        x_hi = float(r["utility_hi"]) - x
        y_lo = y - float(r["leak_lo"])
        y_hi = float(r["leak_hi"]) - y

        color = DEFENSE_COLORS.get(defense, "#888")
        marker = DEFENSE_MARKERS.get(defense, "o")

        ax.errorbar(
            x, y,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt=marker,
            color=color,
            markersize=7,
            capsize=2,
            linewidth=0.8,
            markeredgewidth=0.8,
            markeredgecolor=color,
            markerfacecolor=lighten(color, 0.35),
            alpha=0.95,
            label=DEFENSE_LABELS.get(defense, defense) if model == list({r["model"] for r in rows})[0] else "_nolegend_",
        )
        texts.append(ax.text(x, y, model_label(model), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x)
        ys.append(y)

    by_model = {}
    for r in rows:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {}
        by_model[m][r["defense"]] = (float(r["utility_rate"]), float(r["leak_rate"]))

    for m, d in by_model.items():
        if "none" in d:
            for def_name, (xd, yd) in d.items():
                if def_name != "none":
                    x0, y0 = d["none"]
                    ax.annotate(
                        "", xy=(xd, yd), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#cccccc",
                                        lw=0.7, connectionstyle="arc3,rad=0.1"),
                    )

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leak rate")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    _adjust(texts, xs, ys, ax)

    handles = []
    seen_def = set()
    for r in rows:
        d = r["defense"]
        if d not in seen_def:
            h = matplotlib.lines.Line2D(
                [], [], color=DEFENSE_COLORS.get(d, "#888"),
                marker=DEFENSE_MARKERS.get(d, "o"),
                linestyle="None", markersize=7,
                markeredgewidth=0.8,
                markeredgecolor=DEFENSE_COLORS.get(d, "#888"),
                markerfacecolor=lighten(DEFENSE_COLORS.get(d, "#888"), 0.35),
                label=DEFENSE_LABELS.get(d, d),
            )
            handles.append(h)
            seen_def.add(d)
    ax.legend(handles=handles, loc="upper left")

    _save(fig, out_dir, "defenses_pareto", fmt)
    plt.close(fig)


# ── Figure 4: Defenses per-mode leak reduction bars ──────────────────────────

def plot_defenses_per_mode_bars(agg_dir, out_dir, fmt):
    rows = load_csv(os.path.join(agg_dir, "defenses_per_mode_ci.csv"))

    defenses = ["recipient_typed", "restrictive", "rubric_informed"]
    modes = ["task_ambiguity_overshare", "recipient_misalignment", "visual_co_location"]

    n_modes = len(modes)
    x = np.arange(n_modes)
    width = 0.25

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _add_ygrid(ax)

    row_idx = {(r["defense"], r["failure_mode"]): r for r in rows}

    for mi, mode in enumerate(modes):
        key = ("none", mode)
        if key in row_idx:
            y_none = float(row_idx[key]["leak_rate"])
            ax.plot([mi - 0.45, mi + 0.45], [y_none, y_none],
                    color="#999999", linewidth=1.0, linestyle="--", zorder=4)

    for di, defense in enumerate(defenses):
        leak_vals, lo_errs, hi_errs = [], [], []
        for mode in modes:
            key = (defense, mode)
            if key in row_idx:
                r = row_idx[key]
                l = float(r["leak_rate"])
                leak_vals.append(l)
                lo_errs.append(l - float(r["leak_lo"]))
                hi_errs.append(float(r["leak_hi"]) - l)
            else:
                leak_vals.append(0)
                lo_errs.append(0)
                hi_errs.append(0)

        offset = (di - 1) * width
        color = DEFENSE_COLORS[defense]
        _bar(ax, x + offset, leak_vals, width, color,
             label=DEFENSE_LABELS[defense])
        ax.errorbar(
            x + offset, leak_vals,
            yerr=[lo_errs, hi_errs],
            fmt="none", color="#444444", capsize=2, linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in modes])
    ax.set_ylabel("Leak rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 0.65)

    ax.plot([], [], color="#999999", linewidth=1.0, linestyle="--", label="No defense")
    ax.legend(loc="upper right")

    _save(fig, out_dir, "defenses_per_mode_bars", fmt)
    plt.close(fig)


# ── Figure 5: Defenses macro summary (grouped bar: utility + leak) ───────────

def plot_defenses_macro_bars(agg_dir, out_dir, fmt):
    rows = load_csv(os.path.join(agg_dir, "defenses_ci.csv"))

    defenses = [r["defense"] for r in rows]
    utility = [float(r["utility_rate"]) for r in rows]
    utility_lo = [float(r["utility_rate"]) - float(r["utility_lo"]) for r in rows]
    utility_hi = [float(r["utility_hi"]) - float(r["utility_rate"]) for r in rows]
    leak = [float(r["leak_rate"]) for r in rows]
    leak_lo = [float(r["leak_rate"]) - float(r["leak_lo"]) for r in rows]
    leak_hi = [float(r["leak_hi"]) - float(r["leak_rate"]) for r in rows]

    x = np.arange(len(defenses))
    width = 0.38

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _add_ygrid(ax)

    # Solid borders, slightly less opaque fill (matched to Figure 4 style).
    ax.bar(x - width / 2, utility, width,
           color=lighten(UTILITY_COLOR, 0.72),
           edgecolor=UTILITY_COLOR,
           linewidth=BAR_EDGE_LW,
           linestyle="solid",
           label="Utility")
    ax.errorbar(x - width / 2, utility,
                yerr=[utility_lo, utility_hi],
                fmt="none", color="#444444", capsize=3, linewidth=0.9)

    ax.bar(x + width / 2, leak, width,
           color=lighten(LEAK_COLOR, 0.72),
           edgecolor=LEAK_COLOR,
           linewidth=BAR_EDGE_LW,
           linestyle="solid",
           label="Leak")
    ax.errorbar(x + width / 2, leak,
                yerr=[leak_lo, leak_hi],
                fmt="none", color="#444444", capsize=3, linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([DEFENSE_LABELS.get(d, d) for d in defenses], rotation=15, ha="right")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend()

    _save(fig, out_dir, "defenses_macro_bars", fmt)
    plt.close(fig)


def plot_defenses_macro_bars_engaged(agg_dir, out_dir, fmt):
    """Same as plot_defenses_macro_bars but reports engagement-conditioned
    leak rate (leak rate on the subset of runs the agent did not refuse)."""
    rows = load_csv(os.path.join(agg_dir, "defenses_ci.csv"))

    defenses = [r["defense"] for r in rows]
    utility = [float(r["utility_rate"]) for r in rows]
    utility_lo = [float(r["utility_rate"]) - float(r["utility_lo"]) for r in rows]
    utility_hi = [float(r["utility_hi"]) - float(r["utility_rate"]) for r in rows]
    leak = [float(r["engagement_leak_rate"]) for r in rows]
    leak_lo = [float(r["engagement_leak_rate"]) - float(r["engagement_leak_lo"]) for r in rows]
    leak_hi = [float(r["engagement_leak_hi"]) - float(r["engagement_leak_rate"]) for r in rows]

    x = np.arange(len(defenses))
    width = 0.38

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _add_ygrid(ax)

    ax.bar(x - width / 2, utility, width,
           color=lighten(UTILITY_COLOR, 0.72),
           edgecolor=UTILITY_COLOR,
           linewidth=BAR_EDGE_LW,
           linestyle="solid",
           label="Utility")
    ax.errorbar(x - width / 2, utility,
                yerr=[utility_lo, utility_hi],
                fmt="none", color="#444444", capsize=3, linewidth=0.9)

    ax.bar(x + width / 2, leak, width,
           color=lighten(LEAK_COLOR, 0.72),
           edgecolor=LEAK_COLOR,
           linewidth=BAR_EDGE_LW,
           linestyle="solid",
           label="Engaged Leakage")
    ax.errorbar(x + width / 2, leak,
                yerr=[leak_lo, leak_hi],
                fmt="none", color="#444444", capsize=3, linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([DEFENSE_LABELS.get(d, d) for d in defenses], rotation=15, ha="right")
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend()

    _save(fig, out_dir, "defenses_macro_bars_engaged", fmt)
    plt.close(fig)


# ── Figure 6: E2E preliminary bars ───────────────────────────────────────────

def _e2e_engaged_leak_rates(agg_dir):
    """Return {model_slug: engaged_leak_rate} from visual_mixed JSONL files.

    'Engaged' = agent reached a completion assessment (utility=1 OR
    completion_assessment.completed=True), i.e. it finished or attempted to
    finish the task before being budget-truncated.  Using this broader
    denominator avoids over-inflating the rate when only a handful of runs
    pass the strict utility check.
    """
    visual_dir = os.path.join(os.path.dirname(agg_dir), "visual_mixed")
    result = {}
    for path in glob.glob(os.path.join(visual_dir, "benchmark_results__*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            continue
        model = rows[0]["model_name"]
        engaged = [
            r for r in rows
            if r.get("utility") == 1
            or r.get("completion_assessment", {}).get("completed")
        ]
        leaked = [r for r in engaged if r.get("leaked_items")]
        result[model] = len(leaked) / len(engaged) if engaged else 0.0
    return result


def _e2e_all_leak_rates(agg_dir):
    """Return {model_slug: all_tasks_leak_rate} from visual_mixed JSONL files."""
    visual_dir = os.path.join(os.path.dirname(agg_dir), "visual_mixed")
    result = {}
    for path in glob.glob(os.path.join(visual_dir, "benchmark_results__*.jsonl")):
        rows = [json.loads(l) for l in open(path)]
        if not rows:
            continue
        model = rows[0]["model_name"]
        leaked = [r for r in rows if r.get("leaked_items")]
        result[model] = len(leaked) / len(rows) if rows else 0.0
    return result


def plot_e2e_bars(agg_dir, out_dir, fmt):
    """Grouped bars: S2A leak rate vs E2E engaged-task leak rate, per model."""
    ci_path = os.path.join(agg_dir, "e2e_ci.csv")
    if not os.path.exists(ci_path):
        print("  e2e_ci.csv not found, skipping e2e_bars")
        return
    e2e_rows = load_csv(ci_path)
    if not e2e_rows:
        print("  e2e_ci.csv is empty, skipping e2e_bars")
        return

    s2a_index = {r["model"]: float(r["leak_rate"])
                 for r in load_csv(os.path.join(agg_dir, "s2a_ci.csv"))}
    engaged_rates = _e2e_engaged_leak_rates(agg_dir)

    models = [r["model"] for r in e2e_rows
              if r["model"] in s2a_index and r["model"] in engaged_rates]
    if not models:
        print("  no matching models found, skipping e2e_bars")
        return

    labels = [model_label(m) for m in models]
    s2a_leak = [s2a_index[m] for m in models]
    e2e_leak = [engaged_rates[m] for m in models]

    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    _add_ygrid(ax)

    _bar(ax, x - width / 2, s2a_leak, width, UTILITY_COLOR, label="S2A leak")
    _bar(ax, x + width / 2, e2e_leak, width, LEAK_COLOR, label="E2E leak (engaged)")

    for xi, (s, e) in enumerate(zip(s2a_leak, e2e_leak)):
        ax.text(xi - width / 2, s + 0.02, f"{s:.0%}", ha="center", fontsize=9.5, fontweight=LABEL_FONT_WEIGHT,
                color=UTILITY_COLOR)
        ax.text(xi + width / 2, e + 0.02, f"{e:.0%}", ha="center", fontsize=9.5, fontweight=LABEL_FONT_WEIGHT,
                color=LEAK_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Leak rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9.5)

    _save(fig, out_dir, "e2e_bars", fmt)
    plt.close(fig)


# ── Figure 6b: E2E comparison (S2A / all tasks / completed tasks) ─────────────

def plot_e2e_comparison(agg_dir, out_dir, fmt):
    """3 bars per model: S2A leak | E2E all-tasks | E2E engaged-tasks."""
    ci_path = os.path.join(agg_dir, "e2e_ci.csv")
    if not os.path.exists(ci_path):
        print("  e2e_ci.csv not found, skipping e2e_comparison")
        return
    e2e_rows = load_csv(ci_path)
    if not e2e_rows:
        print("  e2e_ci.csv is empty, skipping e2e_comparison")
        return

    s2a_index = {r["model"]: float(r["leak_rate"])
                 for r in load_csv(os.path.join(agg_dir, "s2a_ci.csv"))}
    all_rates = _e2e_all_leak_rates(agg_dir)
    engaged_rates = _e2e_engaged_leak_rates(agg_dir)

    models = [r["model"] for r in e2e_rows
              if r["model"] in s2a_index
              and r["model"] in all_rates
              and r["model"] in engaged_rates]
    if not models:
        print("  no matching models found, skipping e2e_comparison")
        return

    labels = [model_label(m) for m in models]
    s2a_leak = [s2a_index[m] for m in models]
    e2e_all = [all_rates[m] for m in models]
    e2e_comp = [engaged_rates[m] for m in models]

    x = np.arange(len(models))
    width_s2a = 0.26
    width_bg = 0.28
    width_fg = 0.17
    group_offset = (width_s2a + width_bg) / 4

    x_s2a = x - group_offset
    x_e2e = x + group_offset

    E2E_COLOR = "#e8a020"
    E2E_COMP_COLOR = "#c07840"

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    _add_ygrid(ax)

    # S2A: light fill with solid blue border
    _bar(ax, x_s2a, s2a_leak, width_s2a, UTILITY_COLOR, zorder=3)

    # E2E completed-tasks: wide, hatched background bar
    ax.bar(x_e2e, e2e_comp, width_bg,
           color=lighten(E2E_COMP_COLOR, 0.70),
           hatch="///",
           edgecolor=E2E_COMP_COLOR,
           linewidth=BAR_EDGE_LW,
           zorder=2)

    # E2E all-tasks: narrower, light fill with dark border, drawn on top
    ax.bar(x_e2e, e2e_all, width_fg,
           color=lighten(E2E_COLOR),
           edgecolor=E2E_COLOR,
           linewidth=BAR_EDGE_LW,
           zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Leak rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.0)

    legend_handles = [
        mpatches.Patch(facecolor=lighten(UTILITY_COLOR), edgecolor=UTILITY_COLOR,
                       linewidth=BAR_EDGE_LW, label="S2A"),
        mpatches.Patch(facecolor=lighten(E2E_COLOR), edgecolor=E2E_COLOR,
                       linewidth=BAR_EDGE_LW, label="E2E — all tasks"),
        mpatches.Patch(facecolor=lighten(E2E_COMP_COLOR, 0.70),
                       hatch="///", edgecolor=E2E_COMP_COLOR,
                       linewidth=BAR_EDGE_LW, label="E2E — engaged tasks"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9.5)

    _save(fig, out_dir, "e2e_comparison", fmt)
    plt.close(fig)


# ── Figure 7: S2A per-mode summary (macro, 3 modes) ─────────────────────────

def plot_s2a_mode_summary(agg_dir, out_dir, fmt):
    """Compact 3-bar chart: macro leak rate per failure mode across all models."""
    rows = load_csv(os.path.join(agg_dir, "s2a_per_mode_ci.csv"))

    modes = ["task_ambiguity_overshare", "recipient_misalignment", "visual_co_location"]
    from collections import defaultdict

    mode_vals = defaultdict(list)
    for r in rows:
        mode_vals[r["failure_mode"]].append(float(r["leak_rate"]))

    means = [np.mean(mode_vals[m]) for m in modes]

    rng = np.random.default_rng(42)
    cis = []
    for m in modes:
        vals = np.array(mode_vals[m])
        boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(2000)]
        cis.append((np.quantile(boots, 0.025), np.quantile(boots, 0.975)))

    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    _add_ygrid(ax)

    for i, (mode, mean, (lo, hi)) in enumerate(zip(modes, means, cis)):
        color = MODE_COLORS[mode]
        _bar(ax, i, mean, 0.55, color)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                    fmt="none", color="#444444", capsize=4, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m] for m in modes], rotation=10, ha="right")
    ax.set_ylabel("Mean leak rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.0)

    _save(fig, out_dir, "s2a_mode_summary", fmt)
    plt.close(fig)


# ── Figure 8: Defense Pareto averaged over models (4 dots) ───────────────────

def plot_defenses_pareto_avg(agg_dir, out_dir, fmt):
    """4-dot Pareto: each dot is one defense condition, averaged over models."""
    rows = load_csv(os.path.join(agg_dir, "defenses_ci.csv"))

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    xs, ys, texts = [], [], []
    for r in rows:
        defense = r["defense"]
        x = float(r["utility_rate"])
        y = float(r["leak_rate"])
        x_lo = x - float(r["utility_lo"])
        x_hi = float(r["utility_hi"]) - x
        y_lo = y - float(r["leak_lo"])
        y_hi = float(r["leak_hi"]) - y

        color = DEFENSE_COLORS.get(defense, "#888")
        marker = DEFENSE_MARKERS.get(defense, "o")

        ax.errorbar(
            x, y,
            xerr=[[x_lo], [x_hi]],
            yerr=[[y_lo], [y_hi]],
            fmt=marker,
            color=color,
            markersize=10,
            capsize=3,
            linewidth=1.0,
            markeredgewidth=1.0,
            markeredgecolor=color,
            markerfacecolor=lighten(color, 0.35),
            alpha=0.95,
            label=DEFENSE_LABELS.get(defense, defense),
        )
        texts.append(ax.text(x, y, DEFENSE_LABELS.get(defense, defense), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x)
        ys.append(y)

    px, py = pareto_frontier(xs, ys)
    px, py = list(px), list(py)
    if px:
        ax.plot(px, py, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=0,
                label="Pareto frontier")

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leak rate")
    ax.set_xlim(0.55, 1.05)
    ax.set_ylim(-0.04, 0.55)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper right")

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "defenses_pareto_avg", fmt)
    plt.close(fig)


# ── Figure 9: S2A sweep without CI, with regression line ─────────────────────

def plot_s2a_pareto_regression(agg_dir, out_dir, fmt):
    """S2A scatter (no error bars) with linear regression line + 95% CI band."""
    rows = load_csv(os.path.join(agg_dir, "s2a_ci.csv"))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    xs, ys, texts = [], [], []
    for r in rows:
        x = float(r["utility_rate"])
        y = float(r["leak_rate"])
        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")

        ax.scatter(x, y, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=0.8, s=44, zorder=4)
        texts.append(ax.text(x, y, model_label(r["model"]), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x)
        ys.append(y)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    coeffs = np.polyfit(xs_arr, ys_arr, 1)
    x_fit = np.linspace(xs_arr.min() - 0.05, xs_arr.max() + 0.05, 300)
    y_fit = np.polyval(coeffs, x_fit)

    rng = np.random.default_rng(42)
    n = len(xs_arr)
    boot_lines = np.array([
        np.polyval(np.polyfit(xs_arr[idx := rng.integers(0, n, n)],
                              ys_arr[idx], 1), x_fit)
        for _ in range(2000)
    ])
    y_band_lo = np.quantile(boot_lines, 0.025, axis=0)
    y_band_hi = np.quantile(boot_lines, 0.975, axis=0)

    ax.fill_between(x_fit, y_band_lo, y_band_hi, color="#888888", alpha=0.12, zorder=1)
    ax.plot(x_fit, y_fit, color="#444444", linewidth=1.4, zorder=3,
            label=f"slope={coeffs[0]:.2f}")

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leak rate")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_pareto_regression", fmt)
    plt.close(fig)


# ── Figure 10: S2A leak rate vs engaged leak rate — scatter with arrows ───────

def plot_s2a_leak_vs_engaged(agg_dir, out_dir, fmt):
    """Scatter: overall leak (circle) vs engaged leak (triangle) per model, with arrows."""
    all_rows = load_csv(os.path.join(agg_dir, "main_table.csv"))
    rows = [r for r in all_rows
            if r["setting"] == "reasoning" and r["access_mode"] == "text"]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    xs, ys, texts = [], [], []
    for r in rows:
        x = float(r["utility_rate"])
        y_leak = float(r["leak_rate"])
        y_eng = float(r["engagement_leak_rate"])
        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")

        ax.annotate(
            "", xy=(x, y_eng), xytext=(x, y_leak),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                            mutation_scale=10),
            zorder=2,
        )

        ax.scatter(x, y_leak, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=1.0, s=70, zorder=3, marker="o")
        ax.scatter(x, y_eng, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=1.0, s=70, zorder=3, marker="^")

        texts.append(ax.text(x, y_eng, model_label(r["model"]),
                             fontsize=LABEL_FONT_SIZE,
                             fontweight=LABEL_FONT_WEIGHT, color=color))
        xs.append(x); ys.append(y_eng)

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leak rate")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    import matplotlib.lines as mlines
    h_leak = mlines.Line2D([], [], color="#555555", marker="o", linestyle="None",
                           markersize=5, markeredgewidth=0.8,
                           markeredgecolor="#555555",
                           markerfacecolor=lighten("#555555", 0.45),
                           label="Overall leak rate")
    h_eng = mlines.Line2D([], [], color="#555555", marker="^", linestyle="None",
                          markersize=5, markeredgewidth=0.8,
                          markeredgecolor="#555555",
                          markerfacecolor=lighten("#555555", 0.45),
                          label="Engaged leak rate")
    ax.legend(handles=[h_leak, h_eng], loc="upper left")

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_leak_vs_engaged", fmt)
    plt.close(fig)


# ── Figure 11: Combined panel (regression + Pareto | leak vs engaged) ────────

def plot_s2a_combined_panel(agg_dir, out_dir, fmt):
    """Wide two-panel figure: left = regression scatter + Pareto frontier,
    right = overall vs engaged leak rate scatter with arrows."""
    s2a_rows = load_csv(os.path.join(agg_dir, "s2a_ci.csv"))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    fig.subplots_adjust(wspace=0.32)

    # ── Left panel: regression + Pareto frontier ─────────────────────────────
    xs, ys, texts_l = [], [], []
    for r in s2a_rows:
        x = float(r["utility_rate"])
        y = float(r["leak_rate"])
        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")

        ax_l.scatter(x, y, color=lighten(color, 0.35), edgecolors=color,
                     linewidths=0.8, s=44, zorder=4)
        texts_l.append(ax_l.text(x, y, model_label(r["model"]), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x)
        ys.append(y)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    coeffs = np.polyfit(xs_arr, ys_arr, 1)
    x_fit = np.linspace(xs_arr.min() - 0.05, xs_arr.max() + 0.05, 300)
    y_fit = np.polyval(coeffs, x_fit)

    rng = np.random.default_rng(42)
    n = len(xs_arr)
    boot_lines = np.array([
        np.polyval(np.polyfit(xs_arr[idx := rng.integers(0, n, n)],
                              ys_arr[idx], 1), x_fit)
        for _ in range(2000)
    ])
    y_band_lo = np.quantile(boot_lines, 0.025, axis=0)
    y_band_hi = np.quantile(boot_lines, 0.975, axis=0)

    ax_l.fill_between(x_fit, y_band_lo, y_band_hi, color="#888888", alpha=0.12, zorder=1)
    ax_l.plot(x_fit, y_fit, color="#444444", linewidth=1.4, zorder=3,
              label=f"slope={coeffs[0]:.2f}")

    px, py = pareto_frontier(xs, ys)
    px, py = list(px), list(py)
    if px:
        ax_l.plot(px, py, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=0,
                  label="Pareto frontier")

    ax_l.set_xlabel("Utility rate")
    ax_l.set_ylabel("Leak rate")
    ax_l.set_xlim(-0.02, 1.08)
    ax_l.set_ylim(-0.04, 1.08)
    ax_l.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_l.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_l.legend(loc="upper left", fontsize=9.5)
    ax_l.set_title("(a) Utility–leak trade-off")
    _adjust(texts_l, xs, ys, ax_l)

    # ── Right panel: leak vs engaged leak scatter with arrows ─────────────────
    main_table_path = os.path.join(agg_dir, "main_table.csv")
    if not os.path.exists(main_table_path):
        ax_r.text(0.5, 0.5, "main_table.csv not found",
                  ha="center", va="center", transform=ax_r.transAxes, fontsize=12)
    else:
        all_rows = load_csv(main_table_path)
        eng_rows = [r for r in all_rows
                    if r["setting"] == "reasoning" and r["access_mode"] == "text"]

        rxs, rys, texts_r = [], [], []
        for r in eng_rows:
            x = float(r["utility_rate"])
            y_leak = float(r["leak_rate"])
            y_eng = float(r["engagement_leak_rate"])
            fam = model_family(r["model"])
            color = FAMILY_COLORS.get(fam, "#888888")

            ax_r.annotate(
                "",
                xy=(x, y_eng),
                xytext=(x, y_leak),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=8),
                zorder=2,
            )
            ax_r.scatter(x, y_leak, color=lighten(color, 0.35), edgecolors=color,
                         linewidths=1.0, s=70, zorder=3, marker="o")
            ax_r.scatter(x, y_eng, color=lighten(color, 0.35), edgecolors=color,
                         linewidths=1.0, s=70, zorder=3, marker="^")
            texts_r.append(ax_r.text(x, y_eng, model_label(r["model"]),
                                     fontsize=LABEL_FONT_SIZE,
                                     fontweight=LABEL_FONT_WEIGHT, color=color))
            rxs.append(x); rys.append(y_eng)

        _adjust(texts_r, rxs, rys, ax_r)

        import matplotlib.lines as mlines
        h_leak = mlines.Line2D([], [], color="#555555", marker="o", linestyle="None",
                               markersize=5, markeredgewidth=0.8,
                               markeredgecolor="#555555", markerfacecolor=lighten("#555555", 0.45),
                               label="Overall leak rate")
        h_eng = mlines.Line2D([], [], color="#555555", marker="^", linestyle="None",
                              markersize=5, markeredgewidth=0.8,
                              markeredgecolor="#555555", markerfacecolor=lighten("#555555", 0.45),
                              label="Engaged leak rate")
        ax_r.legend(handles=[h_leak, h_eng], loc="upper left", fontsize=9.5)

    ax_r.set_xlabel("Utility rate")
    ax_r.set_ylabel("Leak rate")
    ax_r.set_xlim(-0.02, 1.08)
    ax_r.set_ylim(-0.04, 1.08)
    ax_r.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_r.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax_r.set_title("(b) Overall vs. engaged leak rate")

    _save(fig, out_dir, "s2a_combined_panel", fmt)
    plt.close(fig)


# ── helpers shared by new plots ──────────────────────────────────────────────

def _load_main_reasoning(agg_dir):
    """Return rows from main_table.csv restricted to reasoning/text setting."""
    rows = load_csv(os.path.join(agg_dir, "main_table.csv"))
    return [r for r in rows if r["setting"] == "reasoning" and r["access_mode"] == "text"]


_QUAD_U_THRESH = 0.75
_QUAD_L_THRESH = 0.50
_QUAD_CAREFUL_COLOR = "#d6ecd6"   # light green
_QUAD_CARELESS_COLOR = "#f4d6d6"  # light red
_QUAD_GREY = "#f0f0f0"


def _draw_quadrants(ax, u_thresh=_QUAD_U_THRESH, l_thresh=_QUAD_L_THRESH,
                    x_max=1.08, y_max=1.08, x_min=-0.02, y_min=-0.04,
                    label_quads=True):
    """Shade the four quadrants and draw threshold guides."""
    # Bottom-right (high U, low L): careful & capable — green
    ax.add_patch(mpatches.Rectangle(
        (u_thresh, y_min), x_max - u_thresh, l_thresh - y_min,
        facecolor=_QUAD_CAREFUL_COLOR, edgecolor="none", alpha=0.55, zorder=0))
    # Top-right (high U, high L): capable but careless — red
    ax.add_patch(mpatches.Rectangle(
        (u_thresh, l_thresh), x_max - u_thresh, y_max - l_thresh,
        facecolor=_QUAD_CARELESS_COLOR, edgecolor="none", alpha=0.55, zorder=0))
    # Top-left (low U, high L): failing — light grey
    ax.add_patch(mpatches.Rectangle(
        (x_min, l_thresh), u_thresh - x_min, y_max - l_thresh,
        facecolor=_QUAD_GREY, edgecolor="none", alpha=0.55, zorder=0))
    # Bottom-left (low U, low L): refusing — slightly lighter
    ax.add_patch(mpatches.Rectangle(
        (x_min, y_min), u_thresh - x_min, l_thresh - y_min,
        facecolor="#fafafa", edgecolor="none", alpha=0.6, zorder=0))

    ax.axvline(u_thresh, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(l_thresh, color="#888888", linewidth=0.7, linestyle="--", zorder=1)

    if label_quads:
        common = dict(fontsize=8, fontweight="bold", color="#555555",
                      zorder=1, alpha=0.85)
        ax.text(u_thresh + 0.01, l_thresh + 0.02, "Capable but careless",
                ha="left", va="bottom", **common)
        ax.text(u_thresh + 0.01, y_min + 0.02, "Capable & careful",
                ha="left", va="bottom", **common)
        ax.text(x_min + 0.01, l_thresh + 0.02, "Failing on both",
                ha="left", va="bottom", **common)
        ax.text(x_min + 0.01, y_min + 0.02, "Refusing",
                ha="left", va="bottom", **common)


# ── Figure 12: Four-quadrant Pareto (utility vs engagement-conditioned leak) ──

def plot_s2a_pareto_quadrants(agg_dir, out_dir, fmt):
    """Pareto plot with four quadrants at U=75% and L_eng=50%."""
    rows = _load_main_reasoning(agg_dir)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    _draw_quadrants(ax)

    xs, ys, texts = [], [], []
    for r in rows:
        x = float(r["utility_rate"])
        y = float(r["engagement_leak_rate"])
        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.scatter(x, y, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=0.9, s=60, zorder=4)
        texts.append(ax.text(x, y, model_label(r["model"]), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x)
        ys.append(y)

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Engagement-conditioned leakage")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_pareto_quadrants", fmt)
    plt.close(fig)


# ── Figure 13: Quadrants + marker size = refusal rate ─────────────────────────

def plot_s2a_pareto_quadrants_size(agg_dir, out_dir, fmt):
    """Four-quadrant Pareto where marker size encodes refusal rate."""
    rows = _load_main_reasoning(agg_dir)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    _draw_quadrants(ax)

    xs, ys, texts = [], [], []
    sizes = []
    for r in rows:
        x = float(r["utility_rate"])
        y = float(r["engagement_leak_rate"])
        refusal = float(r["refusal_rate"])
        # Marker size proportional to refusal; baseline area 40, max area 360
        s = 40 + refusal * 700
        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.scatter(x, y, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=0.9, s=s, zorder=4, alpha=0.9)
        texts.append(ax.text(x, y, model_label(r["model"]), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x); ys.append(y); sizes.append(s)

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Engagement-conditioned leakage")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    # Marker-size legend (refusal = 0%, 20%, 40%)
    legend_handles = []
    for ref in (0.0, 0.20, 0.40):
        s = 40 + ref * 700
        legend_handles.append(plt.scatter(
            [], [], s=s, color="#cccccc", edgecolors="#555555", linewidths=0.8,
            label=f"refusal {ref*100:.0f}%"))
    ax.legend(handles=legend_handles, loc="upper left", labelspacing=1.2,
              borderpad=0.7, fontsize=9.5)

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_pareto_quadrants_size", fmt)
    plt.close(fig)


# ── Figure 14: Quadrants + arrow per agent from raw L to engaged L ────────────

def plot_s2a_pareto_quadrants_arrows(agg_dir, out_dir, fmt):
    """Four-quadrant Pareto with a per-agent vertical arrow from raw L to L_eng."""
    rows = _load_main_reasoning(agg_dir)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    _draw_quadrants(ax)

    xs, ys, texts = [], [], []
    for r in rows:
        x = float(r["utility_rate"])
        y_raw = float(r["leak_rate"])
        y_eng = float(r["engagement_leak_rate"])
        fam = model_family(r["model"])
        color = FAMILY_COLORS.get(fam, "#888888")

        if abs(y_eng - y_raw) > 1e-4:
            ax.annotate(
                "", xy=(x, y_eng), xytext=(x, y_raw),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                mutation_scale=10),
                zorder=2,
            )
        ax.scatter(x, y_raw, color="white", edgecolors=color, linewidths=1.0,
                   s=32, zorder=3, marker="o")
        ax.scatter(x, y_eng, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=1.0, s=46, zorder=4, marker="o")
        texts.append(ax.text(x, y_eng, model_label(r["model"]),
                             fontsize=LABEL_FONT_SIZE,
                             fontweight=LABEL_FONT_WEIGHT, color=color))
        xs.append(x); ys.append(y_eng)

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leakage")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    import matplotlib.lines as mlines
    h_raw = mlines.Line2D([], [], marker="o", linestyle="None", markersize=9,
                          markeredgecolor="#555555", markerfacecolor="white",
                          markeredgewidth=1.0, label="Raw leakage")
    h_eng = mlines.Line2D([], [], marker="o", linestyle="None", markersize=10,
                          markeredgecolor="#555555",
                          markerfacecolor=lighten("#555555", 0.45),
                          markeredgewidth=1.0, label="Engaged leakage")
    ax.legend(handles=[h_raw, h_eng], loc="upper left")

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_pareto_quadrants_arrows", fmt)
    plt.close(fig)


# ── Figure 15: Quadrants + color = refusal rate ───────────────────────────────

def plot_s2a_pareto_quadrants_color(agg_dir, out_dir, fmt):
    """Four-quadrant Pareto with point color encoding refusal rate."""
    rows = _load_main_reasoning(agg_dir)

    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    _draw_quadrants(ax, label_quads=True, x_max=1.15)

    soft_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "soft_refusal",
        ["#fbf5ec", "#f3d9a4", "#d98c5f", "#8c3a2b"],
        N=256,
    )
    cmap = soft_cmap
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=0.5)

    xs, ys, texts = [], [], []
    for r in rows:
        x = float(r["utility_rate"])
        y = float(r["engagement_leak_rate"])
        refusal = float(r["refusal_rate"])
        color = cmap(norm(refusal))
        ax.scatter(x, y, color=color, edgecolors="#333333", linewidths=1.0,
                   s=70, zorder=4)
        texts.append(ax.text(x, y, model_label(r["model"]),
                             fontsize=LABEL_FONT_SIZE,
                             fontweight=LABEL_FONT_WEIGHT, color="#222222"))
        xs.append(x); ys.append(y)

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Engagement-conditioned leakage")
    ax.set_xlim(-0.02, 1.15)
    ax.set_ylim(-0.04, 1.10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.042, pad=0.04)
    cbar.set_label("Refusal rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_pareto_quadrants_color", fmt)
    plt.close(fig)


# ── Figure 16: Capability rank vs leakage rank ────────────────────────────────

def plot_rank_inversion(agg_dir, out_dir, fmt):
    """Rank-rank scatter: agents ordered by utility (x) vs by leakage (y)."""
    rows = _load_main_reasoning(agg_dir)

    by_util = sorted(rows, key=lambda r: -float(r["utility_rate"]))
    by_leak = sorted(rows, key=lambda r: float(r["engagement_leak_rate"]))

    util_rank = {r["model"]: i + 1 for i, r in enumerate(by_util)}
    leak_rank = {r["model"]: i + 1 for i, r in enumerate(by_leak)}

    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    n = len(rows)
    # Diagonal (perfect alignment)
    ax.plot([1, n], [1, n], color="#bbbbbb", linewidth=1.0, linestyle="--",
            zorder=1, label="Perfect alignment")

    xs, ys, texts = [], [], []
    for r in rows:
        m = r["model"]
        x = util_rank[m]
        y = leak_rank[m]
        fam = model_family(m)
        color = FAMILY_COLORS.get(fam, "#888888")
        ax.scatter(x, y, color=lighten(color, 0.35), edgecolors=color,
                   linewidths=0.9, s=70, zorder=3)
        texts.append(ax.text(x, y, model_label(m), fontsize=SCATTER_LABEL_SIZE, color=color, fontweight=LABEL_FONT_WEIGHT))
        xs.append(x); ys.append(y)

    # Spearman rank correlation
    xs_arr = np.array(xs); ys_arr = np.array(ys)
    rho_num = np.sum((xs_arr - xs_arr.mean()) * (ys_arr - ys_arr.mean()))
    rho_den = np.sqrt(np.sum((xs_arr - xs_arr.mean())**2) * np.sum((ys_arr - ys_arr.mean())**2))
    spearman = rho_num / rho_den if rho_den else 0.0

    ax.set_xlabel("Capability rank (1 = highest utility)")
    ax.set_ylabel("Disclosure rank (1 = lowest leakage)")
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9.5,
              title=f"Spearman $\\rho$ = {spearman:.2f}", title_fontsize=10)

    _adjust(texts, xs, ys, ax)

    _save(fig, out_dir, "s2a_rank_inversion", fmt)
    plt.close(fig)


# ── Figure 16b: Bump chart — rank under utility vs rank under leakage ─────────

def plot_rank_bump(agg_dir, out_dir, fmt):
    """Bump chart: each model has its utility rank on the left, leakage rank on
    the right. Lines that cross signal rank inversions (capable but careless)."""
    rows = _load_main_reasoning(agg_dir)

    by_util = sorted(rows, key=lambda r: -float(r["utility_rate"]))
    by_leak = sorted(rows, key=lambda r: float(r["engagement_leak_rate"]))

    util_rank = {r["model"]: i + 1 for i, r in enumerate(by_util)}
    leak_rank = {r["model"]: i + 1 for i, r in enumerate(by_leak)}

    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    x_left, x_right = 0.0, 1.0

    for r in rows:
        m = r["model"]
        yL = util_rank[m]
        yR = leak_rank[m]
        fam = model_family(m)
        color = FAMILY_COLORS.get(fam, "#888888")

        ax.plot([x_left, x_right], [yL, yR], color=color, lw=2.0,
                alpha=0.78, zorder=2)
        ax.scatter([x_left, x_right], [yL, yR],
                   color=lighten(color, 0.35), edgecolors=color,
                   linewidths=1.2, s=42, zorder=3)
        ax.text(x_left - 0.07, yL, model_label(m), ha="right", va="center",
                fontsize=SCATTER_LABEL_SIZE, color=color)
        ax.text(x_right + 0.07, yR, model_label(m), ha="left", va="center",
                fontsize=SCATTER_LABEL_SIZE, color=color)

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["Capability\n(1 = highest utility)",
                        "Disclosure\n(1 = lowest leakage)"],
                       fontsize=10)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(x_left - 0.6, x_right + 0.6)
    ax.set_ylim(n + 0.8, 0.2)
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)

    _save(fig, out_dir, "s2a_rank_bump", fmt)
    plt.close(fig)


# ── Figure 17: Defenses as arrows on the utility-leak plane ───────────────────

def plot_defenses_arrows_pareto(agg_dir, out_dir, fmt):
    """Per-model arrows from no-defense point to each defense point on the
    utility-leak plane, with four quadrants shaded."""
    rows = load_csv(os.path.join(agg_dir, "defenses_per_model_ci.csv"))

    by_model = {}
    for r in rows:
        m = r["model"]
        by_model.setdefault(m, {})[r["defense"]] = (
            float(r["utility_rate"]), float(r["leak_rate"]))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    _draw_quadrants(ax)

    for m, d in by_model.items():
        if "none" not in d:
            continue
        x0, y0 = d["none"]
        fam = model_family(m)
        color = FAMILY_COLORS.get(fam, "#888888")

        # Baseline marker (hollow)
        ax.scatter(x0, y0, color="white", edgecolors=color, linewidths=1.4,
                   s=70, zorder=4, marker="o")
        ax.text(x0, y0 - 0.03, model_label(m), ha="center", va="top",
                fontsize=LABEL_FONT_SIZE, fontweight=LABEL_FONT_WEIGHT,
                color=color, zorder=5)

        # Arrow to each defense
        for def_name, (xd, yd) in d.items():
            if def_name == "none":
                continue
            ax.annotate(
                "", xy=(xd, yd), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=DEFENSE_COLORS.get(def_name, "#888"),
                                lw=1.1, mutation_scale=9, alpha=0.85,
                                connectionstyle="arc3,rad=0.0"),
                zorder=3,
            )
            ax.scatter(xd, yd, color=lighten(DEFENSE_COLORS.get(def_name, "#888"), 0.35),
                       edgecolors=DEFENSE_COLORS.get(def_name, "#888"),
                       linewidths=0.8,
                       s=55, zorder=4,
                       marker=DEFENSE_MARKERS.get(def_name, "s"))

    ax.set_xlabel("Utility rate")
    ax.set_ylabel("Leak rate")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.04, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                             markeredgecolor="#555555", markerfacecolor="white",
                             markeredgewidth=1.0, markersize=7,
                             label="No defense (baseline)")]
    for d in ("recipient_typed", "restrictive", "rubric_informed"):
        handles.append(mlines.Line2D(
            [], [], marker=DEFENSE_MARKERS[d], linestyle="None",
            markeredgecolor=DEFENSE_COLORS[d],
            markerfacecolor=lighten(DEFENSE_COLORS[d], 0.35),
            markeredgewidth=0.8, markersize=7,
            label=DEFENSE_LABELS[d]))
    ax.legend(handles=handles, loc="upper right", fontsize=9.5)

    _save(fig, out_dir, "defenses_arrows_pareto", fmt)
    plt.close(fig)


# ── Figure 18: Behaviour decomposition stacked bars ───────────────────────────

def plot_behavior_decomposition(agg_dir, out_dir, fmt):
    """Stacked bars per model: completed-clean / completed-leak /
    incomplete-clean / incomplete-leak, from confusion.csv."""
    conf_path = os.path.join(agg_dir, "confusion.csv")
    if not os.path.exists(conf_path):
        print("  confusion.csv not found, skipping behavior_decomposition")
        return
    rows = [r for r in load_csv(conf_path)
            if r["setting"] == "reasoning" and r["access_mode"] == "text"]
    if not rows:
        print("  no reasoning/text rows in confusion.csv, skipping")
        return

    s2a_leak = {r["model"]: float(r["leak_rate"])
                for r in load_csv(os.path.join(agg_dir, "s2a_ci.csv"))}
    rows.sort(key=lambda r: s2a_leak.get(r["model"], 0.0))

    labels = [model_label(r["model"]) for r in rows]
    cc = [float(r["rate_completed_clean"]) for r in rows]
    cl = [float(r["rate_completed_leak"]) for r in rows]
    ic = [float(r["rate_incomplete_clean"]) for r in rows]
    il = [float(r["rate_incomplete_leak"]) for r in rows]

    n = len(rows)
    x = np.arange(n)
    width = 0.7

    colors = {
        "cc": "#3aa86b",   # completed-clean — green
        "cl": "#e04040",   # completed-leak — red
        "ic": "#b0b0b0",   # incomplete-clean — grey
        "il": "#e8a020",   # incomplete-leak — amber
    }

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    _add_ygrid(ax)

    bot = np.zeros(n)
    for vals, key, label in [
        (cc, "cc", "Completed, clean"),
        (cl, "cl", "Completed, leak"),
        (ic, "ic", "Incomplete, clean"),
        (il, "il", "Incomplete, leak"),
    ]:
        ax.bar(x, vals, width, bottom=bot,
               color=lighten(colors[key], 0.25),
               edgecolor=colors[key], linewidth=1.0, label=label)
        bot = bot + np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9.5)
    ax.set_ylabel("Share of scenarios")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4,
              fontsize=9.5, frameon=False)

    _save(fig, out_dir, "s2a_behavior_decomposition", fmt)
    plt.close(fig)


# ── Figure 19: Per-mode heatmap (model × failure mode) ────────────────────────

def plot_per_mode_heatmap(agg_dir, out_dir, fmt):
    """Heatmap of leak_rate over (model, failure_mode)."""
    rows = load_csv(os.path.join(agg_dir, "s2a_per_mode_ci.csv"))

    s2a_leak = {r["model"]: float(r["leak_rate"])
                for r in load_csv(os.path.join(agg_dir, "s2a_ci.csv"))}
    models = sorted(s2a_leak.keys(), key=lambda m: s2a_leak[m])
    modes = ["visual_co_location", "task_ambiguity_overshare", "recipient_misalignment"]

    cell = {(r["model"], r["failure_mode"]): float(r["leak_rate"]) for r in rows}
    Z = np.array([[cell.get((m, mode), np.nan) for mode in modes] for m in models])

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    soft_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "soft_heat",
        ["#f5f6fa", "#fdecec", "#f6c8c2", "#e8948a", "#c75a4d"],
        N=256,
    )
    im = ax.imshow(Z, aspect="auto", cmap=soft_cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(["VCL", "TAO", "RMA"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([model_label(m) for m in models], fontsize=10)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > 0.75 else "#333333"
            ax.text(j, i, f"{val*100:.0f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Leak rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    ax.set_xlabel("Failure mode")

    _save(fig, out_dir, "s2a_per_mode_heatmap", fmt)
    plt.close(fig)


# ── Figure 20: Per-family bars (utility and leak averaged by family) ──────────

def plot_per_family_bars(agg_dir, out_dir, fmt):
    """Grouped bars: mean utility and mean engagement-conditioned leakage per
    model family."""
    rows = _load_main_reasoning(agg_dir)

    from collections import defaultdict
    by_fam_u = defaultdict(list)
    by_fam_l = defaultdict(list)
    by_fam_n = defaultdict(int)
    for r in rows:
        fam = model_family(r["model"])
        by_fam_u[fam].append(float(r["utility_rate"]))
        by_fam_l[fam].append(float(r["engagement_leak_rate"]))
        by_fam_n[fam] += 1

    families = sorted(by_fam_u.keys(), key=lambda f: np.mean(by_fam_l[f]))
    util_means = [np.mean(by_fam_u[f]) for f in families]
    leak_means = [np.mean(by_fam_l[f]) for f in families]

    x = np.arange(len(families))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    _add_ygrid(ax)

    _bar(ax, x - width / 2, util_means, width, UTILITY_COLOR, label="Utility")
    _bar(ax, x + width / 2, leak_means, width, LEAK_COLOR, label="Engaged leakage")

    for xi, (u, l) in enumerate(zip(util_means, leak_means)):
        ax.text(xi - width / 2, u + 0.02, f"{u*100:.0f}", ha="center",
                fontsize=SCATTER_LABEL_SIZE, color=UTILITY_COLOR)
        ax.text(xi + width / 2, l + 0.02, f"{l*100:.0f}", ha="center",
                fontsize=SCATTER_LABEL_SIZE, color=LEAK_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n(n={by_fam_n[f]})" for f in families],
                       fontsize=9.5)
    ax.set_ylabel("Rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9.5)

    _save(fig, out_dir, "s2a_per_family_bars", fmt)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agg-dir", default="data/results/aggregated")
    parser.add_argument("--out-dir", default="figures/generated")
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading from: {args.agg_dir}")
    print(f"Writing to:   {args.out_dir}  (format={args.fmt})")
    print()

    print("[S2A]")
    plot_s2a_pareto(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_pareto_regression(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_per_mode_bars(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_mode_summary(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_leak_vs_engaged(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_combined_panel(args.agg_dir, args.out_dir, args.fmt)

    print("\n[S2A new variants]")
    plot_s2a_pareto_quadrants(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_pareto_quadrants_size(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_pareto_quadrants_arrows(args.agg_dir, args.out_dir, args.fmt)
    plot_s2a_pareto_quadrants_color(args.agg_dir, args.out_dir, args.fmt)
    plot_rank_inversion(args.agg_dir, args.out_dir, args.fmt)
    plot_rank_bump(args.agg_dir, args.out_dir, args.fmt)
    plot_behavior_decomposition(args.agg_dir, args.out_dir, args.fmt)
    plot_per_mode_heatmap(args.agg_dir, args.out_dir, args.fmt)
    plot_per_family_bars(args.agg_dir, args.out_dir, args.fmt)

    print("\n[Defenses]")
    plot_defenses_macro_bars(args.agg_dir, args.out_dir, args.fmt)
    plot_defenses_macro_bars_engaged(args.agg_dir, args.out_dir, args.fmt)
    plot_defenses_pareto(args.agg_dir, args.out_dir, args.fmt)
    plot_defenses_pareto_avg(args.agg_dir, args.out_dir, args.fmt)
    plot_defenses_per_mode_bars(args.agg_dir, args.out_dir, args.fmt)
    plot_defenses_arrows_pareto(args.agg_dir, args.out_dir, args.fmt)

    print("\n[E2E]")
    plot_e2e_bars(args.agg_dir, args.out_dir, args.fmt)
    plot_e2e_comparison(args.agg_dir, args.out_dir, args.fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
