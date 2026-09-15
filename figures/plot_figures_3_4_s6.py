#!/usr/bin/env python3
"""Render the frozen Fig. 3 and Fig. 4 designs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd

from palette import (
    AXIS_LABEL_SIZE,
    EXPORT_DPI,
    FONT_FAMILY,
    GRID_WIDTH,
    GROUP_STYLE,
    LEGEND_SIZE,
    OUTER_PAD_POINTS,
    PALETTE,
    PANEL_LABEL_SIZE,
    SPINE_WIDTH,
    TARGET_COLORS,
    TICK_LABEL_SIZE,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_ROOT = PROJECT_ROOT / "reference_results"
SOURCE_DATA_DIR = HERE / "source_data"
MM_TO_INCH = 1.0 / 25.4
PT_TO_INCH = 1.0 / 72.0

TARGETS = (
    ("B5", r"$B_5$"),
    ("L", r"$L$"),
    ("BurB5", r"Buried $B_5$"),
    ("BurL", r"Buried $L$"),
)


def configure_style() -> None:
    """Apply the user-specified print contract before creating any figure."""
    font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    # Times New Roman is the requested publication-safe alternative to Arial/Helvetica.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": SPINE_WIDTH,
            "figure.facecolor": PALETTE["white"],
            "axes.facecolor": PALETTE["white"],
            "savefig.facecolor": PALETTE["white"],
            "savefig.edgecolor": PALETTE["white"],
            "axes.unicode_minus": False,
        }
    )


def in_points(points: float) -> float:
    """Matplotlib line widths are already specified in typographic points."""
    return points


def style_boxed_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    """Use a thin complete frame and quiet grid without decorative elements."""
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis=grid_axis,
        color=PALETTE["grid"],
        linewidth=GRID_WIDTH,
        linestyle="-",
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.6,
        width=SPINE_WIDTH,
        color=PALETTE["ink"],
        labelcolor=PALETTE["ink"],
        pad=2.2,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=1.6,
        width=SPINE_WIDTH,
        color=PALETTE["neutral_mid"],
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13) -> None:
    ax.text(
        x,
        1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, object]:
    """Export editable vector files and a 600 dpi white-background preview."""
    pad = OUTER_PAD_POINTS * PT_TO_INCH
    common = {
        "bbox_inches": "tight",
        "pad_inches": pad,
        "facecolor": PALETTE["white"],
        "edgecolor": PALETTE["white"],
    }
    pdf_path = HERE / f"{stem}.pdf"
    svg_path = HERE / f"{stem}.svg"
    png_path = HERE / f"{stem}.png"
    fig.savefig(pdf_path, **common)
    fig.savefig(svg_path, **common)
    fig.savefig(png_path, dpi=600, **common)
    outputs = {"pdf": pdf_path.name, "svg": svg_path.name, "png": png_path.name}
    width_in, height_in = fig.get_size_inches()
    plt.close(fig)
    return {
        "files": outputs,
        "design_width_mm": round(width_in / MM_TO_INCH, 2),
        "design_height_mm": round(height_in / MM_TO_INCH, 2),
        "png_dpi": EXPORT_DPI,
        "outer_padding_pt": OUTER_PAD_POINTS,
    }


def load_figure3_data() -> pd.DataFrame:
    external = pd.read_csv(
        DATA_ROOT / "supporting" / "external_baselines.csv"
    )
    fixed = pd.read_csv(
        DATA_ROOT
        / "selection"
        / "fixed_5seed_summary.csv"
    )
    nested = pd.read_csv(
        DATA_ROOT
        / "analysis"
        / "nested_target_summary.csv"
    )

    rows = []
    for _, row in external.iterrows():
        item = {
            "group": row["baseline_group"],
            "method": row["method"],
            "provenance": row["provenance"],
            "n_seeds": row["n_seeds_reported"],
        }
        for key, _ in TARGETS:
            item[f"{key}_mean"] = row[f"{key}_mae"]
            item[f"{key}_std"] = row[f"{key}_std"]
        rows.append(item)

    fixed_rows = {
        "B5": fixed.loc[fixed["property"] == "B5"].iloc[0],
        "L": fixed.loc[fixed["property"] == "L"].iloc[0],
        "BurB5": fixed.loc[fixed["property"] == "BurB5-huber"].iloc[0],
        "BurL": fixed.loc[fixed["property"] == "BurL"].iloc[0],
    }
    fixed_item = {
        "group": "fcsr_fixed",
        "method": "FCSR, fixed",
        "provenance": "our_rerun",
        "n_seeds": 5,
    }
    for key, _ in TARGETS:
        fixed_item[f"{key}_mean"] = fixed_rows[key]["mae_mean"]
        fixed_item[f"{key}_std"] = fixed_rows[key]["mae_std"]
    rows.append(fixed_item)

    nested_by_target = nested.set_index("target_short")
    nested_item = {
        "group": "fcsr_nested",
        "method": "FCSR, nested",
        "provenance": "our_clean_room_rerun",
        "n_seeds": 5,
    }
    for key, _ in TARGETS:
        nested_item[f"{key}_mean"] = nested_by_target.loc[key, "mae_mean"]
        nested_item[f"{key}_std"] = nested_by_target.loc[key, "mae_std"]
    rows.append(nested_item)

    result = pd.DataFrame(rows)
    if len(result) != 14:
        raise ValueError(f"Figure 3 requires 14 method rows, found {len(result)}")
    result.to_csv(SOURCE_DATA_DIR / "fig3_source_data.csv", index=False)
    return result


def plot_figure3(data: pd.DataFrame) -> dict[str, object]:
    width_mm = 178
    height_mm = 116
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH),
        sharey=True,
    )
    fig.subplots_adjust(left=0.245, right=0.992, top=0.965, bottom=0.125, wspace=0.20, hspace=0.30)

    y = np.arange(len(data))
    group_breaks = (1.5, 7.5, 11.5)
    panels = ("(a)", "(b)", "(c)", "(d)")

    for panel_index, (ax, (target, target_label)) in enumerate(zip(axes.flat, TARGETS)):
        style_boxed_axis(ax, grid_axis="x")
        for boundary in group_breaks:
            ax.axhline(boundary, color=PALETTE["neutral_light"], linewidth=0.30, zorder=1)

        for yi, row in data.iterrows():
            color, marker = GROUP_STYLE[row["group"]]
            mean = float(row[f"{target}_mean"])
            std = row[f"{target}_std"]
            marker_size = 4.4 if row["group"].startswith("fcsr") else 3.5
            if pd.isna(std):
                ax.plot(
                    mean,
                    yi,
                    marker=marker,
                    markersize=marker_size,
                    markerfacecolor=color,
                    markeredgecolor=PALETTE["ink"],
                    markeredgewidth=0.30,
                    linestyle="none",
                    zorder=4,
                )
            else:
                ax.errorbar(
                    mean,
                    yi,
                    xerr=float(std),
                    fmt=marker,
                    markersize=marker_size,
                    markerfacecolor=color,
                    markeredgecolor=PALETTE["ink"],
                    markeredgewidth=0.30,
                    color=color,
                    ecolor=color,
                    elinewidth=0.43,
                    capsize=1.7,
                    capthick=0.37,
                    zorder=4,
                )

        means = data[f"{target}_mean"].to_numpy(float)
        stds = data[f"{target}_std"].fillna(0.0).to_numpy(float)
        xmax = float(np.max(means + stds)) * 1.07
        ax.set_xlim(0.0, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.set_ylim(len(data) - 0.45, -0.55)
        ax.set_xlabel(f"{target_label} MAE")
        ax.set_yticks(y)
        if panel_index % 2 == 0:
            ax.set_yticklabels(data["method"], fontsize=TICK_LABEL_SIZE)
        else:
            ax.tick_params(axis="y", labelleft=False)
        add_panel_label(ax, panels[panel_index], x=-0.16 if panel_index % 2 == 0 else -0.10)

    legend_items = [
        ("classic", "Classic"),
        ("same_task_sota", "2D/3D learned"),
        ("direct_mechanism", "Conformer/multimodal"),
        ("fcsr_fixed", "FCSR, fixed"),
        ("fcsr_nested", "FCSR, nested"),
    ]
    handles = []
    for key, label in legend_items:
        color, marker = GROUP_STYLE[key]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                markersize=4.3,
                markerfacecolor=color,
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                label=label,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.61, 0.012),
        ncol=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.05,
        borderaxespad=0.0,
    )
    return save_figure(fig, "Fig3_benchmark_performance")


def load_figure4_data() -> tuple[pd.DataFrame, dict[str, float]]:
    data = pd.read_csv(
        DATA_ROOT
        / "analysis"
        / "calibration_size.csv"
    )
    fixed = pd.read_csv(
        DATA_ROOT
        / "selection"
        / "fixed_5seed_summary.csv"
    )
    best = {
        "B5": float(fixed.loc[fixed["property"] == "B5", "best_literature_mae"].iloc[0]),
        "L": float(fixed.loc[fixed["property"] == "L", "best_literature_mae"].iloc[0]),
        "BurB5": float(
            fixed.loc[fixed["property"] == "BurB5-huber", "best_literature_mae"].iloc[0]
        ),
        "BurL": float(fixed.loc[fixed["property"] == "BurL", "best_literature_mae"].iloc[0]),
    }
    if len(data) != 36 or set(data["n_seeds"]) != {5}:
        raise ValueError("Figure 4 requires 36 five-seed summary rows")
    data.to_csv(SOURCE_DATA_DIR / "fig4_source_data.csv", index=False)
    return data, best


def simple_log_formatter(value: float, _position: int) -> str:
    if value >= 1:
        return f"{value:g}"
    if value >= 0.1:
        return f"{value:.1f}"
    if value >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot_figure4(data: pd.DataFrame, best: dict[str, float]) -> dict[str, object]:
    width_mm = 178
    height_mm = 94
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH),
        sharex=True,
    )
    fig.subplots_adjust(left=0.095, right=0.992, top=0.895, bottom=0.155, wspace=0.24, hspace=0.30)

    categories = ("raw_0", "5", "10", "25", "50", "100", "250", "500", "full")
    display = ("0", "5", "10", "25", "50", "100", "250", "500", "Full")
    # The full calibrator uses all 1,086 fitting molecules in the 70/10/20 protocol.
    x = np.asarray([0, 5, 10, 25, 50, 100, 250, 500, 1086], dtype=float)
    panels = ("(a)", "(b)", "(c)", "(d)")

    for panel_index, (ax, (target, target_label)) in enumerate(zip(axes.flat, TARGETS)):
        subset = data.loc[data["target_short"] == target].copy()
        subset["calibration_fit_molecules"] = subset["calibration_fit_molecules"].astype(str)
        subset = subset.set_index("calibration_fit_molecules").loc[list(categories)]
        mean = subset["mae_mean"].to_numpy(float)
        std = subset["mae_std"].to_numpy(float)
        lower = np.maximum(mean - std, np.finfo(float).tiny)
        upper = mean + std
        if np.any(lower <= 0) or np.any(mean <= 0) or best[target] <= 0:
            raise ValueError(f"Log-scale inputs for {target} must be strictly positive")
        color = TARGET_COLORS[target]

        style_boxed_axis(ax, grid_axis="y")
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=5.0, linscale=0.75, base=10.0)
        ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0.0, zorder=2)
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=0.77,
            marker="o",
            markersize=3.2,
            markerfacecolor=color,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.47,
            zorder=4,
        )
        ax.axhline(
            best[target],
            color=PALETTE["gold"],
            linewidth=0.53,
            linestyle=(0, (3.0, 2.2)),
            zorder=3,
        )
        ymin = min(float(np.min(lower)), best[target]) / 1.35
        ymax = max(float(np.max(upper)), best[target]) * 1.35
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-1.2, 1350)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
        ax.yaxis.set_major_formatter(FuncFormatter(simple_log_formatter))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(f"{target_label} MAE")
        ax.set_xticks(x)
        if panel_index >= 2:
            ax.set_xticklabels(display)
            ax.set_xlabel("Calibration labels")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        add_panel_label(ax, panels[panel_index], x=-0.13)

    handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["neutral_dark"],
            linewidth=0.77,
            marker="o",
            markersize=3.2,
            markerfacecolor=PALETTE["neutral_dark"],
            markeredgecolor=PALETTE["ink"],
            label="FCSR, mean ± SD",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["gold"],
            linewidth=0.53,
            linestyle=(0, (3.0, 2.2)),
            label="Best literature MAE",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.998),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    return save_figure(fig, "Fig4_calibration_efficiency")


def load_figure_s6_data() -> pd.DataFrame:
    data = pd.read_csv(
        DATA_ROOT / "supporting" / "conformer_count_5seed.csv"
    )
    if len(data) != 16:
        raise ValueError(f"Supplementary Fig. S6 requires 16 rows, found {len(data)}")
    data.to_csv(SOURCE_DATA_DIR / "FigS6_source_data.csv", index=False)
    return data


def plot_figure_s6(data: pd.DataFrame) -> dict[str, object]:
    width_mm = 178
    height_mm = 90
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH),
        sharex=True,
    )
    fig.subplots_adjust(left=0.095, right=0.992, top=0.955, bottom=0.155, wspace=0.24, hspace=0.30)

    categories = ("Q1_few_confs", "Q2", "Q3", "Q4_many_confs")
    display = ("Q1", "Q2", "Q3", "Q4")
    x = np.arange(len(categories))
    panels = ("(a)", "(b)", "(c)", "(d)")

    for panel_index, (ax, (target, target_label)) in enumerate(zip(axes.flat, TARGETS)):
        subset = data.loc[data["property"] == target].set_index("conformer_count_bucket")
        subset = subset.loc[list(categories)]
        mean = subset["mae_mean"].to_numpy(float)
        std = subset["mae_std"].to_numpy(float)
        color = TARGET_COLORS[target]

        style_boxed_axis(ax, grid_axis="y")
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=color,
            ecolor=color,
            linewidth=0.73,
            elinewidth=0.43,
            capsize=1.8,
            capthick=0.37,
            marker="o",
            markersize=3.3,
            markerfacecolor=color,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.47,
            zorder=4,
        )
        ymin = max(0.0, float(np.min(mean - std)) - 0.12 * float(np.ptp(mean + std)))
        ymax = float(np.max(mean + std)) + 0.14 * float(np.ptp(mean + std))
        if ymax <= ymin:
            ymax = ymin + 0.01
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.set_xlim(-0.22, len(categories) - 0.78)
        ax.set_ylabel(f"{target_label} MAE")
        ax.set_xticks(x)
        if panel_index >= 2:
            ax.set_xticklabels(display)
            ax.set_xlabel("Conformer-count quartile")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        add_panel_label(ax, panels[panel_index], x=-0.13)

    return save_figure(fig, "FigS6_conformer_count_longtail")


def main() -> None:
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    metadata = {
        "backend": "Python/matplotlib",
        "font": FONT_FAMILY,
        "palette_module": "palette.py",
        "data_exclusions": 0,
        "figures": {},
    }

    fig3_data = load_figure3_data()
    metadata["figures"]["Fig3"] = {
        **plot_figure3(fig3_data),
        "source_rows": len(fig3_data),
        "external_methods": 12,
        "fcsr_rows": 2,
    }

    fig4_data, best = load_figure4_data()
    metadata["figures"]["Fig4"] = {
        **plot_figure4(fig4_data, best),
        "source_rows": len(fig4_data),
        "seeds": 5,
    }

    (HERE / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
