#!/usr/bin/env python3
"""Render the final applicability, budget, validation and sensitivity figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from palette import (
    EXPORT_DPI,
    FONT_FAMILY,
    GRID_WIDTH,
    LEGEND_SIZE,
    OUTER_PAD_POINTS,
    PALETTE,
    PANEL_LABEL_SIZE,
    SPINE_WIDTH,
    TARGET_COLORS,
    TICK_LABEL_SIZE,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "reference_results"
SOURCE = HERE / "source_data"
MM_TO_INCH = 1.0 / 25.4
PT_TO_INCH = 1.0 / 72.0
fig_width_mm = 178.0
TARGETS = ("B5", "L", "BurB5", "BurL")
TARGET_LABELS = {
    "B5": r"$B_5$",
    "L": r"$L$",
    "BurB5": r"Buried $B_5$",
    "BurL": r"Buried $L$",
}
HEATMAP_LIMIT = 2.1


def configure_style() -> None:
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


def read_csv(*parts: str) -> pd.DataFrame:
    return pd.read_csv(DATA.joinpath(*parts))


def save_figure(fig: plt.Figure, stem: str) -> dict[str, object]:
    common = {
        "bbox_inches": "tight",
        "pad_inches": OUTER_PAD_POINTS * PT_TO_INCH,
        "facecolor": PALETTE["white"],
        "edgecolor": PALETTE["white"],
    }
    svg_path = HERE / f"{stem}.svg"
    pdf_path = HERE / f"{stem}.pdf"
    png_path = HERE / f"{stem}.png"
    fig.savefig(svg_path, **common)
    fig.savefig(pdf_path, **common)
    fig.savefig(png_path, dpi=600, **common)
    files = {"svg": svg_path.name, "pdf": pdf_path.name, "png": png_path.name}
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {
        "files": files,
        "design_width_mm": round(width / MM_TO_INCH, 2),
        "design_height_mm": round(height / MM_TO_INCH, 2),
        "png_dpi": EXPORT_DPI,
        "outer_padding_pt": OUTER_PAD_POINTS,
    }


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )


def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=GRID_WIDTH)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(SPINE_WIDTH)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=2.4,
        width=SPINE_WIDTH,
        color=PALETTE["ink"],
        labelcolor=PALETTE["ink"],
        pad=2.0,
    )


def ratio_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "fcsr_log2_ratio", [PALETTE["teal"], PALETTE["white"], PALETTE["wine"]]
    )


def validate_ratios(ratios: np.ndarray, name: str, enforce_range: bool = True) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=float)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError(f"{name} contains non-finite or non-positive MAE ratios")
    transformed = np.log2(ratios)
    if enforce_range and np.any(np.abs(transformed) > HEATMAP_LIMIT + 1e-8):
        raise ValueError(f"{name} exceeds the declared heatmap colour range")
    return transformed


def draw_ratio_heatmap(
    ax: plt.Axes,
    ratios: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    norm: Normalize,
    selected_cells: set[tuple[int, int]] | None = None,
    ytick_size: float = TICK_LABEL_SIZE,
) -> None:
    transformed = validate_ratios(ratios, "heatmap")
    ax.imshow(transformed, cmap=ratio_cmap(), norm=norm, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=ytick_size)
    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color=PALETTE["white"], linewidth=0.33)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=2.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(SPINE_WIDTH)
    for row in range(ratios.shape[0]):
        for column in range(ratios.shape[1]):
            ax.text(
                column,
                row,
                f"{ratios[row, column]:.2f}x",
                ha="center",
                va="center",
                fontsize=7.0,
                color=PALETTE["ink"],
            )
    for row, column in selected_cells or set():
        ax.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor=PALETTE["ink"],
                linewidth=0.70,
                zorder=5,
            )
        )


def add_ratio_colorbar(fig: plt.Figure, cax: plt.Axes, norm: Normalize) -> None:
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ratio_cmap()), cax=cax)
    ticks = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    colorbar.set_ticks(ticks, labels=["0.25x", "0.5x", "1x", "2x", "4x"])
    colorbar.set_label("MAE / reference MAE", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=8.0, length=2.2, width=SPINE_WIDTH)
    colorbar.outline.set_linewidth(SPINE_WIDTH)


def matrix_from_rows(
    data: pd.DataFrame, row_field: str, row_order: list[str], value_field: str
) -> np.ndarray:
    matrix = np.empty((len(row_order), len(TARGETS)), dtype=float)
    for i, row_value in enumerate(row_order):
        subset = data[data[row_field] == row_value].set_index("target_short")
        missing = set(TARGETS) - set(subset.index)
        if missing:
            raise ValueError(f"Missing targets for {row_field}={row_value}: {sorted(missing)}")
        matrix[i] = subset.loc[list(TARGETS), value_field].to_numpy(float)
    return matrix


def plot_main_figure5() -> dict[str, object]:
    compact = read_csv("analysis", "perturbation_5seed.csv")
    clean = compact[compact["perturbation_type"] == "clean"].set_index("target_short")["mae_mean"]
    reference = clean.loc[list(TARGETS)].to_numpy(float)

    family = read_csv("analysis", "chemical_families.csv")
    family_order = [
        "mixed_aryl_alkyl_ring",
        "non_tri_carbon_P",
        "non_tri_carbon_P_ring",
        "trialkyl",
        "trialkyl_ring",
        "triaryl_ring",
    ]
    family_labels = [
        "Mixed aryl/alkyl ring",
        "Non-tri-carbon P",
        "Non-tri-carbon P ring",
        "Trialkyl",
        "Trialkyl ring",
        "Triaryl ring",
    ]
    family_matrix = matrix_from_rows(family, "ligand_family", family_order, "mae_mean")

    strata = read_csv("supporting", "conformer_count_5seed.csv")
    strata = strata.rename(columns={"property": "target_short"})
    strata_order = ["Q1_few_confs", "Q2", "Q3", "Q4_many_confs"]
    strata_labels = ["Q1: few", "Q2", "Q3", "Q4: many"]
    strata_matrix = matrix_from_rows(strata, "conformer_count_bucket", strata_order, "mae_mean")

    random = pd.DataFrame({"target_short": TARGETS, "split": "Random", "mae_mean": reference})
    scaffold = read_csv("supporting", "scaffold_split_5seed.csv")
    scaffold = scaffold.rename(columns={"property": "target_short"})
    domain = pd.concat(
        [random, scaffold[["target_short", "mae_mean"]].assign(split="Scaffold")],
        ignore_index=True,
    )
    domain_order = ["Random", "Scaffold"]
    domain_matrix = matrix_from_rows(domain, "split", domain_order, "mae_mean")

    matrices = [family_matrix, strata_matrix, domain_matrix]
    row_orders = [family_order, strata_order, domain_order]
    panel_names = ["chemical_family", "conformer_count", "domain_shift"]
    ratios = [matrix / reference[np.newaxis, :] for matrix in matrices]
    for name, ratio in zip(panel_names, ratios):
        validate_ratios(ratio, name)

    source_rows = []
    for panel, row_order, matrix, ratio in zip(panel_names, row_orders, matrices, ratios):
        for i, stratum in enumerate(row_order):
            for j, target in enumerate(TARGETS):
                source_rows.append(
                    {
                        "panel": panel,
                        "stratum": stratum,
                        "target_short": target,
                        "mae_absolute": matrix[i, j],
                        "reference_mae": reference[j],
                        "mae_ratio": ratio[i, j],
                        "log2_mae_ratio": np.log2(ratio[i, j]),
                    }
                )
    pd.DataFrame(source_rows).to_csv(SOURCE / "fig5_applicability_domain_source_data.csv", index=False)

    ad = pd.read_csv(SOURCE / "fig5_ad_residual_distance_source_data.csv")
    expected_columns = {
        "seed",
        "target_short",
        "tanimoto_dissimilarity",
        "reference_loo_p95",
        "domain",
        "normalised_abs_error",
        "distance_decile",
    }
    if not expected_columns.issubset(ad.columns):
        raise ValueError("Applicability-domain source data are incomplete")
    distance_by_seed = (
        ad.groupby(["target_short", "seed", "distance_decile"], as_index=False)
        .agg(
            median_dissimilarity=("tanimoto_dissimilarity", "median"),
            mean_normalised_error=("normalised_abs_error", "mean"),
        )
    )
    distance_summary = (
        distance_by_seed.groupby(["target_short", "distance_decile"], as_index=False)
        .agg(
            median_dissimilarity=("median_dissimilarity", "mean"),
            mean_normalised_error=("mean_normalised_error", "mean"),
            sd_normalised_error=("mean_normalised_error", "std"),
        )
    )
    domain_seed = (
        ad.groupby(["target_short", "seed", "domain"])["abs_error"]
        .mean()
        .unstack("domain")
        .reset_index()
    )
    if not {"ID", "OOD"}.issubset(domain_seed.columns) or len(domain_seed) != 20:
        raise ValueError("Expected paired ID and OOD errors for five seeds and four targets")
    domain_seed["mae_ratio"] = domain_seed["OOD"] / domain_seed["ID"]
    domain_seed["log2_mae_ratio"] = np.log2(domain_seed["mae_ratio"])
    domain_seed.to_csv(SOURCE / "fig5_ad_domain_seed_ratios_source_data.csv", index=False)

    norm = Normalize(vmin=-HEATMAP_LIMIT, vmax=HEATMAP_LIMIT)
    fig = plt.figure(figsize=(178 * MM_TO_INCH, 136 * MM_TO_INCH))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.28, 1.0, 0.045],
        height_ratios=[1.0, 1.28],
        left=0.205,
        right=0.945,
        top=0.875,
        bottom=0.095,
        wspace=0.38,
        hspace=0.47,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    style_axis(ax_a, "y")
    for target in TARGETS:
        rows = distance_summary[distance_summary["target_short"] == target].sort_values(
            "distance_decile"
        )
        x = rows["median_dissimilarity"].to_numpy(float)
        mean = rows["mean_normalised_error"].to_numpy(float)
        sd = rows["sd_normalised_error"].to_numpy(float)
        color = TARGET_COLORS[target]
        ax_a.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        ax_a.plot(
            x,
            mean,
            color=color,
            linewidth=0.55,
            marker="o",
            markersize=3.0,
            markerfacecolor=color,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            zorder=3,
        )
    thresholds = ad.groupby("seed")["reference_loo_p95"].first().to_numpy(float)
    ax_a.axvspan(
        float(np.min(thresholds)),
        float(np.max(thresholds)),
        color=PALETTE["neutral_light"],
        alpha=0.55,
        linewidth=0,
        zorder=0,
    )
    ax_a.axvline(
        float(np.mean(thresholds)),
        color=PALETTE["neutral_dark"],
        linewidth=0.45,
        linestyle=(0, (3, 2)),
        zorder=2,
    )
    ax_a.axhline(1.0, color=PALETTE["neutral_mid"], linewidth=0.40, linestyle=(0, (3, 2)))
    ax_a.set_xlim(0.02, 0.72)
    ax_a.set_ylim(0.35, 2.02)
    ax_a.set_xlabel("Nearest-reference Morgan dissimilarity")
    ax_a.set_ylabel("Target-normalised absolute error")
    ax_a.xaxis.set_major_locator(MaxNLocator(5))
    ax_a.yaxis.set_major_locator(MaxNLocator(5))
    panel_label(ax_a, "(a)", x=-0.24)

    style_axis(ax_b, "y")
    x = np.asarray([0.0, 1.0])
    all_errors = []
    for target in TARGETS:
        rows = domain_seed[domain_seed["target_short"] == target].sort_values("seed")
        paired = rows[["ID", "OOD"]].to_numpy(float)
        all_errors.extend(paired.ravel())
        color = TARGET_COLORS[target]
        for seed_values in paired:
            ax_b.plot(
                x,
                seed_values,
                color=color,
                linewidth=0.32,
                marker="o",
                markersize=2.5,
                markerfacecolor=color,
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.25,
                alpha=0.42,
                zorder=2,
            )
        means = paired.mean(axis=0)
        standard_deviations = paired.std(axis=0, ddof=1)
        ax_b.plot(
            x,
            means,
            color=color,
            linewidth=0.62,
            marker="o",
            markersize=4.8,
            markerfacecolor=color,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            zorder=4,
        )
        ax_b.errorbar(
            x,
            means,
            yerr=standard_deviations,
            fmt="none",
            ecolor=color,
            elinewidth=0.42,
            capsize=1.8,
            capthick=0.32,
            zorder=3,
        )
    ax_b.set_xlim(-0.18, 1.18)
    ax_b.set_xticks(x, ["ID", "OOD"])
    ax_b.set_ylim(0.0, max(all_errors) * 1.13)
    ax_b.set_ylabel(r"MAE ($\AA$)")
    ax_b.yaxis.set_major_locator(MaxNLocator(5))
    panel_label(ax_b, "(b)", x=-0.18)

    column_labels = [r"$B_5$", r"$L$", "Buried\n" + r"$B_5$", "Buried\n" + r"$L$"]
    draw_ratio_heatmap(ax_c, ratios[0], family_labels, column_labels, norm, ytick_size=8.0)
    combined_ratio = np.vstack([ratios[1], ratios[2]])
    combined_labels = strata_labels + domain_order
    draw_ratio_heatmap(ax_d, combined_ratio, combined_labels, column_labels, norm, ytick_size=8.0)
    ax_d.axhline(3.5, color=PALETTE["ink"], linewidth=0.45)
    panel_label(ax_c, "(c)", x=-0.45)
    panel_label(ax_d, "(d)", x=-0.26)
    add_ratio_colorbar(fig, cax, norm)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TARGET_COLORS[target],
                marker="o",
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                linewidth=0.55,
                markersize=3.2,
                label=TARGET_LABELS[target],
            )
            for target in TARGETS
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    result = save_figure(fig, "Fig5_applicability_domain")
    result["source_rows"] = len(source_rows) + len(ad) + len(domain_seed)
    result["colour_transform"] = "log2(MAE/reference MAE)"
    return result


def plot_s3_budget() -> dict[str, object]:
    data = read_csv("supporting", "conformer_budget_5seed.csv")
    levels = ["L2", "L3_4", "L3_8", "L4_32"]
    budget_labels = ["1", "4", "8", "32"]
    source_rows = []
    ratios = np.empty((len(TARGETS), len(levels)), dtype=float)
    scaled_sd = np.empty_like(ratios)
    for i, target in enumerate(TARGETS):
        subset = data[data["property"] == target].set_index("level").loc[levels]
        baseline = float(subset.iloc[0]["mae_mean"])
        ratios[i] = subset["mae_mean"].to_numpy(float) / baseline
        scaled_sd[i] = subset["mae_std"].to_numpy(float) / baseline
        for j, level in enumerate(levels):
            row = subset.iloc[j]
            source_rows.append(
                {
                    "target_short": target,
                    "level": level,
                    "conformers_used": int(budget_labels[j]),
                    "mae_mean": row["mae_mean"],
                    "mae_std": row["mae_std"],
                    "reference_mae": baseline,
                    "mae_ratio": ratios[i, j],
                    "scaled_sd": scaled_sd[i, j],
                    "n_seeds": 5,
                }
            )
    validate_ratios(ratios, "conformer budget", enforce_range=False)
    pd.DataFrame(source_rows).to_csv(
        SOURCE / "figs3_conformer_budget_relative_source_data.csv", index=False
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(178 * MM_TO_INCH, 88 * MM_TO_INCH),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.105, right=0.992, top=0.93, bottom=0.17, wspace=0.18, hspace=0.28)
    x = np.asarray([1, 4, 8, 32], dtype=float)
    for index, (ax, target) in enumerate(zip(axes.flat, TARGETS)):
        style_axis(ax, "y")
        color = TARGET_COLORS[target]
        ax.errorbar(
            x,
            ratios[index],
            yerr=scaled_sd[index],
            fmt="o-",
            markersize=3.8,
            markerfacecolor=color,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            color=color,
            linewidth=0.55,
            ecolor=color,
            elinewidth=0.45,
            capsize=1.8,
            capthick=0.35,
            zorder=3,
        )
        ax.axhline(1.0, color=PALETTE["neutral_mid"], linewidth=0.40, linestyle=(0, (3, 2)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, budget_labels)
        ax.set_xlim(0.75, 42)
        ax.set_ylim(0.08, 1.15)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + index)})", x=0.0, y=1.035)
        if index >= 2:
            ax.set_xlabel("Conformers used")
    fig.text(
        0.020,
        0.52,
        "MAE / 1-conformer MAE",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.5,
        color=PALETTE["ink"],
    )
    result = save_figure(fig, "FigS3_conformer_budget")
    result["source_rows"] = len(source_rows)
    result["normalisation"] = "five-seed mean and SD divided by K=1 mean"
    return result


def fixed_seed_rows() -> pd.DataFrame:
    data = read_csv(
        "selection",
        "fixed_5seed_rows.csv",
    )
    out = data[data["property"].isin(["B5", "L", "BurB5-huber", "BurL"])].copy()
    out["target_short"] = out["property"].replace({"BurB5-huber": "BurB5"})
    if len(out) != 20 or not (out.groupby("target_short")["seed"].nunique() == 5).all():
        raise ValueError("Fixed protocol must contain five seeds for each target")
    return out


def scaffold_seed_rows() -> pd.DataFrame:
    data = read_csv("supporting", "scaffold_split_raw.csv")
    rules = {
        "B5": ("sterimol_B5", "sum_pc_Bmax_all", "huber"),
        "L": ("sterimol_L", "sum_pc_Lpos_all", "huber"),
        "BurB5": ("sterimol_burB5", "sum_pc_R4.4_touch_Bmax", "huber"),
        "BurL": ("sterimol_burL", "sum_pc_R2.7_touch_Lpos", "isotonic"),
    }
    selected = []
    for target, (full_target, candidate, calibration) in rules.items():
        rows = data[
            (data["target"] == full_target)
            & (data["candidate"] == candidate)
            & (data["calibration"] == calibration)
        ].copy()
        rows["target_short"] = target
        selected.append(rows)
    out = pd.concat(selected, ignore_index=True)
    if len(out) != 20 or not (out.groupby("target_short")["seed"].nunique() == 5).all():
        raise ValueError("Scaffold protocol must contain five frozen rows for each target")
    expected = read_csv(
        "supporting", "scaffold_split_5seed.csv"
    ).set_index("property")["mae_mean"].loc[list(TARGETS)]
    observed = out.groupby("target_short")["mae"].mean().loc[list(TARGETS)]
    if not np.allclose(observed.to_numpy(), expected.to_numpy(), atol=1e-12, rtol=0):
        raise ValueError("Scaffold seed rows do not reproduce the frozen summary")
    return out


def plot_s4_validation() -> dict[str, object]:
    fixed = fixed_seed_rows()
    nested = read_csv(
        "analysis", "nested_candidates_raw.csv"
    )
    scaffold = scaffold_seed_rows()
    nested_pairs = fixed[["target_short", "seed", "mae"]].merge(
        nested[["target_short", "seed", "mae"]],
        on=["target_short", "seed"],
        suffixes=("_denominator", "_numerator"),
        validate="one_to_one",
    )
    nested_pairs["comparison"] = "Nested / fixed"
    scaffold_pairs = fixed[["target_short", "seed", "mae"]].merge(
        scaffold[["target_short", "seed", "mae"]],
        on=["target_short", "seed"],
        suffixes=("_denominator", "_numerator"),
        validate="one_to_one",
    )
    scaffold_pairs["comparison"] = "Scaffold / random"
    source = pd.concat([nested_pairs, scaffold_pairs], ignore_index=True)
    source["mae_ratio"] = source["mae_numerator"] / source["mae_denominator"]
    source["log2_mae_ratio"] = np.log2(source["mae_ratio"])
    source["n_seeds"] = 5
    validate_ratios(source["mae_ratio"].to_numpy(), "validation robustness")
    source.to_csv(SOURCE / "figs4_validation_robustness_source_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(178 * MM_TO_INCH, 76 * MM_TO_INCH), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.992, top=0.83, bottom=0.18, wspace=0.16)
    comparisons = [
        ("Nested / fixed", ["Fixed", "Nested"]),
        ("Scaffold / random", ["Random", "Scaffold"]),
    ]
    x = np.asarray([0.0, 1.0])
    for panel_index, (ax, (comparison, tick_labels)) in enumerate(zip(axes, comparisons)):
        style_axis(ax, "y")
        panel_rows = source[source["comparison"] == comparison]
        for target in TARGETS:
            rows = panel_rows[panel_rows["target_short"] == target].sort_values("seed")
            paired = rows[["mae_denominator", "mae_numerator"]].to_numpy(float)
            color = TARGET_COLORS[target]
            for seed_values in paired:
                ax.plot(
                    x,
                    seed_values,
                    color=color,
                    linewidth=0.32,
                    marker="o",
                    markersize=2.5,
                    markerfacecolor=color,
                    markeredgecolor=PALETTE["ink"],
                    markeredgewidth=0.25,
                    alpha=0.42,
                    zorder=2,
                )
            means = paired.mean(axis=0)
            standard_deviations = paired.std(axis=0, ddof=1)
            ax.plot(
                x,
                means,
                color=color,
                linewidth=0.62,
                marker="o",
                markersize=4.8,
                markerfacecolor=color,
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                zorder=4,
            )
            ax.errorbar(
                x,
                means,
                yerr=standard_deviations,
                fmt="none",
                ecolor=color,
                elinewidth=0.42,
                capsize=1.8,
                capthick=0.32,
                zorder=3,
            )
        ax.set_xlim(-0.16, 1.16)
        ax.set_xticks(x, tick_labels)
        ax.set_ylim(0.0, 0.132)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + panel_index)})", x=-0.13 if panel_index == 0 else -0.09)
    axes[0].set_ylabel(r"MAE ($\AA$)")
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                linewidth=0.55,
                markersize=3.2,
                color=TARGET_COLORS[target],
                label=TARGET_LABELS[target],
            )
            for target in TARGETS
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.99),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
    )
    result = save_figure(fig, "FigS4_validation_robustness")
    result["source_rows"] = len(source)
    result["seed_paths"] = 40
    return result


def perturbation_matrix(
    compact: pd.DataFrame,
    kind: str,
    values: list[float],
    reference: pd.Series,
) -> np.ndarray:
    rows = compact[compact["perturbation_type"] == kind].copy()
    rows["magnitude_num"] = pd.to_numeric(rows["magnitude"])
    matrix = np.empty((len(TARGETS), len(values)), dtype=float)
    for i, target in enumerate(TARGETS):
        subset = rows[rows["target_short"] == target].set_index("magnitude_num")
        for j, value in enumerate(values):
            matrix[i, j] = reference[target] if value == 0.0 and kind != "temperature_K" else subset.loc[value, "mae_mean"]
    return matrix


def plot_s5_sensitivity() -> dict[str, object]:
    compact = read_csv("analysis", "perturbation_5seed.csv")
    clean = compact[compact["perturbation_type"] == "clean"].set_index("target_short")
    reference = clean["mae_mean"]
    geometry_values = [0.0, 0.01, 0.03, 0.05, 0.10]
    energy_values = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
    temperature_values = [250.0, 273.15, 298.15, 323.15, 350.0, 400.0]
    radius = read_csv("analysis", "radius_sensitivity.csv")
    radius_specs = [
        ("BurB5", [3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0], 4.4),
        ("BurL", [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2], 2.7),
    ]
    source_rows = []
    perturbation_specs = [
        ("coordinate", "geometry_noise_A_sd", geometry_values, 0.0),
        ("energy", "energy_noise_kcal_mol_sd", energy_values, 0.0),
        ("temperature", "temperature_K", temperature_values, 298.15),
    ]
    for panel, kind, values, selected in perturbation_specs:
        rows = compact[compact["perturbation_type"] == kind].copy()
        rows["condition_num"] = pd.to_numeric(rows["magnitude"])
        for target in TARGETS:
            target_rows = rows[rows["target_short"] == target].set_index("condition_num")
            for condition in values:
                if condition == 0.0 and panel != "temperature":
                    mae_mean = float(clean.loc[target, "mae_mean"])
                    mae_std = float(clean.loc[target, "mae_std"])
                else:
                    mae_mean = float(target_rows.loc[condition, "mae_mean"])
                    mae_std = float(target_rows.loc[condition, "mae_std"])
                ratio = mae_mean / float(reference[target])
                scaled_sd = mae_std / float(reference[target])
                source_rows.append(
                    {
                        "panel": panel,
                        "target_short": target,
                        "condition": condition,
                        "mae_absolute": mae_mean,
                        "mae_std": mae_std,
                        "reference_condition": selected,
                        "reference_mae": reference[target],
                        "mae_ratio": ratio,
                        "scaled_sd": scaled_sd,
                        "log2_mae_ratio": np.log2(ratio),
                        "selected": bool(np.isclose(condition, selected)),
                    }
                )
    for target, values, selected in radius_specs:
        subset = radius[radius["target_short"] == target].set_index("radius").loc[values]
        selected_mae = float(subset.loc[selected, "mae_mean"])
        for condition in values:
            mae_mean = float(subset.loc[condition, "mae_mean"])
            mae_std = float(subset.loc[condition, "mae_std"])
            ratio = mae_mean / selected_mae
            source_rows.append(
                {
                    "panel": "radius",
                    "target_short": target,
                    "condition": condition,
                    "mae_absolute": mae_mean,
                    "mae_std": mae_std,
                    "reference_condition": selected,
                    "reference_mae": selected_mae,
                    "mae_ratio": ratio,
                    "scaled_sd": mae_std / selected_mae,
                    "log2_mae_ratio": np.log2(ratio),
                    "selected": bool(np.isclose(condition, selected)),
                }
            )
    source = pd.DataFrame(source_rows)
    validate_ratios(source["mae_ratio"].to_numpy(float), "sensitivity", enforce_range=False)
    source.to_csv(SOURCE / "figs5_sensitivity_atlas_source_data.csv", index=False)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(178 * MM_TO_INCH, 103 * MM_TO_INCH),
        sharey=True,
    )
    fig.subplots_adjust(left=0.095, right=0.992, top=0.94, bottom=0.14, wspace=0.22, hspace=0.43)
    plot_specs = [
        (axes[0, 0], "coordinate", r"Coordinate noise SD ($\mathrm{\AA}$)"),
        (axes[0, 1], "energy", r"Energy noise SD (kcal mol$^{-1}$)"),
        (axes[0, 2], "temperature", "Temperature (K)"),
    ]
    for panel_index, (ax, panel, xlabel) in enumerate(plot_specs):
        style_axis(ax, "y")
        for target in TARGETS:
            rows = source[(source["panel"] == panel) & (source["target_short"] == target)].sort_values(
                "condition"
            )
            color = TARGET_COLORS[target]
            ax.errorbar(
                rows["condition"],
                rows["mae_ratio"],
                yerr=rows["scaled_sd"],
                fmt="o-",
                color=color,
                ecolor=color,
                linewidth=0.50,
                elinewidth=0.40,
                capsize=1.5,
                capthick=0.35,
                markersize=3.1,
                markerfacecolor=color,
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                zorder=3,
            )
        selected = float(source[source["panel"] == panel]["reference_condition"].iloc[0])
        ax.axvline(selected, color=PALETTE["neutral_dark"], linewidth=0.40, linestyle=(0, (3, 2)))
        ax.axhline(1.0, color=PALETTE["neutral_mid"], linewidth=0.40, linestyle=(0, (3, 2)))
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + panel_index)})", x=0.018, y=0.91)

    for offset, (ax, (target, values, selected)) in enumerate(zip(axes[1, :2], radius_specs), start=3):
        style_axis(ax, "y")
        rows = source[(source["panel"] == "radius") & (source["target_short"] == target)].sort_values(
            "condition"
        )
        color = TARGET_COLORS[target]
        ax.errorbar(
            rows["condition"],
            rows["mae_ratio"],
            yerr=rows["scaled_sd"],
            fmt="o-",
            color=color,
            ecolor=color,
            linewidth=0.50,
            elinewidth=0.40,
            capsize=1.5,
            capthick=0.35,
            markersize=3.1,
            markerfacecolor=color,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            zorder=3,
        )
        ax.axvline(selected, color=PALETTE["neutral_dark"], linewidth=0.40, linestyle=(0, (3, 2)))
        ax.axhline(1.0, color=PALETTE["neutral_mid"], linewidth=0.40, linestyle=(0, (3, 2)))
        ax.set_xticks(values[::2] if target == "BurL" else values)
        ax.set_xlabel(r"Buried descriptor radius ($\mathrm{\AA}$)")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + offset)})", x=0.018, y=0.91)

    for ax in axes.flat[:5]:
        ax.set_ylim(0.45, 4.10)
    fig.text(
        0.020,
        0.54,
        "MAE / reference MAE",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.5,
        color=PALETTE["ink"],
    )
    axes[1, 2].axis("off")
    axes[1, 2].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=TARGET_COLORS[target],
                marker="o",
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                linewidth=0.50,
                markersize=3.5,
                label=TARGET_LABELS[target],
            )
            for target in TARGETS
        ],
        loc="center",
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.1,
    )
    result = save_figure(fig, "FigS5_sensitivity_atlas")
    result["source_rows"] = len(source_rows)
    result["colour_transform"] = "log2(MAE/reference MAE)"
    return result


def main() -> None:
    configure_style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Fig5_applicability_domain": plot_main_figure5(),
        "FigS3_conformer_budget": plot_s3_budget(),
        "FigS4_validation_robustness": plot_s4_validation(),
        "FigS5_sensitivity_atlas": plot_s5_sensitivity(),
    }
    output = HERE / "reorganized_figure_metadata.json"
    output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
