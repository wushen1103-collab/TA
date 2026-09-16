#!/usr/bin/env python3
"""Render the remaining quantitative main-text and Supplementary figures.

All panels are generated from frozen CSV tables or the original Kraken archive.
The script intentionally contains no manually entered experimental values.
"""

from __future__ import annotations

import json
import math
import pickle
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
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
KRAKEN_ZIP = ROOT / "data" / "Kraken.zip"
MM_TO_INCH = 1.0 / 25.4
PT_TO_INCH = 1.0 / 72.0
fig_width_mm = 178.0

TARGETS = (
    ("B5", r"$B_5$"),
    ("L", r"$L$"),
    ("BurB5", r"Buried $B_5$"),
    ("BurL", r"Buried $L$"),
)
TARGET_NAME = dict(TARGETS)

FRAME_LABELS = {
    "sum_pc": r"Functional axis",
    "pc0": "P\u2013substituent direction 1",
    "pc1": "P\u2013substituent direction 2",
    "pc2": "P\u2013substituent direction 3",
    "pca0": r"Principal component 1",
    "pca1": r"Principal component 2",
    "pca2": r"Principal component 3",
    "lone_pair": r"Sign-reversed functional axis",
    "global_x": r"Global $x$-axis",
}
FRAME_COLORS = {
    "sum_pc": PALETTE["wine"],
    "pc0": PALETTE["blue"],
    "pc1": PALETTE["blue"],
    "pc2": PALETTE["blue"],
    "pca0": PALETTE["gold"],
    "pca1": PALETTE["gold"],
    "pca2": PALETTE["gold"],
    "lone_pair": PALETTE["teal"],
    "global_x": PALETTE["neutral_mid"],
}


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


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.035) -> None:
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


def save_figure(fig: plt.Figure, stem: str) -> dict[str, object]:
    pad = OUTER_PAD_POINTS * PT_TO_INCH
    common = {
        "bbox_inches": "tight",
        "pad_inches": pad,
        "facecolor": PALETTE["white"],
        "edgecolor": PALETTE["white"],
    }
    pdf_path = HERE / f"{stem}.pdf"
    svg_path = HERE / f"{stem}.svg"
    fig.savefig(pdf_path, **common)
    fig.savefig(svg_path, **common)
    png_path = HERE / f"{stem}.png"
    fig.savefig(png_path, dpi=600, **common)
    paths = {"pdf": pdf_path.name, "svg": svg_path.name, "png": png_path.name}
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {
        "files": paths,
        "design_width_mm": round(width / MM_TO_INCH, 2),
        "design_height_mm": round(height / MM_TO_INCH, 2),
        "png_dpi": EXPORT_DPI,
        "outer_padding_pt": OUTER_PAD_POINTS,
    }


def read_csv(*parts: str) -> pd.DataFrame:
    return pd.read_csv(DATA.joinpath(*parts))


def frame_figure(stem: str, include_reversed: bool, height_mm: float) -> dict[str, object]:
    data = read_csv(
        "analysis",
        "frame_controls_5seed.csv",
    )
    order = ["sum_pc", "pc0", "pc1", "pc2", "pca0", "pca1", "pca2"]
    if include_reversed:
        order.append("lone_pair")
    order.append("global_x")
    out = data[data["axis"].isin(order)].copy()
    if np.any(out["mae_mean"].to_numpy(float) <= 0):
        raise ValueError("Logarithmic MAE axes require strictly positive values")
    out["axis"] = pd.Categorical(out["axis"], categories=order, ordered=True)
    out = out.sort_values(["target", "axis"])
    out.to_csv(SOURCE / f"{stem.lower()}_source_data.csv", index=False)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(178 * MM_TO_INCH, height_mm * MM_TO_INCH),
        sharey=True,
    )
    fig.subplots_adjust(left=0.245, right=0.992, top=0.955, bottom=0.18, wspace=0.18)
    target_map = (("sterimol_B5", r"$B_5$ MAE"), ("sterimol_L", r"$L$ MAE"))
    xmin = 0.02
    for idx, (ax, (target, xlabel)) in enumerate(zip(axes, target_map)):
        style_axis(ax, "x")
        rows = out[out["target"] == target].set_index("axis").loc[order].reset_index()
        y = np.arange(len(rows))
        for yi, row in rows.iterrows():
            mean = float(row["mae_mean"])
            color = FRAME_COLORS[str(row["axis"])]
            ax.errorbar(
                mean,
                yi,
                xerr=float(row["mae_std"]),
                fmt="o",
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.30,
                color=color,
                ecolor=color,
                elinewidth=0.50,
                capsize=2.0,
                capthick=0.35,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_xlim(xmin, 1.15)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_xlabel(xlabel)
        ax.set_yticks(y)
        ax.set_ylim(len(rows) - 0.35, -0.65)
        if idx == 0:
            ax.set_yticklabels([FRAME_LABELS[str(x)] for x in rows["axis"]])
        else:
            ax.tick_params(axis="y", labelleft=False)
        panel_label(ax, f"({chr(97 + idx)})", x=-0.18 if idx == 0 else -0.10)
    return save_figure(fig, stem)


def plot_complete_frame_heatmap() -> dict[str, object]:
    data = read_csv(
        "analysis",
        "frame_controls_5seed.csv",
    )
    order = [
        "sum_pc",
        "pc0",
        "pc1",
        "pc2",
        "pca0",
        "pca1",
        "pca2",
        "lone_pair",
        "global_x",
    ]
    targets = ["sterimol_B5", "sterimol_L"]
    out = data[data["axis"].isin(order) & data["target"].isin(targets)].copy()
    expected = {(axis, target) for axis in order for target in targets}
    observed = set(zip(out["axis"], out["target"]))
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError(f"Complete frame heatmap is missing rows: {missing}")

    references = (
        out[out["axis"] == "sum_pc"].set_index("target")["mae_mean"].loc[targets]
    )
    out["reference_mae"] = out["target"].map(references)
    out["mae_ratio"] = out["mae_mean"] / out["reference_mae"]
    out["log2_mae_ratio"] = np.log2(out["mae_ratio"])
    if not np.all(np.isfinite(out["log2_mae_ratio"])) or np.any(out["mae_ratio"] < 1 - 1e-10):
        raise ValueError("Frame-ablation degradation ratios must be finite and at least one")
    out["axis"] = pd.Categorical(out["axis"], categories=order, ordered=True)
    out = out.sort_values(["axis", "target"])
    out.to_csv(SOURCE / "figs2_complete_frame_ablation_source_data.csv", index=False)

    matrix = np.empty((len(order), len(targets)), dtype=float)
    for row_index, axis in enumerate(order):
        rows = out[out["axis"] == axis].set_index("target").loc[targets]
        matrix[row_index] = rows["mae_ratio"].to_numpy(float)
    transformed = np.log2(matrix)
    colour_limit = 5.0
    cmap = LinearSegmentedColormap.from_list(
        "frame_degradation",
        [PALETTE["white"], PALETTE["gold"], PALETTE["wine"]],
    )
    norm = Normalize(vmin=0.0, vmax=colour_limit)

    fig = plt.figure(figsize=(126 * MM_TO_INCH, 98 * MM_TO_INCH))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.055],
        left=0.43,
        right=0.91,
        top=0.965,
        bottom=0.11,
        wspace=0.22,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax.imshow(transformed, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    ax.set_xticks([0, 1], [r"$B_5$", r"$L$"])
    ax.set_yticks(np.arange(len(order)), [FRAME_LABELS[axis] for axis in order])
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=PALETTE["white"], linewidth=0.33)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=2.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(SPINE_WIDTH)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ratio = matrix[row, column]
            label = f"{ratio:.1f}x" if ratio >= 10 else f"{ratio:.2f}x"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.3,
                color=PALETTE["ink"],
            )

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = np.arange(0.0, colour_limit + 0.1, 1.0)
    colorbar.set_ticks(ticks, labels=[f"{int(2**tick)}x" for tick in ticks])
    colorbar.set_label("MAE ratio", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=8.0, length=2.2, width=SPINE_WIDTH)
    colorbar.outline.set_linewidth(SPINE_WIDTH)

    result = save_figure(fig, "FigS2_complete_frame_ablation")
    result["source_rows"] = len(out)
    result["colour_transform"] = "log2(MAE/functional-axis MAE)"
    return result


def plot_conformer_budget() -> dict[str, object]:
    data = read_csv(
        "supporting",
        "conformer_budget_5seed.csv",
    )
    data.to_csv(SOURCE / "figs3_source_data.csv", index=False)
    levels = ["L2", "L3_4", "L3_8", "L4_32"]
    x = np.array([1, 4, 8, 32], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(178 * MM_TO_INCH, 102 * MM_TO_INCH))
    fig.subplots_adjust(left=0.105, right=0.992, top=0.965, bottom=0.14, wspace=0.27, hspace=0.34)
    for idx, (ax, (target, label)) in enumerate(zip(axes.flat, TARGETS)):
        style_axis(ax, "y")
        rows = data[data["property"] == target].set_index("level").loc[levels]
        y = rows["mae_mean"].to_numpy(float)
        err = rows["mae_std"].to_numpy(float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=TARGET_COLORS[target],
            marker="o",
            markersize=3.8,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            linewidth=0.60,
            elinewidth=0.37,
            capsize=1.7,
            capthick=0.30,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, ["1", "4", "8", "32"])
        ax.set_xlabel("Conformers used")
        ax.set_ylabel(f"{label} MAE")
        ax.set_xlim(0.8, 39)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + idx)})")
    return save_figure(fig, "FigS3_conformer_budget")


def load_fixed_seed_rows() -> pd.DataFrame:
    data = read_csv(
        "selection",
        "fixed_5seed_rows.csv",
    )
    properties = ["B5", "L", "BurB5-huber", "BurL"]
    out = data[data["property"].isin(properties)].copy()
    out["target_short"] = out["property"].replace({"BurB5-huber": "BurB5"})
    return out


def plot_cleanroom() -> dict[str, object]:
    fixed = load_fixed_seed_rows()
    nested = read_csv(
        "analysis",
        "nested_candidates_raw.csv",
    )
    fixed[["target_short", "seed", "mae", "candidate", "calibration"]].to_csv(
        SOURCE / "figs4_fixed_source_data.csv", index=False
    )
    nested.to_csv(SOURCE / "figs4_nested_source_data.csv", index=False)

    fig = plt.figure(figsize=(178 * MM_TO_INCH, 126 * MM_TO_INCH))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.48, 1.0, 1.0],
        left=0.10,
        right=0.992,
        top=0.96,
        bottom=0.11,
        wspace=0.27,
        hspace=0.48,
    )
    flow = fig.add_subplot(gs[0, :])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    stages = [
        (0.03, "Training set", "fit candidates"),
        (0.285, "Calibration set", "select once"),
        (0.54, "Frozen pipeline", "lock choices"),
        (0.795, "Test set", "evaluate once"),
    ]
    box_w, box_h = 0.175, 0.48
    for i, (x0, line1, line2) in enumerate(stages):
        face = PALETTE["band"] if i < 3 else PALETTE["white"]
        flow.add_patch(
            Rectangle(
                (x0, 0.24),
                box_w,
                box_h,
                facecolor=face,
                edgecolor=PALETTE["ink"],
                linewidth=0.30,
            )
        )
        flow.text(x0 + box_w / 2, 0.53, line1, ha="center", va="center", fontsize=9.0)
        flow.text(x0 + box_w / 2, 0.38, line2, ha="center", va="center", fontsize=8.0, color=PALETTE["neutral_dark"])
        if i < len(stages) - 1:
            flow.add_patch(
                FancyArrowPatch(
                    (x0 + box_w + 0.012, 0.48),
                    (stages[i + 1][0] - 0.012, 0.48),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    linewidth=0.43,
                    color=PALETTE["neutral_dark"],
                )
            )
    panel_label(flow, "(a)", x=-0.01, y=0.98)

    for idx, (target, label) in enumerate(TARGETS):
        ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])
        style_axis(ax, "y")
        f = fixed[fixed["target_short"] == target].set_index("seed").sort_index()
        n = nested[nested["target_short"] == target].set_index("seed").sort_index()
        seeds = sorted(set(f.index) & set(n.index))
        for seed in seeds:
            y0, y1 = float(f.loc[seed, "mae"]), float(n.loc[seed, "mae"])
            ax.plot([0, 1], [y0, y1], color=PALETTE["neutral_light"], linewidth=0.43, zorder=1)
            ax.plot(0, y0, "o", color=PALETTE["blue"], markeredgecolor=PALETTE["ink"], markeredgewidth=0.30, markersize=3.5, zorder=3)
            ax.plot(1, y1, "o", color=PALETTE["wine"], markeredgecolor=PALETTE["ink"], markeredgewidth=0.30, markersize=3.5, zorder=3)
        ax.set_xticks([0, 1], ["Fixed", "Nested"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(f"{label} MAE")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(98 + idx)})", x=-0.14)
    return save_figure(fig, "FigS4_cleanroom_protocol")


def plot_scaffold() -> dict[str, object]:
    random = read_csv(
        "selection",
        "fixed_5seed_summary.csv",
    )
    random = random[random["property"].isin(["B5", "L", "BurB5-huber", "BurL"])].copy()
    random["target_short"] = random["property"].replace({"BurB5-huber": "BurB5"})
    scaffold = read_csv("supporting", "scaffold_split_5seed.csv")
    source = pd.concat(
        [
            random.assign(split="Random")[["target_short", "split", "mae_mean", "mae_std", "n_seeds"]],
            scaffold.assign(target_short=scaffold["property"], split="Scaffold")[["target_short", "split", "mae_mean", "mae_std"]].assign(n_seeds=5),
        ],
        ignore_index=True,
    )
    source.to_csv(SOURCE / "figs5_source_data.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(178 * MM_TO_INCH, 92 * MM_TO_INCH))
    fig.subplots_adjust(left=0.09, right=0.992, top=0.95, bottom=0.15, wspace=0.30, hspace=0.48)
    for idx, (ax, (target, label)) in enumerate(zip(axes.flat, TARGETS)):
        style_axis(ax, "y")
        rows = source[source["target_short"] == target].set_index("split").loc[["Random", "Scaffold"]]
        x = np.arange(2)
        ax.errorbar(
            x,
            rows["mae_mean"],
            yerr=rows["mae_std"],
            fmt="o",
            color=TARGET_COLORS[target],
            markerfacecolor=TARGET_COLORS[target],
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            markersize=4.2,
            elinewidth=0.40,
            capsize=1.8,
            capthick=0.33,
        )
        ax.set_xticks(x, ["Random", "Scaffold"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(f"{label} MAE")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + idx)})", x=-0.13)
    return save_figure(fig, "FigS5_scaffold_ood")


def plot_radius() -> dict[str, object]:
    data = read_csv("analysis", "radius_sensitivity.csv")
    data.to_csv(SOURCE / "figs7_source_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(178 * MM_TO_INCH, 65 * MM_TO_INCH))
    fig.subplots_adjust(left=0.095, right=0.992, top=0.94, bottom=0.21, wspace=0.27)
    definitions = (("BurB5", 4.4), ("BurL", 2.7))
    for idx, (ax, (target, selected)) in enumerate(zip(axes, definitions)):
        style_axis(ax, "y")
        rows = data[data["target_short"] == target].sort_values("radius")
        ax.errorbar(
            rows["radius"],
            rows["mae_mean"],
            yerr=rows["mae_std"],
            color=TARGET_COLORS[target],
            marker="o",
            markersize=3.7,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            linewidth=0.60,
            elinewidth=0.37,
            capsize=1.6,
            capthick=0.30,
        )
        chosen = rows[np.isclose(rows["radius"], selected)].iloc[0]
        ax.plot(selected, chosen["mae_mean"], "o", color=PALETTE["wine"], markeredgecolor=PALETTE["ink"], markeredgewidth=0.30, markersize=4.3, zorder=4)
        ax.set_xlabel(r"Buried radius ($\AA$)")
        ax.set_ylabel(f"{TARGET_NAME[target]} MAE")
        ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + idx)})")
    return save_figure(fig, "FigS7_radius_sensitivity")


def perturbation_rows(kind: str) -> pd.DataFrame:
    compact = read_csv(
        "analysis", "perturbation_5seed.csv"
    )
    clean = compact[compact["perturbation_type"] == "clean"].copy()
    rows = compact[compact["perturbation_type"] == kind].copy()
    if kind in {"geometry_noise_A_sd", "energy_noise_kcal_mol_sd"}:
        clean["magnitude"] = 0.0
        rows = pd.concat([clean, rows], ignore_index=True)
    rows["magnitude_num"] = pd.to_numeric(rows["magnitude"], errors="coerce")
    return rows.sort_values(["target_short", "magnitude_num"])


def plot_perturbation(kind: str, stem: str, xlabel: str) -> dict[str, object]:
    data = perturbation_rows(kind)
    data.to_csv(SOURCE / f"{stem.lower()}_source_data.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(178 * MM_TO_INCH, 102 * MM_TO_INCH))
    fig.subplots_adjust(left=0.105, right=0.992, top=0.965, bottom=0.14, wspace=0.27, hspace=0.34)
    for idx, (ax, (target, label)) in enumerate(zip(axes.flat, TARGETS)):
        style_axis(ax, "y")
        rows = data[data["target_short"] == target].sort_values("magnitude_num")
        ax.errorbar(
            rows["magnitude_num"],
            rows["mae_mean"],
            yerr=rows["mae_std"],
            color=TARGET_COLORS[target],
            marker="o",
            markersize=3.7,
            markeredgecolor=PALETTE["ink"],
            markeredgewidth=0.30,
            linewidth=0.60,
            elinewidth=0.37,
            capsize=1.6,
            capthick=0.30,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{label} MAE")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        panel_label(ax, f"({chr(97 + idx)})")
    return save_figure(fig, stem)


def load_distribution_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(KRAKEN_ZIP) as archive:
        raw = pickle.loads(archive.read("Kraken.pickle"))
    molecule_rows = []
    conformer_rows = []
    gas_constant = 0.00198720425864083
    temperature = 298.15
    for molecule_id, record in raw.items():
        smiles, targets, conformers = record
        weights = np.asarray([float(item[1]) for item in conformers.values()], dtype=float)
        weights = weights / weights.sum()
        effective_n = 1.0 / np.square(weights).sum()
        molecule_rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "n_conformers": len(conformers),
                "effective_n_conformers": effective_n,
                "B5": targets["sterimol_B5"],
                "L": targets["sterimol_L"],
                "BurB5": targets["sterimol_burB5"],
                "BurL": targets["sterimol_burL"],
            }
        )
        max_weight = weights.max()
        for (conformer_id, conformer), weight in zip(conformers.items(), weights):
            relative_energy = -gas_constant * temperature * math.log(max(weight / max_weight, 1e-300))
            conformer_rows.append(
                {
                    "molecule_id": molecule_id,
                    "conformer_id": conformer_id,
                    "boltzmann_weight": weight,
                    "relative_energy_kcal_mol": relative_energy,
                }
            )
    molecules = pd.DataFrame(molecule_rows).sort_values("molecule_id")
    conformer_data = pd.DataFrame(conformer_rows).sort_values(["molecule_id", "conformer_id"])
    molecules.to_csv(SOURCE / "figs1_molecule_source_data.csv", index=False)
    conformer_data.to_csv(SOURCE / "figs1_conformer_source_data.csv", index=False)
    return molecules, conformer_data


def plot_distributions() -> dict[str, object]:
    molecules, conformers = load_distribution_data()
    fig, axes = plt.subplots(2, 3, figsize=(178 * MM_TO_INCH, 101 * MM_TO_INCH))
    fig.subplots_adjust(left=0.09, right=0.992, top=0.965, bottom=0.14, wspace=0.30, hspace=0.34)
    definitions = [
        ("B5", r"$B_5$ ($\AA$)"),
        ("L", r"$L$ ($\AA$)"),
        ("BurB5", r"Buried $B_5$ ($\AA$)"),
        ("BurL", r"Buried $L$ ($\AA$)"),
    ]
    for idx, (column, xlabel) in enumerate(definitions):
        ax = axes.flat[idx]
        style_axis(ax, "y")
        ax.hist(
            molecules[column],
            bins=34,
            color=TARGET_COLORS[column],
            edgecolor=PALETTE["ink"],
            linewidth=0.30,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Ligands")
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        panel_label(ax, f"({chr(97 + idx)})")

    ax = axes.flat[4]
    style_axis(ax, "y")
    bins = np.arange(0.5, molecules["n_conformers"].max() + 1.5, 3)
    ax.hist(
        molecules["n_conformers"],
        bins=bins,
        color=PALETTE["neutral_mid"],
        edgecolor=PALETTE["ink"],
        linewidth=0.30,
    )
    ax.set_xlabel("Conformers per ligand")
    ax.set_ylabel("Ligands")
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    panel_label(ax, "(e)")

    ax = axes.flat[5]
    style_axis(ax, "y")
    ax.hist(
        conformers["relative_energy_kcal_mol"],
        bins=46,
        color=PALETTE["neutral_dark"],
        edgecolor=PALETTE["ink"],
        linewidth=0.30,
    )
    ax.set_xlabel(r"Relative energy (kcal mol$^{-1}$)")
    ax.set_ylabel("Conformers")
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    panel_label(ax, "(f)")
    return save_figure(fig, "FigS1_dataset_distributions")


def robustness_matrices() -> tuple[list[np.ndarray], list[list[str]], list[str]]:
    compact = read_csv(
        "analysis", "perturbation_5seed.csv"
    )
    clean = compact[compact["perturbation_type"] == "clean"].set_index("target_short")["mae_mean"]
    family = read_csv(
        "analysis", "chemical_families.csv"
    )
    target_order = [x[0] for x in TARGETS]
    matrices: list[np.ndarray] = []
    labels: list[list[str]] = []

    for kind, values, display in (
        ("geometry_noise_A_sd", [0.0, 0.01, 0.03, 0.05, 0.10], ["0", ".01", ".03", ".05", ".10"]),
        ("energy_noise_kcal_mol_sd", [0.0, 0.05, 0.10, 0.25, 0.50, 1.00], ["0", ".05", ".10", ".25", ".50", "1.0"]),
    ):
        rows = compact[compact["perturbation_type"] == kind].copy()
        rows["magnitude_num"] = pd.to_numeric(rows["magnitude"])
        absolute = np.empty((4, len(values)), dtype=float)
        for i, target in enumerate(target_order):
            absolute[i, 0] = clean[target]
            subset = rows[rows["target_short"] == target].set_index("magnitude_num")
            for j, value in enumerate(values[1:], start=1):
                absolute[i, j] = subset.loc[value, "mae_mean"]
        matrices.append(absolute)
        labels.append(display)

    temp_values = [250.0, 273.15, 298.15, 323.15, 350.0, 400.0]
    temp_rows = compact[compact["perturbation_type"] == "temperature_K"].copy()
    temp_rows["magnitude_num"] = pd.to_numeric(temp_rows["magnitude"])
    absolute = np.empty((4, len(temp_values)), dtype=float)
    for i, target in enumerate(target_order):
        subset = temp_rows[temp_rows["target_short"] == target].set_index("magnitude_num")
        for j, value in enumerate(temp_values):
            absolute[i, j] = subset.loc[value, "mae_mean"]
    matrices.append(absolute)
    labels.append(["250", "273", "298", "323", "350", "400"])

    family_order = [
        "triaryl_ring",
        "trialkyl",
        "trialkyl_ring",
        "mixed_aryl_alkyl_ring",
        "non_tri_carbon_P",
        "non_tri_carbon_P_ring",
    ]
    family_labels = ["Triaryl", "Trialkyl", "Cyc. alkyl", "Mixed", "Non-3C", "Cyc. non-3C"]
    absolute = np.empty((4, len(family_order)), dtype=float)
    for i, target in enumerate(target_order):
        subset = family[family["target_short"] == target].set_index("ligand_family")
        absolute[i] = subset.loc[family_order, "mae_mean"].to_numpy(float)
    matrices.append(absolute)
    labels.append(family_labels)
    return matrices, labels, target_order


def plot_robustness_heatmaps() -> dict[str, object]:
    matrices, xlabels, target_order = robustness_matrices()
    clean = matrices[0][:, [0]]
    ratios = [matrices[0] / clean, matrices[1] / clean]
    temp_reference = matrices[2][:, [2]]
    ratios.append(matrices[2] / temp_reference)
    ratios.append(matrices[3] / clean)

    source_rows = []
    conditions = ["geometry", "energy", "temperature", "family"]
    for condition, matrix, ratio, columns in zip(conditions, matrices, ratios, xlabels):
        for i, target in enumerate(target_order):
            for j, column in enumerate(columns):
                source_rows.append(
                    {
                        "panel": condition,
                        "target_short": target,
                        "condition": column.replace("\n", " "),
                        "mae_absolute": matrix[i, j],
                        "mae_relative": ratio[i, j],
                    }
                )
    pd.DataFrame(source_rows).to_csv(SOURCE / "fig5_source_data.csv", index=False)

    all_values = np.concatenate([x.ravel() for x in ratios])
    vmin = max(0.0, float(np.nanmin(all_values)))
    vmax = float(np.nanmax(all_values))
    cmap = LinearSegmentedColormap.from_list(
        "fcsr_relative_mae", [PALETTE["teal"], PALETTE["white"], PALETTE["wine"]]
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(178 * MM_TO_INCH, 112 * MM_TO_INCH))
    fig.subplots_adjust(left=0.125, right=0.90, top=0.96, bottom=0.15, wspace=0.26, hspace=0.38)
    images = []
    for idx, (ax, absolute, ratio, labels) in enumerate(zip(axes.flat, matrices, ratios, xlabels)):
        image = ax.imshow(ratio, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        images.append(image)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_yticks(np.arange(4), [TARGET_NAME[x] for x in target_order])
        if idx % 2:
            ax.tick_params(axis="y", labelleft=False)
        if idx == 3:
            ax.tick_params(axis="x", rotation=32)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        else:
            ax.tick_params(axis="x", rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PALETTE["ink"])
            spine.set_linewidth(SPINE_WIDTH)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color=PALETTE["white"], linewidth=0.33)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(4):
            for col in range(len(labels)):
                rgba = cmap(norm(ratio[row, col]))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = PALETTE["white"] if luminance < 0.48 else PALETTE["ink"]
                ax.text(col, row, f"{absolute[row, col]:.3f}", ha="center", va="center", fontsize=7.0, color=text_color)
        panel_label(ax, f"({chr(97 + idx)})", x=-0.16 if idx % 2 == 0 else -0.10)
    cax = fig.add_axes([0.925, 0.20, 0.015, 0.65])
    colorbar = fig.colorbar(images[0], cax=cax)
    colorbar.set_label("Relative MAE", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=8.0, length=2.3, width=SPINE_WIDTH)
    colorbar.outline.set_linewidth(SPINE_WIDTH)
    return save_figure(fig, "Fig5_robustness_heatmaps")


def plot_temperature() -> dict[str, object]:
    return plot_perturbation(
        "temperature_K", "FigS10_temperature_sensitivity", "Temperature (K)"
    )


def main() -> None:
    configure_style()
    SOURCE.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Fig2_frame_ablation": frame_figure("Fig2_frame_ablation", False, 76),
        "FigS1_dataset_distributions": plot_distributions(),
        "FigS2_complete_frame_ablation": plot_complete_frame_heatmap(),
    }
    metadata_path = HERE / "remaining_data_render_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
