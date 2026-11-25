"""Poster charts from scripts/evaluate.py output: macro AUROC by finding group, per-finding AUROC,
prompt-paraphrase spread. Writes results/figures/*.png.

Usage: uv run python scripts/figures.py [--seed 0]
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from medical_vlm_distillation.data import FINDINGS, ROOT

RES = ROOT / "results"
OUT = RES / "figures"
TEACHER = "Teacher (CheXzero)"
# Teacher is the reference (neutral gray); students take categorical slots in a fixed order.
COLORS = {
    TEACHER: "#8a8984",
    "Supervised": "#2a78d6",
    "A: embedding": "#eb6834",
    "B: score": "#1baf7a",
    "A+B": "#eda100",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def group_bars(boots: dict, boots_iu: dict, meta: dict, seed: int) -> None:
    point, point_iu = meta["point"], meta["point_iu"]
    held = [FINDINGS.index(f) for f in meta["held_out"]]
    seen = [i for i in range(14) if i not in held]
    groups = {
        f"NIH test, seen ({len(seen)})": (boots, point, seen),
        f"NIH test, held-out ({len(held)})": (boots, point, held),
        f"IU-Xray external ({len(meta['iu_findings'])})": (boots_iu, point_iu, list(range(len(meta["iu_findings"])))),
    }
    models = [m for m in COLORS if m in point]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, m in enumerate(models):
        for g, (b, pt, idx) in enumerate(groups.values()):
            x = g + (k - (len(models) - 1) / 2) * width
            v = float(np.mean([pt[m][i] for i in idx]))
            lo, hi = np.percentile(b[m][:, idx].mean(1), [2.5, 97.5])
            ax.bar(x, v, width * 0.9, color=COLORS[m], label=m if g == 0 else None, zorder=2)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, linewidth=1, capsize=2, zorder=3)
            ax.text(x, hi + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_ylim(0.5, max(1.0, ax.get_ylim()[1]))
    ax.set_ylabel("macro AUROC (95% CI)", fontsize=8, color=INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=len(models), loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / f"macro_auroc_s{seed}.png", dpi=250)


def per_finding(meta: dict, seed: int) -> None:
    point = meta["point"]
    models = [m for m in COLORS if m in point]
    order = np.argsort(point[TEACHER])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k, m in enumerate(models):
        y = np.arange(14) + (k - (len(models) - 1) / 2) * 0.13
        ax.scatter(np.array(point[m])[order], y, s=22, color=COLORS[m], label=m, zorder=3,
                   marker="D" if m == TEACHER else "o", edgecolor="white", linewidth=0.8)  # fmt: skip
    labels = [FINDINGS[i] + ("  (held-out)" if FINDINGS[i] in meta["held_out"] else "") for i in order]
    ax.set_yticks(range(14), labels)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("NIH test AUROC", fontsize=8, color=INK)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"per_finding_s{seed}.png", dpi=250)


def paraphrases(seed: int) -> None:
    t = pd.read_csv(RES / f"paraphrase_s{seed}.csv").set_index("model")
    templates = [c for c in t.columns if " / " in c]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for k, m in enumerate(t.index):
        v = t.loc[m, templates].astype(float).to_numpy()
        ax.plot([k, k], [v.min(), v.max()], color=COLORS[m], linewidth=2, zorder=2)
        ax.scatter(np.full(len(v) - 1, k), v[1:], s=18, color=COLORS[m], zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([k], v[:1], s=40, marker="*", color=INK, zorder=4, label="training prompt" if k == 0 else None)
        ax.text(k + 0.12, v.min(), f"spread {v.max() - v.min():.3f}", fontsize=6, color=MUTED, va="center")
    ax.set_xticks(range(len(t)), list(t.index), fontsize=7)
    ax.set_xlim(-0.5, len(t) - 0.3)
    ax.set_ylabel("NIH test macro AUROC", fontsize=8, color=INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f"paraphrases_s{seed}.png", dpi=250)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    seed = p.parse_args().seed
    OUT.mkdir(parents=True, exist_ok=True)
    boots = torch.load(RES / f"boot_s{seed}.pt", weights_only=False)
    meta = json.loads((RES / f"metrics_s{seed}.json").read_text())
    group_bars(boots["test"], boots["iu"], meta, seed)
    per_finding(meta, seed)
    paraphrases(seed)
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
