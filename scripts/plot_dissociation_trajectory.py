"""Two-panel trajectory figure for the sample-size sweep on Moirai-Small/ETTh2 (h=96, condition B).

Panel (a): CKA against the pre-trained encoder, by n.
Panel (b): forgetting by n, held-out (the outcome) and selection-window (for contrast).

Every point is read from the run records through emit_sample_sweep.sweep_data(), the same loader
that writes tables/sample_sweep.tex. The previous version of this figure hard-coded its numbers and
carried two panels of the retracted linear-probe statistic; both are gone.

Output: paper_8/figures/dissociation_trajectory.pdf
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from emit_sample_sweep import sweep_data  # noqa: E402

OUT_PATH = REPO_ROOT / "paper_8" / "figures" / "dissociation_trajectory.pdf"

# Categorical slots 1 and 2 of the default palette, validated light-mode (CVD dE 24.7).
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def mean_sem(rows, key):
    v = [r[key] for r in rows]
    return st.mean(v), (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0)


def main():
    data = sweep_data()
    ns = [n for n, _, _ in data]
    cka = [mean_sem(rows, "final_cka") for _, rows, _ in data]
    held = [mean_sem(rows, "heldout_forgetting_pct") for _, rows, _ in data]
    sel = [mean_sem(rows, "forgetting_pct") for _, rows, _ in data]
    es = [tag == "cuda" for _, _, tag in data]

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.8, 2.4), dpi=150)
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels(["500", "1k", "2k", "5k", "10k"])
        ax.minorticks_off()
        ax.set_xlim(400, 12500)
        ax.set_xlabel("fine-tuning samples $n$")
        ax.grid(True, axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    def series(ax, pts, color, label):
        m = [p[0] for p in pts]
        e = [p[1] for p in pts]
        ax.errorbar(ns, m, yerr=e, color=color, lw=2, capsize=0, elinewidth=1, label=label)
        # The n=10k point is early-stopped (Table tab:trainers); drawn hollow so the protocol
        # change is visible rather than averaged into the line.
        for x, y, stop in zip(ns, m, es):
            ax.plot([x], [y], "o", ms=5, color=color, mfc="white" if stop else color, mew=1.5)

    series(ax_a, cka, SERIES_1, None)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("CKA vs. pre-trained")
    ax_a.set_title("(a) drift grows with $n$", loc="left", color=INK)

    ax_b.axhline(0, color=INK_MUTED, lw=0.8)
    series(ax_b, held, SERIES_1, "held-out (outcome)")
    series(ax_b, sel, SERIES_2, "selection windows")
    ax_b.set_ylabel("forgetting (%), $<0$ is better")
    ax_b.set_title("(b) held-out forgetting falls", loc="left", color=INK)
    ax_b.legend(frameon=False, fontsize=7, loc="lower left", labelcolor=INK)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
