#!/usr/bin/env python3
"""
Section 7.2 figure -- out-of-domain forgetting against CKA on the same inputs.

WHAT IT SHOWS. One point per (cell, out-of-domain dataset) pair, 28 in all: the mean retention loss
under full fine-tuning (positive = the capability got worse) against the mean CKA of the fine-tuned
encoder on that dataset's own inputs. Filled points are the pairs that meet the registered
harmful-forgetting rule (B loses >= 10 pp more than the strict freeze, in every seed). Colour is
model size, because size is what sets CKA's level: Small sits at 0.81-0.96 and carries 7 of the 11
harmful pairs, Base and Large sit lower and forget on Electricity alone.

THE STATISTIC IN THE CORNER is the registered one, computed over the 84 per-seed records, not over
the 28 pair means drawn here; it is read from scored.json rather than recomputed, so the figure and
Section 7.2 cannot disagree. Every count the text quotes is asserted below.

Colours are the dataviz reference palette's first three categorical slots, validated for the light
surface (CVD and normal-vision separation pass; the aqua is below 3:1 contrast, so every group is
also direct-labelled).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import GRID, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, save  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SC = json.loads((ROOT / "results/v59_ood_retention/scored.json").read_text())
PAIRS = SC["pairs"]
SP = SC["spearman"]
assert len(PAIRS) == 28 and SC["n_harmful"] == 11 and SC["outcome"] == "O2", \
    (len(PAIRS), SC["n_harmful"], SC["outcome"])

SIZE_COLOR = {"small": "#2a78d6", "base": "#eb6834", "large": "#1baf7a"}
SIZE_NAME = {"small": "Moirai-Small", "base": "Moirai-Base", "large": "Moirai-Large"}
size_of = lambda p: p["cell"].split("_")[0]  # noqa: E731
assert sum(p["harmful"] for p in PAIRS if size_of(p) == "small") == 7

fig, ax = plt.subplots(figsize=(5.5, 2.35), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(INK_MUTED)
    ax.spines[sp].set_linewidth(0.6)
ax.axhline(0, color=INK_MUTED, lw=0.7, zorder=1)
ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)

for p in PAIRS:
    c = SIZE_COLOR[size_of(p)]
    ax.scatter(p["cka_ood_B"], p["ret"]["B"]["mean"], s=26,
               facecolor=c if p["harmful"] else SURFACE, edgecolor=c, linewidth=1.0, zorder=4)
    # a 2px surface ring keeps overlapping points separable
    ax.scatter(p["cka_ood_B"], p["ret"]["B"]["mean"], s=46, facecolor="none",
               edgecolor=SURFACE, linewidth=1.2, zorder=3)

# direct labels at each group's centre of mass, so identity is never colour alone
for size, (dx, dy) in {"small": (0.0, 62), "base": (-0.03, 100), "large": (0.0, -50)}.items():
    pts = [p for p in PAIRS if size_of(p) == size]
    x = sum(p["cka_ood_B"] for p in pts) / len(pts)
    ax.text(x + dx, dy, SIZE_NAME[size], fontsize=6.4, color=INK_SECONDARY, ha="center", va="center")

ax.set_xlabel("CKA of the fully fine-tuned encoder on the out-of-domain inputs",
              fontsize=7.0, color=INK_SECONDARY, labelpad=2.0)
ax.set_ylabel("retention loss under\nfull fine-tuning (%)", fontsize=7.0, color=INK_SECONDARY,
              labelpad=2.0)
ax.tick_params(labelsize=6.4, colors=INK_SECONDARY, length=2.4, width=0.6, pad=1.4)
ax.set_xlim(0.33, 1.0)
fmt = lambda v: f"{v:+.3f}".replace("-", "\u2212")  # noqa: E731
ax.text(0.01, 1.02, f"Spearman(1\u2212CKA, loss) = {fmt(SP['rho'])}, "
        f"cell-bootstrap 95% CI [{fmt(SP['lo'])}, {fmt(SP['hi'])}] over {SP['n']} records",
        transform=ax.transAxes, fontsize=6.4, color=INK_PRIMARY, va="bottom")

h_filled = plt.Line2D([], [], marker="o", linestyle="none", markersize=4.2, markerfacecolor=INK_SECONDARY,
                      markeredgecolor=INK_SECONDARY, label=f"harmful (registered rule), {SC['n_harmful']}")
h_open = plt.Line2D([], [], marker="o", linestyle="none", markersize=4.2, markerfacecolor="none",
                    markeredgecolor=INK_SECONDARY, label=f"not harmful, {len(PAIRS) - SC['n_harmful']}")
leg = ax.legend(handles=[h_filled, h_open], fontsize=6.2, loc="upper left", frameon=False,
                handletextpad=0.3, borderaxespad=0.2)
for t in leg.get_texts():
    t.set_color(INK_SECONDARY)

out = save(fig, Path(__file__).with_name("fig_ood_retention.png"))
print(f"wrote {out} and .pdf  ({len(PAIRS)} pairs, {SC['n_harmful']} harmful, rho={SP['rho']:+.3f})")
