"""
generate_pipeline.py  –  draw the CNN-LSTM pipeline diagram and save to figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.6)
ax.axis("off")

# ── helper ──────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel=None, color="#dbeafe", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.07",
        linewidth=1.2, edgecolor="#1e40af", facecolor=color
    )
    ax.add_patch(rect)
    ty = y + h / 2 + (0.1 if sublabel else 0)
    ax.text(x + w/2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#1e3a8a")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.15, sublabel, ha="center", va="center",
                fontsize=7.5, color="#374151")

def arrow(ax, x1, x2, y=1.3):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5))

# ── boxes ────────────────────────────────────────────────
box(ax, 0.05, 0.5, 1.3, 1.6,  "Video",     "30 frames\n224×224", color="#e0f2fe")
arrow(ax, 1.4, 1.75)

box(ax, 1.75, 0.5, 2.2, 1.6,  "ResNet-50", "frozen backbone\n→ 2048-dim / frame",
    color="#dbeafe")
arrow(ax, 3.95, 4.3)

box(ax, 4.3, 0.5, 1.3, 1.6,   "Sequence",  "30 × 2048\nfeatures", color="#e0f2fe")
arrow(ax, 5.6, 5.95)

box(ax, 5.95, 0.5, 2.2, 1.6,  "LSTM",      "1 layer, 128 hidden\nglobal avg-pool",
    color="#dbeafe")
arrow(ax, 8.15, 8.5)

box(ax, 8.5, 0.5, 1.45, 1.6,  "Softmax",   "10 classes", color="#e0f2fe")

# ── label ────────────────────────────────────────────────
ax.text(5, 2.45,
        "CNN-LSTM Pipeline  (frozen ResNet-50 backbone + lightweight LSTM head)",
        ha="center", va="center", fontsize=9.5, color="#111827", fontstyle="italic")

plt.tight_layout(pad=0.2)
plt.savefig("figures/pipeline.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved figures/pipeline.png")
