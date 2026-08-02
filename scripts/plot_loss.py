"""
Plot training and validation loss from a nanoGPT training log.

Reads the "step N: train loss X, val loss Y" lines and renders a light-mode and
a dark-mode PNG.

Usage:
    python scripts/plot_loss.py train_8k.log loss_curve
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")  # write to file, never open a window
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")

# Palette roles. Dark is a selected set of steps for the dark surface,
# not an automatic inversion of the light one.
THEMES = {
    "light": dict(
        surface="#fcfcfb", train="#2a78d6", val="#eb6834",
        ink="#0b0b0b", secondary="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
    ),
    "dark": dict(
        surface="#1a1a19", train="#3987e5", val="#d95926",
        ink="#ffffff", secondary="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
    ),
}


def parse(path):
    steps, train, val = [], [], []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                train.append(float(m.group(2)))
                val.append(float(m.group(3)))
    if not steps:
        sys.exit(f"no 'step N: train loss ...' lines found in {path}")
    return steps, train, val


def render(steps, train, val, theme_name, out_path, reference=None):
    c = THEMES[theme_name]
    best_i = min(range(len(val)), key=lambda i: val[i])
    best_step, best_val = steps[best_i], val[best_i]

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    # Recessive grid, drawn under the data
    ax.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(steps, train, color=c["train"], linewidth=2, zorder=3, label="Training loss")
    ax.plot(steps, val, color=c["val"], linewidth=2, zorder=3, label="Validation loss")

    # Mark the best validation point -- the checkpoint that was actually kept.
    # A 2px surface-colored ring separates the marker from the line beneath it.
    ax.plot(best_step, best_val, "o", markersize=9, color=c["val"],
            markeredgecolor=c["surface"], markeredgewidth=2, zorder=5)
    ax.annotate(
        f"best {best_val:.4f} @ step {best_step:,}\ncheckpoint saved here",
        xy=(best_step, best_val), xytext=(-16, 38), textcoords="offset points",
        ha="right", fontsize=9.5, color=c["secondary"], zorder=6,
        arrowprops=dict(arrowstyle="-", color=c["axis"], linewidth=1),
    )

    # Shade the region after the best point: training continued, model got worse.
    # Label runs vertically because the band is narrower than the word.
    if best_step < steps[-1]:
        ax.axvspan(best_step, steps[-1], color=c["muted"], alpha=0.10, zorder=1, lw=0)
        ax.text((best_step + steps[-1]) / 2, ax.get_ylim()[0], " overfitting",
                ha="center", va="bottom", fontsize=8.5, rotation=90,
                color=c["muted"], zorder=2)

    # Reference: where the shorter 3000-iteration run topped out.
    if reference is not None:
        ref_val, ref_label = reference
        ax.axhline(ref_val, color=c["muted"], linewidth=1,
                   linestyle=(0, (4, 4)), zorder=2)
        ax.text(steps[0], ref_val, f" {ref_label}", ha="left", va="bottom",
                fontsize=8.5, color=c["muted"], zorder=2)

    ax.set_title("Character-level GPT on Shakespeare, CPU-only",
                 fontsize=13, color=c["ink"], pad=14, loc="left")
    ax.text(0, 1.015, "2.67M parameters · Intel i5, 4 threads, no GPU",
            transform=ax.transAxes, fontsize=9.5, color=c["muted"], va="bottom")

    ax.set_xlabel("Training iteration", fontsize=10, color=c["secondary"], labelpad=8)
    ax.set_ylabel("Loss (cross-entropy)", fontsize=10, color=c["secondary"], labelpad=8)
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1)

    leg = ax.legend(frameon=False, fontsize=10, loc="center right")
    for text in leg.get_texts():
        text.set_color(c["secondary"])  # text stays ink, the line carries identity

    fig.tight_layout()
    fig.savefig(out_path, facecolor=c["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "train_8k.log"
    stem = sys.argv[2] if len(sys.argv) > 2 else "loss_curve"
    s, t, v = parse(log)
    print(f"parsed {len(s)} eval points from {log}")
    ref = (1.8762, "best of the 3000-iteration run: 1.8762")
    render(s, t, v, "light", f"{stem}.png", reference=ref)
    render(s, t, v, "dark", f"{stem}_dark.png", reference=ref)
