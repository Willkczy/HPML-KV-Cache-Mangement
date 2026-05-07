"""Plot figures for KV Cache Management presentation slides."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIGURE_DIR = Path(__file__).parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "full_cache":    "#4C72B0",
    "h2o":           "#DD8452",
    "streaming_llm": "#55A868",
    "paged_attention":"#C44E52",
}
LABELS = {
    "full_cache":    "Full Cache",
    "h2o":           "H2O",
    "streaming_llm": "StreamingLLM",
    "paged_attention":"PagedAttention",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 (Slide 2)
# KV memory comparison on LongBench Short Answer (log scale)
# Inset: accuracy (all identical)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure1():
    methods = ["full_cache", "h2o", "streaming_llm", "paged_attention"]
    short_labels = [LABELS[m] for m in methods]

    # KV memory per benchmark
    kv_mmlu = [23.8, 17.5, 3.7, 23.8]       # MMLU short
    kv_lb   = [732.8, 17.5, 3.7, 732.9]     # LongBench short

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - w/2, kv_mmlu, w, label="MMLU (acc=56.7%)",
                   color=[COLORS[m] for m in methods], edgecolor="white",
                   linewidth=0.8, alpha=1.0)
    bars2 = ax.bar(x + w/2, kv_lb, w, label="LongBench Short (acc=40.9%)",
                   color=[COLORS[m] for m in methods], edgecolor="white",
                   linewidth=0.8, alpha=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Decode-phase KV Memory (MB, log scale)", fontsize=11)
    ax.set_title("Short-Answer Tasks — All methods achieve identical accuracy\nKV memory comparison: MMLU vs LongBench Short Answer",
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_ylim(0.5, 5000)

    # Value labels
    for bar, val in zip(bars1, kv_mmlu):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, kv_lb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Key reduction note as caption
    ax.text(0.5, -0.18,
            "StreamingLLM: 6.4× less KV than Full Cache on MMLU  |  198× less on LongBench  |  Zero accuracy loss",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color="dimgray")

    # Legend: solid = MMLU, faded = LB
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="gray", alpha=1.0, label="MMLU short (128–512 tok)"),
        Patch(facecolor="gray", alpha=0.5, label="LongBench short (8k–16k tok)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    ax.text(0.01, 0.97, "✓ Same accuracy across all methods — KV reduction is lossless",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(FIGURE_DIR / "figure1_kv_memory_short_tasks.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure1_kv_memory_short_tasks.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 (Slide 3)
# Left: LongBench Explain accuracy (shows H2O collapse)
# Right: GovReport ROUGE-L (shows H2O=0, StreamingLLM recovery)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure2():
    methods = ["full_cache", "h2o", "streaming_llm", "paged_attention"]
    colors  = [COLORS[m] for m in methods]
    short_labels = [LABELS[m] for m in methods]

    govreport_rouge = [0.1859, 0.0000, 0.1832, 0.1850]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    bars = ax.bar(short_labels, govreport_rouge, color=colors,
                  edgecolor="white", width=0.55)
    ax.set_ylabel("ROUGE-L", fontsize=12)
    ax.set_title("GovReport Summarization — Long-Form Generation Quality\n"
                 "(4k–16k input tokens, 512 output tokens, best config per method)",
                 fontsize=11)
    ax.set_ylim(0, 0.25)
    ax.axhline(0.1859, color=COLORS["full_cache"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Full Cache baseline (0.1859)")

    for bar, val in zip(bars, govreport_rouge):
        label = f"{val:.4f}" if val > 0 else "0.0000"
        ypos = bar.get_height() + 0.004 if val > 0 else 0.006
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                label, ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # Key takeaways as figure caption (below x-axis)
    ax.text(0.5, -0.18,
            "H2O: ROUGE-L = 0 at all budgets tested (128–2048 tokens)\n"
            "StreamingLLM (recent=4096): −1.5% vs baseline with 2.3× less KV memory",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            color="dimgray")

    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=11)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(FIGURE_DIR / "figure2_govreport_quality.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure2_govreport_quality.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 (Slide 4)
# GovReport quality-memory frontier: StreamingLLM sweep + baselines
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure3():
    # StreamingLLM sweep
    sllm_kv    = [56.2, 112.2, 224.2]
    sllm_rouge = [0.1292, 0.1583, 0.1832]
    sllm_labels = ["recent=1024", "recent=2048", "recent=4096"]

    # Reference points
    fullcache_kv    = 525.4
    fullcache_rouge = 0.1859
    paged_kv        = 525.8
    paged_rouge     = 0.1850
    h2o_kv          = 17.5
    h2o_rouge       = 0.0000

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # StreamingLLM frontier line
    ax.plot(sllm_kv, sllm_rouge, "o-",
            color=COLORS["streaming_llm"], linewidth=2, markersize=7,
            label="StreamingLLM (window sweep)")
    for x, y, lbl in zip(sllm_kv, sllm_rouge, sllm_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8, color=COLORS["streaming_llm"])

    # Reference: full_cache
    ax.scatter([fullcache_kv], [fullcache_rouge],
               color=COLORS["full_cache"], s=100, zorder=5,
               label=f"Full Cache ({fullcache_rouge:.4f})")
    ax.axhline(fullcache_rouge, color=COLORS["full_cache"],
               linestyle="--", linewidth=1, alpha=0.5)

    # Reference: paged_attention
    ax.scatter([paged_kv], [paged_rouge],
               color=COLORS["paged_attention"], s=100, marker="^", zorder=5,
               label=f"PagedAttention ({paged_rouge:.4f})")

    # Reference: H2O
    ax.scatter([h2o_kv], [h2o_rouge],
               color=COLORS["h2o"], s=100, marker="x", zorder=5,
               linewidths=2, label=f"H2O (0.0000 — collapse)")

    ax.set_xlabel("Decode-phase Peak KV Memory (MB)")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("GovReport: Quality–Memory Trade-off Frontier")
    ax.set_xlim(-10, 600)
    ax.set_ylim(-0.02, 0.22)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    # Highlight sweet spot
    ax.annotate("Sweet spot\n(−1.5%, 2.3× less memory)",
                xy=(224.2, 0.1832), xytext=(300, 0.13),
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "figure3_govreport_frontier.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure3_govreport_frontier.png")


if __name__ == "__main__":
    plot_figure1()
    plot_figure2()
    plot_figure3()
    print("\nAll figures saved.")