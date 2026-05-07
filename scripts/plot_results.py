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
    kv_mem  = [732.8, 17.5, 3.7, 732.9]   # MB, post-eviction decode steady-state
    accuracy = [40.9, 40.9, 40.9, 40.9]   # %

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        [LABELS[m] for m in methods],
        kv_mem,
        color=[COLORS[m] for m in methods],
        edgecolor="white", linewidth=0.8,
        width=0.55,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Decode-phase KV Memory (MB, log scale)")
    ax.set_title("LongBench Short Answer — KV Memory vs Accuracy")
    ax.set_ylim(0.5, 3000)

    # Annotate bars
    for bar, val in zip(bars, kv_mem):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                f"{val:.1f} MB", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Inset: accuracy (all equal)
    ax_in = ax.inset_axes([0.62, 0.55, 0.36, 0.40])
    ax_in.bar([LABELS[m][:4] for m in methods], accuracy,
              color=[COLORS[m] for m in methods], edgecolor="white")
    ax_in.set_ylim(0, 80)
    ax_in.set_ylabel("Accuracy (%)", fontsize=8)
    ax_in.set_title("Accuracy (all equal)", fontsize=8)
    ax_in.tick_params(axis="x", labelsize=7)
    ax_in.tick_params(axis="y", labelsize=7)
    ax_in.axhline(40.9, color="gray", linestyle="--", linewidth=0.8)

    ax.annotate("198× less\nthan baseline",
                xy=(2, 3.7), xytext=(2, 80),
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green", fontsize=9, ha="center")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "figure1_kv_memory_lbshort.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure1_kv_memory_lbshort.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 (Slide 3)
# Left: LongBench Explain accuracy (shows H2O collapse)
# Right: GovReport ROUGE-L (shows H2O=0, StreamingLLM recovery)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure2():
    methods = ["full_cache", "h2o", "streaming_llm", "paged_attention"]
    colors  = [COLORS[m] for m in methods]
    short_labels = [LABELS[m] for m in methods]

    lb_explain_acc  = [27.3, 4.5, 13.6, 18.2]   # %
    govreport_rouge = [0.1859, 0.0000, 0.1832, 0.1850]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: LB Explain accuracy ──
    bars1 = ax1.bar(short_labels, lb_explain_acc, color=colors,
                    edgecolor="white", width=0.55)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("LongBench Explain\n(8k–16k input, 512 output tokens)")
    ax1.set_ylim(0, 40)
    ax1.axhline(27.3, color=COLORS["full_cache"], linestyle="--",
                linewidth=1.2, alpha=0.6, label="full_cache baseline")
    for bar, val in zip(bars1, lb_explain_acc):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Right: GovReport ROUGE-L ──
    bars2 = ax2.bar(short_labels, govreport_rouge, color=colors,
                    edgecolor="white", width=0.55)
    ax2.set_ylabel("ROUGE-L")
    ax2.set_title("GovReport Summarization\n(best config per method)")
    ax2.set_ylim(0, 0.25)
    ax2.axhline(0.1859, color=COLORS["full_cache"], linestyle="--",
                linewidth=1.2, alpha=0.6, label="full_cache baseline")
    for bar, val in zip(bars2, govreport_rouge):
        label = f"{val:.4f}" if val > 0 else "0.0000\n(collapse)"
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 label, ha="center", va="bottom", fontsize=8.5)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate StreamingLLM recovery
    ax2.annotate("−1.5%\nvs baseline",
                 xy=(2, 0.1832), xytext=(2.4, 0.21),
                 arrowprops=dict(arrowstyle="->", color="green"),
                 color="green", fontsize=9)

    plt.suptitle("Long-Context Generation: Where KV Eviction Breaks", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "figure2_longcontext_quality.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure2_longcontext_quality.png")


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