"""Plot figures for KV Cache Management presentation slides."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURE_DIR = Path(__file__).parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

COLORS = {
    "full_cache":     "#4C72B0",
    "h2o":            "#DD8452",
    "streaming_llm":  "#55A868",
    "paged_attention": "#C44E52",
}
LABELS = {
    "full_cache":     "Full Cache",
    "h2o":            "H2O",
    "streaming_llm":  "StreamingLLM",
    "paged_attention": "PagedAttention",
}

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "figure.dpi": 150})


# ── Figure 1: KV memory on short tasks (MMLU + LongBench) ────────────────────
def plot_figure1():
    methods = ["full_cache", "h2o", "streaming_llm", "paged_attention"]
    kv_mmlu = [23.8,  7.0,  3.7, 23.8]
    kv_lb   = [732.8, 7.0,  3.7, 732.9]
    x, w = np.arange(len(methods)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, kv_mmlu, w, label="MMLU (acc=56.7%)",
                   color=[COLORS[m] for m in methods], edgecolor="white", alpha=1.0)
    bars2 = ax.bar(x + w/2, kv_lb,   w, label="LongBench Short (acc=40.9%)",
                   color=[COLORS[m] for m in methods], edgecolor="white", alpha=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Decode-phase KV Memory (MB, log scale)", fontsize=11)
    ax.set_title("Short-Answer Tasks — All methods achieve identical accuracy\n"
                 "KV memory comparison: MMLU vs LongBench Short Answer", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=11)
    ax.set_ylim(0.5, 5000)

    for bar, val in zip(bars1, kv_mmlu):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, kv_lb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.4,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="gray", alpha=1.0, label="MMLU short (128–512 tok)"),
        Patch(facecolor="gray", alpha=0.5, label="LongBench short (8k–16k tok)"),
    ], fontsize=9, loc="upper right")

    ax.text(0.5, -0.14,
            "StreamingLLM: 6.4× less KV on MMLU  |  198× less on LongBench  |  Zero accuracy loss",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="dimgray")
    ax.text(0.01, 0.97, "✓ Same accuracy across all methods — KV reduction is lossless",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(FIGURE_DIR / "figure1_kv_memory_short_tasks.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure1_kv_memory_short_tasks.png")


# ── Figure 1b: Throughput on short tasks ─────────────────────────────────────
def plot_figure1b():
    methods = ["full_cache", "h2o", "streaming_llm", "paged_attention"]
    thr_mmlu = [40.2, 30.2, 34.5, 43.5]
    thr_lb   = [4.6,  4.0,  4.1,  5.4]
    x, w = np.arange(len(methods)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, thr_mmlu, w, label="MMLU (128–512 tok)",
                   color=[COLORS[m] for m in methods], edgecolor="white", alpha=1.0)
    bars2 = ax.bar(x + w/2, thr_lb,   w, label="LongBench Short (8k–16k tok)",
                   color=[COLORS[m] for m in methods], edgecolor="white", alpha=0.5)

    ax.set_ylabel("Throughput (tok/s)", fontsize=12)
    ax.set_title("Short-Answer Tasks — Throughput Comparison\nSame accuracy across all methods", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=11)

    for bar, val in zip(bars1, thr_mmlu):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, thr_lb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="gray", alpha=1.0, label="MMLU (128–512 tok)"),
        Patch(facecolor="gray", alpha=0.5, label="LongBench Short (8k–16k tok)"),
    ], fontsize=9, loc="upper right")

    ax.text(0.5, -0.14,
            "H2O is ~25% slower due to unfused eager attention kernel required for eviction scoring",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="dimgray")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(FIGURE_DIR / "figure1b_throughput_short_tasks.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure1b_throughput_short_tasks.png")


# ── Figure 2: GovReport ROUGE-L ──────────────────────────────────────────────
def plot_figure2():
    methods = ["full_cache", "h2o", "streaming_llm", "paged_attention"]
    govreport_rouge = [0.1859, 0.0000, 0.1832, 0.1850]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar([LABELS[m] for m in methods], govreport_rouge,
                  color=colors, edgecolor="white", width=0.55)
    ax.set_ylabel("ROUGE-L", fontsize=12)
    ax.set_title("GovReport Summarization — Long-Form Generation Quality\n"
                 "(4k–16k input tokens, 512 output tokens, best config per method)", fontsize=11)
    ax.set_ylim(0, 0.25)
    ax.axhline(0.1859, color=COLORS["full_cache"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Full Cache baseline (0.1859)")

    for bar, val in zip(bars, govreport_rouge):
        label = f"{val:.4f}" if val > 0 else "0.0000"
        ypos = bar.get_height() + 0.004 if val > 0 else 0.006
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.text(0.5, -0.18,
            "H2O: ROUGE-L = 0 at all budgets tested (128–2048 tokens)\n"
            "StreamingLLM (recent=4096): −1.5% vs baseline with 2.3× less KV memory",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="dimgray")

    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=11)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(FIGURE_DIR / "figure2_govreport_quality.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure2_govreport_quality.png")


# ── Figure 3: GovReport quality-memory frontier ───────────────────────────────
def plot_figure3():
    sllm_kv    = [56.2, 112.2, 224.2]
    sllm_rouge = [0.1292, 0.1583, 0.1832]
    sllm_labels = ["recent=1024", "recent=2048", "recent=4096"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(sllm_kv, sllm_rouge, "o-", color=COLORS["streaming_llm"],
            linewidth=2, markersize=7, label="StreamingLLM (window sweep)")
    for x, y, lbl in zip(sllm_kv, sllm_rouge, sllm_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(5, 6), fontsize=8, color=COLORS["streaming_llm"])

    ax.scatter([525.4], [0.1859], color=COLORS["full_cache"], s=100, zorder=5,
               label="Full Cache (0.1859)")
    ax.axhline(0.1859, color=COLORS["full_cache"], linestyle="--", linewidth=1, alpha=0.5)
    ax.scatter([525.8], [0.1850], color=COLORS["paged_attention"], s=100,
               marker="^", zorder=5, label="PagedAttention (0.1850)")
    ax.scatter([7.0], [0.0], color=COLORS["h2o"], s=100, marker="x",
               zorder=5, linewidths=2, label="H2O (0.0000 — collapse)")

    ax.set_xlabel("Decode-phase Peak KV Memory (MB)", fontsize=11)
    ax.set_ylabel("ROUGE-L", fontsize=11)
    ax.set_title("GovReport: Quality–Memory Trade-off Frontier", fontsize=11)
    ax.set_xlim(-10, 600)
    ax.set_ylim(-0.02, 0.22)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(linestyle="--", alpha=0.4)
    ax.annotate("Sweet spot\n(−1.5%, 2.3× less memory)",
                xy=(224.2, 0.1832), xytext=(300, 0.13),
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "figure3_govreport_frontier.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure3_govreport_frontier.png")


# ── Figure 4: Serving P99 latency + KV util ──────────────────────────────────
def plot_figure4():
    rates = [0.5, 1.0, 2.0, 4.0]
    p99 = {
        "v0_eager": [25.2,  60.5,  202.8, 256.7],
        "v0_graph": [30.4,  75.5,  230.5, 294.3],
        "v1_graph": [26.1,  73.0,  274.5, 345.6],
        "v1_eager": [26.3,  66.6,  276.0, 347.0],
    }
    kv_util_v0_eager = [18.7, 33.9, 79.7, 79.7]
    styles = {
        "v0_eager": ("solid",  "o", "#4C72B0", "v0 eager"),
        "v0_graph": ("dashed", "s", "#4C72B0", "v0 graph"),
        "v1_graph": ("solid",  "^", "#DD8452", "v1 graph"),
        "v1_eager": ("dashed", "D", "#DD8452", "v1 eager"),
    }

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    for key, (ls, mk, col, lbl) in styles.items():
        ax1.plot(rates, p99[key], linestyle=ls, marker=mk,
                 color=col, linewidth=2, markersize=7, label=lbl)
    ax2.bar(rates, kv_util_v0_eager, width=0.18, alpha=0.25,
            color="#4C72B0", label="KV util (v0 eager)")
    ax2.set_ylabel("Peak KV Block Utilization % (v0 only)", fontsize=10, color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.set_ylim(0, 120)

    ax1.set_xlabel("Arrival Rate (req/s)", fontsize=11)
    ax1.set_ylabel("P99 E2E Latency (s)", fontsize=11)
    ax1.set_title("Concurrent Serving — P99 Latency & KV Block Utilization\n"
                  "PagedAttention: v0/v1 × graph/eager", fontsize=11)
    ax1.set_xticks(rates)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.axvspan(1.0, 2.0, alpha=0.07, color="red")
    ax1.text(1.5, 310, "Saturation\nzone", ha="center", fontsize=8, color="red")

    from matplotlib.patches import Patch
    lines1, labels1 = ax1.get_legend_handles_labels()
    kv_patch = Patch(facecolor="#4C72B0", alpha=0.25, label="KV util % (v0 eager, right axis)")
    ax1.legend(handles=lines1 + [kv_patch], fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "figure4_serving_p99_kv.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure4_serving_p99_kv.png")


# ── Figure 5: Serving throughput at saturation ───────────────────────────────
def plot_figure5():
    configs = ["v0\neager", "v0\ngraph", "v1\ngraph", "v1\neager"]
    throughput = [1.434, 1.296, 1.147, 1.142]
    colors = ["#4C72B0", "#7BA7D6", "#DD8452", "#F0B482"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(configs, throughput, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Actual Throughput (req/s)", fontsize=12)
    ax.set_title("Serving Throughput at 4.0 req/s Arrival\n"
                 "(system saturated — actual throughput ≪ arrival rate)", fontsize=11)
    ax.set_ylim(0, 1.8)
    ax.axhline(4.0, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Arrival rate (4.0 req/s)")
    for bar, val in zip(bars, throughput):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.16,
            "v0 eager is 24% faster than v1 — CUDA graphs hurt variable-length batches; "
            "v1 chunked prefill adds scheduling overhead",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="dimgray")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(FIGURE_DIR / "figure5_serving_throughput.png", bbox_inches="tight")
    plt.show()
    print("Saved: figure5_serving_throughput.png")


if __name__ == "__main__":
    plot_figure1()
    plot_figure1b()
    plot_figure2()
    plot_figure3()
    plot_figure4()
    plot_figure5()
    print("\nAll figures saved to figures/")