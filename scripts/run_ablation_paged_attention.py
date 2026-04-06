"""Ablation study for PagedAttention parameters.

Varies one of {gpu_memory_utilization, max_model_len, block_size} at a time
while holding the other two at their defaults.  Runs across two configs
(experiment1_short + experiment2_long) and prints results as ablation tables.

Usage:
    python scripts/run_ablation_paged_attention.py

    # Smoke test (3 samples per setting):
    python scripts/run_ablation_paged_attention.py --smoke_test
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipeline import load_samples
from methods.paged_attention import PagedAttentionMethod


# ── Defaults (held constant when not being varied) ────────────────────────────

DEFAULTS = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 4096,
    "block_size": 16,
}

# Long-context config needs a larger max_model_len default
DEFAULTS_LONG = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 32768,
    "block_size": 16,
}

# ── Ablation ranges ──────────────────────────────────────────────────────────

ABLATION_RANGES = {
    "gpu_memory_utilization": [0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    "max_model_len": {
        "short": [1024, 2048, 4096, 8192, 16384],
        "long":  [8192, 16384, 32768, 65536],
    },
    "block_size": [8, 16, 32],
}

EXPERIMENTS = [
    {
        "label": "Exp1 Short (MMLU)",
        "config": "configs/experiment1_short.yaml",
        "key": "short",
    },
    {
        "label": "Exp2 Long (LongBench v2)",
        "config": "configs/experiment2_long.yaml",
        "key": "long",
    },
]


# ── Reused helpers from run_experiment1.py ────────────────────────────────────

def extract_answer(generated_text: str) -> str:
    text = generated_text.strip()
    patterns = [
        r'[Aa]nswer\s*(?:is|:)\s*([A-D])',
        r'\b([A-D])\b\s*$',
        r'^\s*([A-D])\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()
    return ""


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def compute_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Run one configuration ────────────────────────────────────────────────────

def run_single(config_path: str, samples, model_name: str, device: str,
               max_new_tokens: int, dataset_name: str,
               gpu_memory_utilization: float, max_model_len: int,
               block_size: int) -> dict:
    """Instantiate PagedAttention with given params, run all samples, return summary."""

    method = PagedAttentionMethod()
    try:
        method.setup(
            model_name=model_name,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            block_size=block_size,
        )
    except Exception as e:
        print(f"    [SETUP FAILED] {e}")
        return {
            "n_samples": len(samples), "n_oom": len(samples),
            "accuracy": 0.0, "rouge_l": 0.0,
            "avg_ttft_ms": 0.0, "avg_decode_latency_ms": 0.0,
            "avg_total_time_ms": 0.0, "avg_throughput_tok_s": 0.0,
            "avg_peak_kv_memory_mb": 0.0, "error": str(e),
        }

    use_rouge = dataset_name == "govreport"
    records = []
    n = len(samples)

    for i, sample in enumerate(samples):
        try:
            output = method.generate(sample.prompt, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            records.append({"oom": True})
            continue
        except Exception as e:
            records.append({"oom": True, "error": str(e)})
            continue

        throughput = (output.generated_tokens / output.total_time_ms * 1000
                      if output.total_time_ms > 0 else 0.0)

        record = {
            "oom": False,
            "ttft_ms": output.ttft_ms,
            "decode_latency_ms": output.decode_latency_ms,
            "total_time_ms": output.total_time_ms,
            "throughput_tok_s": throughput,
            "peak_kv_memory_mb": output.peak_kv_memory_mb,
        }

        if use_rouge:
            record["rouge_l"] = compute_rouge_l(output.generated_text, sample.reference)
        else:
            predicted = extract_answer(output.generated_text)
            record["correct"] = predicted == sample.reference

        records.append(record)

    method.teardown()

    valid = [r for r in records if not r.get("oom")]
    n_oom = len(records) - len(valid)
    avg = lambda key: (sum(r[key] for r in valid) / len(valid)) if valid else 0.0

    summary = {
        "n_samples": n,
        "n_oom": n_oom,
        "avg_ttft_ms": round(avg("ttft_ms"), 2),
        "avg_decode_latency_ms": round(avg("decode_latency_ms"), 2),
        "avg_total_time_ms": round(avg("total_time_ms"), 2),
        "avg_throughput_tok_s": round(avg("throughput_tok_s"), 2),
        "avg_peak_kv_memory_mb": round(avg("peak_kv_memory_mb"), 2),
    }

    if use_rouge:
        summary["avg_rouge_l"] = round(avg("rouge_l"), 4)
    else:
        n_correct = sum(r.get("correct", False) for r in valid)
        summary["accuracy"] = round(n_correct / len(valid), 4) if valid else 0.0

    return summary


# ── Table printer ─────────────────────────────────────────────────────────────

def print_ablation_table(title: str, param_name: str, rows: list[dict],
                         quality_key: str):
    """Pretty-print an ablation table to stdout."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"  Varying: {param_name}")
    print(f"{'='*90}")

    quality_hdr = "Accuracy" if quality_key == "accuracy" else "ROUGE-L"
    header = (f"{'Value':>12} | {quality_hdr:>10} | {'TTFT(ms)':>10} | "
              f"{'Decode(ms)':>10} | {'Tok/s':>8} | {'KV Mem(MB)':>11} | "
              f"{'OOM':>4} | {'Samples':>7}")
    print(header)
    print("-" * len(header))

    for row in rows:
        quality_val = row["summary"].get(quality_key, 0.0)
        s = row["summary"]
        val_str = str(row["value"])
        print(f"{val_str:>12} | {quality_val:>10.4f} | {s['avg_ttft_ms']:>10.2f} | "
              f"{s['avg_decode_latency_ms']:>10.2f} | {s['avg_throughput_tok_s']:>8.2f} | "
              f"{s['avg_peak_kv_memory_mb']:>11.2f} | {s['n_oom']:>4} | {s['n_samples']:>7}")

    print(f"{'='*90}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study for PagedAttention.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use only 3 samples per setting for quick testing.")
    args = parser.parse_args()

    all_results = {}
    results_dir = Path("results/ablation_paged_attention")
    results_dir.mkdir(parents=True, exist_ok=True)

    for experiment in EXPERIMENTS:
        label = experiment["label"]
        config_path = experiment["config"]
        exp_key = experiment["key"]
        defaults = DEFAULTS_LONG if exp_key == "long" else DEFAULTS

        print(f"\n{'#'*90}")
        print(f"# EXPERIMENT: {label}")
        print(f"# Config:     {config_path}")
        print(f"{'#'*90}")

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        dataset_name = config["data"]["dataset"]
        model_name = config["model"].get("local_path") or config["model"]["name"]
        model_name = os.path.expanduser(model_name)
        device = config["model"]["device"]
        max_new_tokens = config["generation"]["max_new_tokens"]

        # Load samples once per experiment
        print(f"\n[ablation] Loading samples from {config_path} ...")
        samples = load_samples(config_path)
        if args.smoke_test:
            samples = samples[:3]
            print(f"[ablation] Smoke test — using {len(samples)} samples.")

        use_rouge = dataset_name == "govreport"
        quality_key = "avg_rouge_l" if use_rouge else "accuracy"

        exp_results = {}

        # ── Ablation 1: gpu_memory_utilization ────────────────────────
        param = "gpu_memory_utilization"
        values = ABLATION_RANGES[param]
        rows = []
        print(f"\n[ablation] Sweeping {param}: {values}")
        for val in values:
            print(f"\n  >> {param}={val}  (max_model_len={defaults['max_model_len']}, "
                  f"block_size={defaults['block_size']})")
            summary = run_single(
                config_path, samples, model_name, device, max_new_tokens,
                dataset_name,
                gpu_memory_utilization=val,
                max_model_len=defaults["max_model_len"],
                block_size=defaults["block_size"],
            )
            rows.append({"value": val, "summary": summary})
            print(f"     -> {quality_key}={summary.get(quality_key, summary.get('accuracy', 0.0)):.4f}  "
                  f"TTFT={summary['avg_ttft_ms']:.1f}ms  "
                  f"KV={summary['avg_peak_kv_memory_mb']:.1f}MB  "
                  f"OOM={summary['n_oom']}")

        print_ablation_table(label, param, rows, quality_key)
        exp_results[param] = rows

        # ── Ablation 2: max_model_len ─────────────────────────────────
        param = "max_model_len"
        values = ABLATION_RANGES[param][exp_key]
        rows = []
        print(f"\n[ablation] Sweeping {param}: {values}")
        for val in values:
            print(f"\n  >> {param}={val}  (gpu_memory_utilization={defaults['gpu_memory_utilization']}, "
                  f"block_size={defaults['block_size']})")
            summary = run_single(
                config_path, samples, model_name, device, max_new_tokens,
                dataset_name,
                gpu_memory_utilization=defaults["gpu_memory_utilization"],
                max_model_len=val,
                block_size=defaults["block_size"],
            )
            rows.append({"value": val, "summary": summary})
            print(f"     -> {quality_key}={summary.get(quality_key, summary.get('accuracy', 0.0)):.4f}  "
                  f"TTFT={summary['avg_ttft_ms']:.1f}ms  "
                  f"KV={summary['avg_peak_kv_memory_mb']:.1f}MB  "
                  f"OOM={summary['n_oom']}")

        print_ablation_table(label, param, rows, quality_key)
        exp_results[param] = rows

        # ── Ablation 3: block_size ────────────────────────────────────
        param = "block_size"
        values = ABLATION_RANGES[param]
        rows = []
        print(f"\n[ablation] Sweeping {param}: {values}")
        for val in values:
            print(f"\n  >> {param}={val}  (gpu_memory_utilization={defaults['gpu_memory_utilization']}, "
                  f"max_model_len={defaults['max_model_len']})")
            summary = run_single(
                config_path, samples, model_name, device, max_new_tokens,
                dataset_name,
                gpu_memory_utilization=defaults["gpu_memory_utilization"],
                max_model_len=defaults["max_model_len"],
                block_size=val,
            )
            rows.append({"value": val, "summary": summary})
            print(f"     -> {quality_key}={summary.get(quality_key, summary.get('accuracy', 0.0)):.4f}  "
                  f"TTFT={summary['avg_ttft_ms']:.1f}ms  "
                  f"KV={summary['avg_peak_kv_memory_mb']:.1f}MB  "
                  f"OOM={summary['n_oom']}")

        print_ablation_table(label, param, rows, quality_key)
        exp_results[param] = rows

        all_results[exp_key] = exp_results

    # ── Save all results to JSON ──────────────────────────────────────────
    tag = "smoke" if args.smoke_test else "full"
    out_path = results_dir / f"ablation_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[ablation] All results saved to {out_path}")

    # ── Final combined summary ────────────────────────────────────────────
    print(f"\n{'#'*90}")
    print(f"# ABLATION STUDY COMPLETE")
    print(f"# Total configurations tested: "
          f"{sum(len(rows) for exp in all_results.values() for rows in exp.values())}")
    print(f"# Results: {out_path}")
    print(f"{'#'*90}\n")


if __name__ == "__main__":
    main()
