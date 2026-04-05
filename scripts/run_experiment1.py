"""Experiment runner — supports MMLU, LongBench v2, and GovReport.

Usage:
    python scripts/run_experiment1.py \
        --config configs/experiment1_short.yaml \
        --method streaming_llm

    # StreamingLLM with custom window sizes:
    python scripts/run_experiment1.py \
        --config configs/experiment1_short.yaml \
        --method streaming_llm \
        --start_size 4 \
        --recent_size 256

    # Smoke test (first 3 samples only):
    python scripts/run_experiment1.py \
        --config configs/experiment1_short.yaml \
        --method streaming_llm \
        --smoke_test
"""

import argparse
import json
import os
import re
import sys
import time
import torch
from pathlib import Path

# Allow running as `python scripts/run_experiment1.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.pipeline import load_samples
from methods import METHODS


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single KV cache method.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()),
                        help="Which method to run.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run only the first 3 samples for quick validation.")
    # StreamingLLM-specific (ignored by other methods via **kwargs)
    parser.add_argument("--start_size", type=int, default=4,
                        help="StreamingLLM: number of attention sink tokens.")
    parser.add_argument("--recent_size", type=int, default=256,
                        help="StreamingLLM: size of the recent token window.")
    # PagedAttention-specific (ignored by other methods via **kwargs)
    parser.add_argument("--block_size", type=int, default=16,
                        help="PagedAttention: tokens per KV block.")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="PagedAttention: max sequence length for vLLM engine.")
    return parser.parse_args()


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(generated_text: str) -> str:
    """Extract A/B/C/D answer from generated text.

    Handles short answers ("B"), preambles ("The answer is B"),
    and chain-of-thought ("...therefore the answer is D").
    Prefers structured patterns; falls back to last standalone letter.
    """
    text = generated_text.strip()

    # Try structured patterns first (last match wins)
    patterns = [
        r'[Aa]nswer\s*(?:is|:)\s*([A-D])',
        r'\b([A-D])\b\s*$',              # single letter at end
        r'^\s*([A-D])\b',                # single letter at start
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()

    # Fallback: last standalone A/B/C/D
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1].upper()

    return ""


# ── ROUGE-L for summarization ────────────────────────────────────────────────

def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of the longest common subsequence."""
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
    """Compute ROUGE-L F1 score between prediction and reference."""
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["data"]["dataset"]
    model_name  = config["model"].get("local_path") or config["model"]["name"]
    model_name  = os.path.expanduser(model_name)
    device      = config["model"]["device"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine evaluation mode based on dataset
    use_rouge = dataset_name == "govreport"

    # Load samples
    print(f"\n[runner] Loading samples from {args.config} ...")
    samples = load_samples(args.config)
    if args.smoke_test:
        samples = samples[:3]
        print(f"[runner] Smoke test — using {len(samples)} samples only.")

    # Instantiate and set up method
    method_cls = METHODS[args.method]
    method = method_cls()

    print(f"\n[runner] Setting up method '{args.method}' with model '{model_name}' ...")
    method.setup(
        model_name=model_name,
        device=device,
        start_size=args.start_size,
        recent_size=args.recent_size,
        hh_size=args.hh_size,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
    )

    # Experiment loop
    records = []
    n = len(samples)
    print(f"\n[runner] Running {n} samples ...\n")

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()

        try:
            output = method.generate(sample.prompt, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [{i+1:>3}/{n}] {sample.id:<45} "
                  f"OOM at {sample.token_count} prompt tokens — skipped")
            oom_record = {
                "id":                 sample.id,
                "subset":             sample.subset,
                "dataset":            dataset_name,
                "prompt_tokens":      sample.token_count,
                "generated_tokens":   0,
                "ttft_ms":            0,
                "decode_latency_ms":  0,
                "total_time_ms":      0,
                "throughput_tok_s":   0,
                "peak_kv_memory_mb":  0,
                "generated_text":     "",
                "reference":          sample.reference,
                "oom":                True,
            }
            if use_rouge:
                oom_record["rouge_l"] = 0.0
            else:
                oom_record["predicted"] = ""
                oom_record["correct"] = False
            records.append(oom_record)
            continue

        wall_s = time.perf_counter() - t0

        throughput = (output.generated_tokens / output.total_time_ms * 1000
                      if output.total_time_ms > 0 else 0.0)
 
        record = {
            "id":                 sample.id,
            "subset":             sample.subset,
            "dataset":            dataset_name,
            "prompt_tokens":      output.prompt_tokens,
            "generated_tokens":   output.generated_tokens,
            "ttft_ms":            round(output.ttft_ms, 3),
            "decode_latency_ms":  round(output.decode_latency_ms, 3),
            "total_time_ms":      round(output.total_time_ms, 3),
            "throughput_tok_s":   round(throughput, 2),
            "peak_kv_memory_mb":  round(output.peak_kv_memory_mb, 3),
            "prefill_peak_kv_memory_mb": round(output.metadata.get("prefill_peak_kv_memory_mb", output.peak_kv_memory_mb), 3),
            "generated_text":     output.generated_text,
            "reference":          sample.reference,
            "oom":                False,
        }

        if use_rouge:
            rouge_l = compute_rouge_l(output.generated_text, sample.reference)
            record["rouge_l"] = round(rouge_l, 4)
            quality_str = f"ROUGE-L={rouge_l:.3f}"
        else:
            predicted = extract_answer(output.generated_text)
            correct = predicted == sample.reference
            record["predicted"] = predicted
            record["correct"] = correct
            quality_str = f"{'✓' if correct else '✗'} ({predicted or '?'} vs {sample.reference})"

        records.append(record)

        print(f"  [{i+1:>3}/{n}] {sample.id:<45} "
              f"TTFT={output.ttft_ms:6.1f}ms  "
              f"decode={output.decode_latency_ms:6.1f}ms  "
              f"mem={output.peak_kv_memory_mb:6.1f}MB  "
              f"{quality_str}")

    # Teardown
    method.teardown()

    # Aggregate metrics
    valid_records = [r for r in records if not r.get("oom")]
    n_oom = len(records) - len(valid_records)
    avg = lambda key: (sum(r[key] for r in valid_records) / len(valid_records)
                       if valid_records else 0.0)

    summary = {
        "method":              args.method,
        "config":              args.config,
        "dataset":             dataset_name,
        "model":               model_name,
        "n_samples":           len(records),
        "n_oom":               n_oom,
        "avg_ttft_ms":         round(avg("ttft_ms"), 3),
        "avg_decode_latency_ms": round(avg("decode_latency_ms"), 3),
        "avg_total_time_ms":   round(avg("total_time_ms"), 3),
        "avg_throughput_tok_s": round(avg("throughput_tok_s"), 2),
        "avg_peak_kv_memory_mb": round(avg("peak_kv_memory_mb"), 3),
    }

    print(f"\n{'='*60}")
    print(f"  Method:    {args.method}")
    print(f"  Dataset:   {dataset_name}")
    print(f"  Samples:   {len(records)} ({n_oom} OOM)")

    if use_rouge:
        avg_rouge = avg("rouge_l")
        summary["avg_rouge_l"] = round(avg_rouge, 4)
        print(f"  Avg ROUGE-L:       {avg_rouge:.4f}")
    else:
        n_correct = sum(r["correct"] for r in valid_records)
        accuracy = n_correct / len(valid_records) if valid_records else 0.0
        summary["accuracy"] = round(accuracy, 4)
        print(f"  Accuracy:  {accuracy:.1%}  ({n_correct}/{len(valid_records)})")
 
    print(f"  Avg TTFT:          {summary['avg_ttft_ms']:.1f} ms")
    print(f"  Avg decode:        {summary['avg_decode_latency_ms']:.1f} ms")
    print(f"  Avg throughput:    {summary['avg_throughput_tok_s']:.1f} tok/s")
    print(f"  Avg peak KV mem:   {summary['avg_peak_kv_memory_mb']:.1f} MB")
    print(f"{'='*60}\n")

    # Save results
    tag = "smoke" if args.smoke_test else "full"
    out_path = results_dir / f"{args.method}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[runner] Results saved to {out_path}")


if __name__ == "__main__":
    main()
