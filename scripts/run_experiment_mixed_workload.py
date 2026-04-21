"""Mixed realistic workload runner — replays a pre-generated trace file.

Usage:
    # Step 1: Generate the trace (once, shared across all methods)
    python -m data.realistic_trace \
        --config configs/experiment_mixed_workload.yaml

    # Step 2: Run a method against the trace
    python scripts/run_experiment_mixed_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method full_cache

    # With custom trace path:
    python scripts/run_experiment_mixed_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method paged_attention \
        --trace results/traces/mixed_workload_trace.jsonl

    # Smoke test (first 3 requests only):
    python scripts/run_experiment_mixed_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method h2o \
        --smoke_test
"""

import argparse
import json
import os
import re
import sys
import torch
from pathlib import Path

# Allow running as `python scripts/run_experiment_mixed_workload.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from methods import METHODS

# vLLM raises VLLMValidationError (not torch.cuda.OutOfMemoryError) when a prompt
# exceeds max_model_len. Treat it as a per-sample OOM/skip rather than crashing
# the whole run. Import is conditional so non-vllm methods still work.
try:
    from vllm.exceptions import VLLMValidationError
    _SKIPPABLE_ERRORS = (torch.cuda.OutOfMemoryError, VLLMValidationError)
except ImportError:
    _SKIPPABLE_ERRORS = (torch.cuda.OutOfMemoryError,)


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a mixed realistic workload trace against a KV cache method.")
    parser.add_argument("--config", required=True,
                        help="Path to experiment_mixed_workload.yaml config.")
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()),
                        help="Which method to run.")
    parser.add_argument("--trace", default=None,
                        help="Path to trace JSONL file (overrides config).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run only the first 3 requests for quick validation.")
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


# ── Answer extraction (copied from run_experiment1.py) ─────────────────────────

def extract_answer(generated_text: str) -> str:
    """Extract A/B/C/D answer from generated text."""
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


# ── ROUGE-L (copied from run_experiment1.py) ──────────────────────────────────

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


# ── Percentile helper ─────────────────────────────────────────────────────────

def percentile(values: list[float], p: int) -> float:
    """Compute the p-th percentile from a sorted list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ── Trace loading ─────────────────────────────────────────────────────────────

def load_trace(trace_path: str) -> list[dict]:
    """Load a JSONL trace file into a list of request dicts."""
    requests = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    return requests


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"].get("local_path") or config["model"]["name"]
    model_name = os.path.expanduser(model_name)
    device     = config["model"]["device"]
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    wl = config["workload"]
    seed = wl["seed"]
    trace_path = args.trace or wl["trace_output_path"]

    # Load trace
    print(f"\n[runner] Loading trace from {trace_path} ...")
    trace = load_trace(trace_path)
    if args.smoke_test:
        trace = trace[:3]
        print(f"[runner] Smoke test — using {len(trace)} requests only.")
    print(f"[runner] Trace loaded: {len(trace)} requests")

    # Instantiate and set up method
    method_cls = METHODS[args.method]
    method = method_cls()

    print(f"\n[runner] Setting up method '{args.method}' with model '{model_name}' ...")
    method.setup(
        model_name=model_name,
        device=device,
        start_size=args.start_size,
        recent_size=args.recent_size,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
    )

    # Replay loop
    records = []
    n = len(trace)

    print(f"\n[runner] Replaying {n} requests ...\n")

    for i, req in enumerate(trace):
        max_new_tokens = req["max_new_tokens"]
        prompt = req["prompt"]
        source_dataset = req["source_dataset"]
        bucket = req["bucket"]
        reference = req["reference"]
        subset = req.get("subset", "")
        request_id = req["request_id"]

        use_rouge = source_dataset in ("govreport", "longbench")

        try:
            output = method.generate(prompt, max_new_tokens=max_new_tokens)
        except _SKIPPABLE_ERRORS as e:
            torch.cuda.empty_cache()
            err_kind = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "OVERLEN"
            print(f"  [{i+1:>3}/{n}] req={request_id:<6} {bucket:<10} "
                  f"{err_kind} — skipped")
            oom_record = {
                "id":                 f"req_{request_id}",
                "subset":             subset,
                "dataset":            source_dataset,
                "prompt_tokens":      0,
                "generated_tokens":   0,
                "ttft_ms":            0,
                "decode_latency_ms":  0,
                "total_time_ms":      0,
                "throughput_tok_s":   0,
                "peak_kv_memory_mb":  0,
                "prefill_peak_kv_memory_mb": 0,
                "generated_text":     "",
                "reference":          reference,
                "oom":                True,
                "bucket":             bucket,
            }
            if use_rouge:
                oom_record["rouge_l"] = 0.0
            else:
                oom_record["predicted"] = ""
                oom_record["correct"] = False
            records.append(oom_record)
            continue

        throughput = (output.generated_tokens / output.total_time_ms * 1000
                      if output.total_time_ms > 0 else 0.0)

        record = {
            "id":                 f"req_{request_id}",
            "subset":             subset,
            "dataset":            source_dataset,
            "prompt_tokens":      output.prompt_tokens,
            "generated_tokens":   output.generated_tokens,
            "ttft_ms":            round(output.ttft_ms, 3),
            "decode_latency_ms":  round(output.decode_latency_ms, 3),
            "total_time_ms":      round(output.total_time_ms, 3),
            "throughput_tok_s":   round(throughput, 2),
            "peak_kv_memory_mb":  round(output.peak_kv_memory_mb, 3),
            "prefill_peak_kv_memory_mb": round(output.metadata.get("prefill_peak_kv_memory_mb", output.peak_kv_memory_mb), 3),
            "generated_text":     output.generated_text,
            "reference":          reference,
            "oom":                False,
            "bucket":             bucket,
        }

        if use_rouge:
            rouge_l = compute_rouge_l(output.generated_text, reference)
            record["rouge_l"] = round(rouge_l, 4)
            quality_str = f"ROUGE-L={rouge_l:.3f}"
        else:
            predicted = extract_answer(output.generated_text)
            correct = predicted == reference
            record["predicted"] = predicted
            record["correct"] = correct
            quality_str = f"{'✓' if correct else '✗'} ({predicted or '?'} vs {reference})"

        records.append(record)

        print(f"  [{i+1:>3}/{n}] req={request_id:<6} {bucket:<10} "
              f"TTFT={output.ttft_ms:6.1f}ms  "
              f"decode={output.decode_latency_ms:6.1f}ms  "
              f"mem={output.peak_kv_memory_mb:6.1f}MB  "
              f"{quality_str}")

    # Teardown
    method.teardown()

    # ── Aggregate metrics ──────────────────────────────────────────────────────

    valid_records = [r for r in records if not r.get("oom")]
    n_oom = len(records) - len(valid_records)
    avg = lambda key: (sum(r[key] for r in valid_records) / len(valid_records)
                       if valid_records else 0.0)

    # Percentiles (overall)
    ttft_vals = [r["ttft_ms"] for r in valid_records]
    total_vals = [r["total_time_ms"] for r in valid_records]

    # Per-bucket breakdown
    bucket_names = ["short", "medium", "long", "very_long"]
    per_bucket = {}
    for bname in bucket_names:
        b_records = [r for r in valid_records if r["bucket"] == bname]
        if not b_records:
            per_bucket[bname] = {
                "count": 0, "n_oom": 0,
                "avg_ttft_ms": 0, "avg_total_time_ms": 0,
                "p50_ttft_ms": 0, "p95_ttft_ms": 0, "p99_ttft_ms": 0,
                "p50_total_time_ms": 0, "p95_total_time_ms": 0, "p99_total_time_ms": 0,
            }
            continue

        b_ttft = [r["ttft_ms"] for r in b_records]
        b_total = [r["total_time_ms"] for r in b_records]
        b_all = [r for r in records if r["bucket"] == bname]
        b_oom = len(b_all) - len(b_records)

        bucket_stats = {
            "count":              len(b_records),
            "n_oom":              b_oom,
            "avg_ttft_ms":        round(sum(b_ttft) / len(b_ttft), 3),
            "avg_total_time_ms":  round(sum(b_total) / len(b_total), 3),
            "p50_ttft_ms":        round(percentile(b_ttft, 50), 3),
            "p95_ttft_ms":        round(percentile(b_ttft, 95), 3),
            "p99_ttft_ms":        round(percentile(b_ttft, 99), 3),
            "p50_total_time_ms":  round(percentile(b_total, 50), 3),
            "p95_total_time_ms":  round(percentile(b_total, 95), 3),
            "p99_total_time_ms":  round(percentile(b_total, 99), 3),
        }

        # Quality per bucket
        mmlu_recs = [r for r in b_records if r["dataset"] == "mmlu"]
        rouge_recs = [r for r in b_records if r["dataset"] in ("govreport", "longbench")]

        if mmlu_recs:
            n_correct = sum(r["correct"] for r in mmlu_recs)
            bucket_stats["accuracy"] = round(n_correct / len(mmlu_recs), 4)
        if rouge_recs:
            avg_rouge = sum(r["rouge_l"] for r in rouge_recs) / len(rouge_recs)
            bucket_stats["avg_rouge_l"] = round(avg_rouge, 4)

        per_bucket[bname] = bucket_stats

    summary = {
        "method":              args.method,
        "config":              args.config,
        "trace_path":          trace_path,
        "seed":                seed,
        "model":               model_name,
        "n_samples":           len(records),
        "n_oom":               n_oom,
        "avg_ttft_ms":         round(avg("ttft_ms"), 3),
        "avg_decode_latency_ms": round(avg("decode_latency_ms"), 3),
        "avg_total_time_ms":   round(avg("total_time_ms"), 3),
        "avg_throughput_tok_s": round(avg("throughput_tok_s"), 2),
        "avg_peak_kv_memory_mb": round(avg("peak_kv_memory_mb"), 3),
        "p50_ttft_ms":         round(percentile(ttft_vals, 50), 3),
        "p95_ttft_ms":         round(percentile(ttft_vals, 95), 3),
        "p99_ttft_ms":         round(percentile(ttft_vals, 99), 3),
        "p50_total_time_ms":   round(percentile(total_vals, 50), 3),
        "p95_total_time_ms":   round(percentile(total_vals, 95), 3),
        "p99_total_time_ms":   round(percentile(total_vals, 99), 3),
        "per_bucket":          per_bucket,
    }

    # Overall quality
    all_mmlu = [r for r in valid_records if r["dataset"] == "mmlu"]
    all_rouge = [r for r in valid_records if r["dataset"] in ("govreport", "longbench")]
    if all_mmlu:
        n_correct = sum(r["correct"] for r in all_mmlu)
        summary["overall_accuracy"] = round(n_correct / len(all_mmlu), 4)
    if all_rouge:
        avg_rouge = sum(r["rouge_l"] for r in all_rouge) / len(all_rouge)
        summary["overall_avg_rouge_l"] = round(avg_rouge, 4)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Method:    {args.method}")
    print(f"  Trace:     {trace_path}")
    print(f"  Requests:  {len(records)} ({n_oom} OOM)")
    print(f"  Avg TTFT:          {summary['avg_ttft_ms']:.1f} ms")
    print(f"  Avg decode:        {summary['avg_decode_latency_ms']:.1f} ms")
    print(f"  Avg throughput:    {summary['avg_throughput_tok_s']:.1f} tok/s")
    print(f"  Avg peak KV mem:   {summary['avg_peak_kv_memory_mb']:.1f} MB")
    print(f"  P50 TTFT:          {summary['p50_ttft_ms']:.1f} ms")
    print(f"  P95 TTFT:          {summary['p95_ttft_ms']:.1f} ms")
    print(f"  P99 TTFT:          {summary['p99_ttft_ms']:.1f} ms")
    print(f"  P50 total:         {summary['p50_total_time_ms']:.1f} ms")
    print(f"  P95 total:         {summary['p95_total_time_ms']:.1f} ms")
    print(f"  P99 total:         {summary['p99_total_time_ms']:.1f} ms")

    if all_mmlu:
        print(f"  MMLU accuracy:     {summary['overall_accuracy']:.1%} ({len(all_mmlu)} samples)")
    if all_rouge:
        print(f"  Avg ROUGE-L:       {summary['overall_avg_rouge_l']:.4f} ({len(all_rouge)} samples)")

    print(f"\n  Per-bucket breakdown:")
    for bname in bucket_names:
        bs = per_bucket[bname]
        if bs["count"] == 0:
            print(f"    {bname:>10}: no valid samples")
            continue
        line = (f"    {bname:>10}: n={bs['count']:<4} "
                f"TTFT p50={bs['p50_ttft_ms']:>7.1f} p95={bs['p95_ttft_ms']:>7.1f} p99={bs['p99_ttft_ms']:>7.1f}  "
                f"total p50={bs['p50_total_time_ms']:>7.1f} p95={bs['p95_total_time_ms']:>7.1f} p99={bs['p99_total_time_ms']:>7.1f}")
        if "accuracy" in bs:
            line += f"  acc={bs['accuracy']:.1%}"
        if "avg_rouge_l" in bs:
            line += f"  rouge={bs['avg_rouge_l']:.4f}"
        print(line)

    print(f"{'='*60}\n")

    # Save results
    tag = "smoke" if args.smoke_test else "full"
    out_path = results_dir / f"mixed_{args.method}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[runner] Results saved to {out_path}")


if __name__ == "__main__":
    main()
