"""Experiment 1 runner — MMLU short bucket.

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
import time
from pathlib import Path

import yaml

from data.pipeline import load_samples
from methods import METHODS


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single KV cache method on MMLU.")
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
    return parser.parse_args()


# ── Accuracy helpers ───────────────────────────────────────────────────────────

def extract_mmlu_answer(generated_text: str) -> str:
    """Return the first A/B/C/D found in the generated text, or '' if none."""
    for char in generated_text.strip():
        if char in ("A", "B", "C", "D"):
            return char
    return ""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name  = config["model"].get("local_path") or config["model"]["name"]
    model_name  = os.path.expanduser(model_name)
    device      = config["model"]["device"]
    max_new_tokens = config["generation"]["max_new_tokens"]
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

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
    )

    # Experiment loop
    records = []
    n = len(samples)
    print(f"\n[runner] Running {n} samples ...\n")

    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        output = method.generate(sample.prompt, max_new_tokens=max_new_tokens)
        wall_s = time.perf_counter() - t0

        predicted = extract_mmlu_answer(output.generated_text)
        correct   = predicted == sample.reference

        throughput = (output.generated_tokens / output.total_time_ms * 1000
                      if output.total_time_ms > 0 else 0.0)

        record = {
            "id":                 sample.id,
            "subset":             sample.subset,
            "prompt_tokens":      output.prompt_tokens,
            "generated_tokens":   output.generated_tokens,
            "ttft_ms":            round(output.ttft_ms, 3),
            "decode_latency_ms":  round(output.decode_latency_ms, 3),
            "total_time_ms":      round(output.total_time_ms, 3),
            "throughput_tok_s":   round(throughput, 2),
            "peak_kv_memory_mb":  round(output.peak_kv_memory_mb, 3),
            "predicted":          predicted,
            "reference":          sample.reference,
            "correct":            correct,
        }
        records.append(record)

        print(f"  [{i+1:>3}/{n}] {sample.id:<45} "
              f"TTFT={output.ttft_ms:6.1f}ms  "
              f"decode={output.decode_latency_ms:6.1f}ms  "
              f"mem={output.peak_kv_memory_mb:6.1f}MB  "
              f"{'✓' if correct else '✗'} ({predicted or '?'} vs {sample.reference})")

    # Teardown
    method.teardown()

    # Aggregate metrics
    n_correct = sum(r["correct"] for r in records)
    accuracy  = n_correct / len(records) if records else 0.0
    avg = lambda key: sum(r[key] for r in records) / len(records) if records else 0.0

    summary = {
        "method":              args.method,
        "config":              args.config,
        "model":               model_name,
        "n_samples":           len(records),
        "accuracy":            round(accuracy, 4),
        "avg_ttft_ms":         round(avg("ttft_ms"), 3),
        "avg_decode_latency_ms": round(avg("decode_latency_ms"), 3),
        "avg_total_time_ms":   round(avg("total_time_ms"), 3),
        "avg_throughput_tok_s": round(avg("throughput_tok_s"), 2),
        "avg_peak_kv_memory_mb": round(avg("peak_kv_memory_mb"), 3),
    }

    print(f"\n{'='*60}")
    print(f"  Method:    {args.method}")
    print(f"  Samples:   {len(records)}")
    print(f"  Accuracy:  {accuracy:.1%}  ({n_correct}/{len(records)})")
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
