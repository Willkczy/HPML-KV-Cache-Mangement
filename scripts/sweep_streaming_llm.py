"""Window size sweep for StreamingLLM.

Runs StreamingLLM across increasing recent_size values on one or more
benchmarks to characterise the memory-quality tradeoff curve.

Usage:
    # Smoke (3 samples per config) — quick sanity check
    python scripts/sweep_streaming_llm.py --smoke_test

    # Full sweep on LongBench only
    python scripts/sweep_streaming_llm.py \
        --configs configs/experiment2_long.yaml

    # Full sweep on all benchmarks
    python scripts/sweep_streaming_llm.py \
        --configs configs/experiment1_short.yaml \
                  configs/experiment2_long.yaml \
                  configs/experiment2_long_explain.yaml \
                  configs/experiment2_summarization.yaml

Results are saved to results/sweep_streaming_llm/<config_name>/<recent_size>.json
Summary table is printed at the end.
"""

import argparse
import json
import os
import re
import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.pipeline import load_samples
from methods.streaming_llm import StreamingLLMMethod


# ── Window sizes to sweep ─────────────────────────────────────────────────────

WINDOW_SIZES = [256, 512, 1024, 2048, 4096, 8192]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
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
    return matches[-1].upper() if matches else ""


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def run_one(method, samples, dataset_name, max_new_tokens, use_rouge):
    """Run method on samples, return list of records."""
    records = []
    for sample in samples:
        try:
            out = method.generate(sample.prompt, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            records.append({"id": sample.id, "oom": True,
                            "prompt_tokens": sample.token_count})
            continue

        throughput = (out.generated_tokens / out.total_time_ms * 1000
                      if out.total_time_ms > 0 else 0.0)

        record = {
            "id":                        sample.id,
            "subset":                    sample.subset,
            "prompt_tokens":             out.prompt_tokens,
            "generated_tokens":          out.generated_tokens,
            "ttft_ms":                   round(out.ttft_ms, 3),
            "decode_latency_ms":         round(out.decode_latency_ms, 3),
            "throughput_tok_s":          round(throughput, 2),
            "peak_kv_memory_mb":         round(out.peak_kv_memory_mb, 3),
            "prefill_kv_memory_mb": round(out.metadata.get("prefill_kv_memory_mb", 0), 3),
            "oom":                       False,
        }

        if use_rouge:
            record["rouge_l"] = round(compute_rouge_l(out.generated_text, sample.reference), 4)
        else:
            predicted = extract_answer(out.generated_text)
            record["predicted"] = predicted
            record["correct"]   = predicted == sample.reference

        records.append(record)

    return records


def summarise(records, use_rouge):
    valid = [r for r in records if not r.get("oom")]
    n_oom = len(records) - len(valid)
    avg = lambda k: sum(r[k] for r in valid) / len(valid) if valid else 0.0

    s = {
        "n_samples":               len(records),
        "n_oom":                   n_oom,
        "avg_ttft_ms":             round(avg("ttft_ms"), 1),
        "avg_decode_latency_ms":   round(avg("decode_latency_ms"), 1),
        "avg_throughput_tok_s":    round(avg("throughput_tok_s"), 2),
        "avg_peak_kv_memory_mb":   round(avg("peak_kv_memory_mb"), 1),
        "avg_prefill_kv_memory_mb": round(avg("prefill_kv_memory_mb"), 1),
    }
    if use_rouge:
        s["avg_rouge_l"] = round(avg("rouge_l"), 4)
    else:
        n_correct = sum(r.get("correct", False) for r in valid)
        s["accuracy"] = round(n_correct / len(valid), 4) if valid else 0.0

    return s


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="StreamingLLM window size sweep.")
    parser.add_argument("--configs", nargs="+",
                        default=["configs/experiment2_long.yaml"],
                        help="One or more config YAML files to sweep over.")
    parser.add_argument("--window_sizes", nargs="+", type=int,
                        default=WINDOW_SIZES,
                        help="recent_size values to sweep.")
    parser.add_argument("--start_size", type=int, default=4,
                        help="Number of attention sink tokens (fixed).")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Use only 3 samples per config for quick validation.")
    return parser.parse_args()


def main():
    args = parse_args()

    all_summaries = {}  # config_name → {recent_size → summary}

    for config_path in args.configs:
        config_name = Path(config_path).stem
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}")
        print(f"{'='*60}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_name     = config["model"].get("local_path") or config["model"]["name"]
        model_name     = os.path.expanduser(model_name)
        device         = config["model"]["device"]
        max_new_tokens = config["generation"]["max_new_tokens"]
        dataset_name   = config["data"]["dataset"]
        use_rouge      = dataset_name == "govreport"

        results_dir = Path("results/sweep_streaming_llm") / config_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load samples once per config
        print(f"\n[sweep] Loading samples ...")
        samples = load_samples(config_path)
        if args.smoke_test:
            samples = samples[:3]
            print(f"[sweep] Smoke test — {len(samples)} samples only.")

        config_summaries = {}

        for recent_size in args.window_sizes:
            window = args.start_size + recent_size
            print(f"\n── recent_size={recent_size}  (window={window} tokens) ──")

            method = StreamingLLMMethod()
            method.setup(model_name, device=device,
                         start_size=args.start_size, recent_size=recent_size)

            records = run_one(method, samples, dataset_name, max_new_tokens, use_rouge)
            method.teardown()

            summary = summarise(records, use_rouge)
            config_summaries[recent_size] = summary

            # Print one-line result
            quality = (f"ROUGE-L={summary['avg_rouge_l']:.3f}" if use_rouge
                       else f"acc={summary['accuracy']:.1%}")
            print(f"  {quality}  |  "
                  f"decode_mem={summary['avg_peak_kv_memory_mb']:.1f}MB  |  "
                  f"prefill_mem={summary['avg_prefill_kv_memory_mb']:.1f}MB  |  "
                  f"TTFT={summary['avg_ttft_ms']:.0f}ms  |  "
                  f"OOM={summary['n_oom']}/{summary['n_samples']}")

            # Save per-window results
            tag = "smoke" if args.smoke_test else "full"
            out_path = results_dir / f"recent{recent_size}_{tag}.json"
            with open(out_path, "w") as f:
                json.dump({"summary": summary, "records": records}, f, indent=2)

        all_summaries[config_name] = config_summaries

    # ── Final comparison table ─────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  SWEEP SUMMARY")
    print(f"{'='*60}")

    for config_name, summaries in all_summaries.items():
        print(f"\n  {config_name}")
        use_rouge = "avg_rouge_l" in next(iter(summaries.values()))
        quality_key = "avg_rouge_l" if use_rouge else "accuracy"
        quality_label = "ROUGE-L" if use_rouge else "Accuracy"

        print(f"  {'recent_size':>12} | {quality_label:>10} | {'Decode Mem':>12} | {'Prefill Mem':>12} | {'TTFT':>8} | {'OOM':>5}")
        print(f"  {'-'*70}")
        for recent_size, s in sorted(summaries.items()):
            q = s.get(quality_key, 0)
            print(f"  {recent_size:>12} | {q:>10.3f} | "
                  f"{s['avg_peak_kv_memory_mb']:>10.1f}MB | "
                  f"{s['avg_prefill_kv_memory_mb']:>10.1f}MB | "
                  f"{s['avg_ttft_ms']:>6.0f}ms | "
                  f"{s['n_oom']:>2}/{s['n_samples']}")


if __name__ == "__main__":
    main()
