"""Experiment 1 — Controlled Long Context runner.

Usage:
    python scripts/run_experiment1.py \
        --config configs/experiment1_short.yaml \
        --method h2o \
        [--hh_size 64] [--recent_size 64]

Outputs a JSON results file to the directory specified in the config.
"""

import argparse
import json
import time
from pathlib import Path

import yaml

from data.pipeline import load_samples
from eval.metrics import evaluate_mmlu, print_results
from methods import METHODS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", required=True, choices=list(METHODS.keys()))
    p.add_argument("--hh_size", type=int, default=64, help="H2O heavy-hitter budget")
    p.add_argument("--recent_size", type=int, default=64, help="H2O recent-window budget")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    device = config["model"].get("device", "cuda")
    max_new_tokens = config["generation"]["max_new_tokens"]
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] method={args.method}  model={model_name}  device={device}")

    # ── Load data ────────────────────────────────────────────────────────────
    samples = load_samples(args.config)
    print(f"[run] {len(samples)} samples loaded")

    # ── Setup method ─────────────────────────────────────────────────────────
    method = METHODS[args.method]()
    method_kwargs = {}
    if args.method == "h2o":
        method_kwargs = {"hh_size": args.hh_size, "recent_size": args.recent_size}
        print(f"[run] H2O budget: hh={args.hh_size}  recent={args.recent_size}")

    method.setup(model_name, device=device, **method_kwargs)

    # ── Run inference ─────────────────────────────────────────────────────────
    outputs = []
    references = []
    t_start = time.perf_counter()

    for i, sample in enumerate(samples):
        out = method.generate(sample.prompt, max_new_tokens=max_new_tokens, **method_kwargs)
        outputs.append(out)
        references.append(sample.reference)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] ttft={out.ttft_ms:.0f}ms  "
                  f"kv={out.peak_kv_memory_mb:.1f}MB  pred={repr(out.generated_text[:20])}")

    t_total = time.perf_counter() - t_start
    print(f"[run] finished {len(samples)} samples in {t_total:.1f}s")

    method.teardown()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    dataset = config["data"]["dataset"]
    if dataset == "mmlu":
        result = evaluate_mmlu(args.method, outputs, references)
    else:
        raise NotImplementedError(f"No evaluator for dataset '{dataset}'")

    print_results(result)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = results_dir / f"{args.method}.json"
    record = {
        "method": result.method,
        "dataset": dataset,
        "num_samples": result.num_samples,
        "accuracy": result.accuracy,
        "avg_ttft_ms": result.avg_ttft_ms,
        "avg_decode_latency_ms": result.avg_decode_latency_ms,
        "avg_throughput_tps": result.avg_throughput_tps,
        "avg_peak_kv_mb": result.avg_peak_kv_mb,
        "hh_size": args.hh_size if args.method == "h2o" else None,
        "recent_size": args.recent_size if args.method == "h2o" else None,
        "per_sample": [
            {
                "id": s.id,
                "reference": s.reference,
                "prediction": o.generated_text,
                "ttft_ms": o.ttft_ms,
                "decode_latency_ms": o.decode_latency_ms,
                "peak_kv_mb": o.peak_kv_memory_mb,
                "generated_tokens": o.generated_tokens,
            }
            for s, o in zip(samples, outputs)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[run] results saved to {out_path}")


if __name__ == "__main__":
    main()
