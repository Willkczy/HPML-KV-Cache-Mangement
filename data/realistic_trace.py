"""Realistic workload trace generator.

Loads samples from MMLU, GovReport, and LongBench v2 via the existing
data pipeline, buckets them by token count, mixes according to configured
ratios, and writes a deterministic JSONL trace file that can be replayed
identically by every method.

Usage:
    python -m data.realistic_trace \
        --config configs/experiment_mixed_workload.yaml

    # Override output path:
    python -m data.realistic_trace \
        --config configs/experiment_mixed_workload.yaml \
        --output results/traces/custom_trace.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.pipeline import load_samples


# ── Bucket definitions ─────────────────────────────────────────────────────────

# Maps bucket name to (source_dataset_key, config_key_in_source_configs)
BUCKET_SOURCES = {
    "short":     "mmlu",
    "medium":    "govreport",
    "long":      "govreport",
    "very_long": "longbench",
}


def _filter_by_token_range(samples, tok_min, tok_max):
    """Filter samples to those within [tok_min, tok_max]."""
    return [s for s in samples if tok_min <= s.token_count <= tok_max]


# ── Main trace generation ─────────────────────────────────────────────────────

def generate_trace(config_path: str, output_override: str = None) -> str:
    """Generate a JSONL trace file from a mixed-workload config.

    Args:
        config_path: path to the experiment_mixed_workload.yaml config.
        output_override: optional path to override trace_output_path in config.

    Returns:
        Path to the written trace file.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wl = config["workload"]
    total_requests = wl["total_requests"]
    seed = wl["seed"]
    bucket_ratios = wl["bucket_ratios"]
    bucket_ranges = wl["bucket_token_ranges"]
    max_new_tokens_map = wl["max_new_tokens_per_bucket"]
    source_configs = wl["source_configs"]
    trace_path = output_override or wl["trace_output_path"]

    rng = random.Random(seed)

    # Compute per-bucket request counts (distribute remainder to largest bucket)
    bucket_names = ["short", "medium", "long", "very_long"]
    bucket_counts = {}
    assigned = 0
    for name in bucket_names:
        count = int(total_requests * bucket_ratios[name])
        bucket_counts[name] = count
        assigned += count
    # Assign remainder to the first (largest) bucket
    remainder = total_requests - assigned
    bucket_counts[bucket_names[0]] += remainder

    print(f"\n[trace] Generating trace with {total_requests} requests (seed={seed})")
    for name in bucket_names:
        print(f"  {name:>10}: {bucket_counts[name]:>4} requests "
              f"({bucket_ranges[name]['min']}-{bucket_ranges[name]['max']} tokens, "
              f"max_new_tokens={max_new_tokens_map[name]})")

    # Load samples from each source dataset (deduplicate loads for same config)
    loaded_cache = {}
    for name in bucket_names:
        src_key = BUCKET_SOURCES[name]
        cfg_path = source_configs[src_key]
        if cfg_path not in loaded_cache:
            print(f"\n[trace] Loading samples from {cfg_path} ...")
            loaded_cache[cfg_path] = load_samples(cfg_path)

    # Bucket and select samples
    all_requests = []
    request_id = 0

    for name in bucket_names:
        src_key = BUCKET_SOURCES[name]
        cfg_path = source_configs[src_key]
        pool = loaded_cache[cfg_path]

        tok_min = bucket_ranges[name]["min"]
        tok_max = bucket_ranges[name]["max"]
        filtered = _filter_by_token_range(pool, tok_min, tok_max)

        needed = bucket_counts[name]
        if len(filtered) == 0:
            print(f"[trace] WARNING: no samples for bucket '{name}' "
                  f"({tok_min}-{tok_max} tokens from {src_key})")
            continue

        print(f"[trace] Bucket '{name}': {len(filtered)} samples available, "
              f"need {needed}")

        # Sample with replacement if needed > available, without otherwise
        if needed <= len(filtered):
            selected = rng.sample(filtered, needed)
        else:
            selected = [rng.choice(filtered) for _ in range(needed)]

        max_nt = max_new_tokens_map[name]
        for sample in selected:
            all_requests.append({
                "request_id": request_id,
                "prompt": sample.prompt,
                "max_new_tokens": max_nt,
                "source_dataset": sample.dataset,
                "bucket": name,
                "reference": sample.reference,
                "subset": sample.subset,
            })
            request_id += 1

    # Shuffle to interleave buckets (deterministic with seed)
    rng.shuffle(all_requests)

    # Re-assign sequential request_ids after shuffle
    for i, req in enumerate(all_requests):
        req["request_id"] = i

    # Write trace to disk
    trace_file = Path(trace_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    with open(trace_file, "w") as f:
        for req in all_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    # Print summary
    bucket_summary = {}
    for req in all_requests:
        b = req["bucket"]
        bucket_summary[b] = bucket_summary.get(b, 0) + 1

    print(f"\n[trace] Trace written to {trace_file}")
    print(f"[trace] Total requests: {len(all_requests)}")
    for name in bucket_names:
        print(f"  {name:>10}: {bucket_summary.get(name, 0)}")

    return str(trace_file)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a realistic mixed-workload trace file.")
    parser.add_argument("--config", required=True,
                        help="Path to experiment_mixed_workload.yaml config.")
    parser.add_argument("--output", default=None,
                        help="Override trace output path from config.")
    args = parser.parse_args()

    generate_trace(args.config, output_override=args.output)


if __name__ == "__main__":
    main()
