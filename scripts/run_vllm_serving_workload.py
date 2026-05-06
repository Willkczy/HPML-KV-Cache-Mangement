"""True vLLM concurrent serving benchmark.

Submits trace requests concurrently via AsyncLLMEngine so vLLM's continuous
batching scheduler can batch decode steps across active requests. This is the
correct way to measure PagedAttention's serving advantage.

Usage:
    # PagedAttention (vLLM v0)
    python scripts/run_vllm_serving_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method paged_attention

    # StreamingLLM-vLLM (vLLM v0, with KV eviction)
    python scripts/run_vllm_serving_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method streaming_llm_vllm

    # Sweep arrival rates
    python scripts/run_vllm_serving_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method paged_attention \
        --arrival_rates 0.5 1.0 2.0 4.0

    # Smoke test (3 requests)
    python scripts/run_vllm_serving_workload.py \
        --config configs/experiment_mixed_workload.yaml \
        --method paged_attention \
        --smoke_test
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM concurrent serving benchmark using AsyncLLMEngine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True,
                        choices=["paged_attention", "streaming_llm_vllm"])
    parser.add_argument("--trace", default=None)
    parser.add_argument("--arrival_rates", nargs="+", type=float, default=None,
                        help="Arrival rates to sweep (req/s). Default: use trace rate.")
    parser.add_argument("--max_model_len", type=int, default=34000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=1024)
    parser.add_argument("--num_requests", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--vllm_use_v1", action="store_true",
                        help="Use vLLM v1 engine (for experiment 5).")
    parser.add_argument("--enforce_eager", action="store_true", default=False,
                        help="Disable CUDA graphs. Use for HF-comparable benchmarks. Default: False (CUDA graphs on).")
    return parser.parse_args()


# ── Quality helpers ────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    text = text.strip()
    for pattern in [r'[Aa]nswer\s*(?:is|:)\s*([A-D])', r'\b([A-D])\b\s*$', r'^\s*([A-D])\b']:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    matches = re.findall(r'\b([A-D])\b', text)
    return matches[-1].upper() if matches else ""


def compute_rouge_l(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev = curr
    lcs = prev[n]
    p = lcs / len(pred_tokens)
    r = lcs / len(ref_tokens)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def percentile(values: list, p: int) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sv) - 1)
    return sv[f] + (k - f) * (sv[c] - sv[f])


# ── Trace loading ──────────────────────────────────────────────────────────────

def load_trace(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def scale_arrivals(trace: list[dict], original_rate: float, target_rate: float) -> list[dict]:
    """Scale arrival_time_s to match a different request rate."""
    if abs(original_rate - target_rate) < 1e-6:
        return trace
    scale = original_rate / target_rate
    return [{**r, "arrival_time_s": r["arrival_time_s"] * scale} for r in trace]


# ── Async request task ─────────────────────────────────────────────────────────

async def run_request(engine: AsyncLLMEngine, req: dict, experiment_start: float) -> dict:
    """Submit one request at its arrival time and collect metrics."""
    target_arrival = experiment_start + req["arrival_time_s"]
    now = time.perf_counter()
    if now < target_arrival:
        await asyncio.sleep(target_arrival - now)

    arrival_time = time.perf_counter()
    first_token_time = None
    final_output = None

    sampling_params = SamplingParams(
        max_tokens=req["max_new_tokens"],
        temperature=0.0,
    )

    try:
        async for output in engine.generate(
            req["prompt"],
            sampling_params,
            request_id=str(req["request_id"]),
        ):
            if first_token_time is None and output.outputs and output.outputs[0].token_ids:
                first_token_time = time.perf_counter()
            final_output = output

        finish_time = time.perf_counter()
        completion = final_output.outputs[0]
        generated_text = completion.text
        generated_tokens = len(completion.token_ids)

        ttft_ms = (first_token_time - arrival_time) * 1000 if first_token_time else 0.0
        e2e_ms = (finish_time - arrival_time) * 1000
        service_ms = (finish_time - arrival_time) * 1000

        use_rouge = req["source_dataset"] == "govreport"
        if use_rouge:
            quality = {"rouge_l": round(compute_rouge_l(generated_text, req["reference"]), 4)}
        else:
            predicted = extract_answer(generated_text)
            quality = {"predicted": predicted, "correct": predicted == req["reference"]}

        return {
            "id": f"req_{req['request_id']}",
            "request_id": req["request_id"],
            "bucket": req["bucket"],
            "dataset": req["source_dataset"],
            "prompt_tokens": req.get("token_count", 0),
            "generated_tokens": generated_tokens,
            "ttft_ms": round(ttft_ms, 3),
            "e2e_latency_ms": round(e2e_ms, 3),
            "service_time_ms": round(service_ms, 3),
            "generated_text": generated_text,
            "reference": req["reference"],
            "oom": False,
            **quality,
        }

    except Exception as e:
        return {
            "id": f"req_{req['request_id']}",
            "request_id": req["request_id"],
            "bucket": req["bucket"],
            "dataset": req["source_dataset"],
            "prompt_tokens": req.get("token_count", 0),
            "generated_tokens": 0,
            "ttft_ms": 0.0,
            "e2e_latency_ms": 0.0,
            "service_time_ms": 0.0,
            "generated_text": "",
            "reference": req["reference"],
            "oom": True,
            "error": str(e),
        }


# ── Block utilization monitor ──────────────────────────────────────────────────

async def monitor_blocks(engine: AsyncLLMEngine, interval: float = 0.5) -> list[dict]:
    """Poll KV block utilization every interval seconds until cancelled."""
    samples = []
    try:
        # Try to access block manager via scheduler (works for v0)
        # v0: engine.engine.scheduler  v1: engine.llm_engine or different path
        inner = (getattr(engine, "engine", None) or
                 getattr(engine, "llm_engine", None))
        if inner is None:
            raise AttributeError(f"Cannot find inner engine on {type(engine).__name__}")
        sched = getattr(inner, "scheduler", None)
        if sched is None:
            raise AttributeError("Cannot find scheduler on inner engine")
        scheduler = sched[0] if isinstance(sched, list) else sched
        total = scheduler.block_manager.num_total_gpu_blocks

        # Block-level utilization (accurate, v0 only)
        while True:
            free = scheduler.block_manager.get_num_free_gpu_blocks()
            samples.append({
                "t": round(time.perf_counter(), 3),
                "utilization": round(1.0 - free / total, 4),
                "used_blocks": total - free,
                "total_blocks": total,
                "source": "block_manager",
            })
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[monitor] Block manager unavailable ({e}), falling back to nvidia-smi GPU memory")
        # Fallback: poll total GPU memory used via nvidia-smi
        import subprocess
        try:
            # Get total GPU memory once for utilization calculation
            total_mem_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True)
            total_mb = float(total_mem_result.stdout.strip().split('\n')[0])

            while True:
                used_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True)
                used_mb = float(used_result.stdout.strip().split('\n')[0])
                samples.append({
                    "t": round(time.perf_counter(), 3),
                    "utilization": round(used_mb / total_mb, 4),
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "source": "nvidia-smi",
                })
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception as e2:
            print(f"[monitor] nvidia-smi fallback also failed: {e2}")
    return samples


# ── Async serving runner ───────────────────────────────────────────────────────

async def run_serving(engine: AsyncLLMEngine, trace: list[dict]) -> tuple[list[dict], float, list[dict]]:
    """Submit all requests concurrently and collect results + block utilization."""
    experiment_start = time.perf_counter()

    # Start block utilization monitor as background task
    monitor_task = asyncio.create_task(monitor_blocks(engine, interval=0.5))

    request_tasks = [
        asyncio.create_task(run_request(engine, req, experiment_start))
        for req in trace
    ]
    records = await asyncio.gather(*request_tasks)

    # Stop monitor after all requests finish
    monitor_task.cancel()
    block_samples = await monitor_task

    total_wall_s = time.perf_counter() - experiment_start
    return list(records), total_wall_s, block_samples


# ── Summary computation ────────────────────────────────────────────────────────

def compute_summary(method: str, records: list[dict], trace: list[dict],
                    arrival_rate: float, total_wall_s: float,
                    config_path: str, model_name: str,
                    block_samples: list[dict] = None) -> dict:
    valid = [r for r in records if not r.get("oom")]
    n_oom = len(records) - len(valid)

    avg = lambda k: sum(r[k] for r in valid) / len(valid) if valid else 0.0

    ttft_vals = [r["ttft_ms"] for r in valid]
    e2e_vals = [r["e2e_latency_ms"] for r in valid]
    svc_vals = [r["service_time_ms"] for r in valid]
    tok_vals = [r["generated_tokens"] for r in valid]

    total_output_tokens = sum(tok_vals)
    request_throughput = len(records) / total_wall_s if total_wall_s > 0 else 0.0
    token_throughput = total_output_tokens / total_wall_s if total_wall_s > 0 else 0.0

    bucket_names = ["short", "medium", "long", "very_long"]
    per_bucket = {}
    for bname in bucket_names:
        b = [r for r in valid if r["bucket"] == bname]
        if not b:
            per_bucket[bname] = {"count": 0, "n_oom": 0}
            continue
        b_ttft = [r["ttft_ms"] for r in b]
        b_e2e = [r["e2e_latency_ms"] for r in b]
        b_oom = len([r for r in records if r["bucket"] == bname]) - len(b)
        stats = {
            "count": len(b), "n_oom": b_oom,
            "avg_ttft_ms": round(sum(b_ttft)/len(b_ttft), 3),
            "p50_ttft_ms": round(percentile(b_ttft, 50), 3),
            "p95_ttft_ms": round(percentile(b_ttft, 95), 3),
            "p99_ttft_ms": round(percentile(b_ttft, 99), 3),
            "p50_e2e_ms": round(percentile(b_e2e, 50), 3),
            "p95_e2e_ms": round(percentile(b_e2e, 95), 3),
            "p99_e2e_ms": round(percentile(b_e2e, 99), 3),
        }
        mcq = [r for r in b if r["dataset"] in ("mmlu", "longbench")]
        rouge = [r for r in b if r["dataset"] == "govreport"]
        if mcq:
            stats["accuracy"] = round(sum(r["correct"] for r in mcq) / len(mcq), 4)
        if rouge:
            stats["avg_rouge_l"] = round(sum(r["rouge_l"] for r in rouge) / len(rouge), 4)
        per_bucket[bname] = stats

    all_mcq = [r for r in valid if r["dataset"] in ("mmlu", "longbench")]
    all_rouge = [r for r in valid if r["dataset"] == "govreport"]

    # Block utilization metrics (KV memory pressure)
    block_util = {}
    if block_samples:
        util_vals = [s["utilization"] for s in block_samples]
        block_util = {
            "peak_block_utilization": round(max(util_vals), 4),
            "avg_block_utilization": round(sum(util_vals) / len(util_vals), 4),
            "p95_block_utilization": round(percentile(util_vals, 95), 4),
            "n_block_samples": len(block_samples),
        }

    summary = {
        "method": method,
        "config": config_path,
        "model": model_name,
        "arrival_rate_req_s": arrival_rate,
        "n_samples": len(records),
        "n_oom": n_oom,
        "total_wall_s": round(total_wall_s, 3),
        "request_throughput_req_s": round(request_throughput, 4),
        "token_throughput_tok_s": round(token_throughput, 2),
        "avg_ttft_ms": round(avg("ttft_ms"), 3),
        "avg_e2e_latency_ms": round(avg("e2e_latency_ms"), 3),
        "p50_ttft_ms": round(percentile(ttft_vals, 50), 3),
        "p95_ttft_ms": round(percentile(ttft_vals, 95), 3),
        "p99_ttft_ms": round(percentile(ttft_vals, 99), 3),
        "p50_e2e_latency_ms": round(percentile(e2e_vals, 50), 3),
        "p95_e2e_latency_ms": round(percentile(e2e_vals, 95), 3),
        "p99_e2e_latency_ms": round(percentile(e2e_vals, 99), 3),
        **block_util,
        "per_bucket": per_bucket,
    }
    if all_mcq:
        summary["overall_accuracy"] = round(sum(r["correct"] for r in all_mcq) / len(all_mcq), 4)
    if all_rouge:
        summary["overall_avg_rouge_l"] = round(sum(r["rouge_l"] for r in all_rouge) / len(all_rouge), 4)

    return summary


# ── Engine factory ─────────────────────────────────────────────────────────────

def build_engine(method: str, model_name: str, args) -> AsyncLLMEngine:
    """Apply patches if needed, then create AsyncLLMEngine."""
    if method == "streaming_llm_vllm":
        import math as _math
        from methods.streaming_llm_vllm import _apply_vllm_patches, SinkRecentPolicy
        block_size = args.block_size
        sink_blocks = _math.ceil(args.start_size / block_size)
        recent_blocks = _math.ceil(args.recent_size / block_size)
        policy = SinkRecentPolicy(
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            block_size=block_size,
        )
        _apply_vllm_patches(policy)
        print(f"[serving] StreamingLLM patches applied — "
              f"sink={sink_blocks} blocks, recent={recent_blocks} blocks")

    engine_args = AsyncEngineArgs(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        block_size=args.block_size,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        disable_log_requests=True,
    )
    print(f"[serving] enforce_eager={args.enforce_eager}, VLLM_USE_V1={os.environ.get('VLLM_USE_V1','0')}")
    return AsyncLLMEngine.from_engine_args(engine_args)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.vllm_use_v1:
        os.environ["VLLM_USE_V1"] = "1"

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model"].get("local_path") or config["model"]["name"]
    model_name = os.path.expanduser(model_name)
    wl = config["workload"]
    trace_path = args.trace or wl["trace_output_path"]
    original_rate = wl.get("arrival_rate", 2.0)
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and optionally cap trace
    full_trace = load_trace(trace_path)
    if args.smoke_test:
        full_trace = full_trace[:3]
    elif args.num_requests:
        full_trace = full_trace[:args.num_requests]

    print(f"[serving] Method: {args.method}")
    print(f"[serving] Model:  {model_name}")
    print(f"[serving] Trace:  {len(full_trace)} requests from {trace_path}")

    arrival_rates = args.arrival_rates or [original_rate]

    for rate in arrival_rates:
        print(f"\n[serving] === Arrival rate: {rate:.2f} req/s ===")
        trace = scale_arrivals(full_trace, original_rate, rate)

        # Fresh engine per rate for independent results
        engine = build_engine(args.method, model_name, args)
        records, wall_s, block_samples = asyncio.run(run_serving(engine, trace))
        del engine

        summary = compute_summary(
            method=args.method,
            records=records,
            trace=trace,
            arrival_rate=rate,
            total_wall_s=wall_s,
            config_path=args.config,
            model_name=model_name,
            block_samples=block_samples,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"  Method:         {args.method}")
        print(f"  Arrival rate:   {rate:.2f} req/s")
        print(f"  Requests:       {summary['n_samples']} ({summary['n_oom']} failed)")
        print(f"  Wall time:      {summary['total_wall_s']:.1f}s")
        print(f"  Req throughput: {summary['request_throughput_req_s']:.3f} req/s")
        print(f"  Tok throughput: {summary['token_throughput_tok_s']:.1f} tok/s")
        print(f"  P50 TTFT:       {summary['p50_ttft_ms']:.1f} ms")
        print(f"  P95 TTFT:       {summary['p95_ttft_ms']:.1f} ms")
        print(f"  P99 TTFT:       {summary['p99_ttft_ms']:.1f} ms")
        print(f"  P50 E2E:        {summary['p50_e2e_latency_ms']:.1f} ms")
        print(f"  P95 E2E:        {summary['p95_e2e_latency_ms']:.1f} ms")
        print(f"  P99 E2E:        {summary['p99_e2e_latency_ms']:.1f} ms")
        if "peak_block_utilization" in summary:
            print(f"  Peak KV util:   {summary['peak_block_utilization']:.1%}")
            print(f"  Avg KV util:    {summary['avg_block_utilization']:.1%}")
        if "overall_accuracy" in summary:
            print(f"  MCQ accuracy:   {summary['overall_accuracy']:.1%}")
        if "overall_avg_rouge_l" in summary:
            print(f"  Avg ROUGE-L:    {summary['overall_avg_rouge_l']:.4f}")
        print(f"{'='*60}\n")

        # Save
        engine_ver = "v1" if os.environ.get("VLLM_USE_V1", "0") == "1" else "v0"
        attn_mode = "eager" if args.enforce_eager else "graph"
        tag = "smoke" if args.smoke_test else f"rate{rate:.2f}"
        out_path = results_dir / f"serving_{args.method}_{engine_ver}_{attn_mode}_{tag}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"[serving] Results saved to {out_path}")


if __name__ == "__main__":
    main()
