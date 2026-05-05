# Realistic Workload And Serving Handoff

This note is a handoff for continuing the realistic-workload work from the
`feat/fairness-realistic-workload` branch. It explains what has already been
implemented, what the current benchmark actually measures, how to reproduce it,
and how to extend it into a true vLLM continuous-batching serving benchmark.

## Branch Context

Base branch:

```text
origin/feat/fairness-fixes
```

Integration branch:

```text
feat/fairness-realistic-workload
```

The key rule for this branch is:

```text
Use feat/fairness-fixes as the source of truth for methods and metrics.
Only port the realistic workload trace/runner pieces from realistic_workload.
```

Do not overwrite the fairness-branch method implementations with older files
from `origin/realistic_workload`. In particular, keep the fairness versions of:

- `methods/__init__.py`
- `methods/paged_attention.py`
- `methods/h2o.py`
- `methods/streaming_llm.py`
- `methods/streaming_llm_vllm.py`
- `scripts/run_experiment1.py`

## Files Added For Mixed Replay

Trace/config:

- `configs/experiment_mixed_workload.yaml`
- `configs/trace_govreport_wide.yaml`
- `configs/trace_longbench_verylong.yaml`
- `data/realistic_trace.py`

Runner/infrastructure:

- `scripts/run_experiment_mixed_workload.py`
- `slurm/run_mixed_workload.sbatch`
- `REALISTIC_WORKLOAD.md`

Results/docs:

- `results/experiment_mixed_workload/*.json`
- `MIXED_WORKLOAD_RESULTS.md`

`MIXED_WORKLOAD_RESULTS.md` is the easiest human-readable entry point for the
committed benchmark numbers. The JSON files remain the source of truth.

## Current Workload Design

The mixed replay workload has 500 total requests with a deterministic seed.
It uses one shared trace for every method.

| Bucket | Ratio | Count | Dataset | Prompt Tokens | Max New Tokens | Metric |
|---|---:|---:|---|---:|---:|---|
| `short` | 50% | 250 | MMLU | 0-512 | 8 | accuracy |
| `medium` | 30% | 150 | GovReport | 1k-4k | 256 | ROUGE-L |
| `long` | 15% | 75 | GovReport | 4k-16k | 512 | ROUGE-L |
| `very_long` | 5% | 25 | LongBench v2 | 16k-32k | 10 | accuracy |

The trace generator stores:

- `request_id`
- `prompt`
- `token_count`
- `max_new_tokens`
- `source_dataset`
- `bucket`
- `reference`
- `subset`
- `arrival_time_s`

`arrival_time_s` is generated from a Poisson process using:

```text
arrival_rate: 2.0 requests/sec
```

The trace path is:

```text
results/traces/mixed_workload_trace.jsonl
```

## What The Current Benchmark Measures

The current runner is a deterministic mixed-length sequential replay:

```text
for request in trace:
    method.generate(prompt, max_new_tokens)
```

It measures:

- per-request TTFT
- decode latency
- total generation time
- simulated queue wait from the trace arrival time
- simulated e2e latency
- per-request KV memory
- prefill KV memory when available
- OOM / over-length skips
- per-bucket quality
- overall MMLU/LongBench accuracy
- overall GovReport ROUGE-L

This is useful for cross-method comparison because every method sees the same
request distribution and the same request order.

## Important Limitation

This benchmark is not a true continuous-batching serving benchmark.

Why:

- Requests are replayed one at a time.
- `arrival_time_s` is used to estimate queue wait, not to concurrently submit
  requests into a live server.
- vLLM sees one prompt per `LLM.generate([prompt])` call.
- There is no multi-request resident KV pressure inside one engine.
- The vLLM scheduler does not get to batch decode steps across active requests.

Use this wording in reports:

```text
mixed sequential replay workload
```

Avoid calling this:

```text
true continuous batching
```

## Current Completed Runs

The following runs have been completed on Insomnia:

Full 500-request runs:

- `mixed_full_cache_full.json`
- `mixed_paged_attention_full.json`
- `mixed_streaming_llm_full.json`
- `mixed_streaming_llm_vllm_full.json`

50-request trials:

- `mixed_full_cache_50.json`
- `mixed_h2o_50.json`
- `mixed_paged_attention_50.json`
- `mixed_streaming_llm_50.json`
- `mixed_streaming_llm_vllm_50.json`

Smoke tests:

- `mixed_paged_attention_smoke.json`
- `mixed_streaming_llm_vllm_smoke.json`

Smoke files are sanity checks and usually should not be used in final tables.
The JSON summaries are the source of truth for exact numbers.

Committed full-run result files:

| Method | Requests | OOM | Avg KV MB | MCQ Acc | ROUGE-L | Notes |
|---|---:|---:|---:|---:|---:|---|
| `full_cache` | 500 | 0 | 200.347 | 0.5636 | 0.1945 | HF full-cache baseline |
| `paged_attention` | 500 | 0 | 200.402 | 0.5673 | 0.1948 | vLLM full-cache baseline |
| `streaming_llm` | 500 | 0 | 39.884 | 0.5600 | 0.1414 | HF sink+recent eviction |
| `streaming_llm_vllm` | 500 | 0 | 40.183 | 0.5273 | 0.0718 | vLLM sink+recent eviction |

Committed 50-request trial result files:

| Method | Requests | OOM | Avg KV MB | MCQ Acc | ROUGE-L | Notes |
|---|---:|---:|---:|---:|---:|---|
| `full_cache` | 50 | 0 | 213.668 | 0.6129 | 0.1549 | short sanity trial |
| `h2o` | 50 | 0 | 7.000 | 0.6129 | 0.0000 | 128-token H2O budget collapses summarization |
| `paged_attention` | 50 | 0 | 213.300 | 0.6129 | 0.1554 | matches full-cache quality in trial |
| `streaming_llm` | 50 | 0 | 38.000 | 0.6129 | 0.1308 | moderate summarization loss |
| `streaming_llm_vllm` | 50 | 0 | 38.259 | 0.5806 | 0.0558 | large summarization loss in trial |

The large p95/p99 e2e latency values in `MIXED_WORKLOAD_RESULTS.md` include
simulated queue wait at `arrival_rate=2.0`. They should be interpreted as
sequential-replay overload behavior, not as true vLLM serving latency.

## How To Reproduce Current Mixed Replay

Generate trace:

```bash
python -m data.realistic_trace \
    --config configs/experiment_mixed_workload.yaml
```

PagedAttention full run:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
  --export=ALL,METHOD=paged_attention,MAX_MODEL_LEN=34000,GPU_MEMORY_UTILIZATION=0.90 \
  slurm/run_mixed_workload.sbatch
```

StreamingLLM-vLLM full run:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
  --export=ALL,METHOD=streaming_llm_vllm,START_SIZE=4,RECENT_SIZE=1024,MAX_MODEL_LEN=34000,GPU_MEMORY_UTILIZATION=0.90 \
  slurm/run_mixed_workload.sbatch
```

HF StreamingLLM full run:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
  --export=ALL,METHOD=streaming_llm,START_SIZE=4,RECENT_SIZE=1024 \
  slurm/run_mixed_workload.sbatch
```

FullCache full run:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
  --export=ALL,METHOD=full_cache \
  slurm/run_mixed_workload.sbatch
```

H2O 50-request trial:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
  --export=ALL,METHOD=h2o,NUM_REQUESTS=50,HH_SIZE=64,RECENT_SIZE=64 \
  slurm/run_mixed_workload.sbatch
```

Check summaries:

```bash
python - <<'PY'
import json, glob
for path in sorted(glob.glob("results/experiment_mixed_workload/*.json")):
    if "_smoke" in path:
        continue
    s = json.load(open(path))["summary"]
    print("\n", path)
    for k in [
        "method", "n_samples", "n_oom", "avg_ttft_ms",
        "p95_e2e_latency_ms", "p99_e2e_latency_ms",
        "avg_peak_kv_memory_mb", "overall_accuracy",
        "overall_avg_rouge_l",
    ]:
        if k in s:
            print(f"{k}: {s[k]}")
PY
```

## What A True vLLM Serving Benchmark Should Do

To actually exercise vLLM continuous batching, do not use `BaseMethod.generate`
or one-request-at-a-time `LLM.generate([prompt])`.

Instead, build a vLLM-only async runner that compares:

- `paged_attention`
- `streaming_llm_vllm`

The runner should:

1. Load the same trace JSONL.
2. Start one shared vLLM async engine.
3. Submit requests according to `arrival_time_s`.
4. Keep multiple requests active at once.
5. Let vLLM batch decode steps across active requests.
6. Record per-request first-token and finish times.

Target architecture:

```text
trace JSONL
  -> asyncio arrival scheduler
  -> shared vLLM AsyncLLMEngine
  -> concurrent active requests
  -> vLLM continuous batching
  -> per-request metrics
```

Conceptual request task:

```python
async def run_request(engine, req, experiment_start):
    arrival = experiment_start + req["arrival_time_s"]
    now = time.perf_counter()
    if now < arrival:
        await asyncio.sleep(arrival - now)

    submit_time = time.perf_counter()
    first_token_time = None
    final_output = None

    sampling_params = SamplingParams(
        max_tokens=req["max_new_tokens"],
        temperature=0.0,
    )

    async for output in engine.generate(
        req["prompt"],
        sampling_params,
        request_id=str(req["request_id"]),
    ):
        if first_token_time is None and output.outputs[0].token_ids:
            first_token_time = time.perf_counter()
        final_output = output

    finish_time = time.perf_counter()
    return {
        "request_id": req["request_id"],
        "ttft_ms": (first_token_time - arrival) * 1000,
        "e2e_latency_ms": (finish_time - arrival) * 1000,
        "service_time_ms": (finish_time - submit_time) * 1000,
        "generated_text": final_output.outputs[0].text,
        "generated_tokens": len(final_output.outputs[0].token_ids),
    }
```

The top-level runner should create tasks for all trace requests:

```python
experiment_start = time.perf_counter()
tasks = [
    asyncio.create_task(run_request(engine, req, experiment_start))
    for req in trace
]
records = await asyncio.gather(*tasks)
```

This is what allows vLLM to see overlapping requests and perform continuous
batching.

## StreamingLLM-vLLM Notes For Async Serving

`methods/streaming_llm_vllm.py` currently applies monkey patches before
constructing a vLLM `LLM` object.

For an async serving runner, the follow-up developer should verify whether the
same patch points apply to `AsyncLLMEngine` in vLLM 0.8.5:

- `SelfAttnBlockSpaceManager.trim_request_blocks`
- `ModelInputForGPUBuilder._compute_lens`
- `Scheduler._append_slots`

Recommended implementation step:

```text
Factor patch setup out of StreamingLLMvLLMMethod so both LLM and
AsyncLLMEngine paths can reuse the same policy/patch code.
```

Then instantiate the async engine only after the patches are applied.

## Serving Metrics To Add

Per request:

- `arrival_time_s`
- `submit_time_s`
- `first_token_time_s`
- `finish_time_s`
- `ttft_ms`
- `e2e_latency_ms`
- `service_time_ms`
- `prompt_tokens`
- `generated_tokens`
- `bucket`
- quality metric

Overall:

- request throughput
- output token throughput
- p50/p95/p99 TTFT
- p50/p95/p99 e2e latency
- goodput under an SLO
- peak GPU memory under load
- `n_oom`
- `n_overlen`
- `n_timeout`
- per-bucket breakdown

## Arrival-Rate Sweep

A true serving benchmark should sweep arrival rate rather than use only one
fixed rate:

```text
0.25 req/s
0.5 req/s
1.0 req/s
2.0 req/s
4.0 req/s
```

Useful plots:

- arrival rate vs p95 e2e latency
- arrival rate vs request throughput
- arrival rate vs token throughput
- arrival rate vs goodput
- arrival rate vs peak GPU memory

This is where the advantage of vLLM PagedAttention and the memory benefit of
StreamingLLM-vLLM should become much clearer.

## Suggested Next PR Scope

1. Add `scripts/run_vllm_serving_workload.py`.
2. Reuse `results/traces/mixed_workload_trace.jsonl`.
3. Implement async request scheduling with `arrival_time_s`.
4. Support `--method paged_attention` and `--method streaming_llm_vllm`.
5. Add `--arrival_rate` override and optionally regenerate trace per rate.
6. Add timeout handling.
7. Add result JSON summaries compatible with the current mixed replay format.
8. Add Slurm script `slurm/run_vllm_serving_workload.sbatch`.
9. Run smoke tests at low request counts.
10. Sweep arrival rates for final comparison.
