# Running The Realistic Workload

This workload should be run from a branch based on `feat/fairness-fixes`.
The trace generator and mixed runner are used here, while method
implementations and KV-memory semantics stay aligned with the fairness branch.

For the handoff notes and the plan for a true vLLM continuous-batching serving
benchmark, see `docs/REALISTIC_SERVING_HANDOFF.md`.

## 1. Generate The Shared Trace

Generate once and reuse the same JSONL file for every method:

```bash
python -m data.realistic_trace \
    --config configs/experiment_mixed_workload.yaml
```

The default trace is written to:

```text
results/traces/mixed_workload_trace.jsonl
```

Each request records its prompt, bucket, source dataset, reference answer,
token count, per-bucket `max_new_tokens`, and optional Poisson arrival time.

## 2. Smoke Test

Run a small test before launching full jobs:

```bash
VLLM_USE_V1=0 python scripts/run_experiment_mixed_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method paged_attention \
    --max_model_len 34000 \
    --gpu_memory_utilization 0.90 \
    --block_size 16 \
    --smoke_test
```

For StreamingLLM on vLLM:

```bash
VLLM_USE_V1=0 python scripts/run_experiment_mixed_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method streaming_llm_vllm \
    --start_size 4 \
    --recent_size 1024 \
    --max_model_len 34000 \
    --gpu_memory_utilization 0.90 \
    --block_size 16 \
    --smoke_test
```

## 3. Run A Capped Trial

Use a capped run before the full 500-request trace:

```bash
VLLM_USE_V1=0 python scripts/run_experiment_mixed_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method paged_attention \
    --max_model_len 34000 \
    --gpu_memory_utilization 0.90 \
    --num_requests 50
```

## 4. Run The Full Trace

```bash
VLLM_USE_V1=0 python scripts/run_experiment_mixed_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method paged_attention \
    --max_model_len 34000 \
    --gpu_memory_utilization 0.90
```

Replace `paged_attention` with `full_cache`, `h2o`, `streaming_llm`, or
`streaming_llm_vllm` as needed. HF methods may OOM on the longest buckets;
vLLM methods should be tested first.

## 5. Run On Insomnia

Generate the trace and run a smoke test:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=paged_attention,GENERATE_TRACE=1,SMOKE_TEST=1 \
    slurm/run_mixed_workload.sbatch
```

Run the full trace:

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=paged_attention,MAX_MODEL_LEN=34000,GPU_MEMORY_UTILIZATION=0.90 \
    slurm/run_mixed_workload.sbatch
```

Results are saved to:

```text
results/experiment_mixed_workload/mixed_<method>_<smoke|full>.json
```

## Metrics To Compare

- `p95_e2e_latency_ms` and `p99_e2e_latency_ms`
- `p95_ttft_ms` and `p99_ttft_ms`
- `avg_peak_kv_memory_mb`
- `n_oom`
- `per_bucket`
- MCQ accuracy for MMLU/LongBench
- ROUGE-L for GovReport

## Important Caveat

This runner replays requests sequentially and estimates queue wait from the
configured arrival times. It is useful for deterministic single-server workload
comparison, but it is not a true continuous-batching server benchmark.

The `very_long` bucket currently uses LongBench short-answer MCQ prompts, so
its generation budget is 10 tokens. GovReport buckets provide the long-output
summarization requests.
