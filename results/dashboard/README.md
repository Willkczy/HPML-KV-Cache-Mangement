# Experiment Results Dashboard

Static export of all experiment runs. Source of truth: JSON files in `results/`.

---

## Experiment 1 — Static Single-Request Benchmark

**Hardware:** NVIDIA RTX L40S (48 GB) · **Model:** Qwen2.5-7B · **dtype:** float16

### MMLU Short Answer (128–512 tokens, 10 output tokens)

| Method | Config | Accuracy | Avg TTFT | Throughput | Decode KV Mem |
|--------|--------|----------|----------|------------|---------------|
| full_cache | — | 56.7% | 44.7 ms | 40.2 tok/s | 23.8 MB |
| h2o | hh=64, r=64 | 56.7% | 45.1 ms | 29.9 tok/s | 7.0 MB |
| streaming_llm | recent=64 | 56.7% | 44.7 ms | 40.4 tok/s | 3.7 MB |
| paged_attention | eager, v0 | 57.3% | 37.1 ms | 43.5 tok/s | 23.8 MB |

### LongBench Short Answer (8k–16k tokens, 10 output tokens)

| Method | Config | Accuracy | Avg TTFT | Throughput | Decode KV Mem |
|--------|--------|----------|----------|------------|---------------|
| full_cache | — | 40.9% | 1435 ms | 4.6 tok/s | 732.8 MB |
| h2o | hh=64, r=64 | 40.9% | 1395 ms | 4.0 tok/s | 7.0 MB |
| streaming_llm | recent=64 | 40.9% | 1383 ms | 5.1 tok/s | 3.7 MB |
| paged_attention | eager, v0 | 40.9% | 1198 ms | 5.4 tok/s | 732.9 MB |

### GovReport Summarization (4k–16k tokens, 512 output tokens)

| Method | Config | ROUGE-L | Avg TTFT | Throughput | KV Mem |
|--------|--------|---------|----------|------------|--------|
| full_cache | — | 0.1859 | 960 ms | 36.4 tok/s | 525.4 MB |
| h2o | hh=64, r=64 | 0.0000 | 932 ms | 16.0 tok/s | 7.0 MB |
| streaming_llm | recent=4096 | 0.1832 | 967 ms | 35.1 tok/s | 224.2 MB |
| paged_attention | eager, v0 | 0.1850 | 845 ms | 40.0 tok/s | 525.8 MB |

### StreamingLLM Window Sweep — GovReport

| recent_size | KV Mem | ROUGE-L | vs baseline |
|-------------|--------|---------|-------------|
| 1024 | 56.2 MB | 0.1292 | −30% |
| 2048 | 112.2 MB | 0.1583 | −15% |
| 4096 | 224.2 MB | 0.1832 | −1.5% |
| full_cache | 525.4 MB | 0.1859 | baseline |

---

## Experiment 2 — Concurrent Serving Benchmark

**Hardware:** NVIDIA RTX A6000 (48 GB) · **vLLM:** 0.8.5 · **Requests:** 500 · **Arrivals:** Poisson

### PagedAttention: v0/v1 × graph/eager

| Config | Rate (req/s) | ReqThr | P50 TTFT | P99 E2E | Peak KV Util |
|--------|-------------|--------|----------|---------|--------------|
| v0_eager | 0.5 | 0.506 | 0.47s | 25.2s | 18.7% |
| v0_eager | 1.0 | 0.990 | 1.39s | 60.5s | 33.9% |
| v0_eager | 2.0 | 1.416 | 29.7s | 202.8s | 79.7% |
| v0_eager | 4.0 | 1.434 | 91.0s | 256.7s | 79.7% |
| v0_graph | 2.0 | 1.292 | 44.6s | 230.5s | 79.7% |
| v0_graph | 4.0 | 1.296 | 108.8s | 294.3s | 79.7% |
| v1_graph | 2.0 | 1.146 | 61.2s | 274.5s | N/A† |
| v1_graph | 4.0 | 1.147 | 122.4s | 345.6s | N/A† |
| v1_eager | 2.0 | 1.139 | 62.9s | 276.0s | N/A† |
| v1_eager | 4.0 | 1.142 | 124.6s | 347.0s | N/A† |

† vLLM v1 engine runs as a subprocess; KV block manager not accessible in offline mode.

---

## Raw Result Files

| Path | Description |
|------|-------------|
| `results/exp1_short/` | MMLU results per method |
| `results/exp2_long/` | LongBench short answer results |
| `results/exp2_summarization/` | GovReport results |
| `results/experiment_mixed_workload/mixed_*_full.json` | Sequential mixed workload |
| `results/experiment_mixed_workload/serving_*.json` | Concurrent serving results |

Full analysis: [STATIC_BENCHMARK_RESULTS.md](../../STATIC_BENCHMARK_RESULTS.md) · [SERVING_RESULTS.md](../../SERVING_RESULTS.md)