# Realistic Serving Workload Results

Concurrent serving benchmark using `AsyncLLMEngine` — true continuous batching where multiple requests are in-flight simultaneously. Results compare PagedAttention (vLLM v0 vs v1) across 4 arrival rates.

Hardware: NVIDIA RTX A6000 (48GB), Insomnia HPC cluster.  
Branch: `feat/fairness-realistic-workload`

---

## Workload Design

| Bucket | Ratio | Count | Dataset | Prompt Tokens | Max New Tokens |
|--------|------:|------:|---------|-------------:|---------------:|
| short | 50% | 250 | MMLU | 128–512 | 8 |
| medium | 30% | 150 | GovReport | 1k–4k | 256 |
| long | 15% | 75 | GovReport | 4k–16k | 512 |
| very_long | 5% | 25 | LongBench v2 | 16k–32k | 10 |

Total: 500 requests, Poisson arrivals, seed=42. Arrival rate swept: 0.5 / 1.0 / 2.0 / 4.0 req/s. Each rate uses a fresh engine instance for independent results.

---

## Engine Configurations

| Config | Engine | CUDA Graphs | Chunked Prefill | Prefix Caching |
|--------|--------|-------------|-----------------|----------------|
| v0_graph | vLLM v0 | ✅ On (default) | ❌ Off (default) | ❌ Off |
| v0_eager | vLLM v0 | ❌ Off | ❌ Off (default) | ❌ Off |
| v1_graph | vLLM v1 | ✅ On (default) | ✅ On (2048 tok/chunk, default) | ❌ Off |
| v1_eager | vLLM v1 | ❌ Off | ✅ On (2048 tok/chunk, default) | ❌ Off |

All configs: `dtype=float16`, `max_model_len=34000`, `gpu_memory_utilization=0.90`, `block_size=16`.

> **KV utilization note:** v0 reports block-level utilization via block manager. v1 reports total GPU memory (nvidia-smi fallback) because the v1 engine runs in a subprocess — block manager is not accessible from the parent process in offline mode.

---

## Results

### paged_attention v0_graph

| Rate | ReqThr | TokThr | P50 TTFT | P95 TTFT | P99 TTFT | P50 E2E | P95 E2E | P99 E2E | Peak KV | Avg KV | MCQ Acc | ROUGE-L |
|-----:|-------:|-------:|---------:|---------:|---------:|--------:|--------:|--------:|--------:|-------:|--------:|--------:|
| 0.5 | 0.506 | 58.5 | 0.56s | 5.87s | 7.48s | 4.6s | 17.9s | 30.4s | 18.7% | 3.5% | 56.7% | 0.1946 |
| 1.0 | 0.987 | 113.7 | 2.19s | 8.93s | 14.67s | 8.0s | 53.5s | 75.5s | 40.9% | 13.6% | 56.7% | 0.1948 |
| 2.0 | 1.292 | 148.3 | 44.59s | 114.34s | 121.80s | 97.4s | 190.8s | 230.5s | 79.7% | 57.5% | 56.7% | 0.1949 |
| 4.0 | 1.296 | 148.2 | 108.82s | 231.01s | 243.76s | 161.9s | 273.3s | 294.3s | 79.7% | 58.1% | 56.7% | 0.1945 |

### paged_attention v0_eager

| Rate | ReqThr | TokThr | P50 TTFT | P95 TTFT | P99 TTFT | P50 E2E | P95 E2E | P99 E2E | Peak KV | Avg KV | MCQ Acc | ROUGE-L |
|-----:|-------:|-------:|---------:|---------:|---------:|--------:|--------:|--------:|--------:|-------:|--------:|--------:|
| 0.5 | 0.506 | 58.9 | 0.47s | 5.12s | 6.29s | 4.4s | 16.5s | 25.2s | 18.7% | 3.3% | 56.7% | 0.1951 |
| 1.0 | 0.990 | 114.2 | 1.39s | 7.23s | 12.58s | 6.9s | 36.9s | 60.5s | 33.9% | 11.0% | 56.7% | 0.1945 |
| 2.0 | 1.416 | 163.7 | 29.67s | 81.91s | 87.74s | 72.0s | 158.1s | 202.8s | 79.7% | 56.7% | 56.7% | 0.1954 |
| 4.0 | 1.434 | 163.8 | 91.03s | 194.70s | 205.54s | 137.2s | 236.4s | 256.7s | 79.7% | 57.8% | 56.7% | 0.1944 |

### paged_attention v1_graph

| Rate | ReqThr | TokThr | P50 TTFT | P95 TTFT | P99 TTFT | P50 E2E | P95 E2E | P99 E2E | Peak KV | Avg KV | MCQ Acc | ROUGE-L |
|-----:|-------:|-------:|---------:|---------:|---------:|--------:|--------:|--------:|--------:|-------:|--------:|--------:|
| 0.5 | 0.507 | 59.1 | 0.48s | 5.30s | 6.44s | 4.4s | 17.5s | 26.1s | N/A | N/A | 56.7% | 0.1957 |
| 1.0 | 0.991 | 115.2 | 1.80s | 8.02s | 14.09s | 7.3s | 50.3s | 73.0s | N/A | N/A | 56.7% | 0.1954 |
| 2.0 | 1.146 | 132.7 | 61.16s | 162.38s | 173.32s | 121.7s | 243.1s | 274.5s | N/A | N/A | 56.7% | 0.1944 |
| 4.0 | 1.147 | 132.6 | 122.44s | 278.86s | 295.69s | 184.7s | 324.7s | 345.6s | N/A | N/A | 56.7% | 0.1944 |

### paged_attention v1_eager

| Rate | ReqThr | TokThr | P50 TTFT | P95 TTFT | P99 TTFT | P50 E2E | P95 E2E | P99 E2E | Peak KV | Avg KV | MCQ Acc | ROUGE-L |
|-----:|-------:|-------:|---------:|---------:|---------:|--------:|--------:|--------:|--------:|-------:|--------:|--------:|
| 0.5 | 0.507 | 58.3 | 0.48s | 5.39s | 6.44s | 4.4s | 16.9s | 26.3s | N/A | N/A | 56.7% | 0.1941 |
| 1.0 | 0.991 | 114.5 | 1.79s | 7.89s | 14.02s | 7.3s | 50.8s | 66.6s | N/A | N/A | 56.7% | 0.1951 |
| 2.0 | 1.139 | 132.1 | 62.93s | 165.92s | 176.21s | 125.1s | 245.9s | 276.0s | N/A | N/A | 56.7% | 0.1956 |
| 4.0 | 1.142 | 132.8 | 124.60s | 280.97s | 297.48s | 190.3s | 326.3s | 347.0s | N/A | N/A | 56.7% | 0.1959 |

---

## Key Findings

### 1. v0_eager is the best performer — CUDA graphs hurt heterogeneous workloads

At saturation (2.0 req/s), v0_eager achieves 1.416 req/s vs v0_graph's 1.292 req/s (**+10% throughput**). Our workload has extreme length variance (128 to 32k tokens). CUDA graphs require fixed batch shapes — when shapes don't match a captured graph, vLLM falls back to eager execution while also paying graph-lookup overhead.

### 2. v1 underperforms v0 — chunked prefill default is the bottleneck

At 2.0 req/s, v0_eager P50 TTFT = 29.67s vs v1_graph P50 TTFT = 61.16s (**2× worse**). vLLM v1 enables chunked prefill by default (`max_num_batched_tokens=2048`). A 16k-token prompt requires 8 scheduling steps in v1 vs 1 step in v0. Each step involves IPC to the engine subprocess, compounding for long requests (15–20% of our workload).

### 3. v1_graph ≈ v1_eager — CUDA graphs irrelevant for v1 here

At 2.0 req/s: v1_graph TokThr=132.7 vs v1_eager=132.1. The chunked prefill overhead dominates; CUDA graphs add no benefit on top.

### 4. Quality identical across all configurations

56.7% MCQ accuracy and ~0.195 ROUGE-L across all configs and all rates. Architecture choice affects latency/throughput only — correctness is unaffected.

### 5. KV pressure (v0 only)

Saturation at 79.7% block utilization for both v0 configs at 2.0+ req/s — the KV pool becomes the bottleneck. At 0.5 req/s, only 18.7% utilization showing headroom.

---

## Limitations

- **streaming_llm_vllm concurrent serving**: Patches cause CUDA illegal memory access due to race condition between block freeing and concurrent kernel execution. Sequential serving results remain valid.
- **v1 KV utilization**: Not measurable in offline mode — v1 engine runs as a subprocess; block manager is inaccessible from the parent process. Production measurement requires Prometheus scraping via the API server.
- **Single GPU (A6000 48GB)**: v1's improvements are designed primarily for multi-GPU deployments with tensor/pipeline parallelism.
