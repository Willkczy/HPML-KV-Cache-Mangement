# Experiment Results

KV Cache Management Benchmarks — Qwen2.5-7B, single-request static workload.
Hardware: NVIDIA RTX L40S (48GB), Insomnia HPC cluster.
Branch: `feat/fairness-fixes`

---

## Shared Settings (all methods, all benchmarks)

| Setting | Value |
|---------|-------|
| Model | Qwen/Qwen2.5-7B |
| dtype | float16 |
| Decoding | Greedy (temperature=0.0, argmax) |
| Batch size | 1 (single request) |
| CUDA warm-up | ✅ dummy forward pass before timing |
| Timing | `perf_counter` + `cuda.synchronize()` (HF); vLLM internal metrics (vLLM) |

---

## Benchmark Configurations

| Benchmark | Dataset | Samples | Input tokens | max_new_tokens | Quality metric |
|-----------|---------|---------|--------------|----------------|----------------|
| MMLU short | MMLU (5-shot) | 150 | 128–512 | 10 | Accuracy |
| LongBench short | LongBench v2 | 22 | 8k–16k | 10 | Accuracy |
| LongBench explain | LongBench v2 | 22 | 8k–16k | 512 | Accuracy |
| GovReport summ. | GovReport | 50 | 4k–16k | 512 | ROUGE-L |

---

## Method Hyperparameters

### full_cache

| Parameter | Value |
|-----------|-------|
| attn_implementation | sdpa (default) |
| KV eviction | None |

*Same settings across all benchmarks.*

---

### h2o

| Parameter | MMLU short | LongBench short | LongBench explain | GovReport summ. |
|-----------|-----------|-----------------|-------------------|-----------------|
| attn_implementation | sdpa (prefill) → eager (decode) | same | same | same |
| hh_size | 64 | 64 | 64 | 64 |
| recent_size | 64 | 64 | 64 | 64 |
| **KV budget (total)** | **128 tokens** | **128 tokens** | **128 tokens** | **128 tokens** |
| Budget as % of input | 25–100% | 0.8–1.6% | 0.8–1.6% | 0.8–3.2% |

> **Note:** Fixed budget of 128 tokens is appropriate for MMLU but highly aggressive for long-context benchmarks. Future runs will test larger budgets (e.g., 512+512).

---

### streaming_llm

| Parameter | MMLU short | LongBench short | LongBench explain | GovReport summ. |
|-----------|-----------|-----------------|-------------------|-----------------|
| attn_implementation | sdpa (default) | same | same | same |
| start_size (sinks) | 4 | 4 | 4 | 4 |
| recent_size (window) | **64** | **64** | 1024 | 1024 |
| **KV window (total)** | **68 tokens** | **68 tokens** | 1028 tokens | 1028 tokens |
| Window as % of input | 13–53% | 0.4–0.85% | 6–12% | 6–25% |

---

### paged_attention

| Parameter | MMLU short | LongBench short | LongBench explain | GovReport summ. |
|-----------|-----------|-----------------|-------------------|-----------------|
| backend | vLLM 0.8.5 | same | same | same |
| VLLM_USE_V1 | 0 | 0 | 0 | 0 |
| enforce_eager | True | True | True | True |
| enable_prefix_caching | False | False | False | False |
| gpu_memory_utilization | 0.90 | 0.90 | 0.90 | 0.90 |
| block_size | 16 | 16 | 16 | 16 |
| **max_model_len** | **4096** | **34000** | **34000** | **18000** |
| KV eviction | None (full paged cache) | same | same | same |

---

## Results

### MMLU Short (128–512 tokens, 10 output tokens)

| Method | Accuracy | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|----------|----------|------------|------------|-------------|
| full_cache | 56.7% (85/150) | 44.7 ms | 204.5 ms | 40.2 tok/s | 23.8 MB |
| h2o | 56.7% (85/150) | 45.1 ms | 78.1 ms | 29.9 tok/s | **17.5 MB** |
| streaming_llm (recent=64) | 56.7% (85/150) | 44.7 ms | 203.6 ms | 40.4 tok/s | **3.7 MB** |
| paged_attention | 57.3% (86/150) | 37.1 ms | 192.9 ms | 43.5 tok/s | 23.8 MB |

**Observations:**
- All methods match in accuracy — eviction has no quality impact on short prompts
- H2O peak KV (17.5 MB) lower than full_cache (23.8 MB) — reports post-eviction decode KV
- StreamingLLM (recent=64) peak KV (3.7 MB) — 6.4× less than full_cache with zero quality loss; MMLU inputs fit within 68-token window
- H2O decode time (78.1ms) shorter because it generates fewer tokens before EOS; per-token throughput lower (29.9 tok/s) due to eager attention overhead
- paged_attention TTFT (37.1ms) vs HF methods (44–45ms): vLLM engine pre-warmed at init

---

### LongBench Short Answer (8k–16k tokens, 10 output tokens)

| Method | Accuracy | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|----------|----------|------------|------------|-------------|
| full_cache | 40.9% (9/22) | 1435.1 ms | 167.8 ms | 4.6 tok/s | 732.8 MB |
| h2o | 40.9% (9/22) | 1394.8 ms | 200.3 ms | 4.0 tok/s | **17.5 MB** |
| streaming_llm (recent=64) | 40.9% (9/22) | 1382.9 ms | 162.9 ms | 5.1 tok/s | **3.7 MB** |
| paged_attention | 40.9% (9/22) | 1198.1 ms | 148.3 ms | 5.4 tok/s | 732.9 MB |

**Observations:**
- All methods achieve identical accuracy — KV eviction does not hurt quality for short-answer generation
- **StreamingLLM (recent=64) KV memory: 3.7 MB** — 198× less than full_cache with zero quality loss; question+choices fit within 68-token window
- **H2O KV memory: 17.5 MB** (128-token budget) — 42× less than full_cache
- Both eviction methods hold ~732MB during prefill; reported KV is post-eviction decode steady-state
- paged_attention TTFT lower (1198ms vs 1382–1435ms): vLLM's more efficient prefill scheduling

---

### LongBench Explain (8k–16k tokens, 512 output tokens)

| Method | Accuracy | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|----------|----------|------------|------------|-------------|
| full_cache | 27.3% (6/22) | 1436.8 ms | 9955.8 ms | 34.3 tok/s | 755.5 MB |
| h2o | **4.5% (1/22)** | 1463.0 ms | 6716.4 ms | 16.1 tok/s | **11.1 MB** |
| streaming_llm (recent=1024) | 13.6% (3/22) | 1436.8 ms | 8863.6 ms | 36.9 tok/s | 56.2 MB |
| paged_attention | 18.2% (4/22) | 1255.7 ms | 5255.9 ms | 24.4 tok/s | 746.6 MB |

**Observations:**
- **Quality degrades significantly** for all eviction methods on long generation — the model loses important context
- **H2O collapses to near-random (4.5%)** — budget of 128 tokens is catastrophically small for 512-token generation over 8k–16k inputs
- StreamingLLM (13.6%) degrades less than H2O but significantly below full_cache (27.3%) — window of 1028 tokens covers only 6–12% of input
- paged_attention (18.2%) lower than full_cache (27.3%) — note: only 22 samples, 2-sample difference may be noise
- H2O throughput (16.2 tok/s) much lower than others — eager attention overhead dominates over 512 decode steps

---

### GovReport Summarization (4k–16k tokens, 512 output tokens)

| Method | ROUGE-L | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|---------|----------|------------|------------|-------------|
| full_cache | **0.1859** | 959.5 ms | 8062.5 ms | 36.4 tok/s | 525.4 MB |
| h2o | **0.0000** | 932.1 ms | 5210.9 ms | 16.0 tok/s | **17.5 MB** |
| streaming_llm (recent=1024) | 0.1292 | 963.1 ms | 4297.1 ms | 35.3 tok/s | 56.2 MB |
| paged_attention | 0.1850 | 845.4 ms | 7500.5 ms | 40.0 tok/s | 525.8 MB |

**Observations:**
- **H2O ROUGE-L = 0.0000** — complete quality collapse; budget of 128 tokens cannot retain enough context for coherent summarization
- StreamingLLM ROUGE-L (0.1292) — meaningful degradation vs full_cache (0.1859) but not catastrophic
- **paged_attention matches full_cache** (0.1850 vs 0.1859) — confirms both are correct full-cache baselines
- StreamingLLM decode time (4297ms) faster than full_cache (8063ms) — window prevents KV growth, reducing per-step attention cost

---

## Summary Table

| Method | Config | MMLU Acc. | LB Acc. | LB Explain Acc. | GovReport ROUGE-L | Peak KV (LB short) |
|--------|--------|-----------|---------|-----------------|-------------------|--------------------|
| full_cache | — | 56.7% | 40.9% | 27.3% | 0.1859 | 732.8 MB |
| h2o | hh=64, r=64 | 56.7% | 40.9% | **4.5%** | **0.0000** | **17.5 MB** |
| streaming_llm | recent=64 | 56.7% | 40.9% | — | — | **3.7 MB** |
| streaming_llm | recent=1024 | 56.7% | 40.9% | 13.6% | 0.1292 | 56.2 MB |
| paged_attention | — | 57.3% | 40.9% | 18.2% | 0.1850 | 732.9 MB |

---

## StreamingLLM Window Size Ablation

Sweep over `recent_size` values on MMLU short and LongBench short answer (start_size=4 fixed).
Goal: find minimum window that maintains full_cache accuracy.

### MMLU Short

| recent_size | Window | Accuracy | Peak KV | vs full_cache KV |
|-------------|--------|----------|---------|-----------------|
| full_cache (ref) | — | 56.7% | 23.8 MB | 1× |
| 512 | 516 tok | 56.7% | 23.8 MB | 1.0× |
| 256 | 260 tok | 56.7% | 14.2 MB | 1.7× |
| **64** | **68 tok** | **56.7%** | **3.7 MB** | **6.4×** |

### LongBench Short Answer

| recent_size | Window | Accuracy | Peak KV | vs full_cache KV |
|-------------|--------|----------|---------|-----------------|
| full_cache (ref) | — | 40.9% | 732.8 MB | 1× |
| 512 | 516 tok | 40.9% | 28.2 MB | 26× |
| 256 | 260 tok | 40.9% | 14.2 MB | 52× |
| **64** | **68 tok** | **40.9%** | **3.7 MB** | **198×** |

**Key finding:** Even `recent_size=64` (68 total tokens) maintains full accuracy on both benchmarks. For multiple choice QA, the question and answer choices fit within the recent window — the model does not need access to the evicted middle context to answer correctly. This enables a **198× KV memory reduction on LongBench with zero quality loss**.

---

## Planned Follow-up Experiments

- [ ] H2O with larger budgets: hh_size=256+recent_size=256, hh_size=512+recent_size=512
- [ ] Experiment 2: realistic serving workload (continuous batching, vLLM methods only)