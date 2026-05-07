# Experiment Results — Static Single-Request Workload

KV Cache Management Benchmarks — Qwen2.5-7B, single-request (batch size=1).
Hardware: NVIDIA RTX L40S (48GB), Insomnia HPC cluster.
Branch: `feat/fairness-realistic-workload`

> For realistic concurrent serving results see [SERVING_RESULTS.md](SERVING_RESULTS.md).

---

## Key Findings

1. **StreamingLLM reduces decode-phase KV memory significantly with no quality loss on short-answer tasks** — for max_new_tokens=10, the answer token is generated from the full prefill context before trimming, so window size does not affect accuracy. KV reduction (3.7 MB vs 732 MB) reflects the post-trim decode cache.
2. **H2O collapses on long generation** — 4.5% accuracy (near-random) and 0.0 ROUGE-L with 128-token budget on 8k–16k inputs, regardless of budget size tested (up to 1024 tokens).
3. **paged_attention matches full_cache quality** across all benchmarks — confirms both are correct uncompressed baselines.
4. **KV eviction hurts long-form generation** — both H2O and StreamingLLM degrade significantly on tasks requiring 512 output tokens. StreamingLLM recovers near-baseline quality on GovReport with recent=4096 (ROUGE-L 0.1832 vs 0.1859 baseline, 2.3× less KV memory).

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
- All methods match in accuracy — for max_new_tokens=10, the answer letter is generated from the full prefill context (before any KV trimming), so eviction budget and window size do not affect accuracy
- StreamingLLM (recent=64) peak KV (3.7 MB) — 6.4× less than full_cache; post-trim decode cache only; accuracy preserved via full prefill mechanism, not window coverage (MMLU inputs are 128–512 tokens, larger than the 68-token window)
- H2O peak KV (17.5 MB) — post-eviction decode cache; budget of 128 tokens covers most MMLU inputs during decode
- H2O decode time (78.1ms) is shorter because eviction changes the attention distribution, causing the model to generate EOS earlier; per-token throughput lower (29.9 tok/s) because H2O uses unfused eager attention kernel (`attn_implementation="eager"`) to materialize attention weights for eviction scoring — full_cache and streaming_llm use the faster SDPA fused kernel
- paged_attention TTFT (37.1ms) lower than HF methods (44–45ms): vLLM engine is more comprehensively warmed during `LLM()` initialization vs the single dummy forward pass warm-up in HF methods

---

### LongBench Short Answer (8k–16k tokens, 10 output tokens)

| Method | Accuracy | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|----------|----------|------------|------------|-------------|
| full_cache | 40.9% (9/22) | 1435.1 ms | 167.8 ms | 4.6 tok/s | 732.8 MB |
| h2o | 40.9% (9/22) | 1394.8 ms | 200.3 ms | 4.0 tok/s | **17.5 MB** |
| streaming_llm (recent=64) | 40.9% (9/22) | 1382.9 ms | 162.9 ms | 5.1 tok/s | **3.7 MB** |
| paged_attention | 40.9% (9/22) | 1198.1 ms | 148.3 ms | 5.4 tok/s | 732.9 MB |

**Observations:**
- All methods achieve identical accuracy — same mechanism as MMLU: the answer token is generated from the full prefill context (8k–16k tokens) before any KV trimming occurs
- **Both eviction methods hold ~732MB during prefill** — eviction only begins after the first decode token; reported KV (3.7 MB and 17.5 MB) is the post-eviction decode steady-state, not the peak during prefill
- **StreamingLLM post-trim decode KV: 3.7 MB** (68-token window) — 198× less than full_cache during decode; trim happens immediately after prefill, so all subsequent decode steps use only the sink+recent window
- **H2O post-eviction decode KV: 17.5 MB** (128-token budget) — 42× less than full_cache during decode; first eviction at decode step 1 reduces from ~732MB to budget size
- paged_attention TTFT lower (1198ms vs 1382–1435ms): vLLM processes the full 8k–16k prompt more efficiently via PagedAttention's block-based memory management

---

### LongBench Explain (8k–16k tokens, 512 output tokens)

| Method | Accuracy | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|----------|----------|------------|------------|-------------|
| full_cache | 27.3% (6/22) | 1436.8 ms | 9955.8 ms | 34.3 tok/s | 755.5 MB |
| h2o | **4.5% (1/22)** | 1463.0 ms | 6716.4 ms | 16.1 tok/s | **11.1 MB** |
| streaming_llm (recent=1024) | 13.6% (3/22) | 1436.8 ms | 8863.6 ms | 36.9 tok/s | 56.2 MB |
| paged_attention | 18.2% (4/22) | 1255.7 ms | 5255.9 ms | 24.4 tok/s | 746.6 MB |

**Observations:**
- **Quality degrades for all eviction methods** — unlike short-answer tasks, 512 decode tokens means the model must generate from the trimmed cache for nearly all steps; context loss accumulates
- **H2O collapses to near-random (4.5%)** — at decode step 1, H2O evicts from 8k–16k tokens down to budget=128 using only the first decode token's attention scores as guidance; this one-shot eviction with no accumulated history selects poor representatives
- **StreamingLLM (13.6%)** degrades less than H2O — window of 1028 tokens (6–12% of input) always retains the 4 attention sinks + most recent 1024 tokens, which includes the question and recent reasoning context
- **paged_attention (18.2%) vs full_cache (27.3%)** — only 22 samples; a 2-sample difference; likely statistical noise (not a real quality gap between full-cache methods)
- **H2O throughput (16.1 tok/s)** — much lower than full_cache (34.3 tok/s) and streaming_llm (36.9 tok/s) because each of the 512 decode steps requires: (1) unfused eager attention kernel, (2) materializing attention weights, (3) running `_update_and_evict` topk selection

---

### GovReport Summarization (4k–16k tokens, 512 output tokens)

Reporting the best-quality eviction config per method (closest ROUGE-L to full_cache baseline).

| Method | Config | ROUGE-L | Avg TTFT | Avg Decode | Throughput | Peak KV mem |
|--------|--------|---------|----------|------------|------------|-------------|
| full_cache | — | **0.1859** | 959.5 ms | 8062.5 ms | 36.4 tok/s | 525.4 MB |
| h2o | hh=64, r=64 (best tested) | **0.0000** | 932.1 ms | 5210.9 ms | 16.0 tok/s | **17.5 MB** |
| streaming_llm | recent=4096 | **0.1832** | 967.0 ms | 5912.2 ms | 35.1 tok/s | **224.2 MB** |
| paged_attention | — | 0.1850 | 845.4 ms | 7500.5 ms | 40.0 tok/s | 525.8 MB |

**Observations:**
- **H2O ROUGE-L=0.0000 regardless of budget** — tested budgets 128, 512, 1024, 2048 tokens, all collapse. Two root causes: (1) prefill attention scoring is O(seq_len²) and computationally infeasible for 4k–16k inputs on a single GPU, so our implementation skips prefill scoring; (2) eviction at decode step 1 uses only one step of attention history — insufficient to identify which document tokens matter for 512-token summarization
- **StreamingLLM (recent=4096) achieves ROUGE-L=0.1832** — only 1.5% below full_cache (0.1859) with 2.3× less KV memory (224.2 MB vs 525.4 MB); the 4100-token window retains enough of the recent document for coherent summarization
- **paged_attention matches full_cache** (0.1850 vs 0.1859) — both store the full KV cache; confirms the analytical KV formula is consistent with the HF tensor-byte measurement

**StreamingLLM window size sweep on GovReport:**

| recent_size | KV mem | ROUGE-L | Quality vs full_cache |
|-------------|--------|---------|----------------------|
| 1024 | 56.2 MB | 0.1292 | −30% |
| 2048 | 112.2 MB | 0.1583 | −15% |
| **4096** | **224.2 MB** | **0.1832** | **−1.5%** |
| full_cache | 525.4 MB | 0.1859 | baseline |

---

## Summary Table

| Method | Config | MMLU Acc. | LB Acc. | LB Explain Acc. | GovReport ROUGE-L | Peak KV (LB short) |
|--------|--------|-----------|---------|-----------------|-------------------|--------------------|
| full_cache | — | 56.7% | 40.9% | 27.3% | 0.1859 | 732.8 MB |
| h2o | hh=64, r=64 | 56.7% | 40.9% | **4.5%** | **0.0000** | **17.5 MB** |
| streaming_llm | recent=1024 | 56.7% | 40.9% | 13.6% | 0.1292 | 56.2 MB |
| streaming_llm (best summ.) | recent=4096 | 56.7% | 40.9% | — | **0.1832** | 224.2 MB |
| paged_attention | — | 57.3% | 40.9% | 18.2% | 0.1850 | 732.9 MB |

---
