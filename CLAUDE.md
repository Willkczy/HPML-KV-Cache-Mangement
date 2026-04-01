# CLAUDE.md — KV Cache Management for LLM Inference

Columbia University HPML Final Project (Spring Semester)

## What This Project Is

A head-to-head benchmark of four KV cache management strategies for transformer inference, using **Qwen2.5-7B** as the shared model. The goal is to measure memory efficiency, latency, throughput, and generation quality trade-offs across realistic context lengths.

## Team & Ownership

| Method | Owner | Branch |
|--------|-------|--------|
| Full KV Cache (HuggingFace baseline) | Hung-Kai Huang | `feat/full-cache` |
| PagedAttention (vLLM) | Kane Wang (Yipeng Wang) | `feat/paged-attention` |
| H2O (official impl) | **Sripad Karne** | `feat/h2o` |
| StreamingLLM (official impl) | Ting-Feng Huang | `feat/streaming-llm` |

**Sripad owns the H2O method** (`methods/h2o.py`) and contributes to data pipeline / eval.

## Project Structure

```
HPML-KV-Cache-Mangement/
├── methods/            ← one file per KV cache strategy
│   ├── base.py         ← BaseMethod ABC + MethodOutput dataclass (key interface)
│   ├── full_cache.py
│   ├── paged_attention.py
│   ├── h2o.py          ← Sripad's ownership
│   └── streaming_llm.py
├── data/
│   └── pipeline.py     ← loads datasets → returns list[Sample]
├── eval/
│   └── metrics.py      ← accuracy, ROUGE-L, perplexity, latency stats
├── scripts/
│   └── run_experiment1.py
├── configs/
│   └── experiment1_short.yaml   ← MMLU, 128-512 tokens, greedy decode
└── results/            ← gitignored; upload to GCS
```

## Core Interfaces (never break these)

### MethodOutput — what every method returns
```python
@dataclass
class MethodOutput:
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float           # time to first token
    total_time_ms: float     # end-to-end
    decode_latency_ms: float # total - ttft
    peak_kv_memory_mb: float # peak KV cache GPU memory
    metadata: dict           # method-specific extras
```

### BaseMethod — what every method implements
```python
class BaseMethod(ABC):
    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None: ...
    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput: ...
    def teardown(self) -> None: ...  # optional
```

### Sample — what the data pipeline returns
```python
@dataclass
class Sample:
    id: str           # unique identifier
    prompt: str       # formatted, ready for model
    reference: str    # ground truth
    dataset: str      # "mmlu" | "cnn_dailymail" | "longbench"
    subset: str       # e.g. MMLU subject
    token_count: int  # pre-computed
    metadata: dict
```

## Experiments

### Experiment 1 — Controlled Long Context (single request, no batching)

| Bucket | Dataset | Token Range | Quality Metric |
|--------|---------|-------------|----------------|
| Short | MMLU | 128–256 | Accuracy |
| Medium | CNN/DailyMail | 512–1024 | ROUGE-L |
| Long | LongBench | 2048+ | Task-specific / perplexity |

**Metrics:** Peak KV memory, TTFT, decode latency, throughput (tok/s), quality

### Experiment 2 — Realistic Serving Workload (continuous batching)

Mixed request workload: 20% ~1k, 30% ~2k, 30% ~4k, 20% ~8k+ tokens with long RAG-style prompts.

**Additional metrics:** P95/P99 latency, goodput, memory under load, throughput stability

## Running Experiments

```bash
# Short bucket (MMLU) — H2O method
python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method h2o
```

Configs drive everything: model path, dataset, token range, generation params, output dir.

## Infrastructure

- **Compute:** GCP VMs (A100 40GB or L4), one per team member for dev
- **Official benchmarks:** All methods run sequentially on a single standardized VM (A100) to eliminate hardware variance
- **Model weights:** `~/models/qwen2.5-7b` (never committed)
- **Results:** Local `results/` → upload to shared GCS bucket with `gsutil cp`
- **Profiling:** PyTorch Profiler + NVIDIA Nsight Systems

## Git Rules

- Branch off `dev`: `git checkout -b feat/your-feature dev`
- Never push directly to `main` or `dev` — always open a PR
- PRs require 1 approval before merging to `dev`
- `dev → main` only before official experiment runs
- Rebase on `dev` before opening PR: `git pull --rebase origin dev`
- Commit format: `feat(h2o): add cumulative attention score eviction`

## What NOT to Commit

- Model weights or checkpoints
- Large data files (use HuggingFace `datasets` at runtime)
- Profiling traces (`.nsys-rep`, PyTorch trace JSONs) — upload to GCS
- `.venv/`, `__pycache__/`, anything in `results/`

## H2O Method Notes (Sripad's component)

H2O (Heavy-Hitter Oracle) evicts KV cache entries based on cumulative attention scores:
- Tracks attention scores across all layers and heads
- Preserves "heavy hitter" tokens (high cumulative attention) + recent context window
- Target: 40-60% memory reduction with controlled quality trade-offs
- Reference: Zhang et al., NeurIPS 2023 — official implementation repo

## Timeline (as of March 2026)

| Date | Milestone |
|------|-----------|
| 3/29 | Implement all four methods on MMLU (past) |
| 4/5 | Finish implementation + complete Experiment 1 |
| 4/12 | Finalize report draft + start Experiment 2 |
| 4/13 | Mid-point project report due |
| 5/4 | Final project presentation |
