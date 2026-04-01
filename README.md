# KV Cache Management for LLM Inference

A comparative study of four KV cache management strategies under realistic serving workloads, using **Qwen2.5-7B** as the primary model.

## Why This Matters

KV cache is the dominant memory bottleneck during LLM inference. As context lengths grow and concurrent requests increase, naive full-cache approaches exhaust GPU memory quickly. This project benchmarks four strategies that make different tradeoffs between memory efficiency, latency, and output quality.

## Methods

| Method | Strategy | Implementation | Owner |
|--------|----------|---------------|-------|
| **Full Cache** | Store all KV pairs (baseline) | HuggingFace Transformers | Hung-Kai Huang |
| **PagedAttention** | Block-based non-contiguous allocation | vLLM | Kane Wang |
| **H2O** | Attention-driven selective eviction | Official H2O repo | Sripad Karne |
| **StreamingLLM** | Sliding window + attention sinks | Official StreamingLLM repo | Ting-Feng Huang |

Every method implements the same `BaseMethod` interface (see `methods/base.py`), taking identical inputs and returning a standardized `MethodOutput` so that data loading, experiment orchestration, and evaluation are fully shared.

## Project Structure

```
kv-cache-bench/
├── README.md                  ← you are here
├── CONTRIBUTING.md            ← branching rules, GCP setup, collaboration guide
├── requirements.txt
├── .gitignore
│
├── configs/                   ← experiment configs (YAML)
│   └── experiment1_short.yaml
│
├── data/                      ← shared data pipeline
│   ├── __init__.py
│   └── pipeline.py            ← loads datasets, returns standardized samples
│
├── methods/                   ← one file per KV cache strategy
│   ├── __init__.py
│   ├── base.py                ← BaseMethod ABC + MethodOutput dataclass
│   ├── full_cache.py
│   ├── paged_attention.py
│   ├── h2o.py
│   └── streaming_llm.py
│
├── eval/                      ← shared evaluation
│   ├── __init__.py
│   └── metrics.py             ← accuracy, ROUGE-L, perplexity, latency stats
│
├── scripts/                   ← experiment runners
│   └── run_experiment1.py     ← loads config → loads data → runs method → evals
│
└── results/                   ← local only (gitignored), upload to GCS
    └── .gitkeep
```

## Quick Start

```bash
# 1. Clone
git clone <repo-url> && cd kv-cache-bench

# 2. Environment
conda create -n kvcache python=3.10 -y && conda activate kvcache
pip install -r requirements.txt

# 3. Download model weights (on your GCP VM)
mkdir -p ~/models
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ~/models/qwen2.5-7b

# 4. Run an experiment
python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method full_cache
```

## Experiments

### Experiment 1 — Controlled Long Context

Single-request evaluation (no batching) across three context-length buckets. Each method processes the same samples.

| Bucket | Dataset | Token Range |
|--------|---------|-------------|
| Short | MMLU | 128–256 |
| Medium | CNN/DailyMail | 512–1024 |
| Long | LongBench | 2048+ |

**Metrics collected:**

| Metric | Description |
|--------|-------------|
| Peak KV Memory (MB) | Max GPU memory used by KV cache |
| TTFT (ms) | Time to first token |
| Decode Latency (ms) | Total generation time minus TTFT |
| Throughput (tokens/sec) | Generated tokens per second |
| Quality | MMLU → Accuracy, CNN/DM → ROUGE-L, LongBench → task-specific |

**Profiling tools:** PyTorch Profiler, NVIDIA Nsight Systems

## How It Fits Together

```
configs/experiment1_short.yaml
        │
        ▼
scripts/run_experiment1.py
        │
        ├──► data/pipeline.py        → loads MMLU samples
        │
        ├──► methods/full_cache.py    → runs inference, returns MethodOutput
        │    methods/paged_attention.py
        │    methods/h2o.py
        │    methods/streaming_llm.py
        │
        └──► eval/metrics.py          → computes accuracy, latency stats
                    │
                    ▼
              results/exp1_short/     → JSON + profiling traces
```

## Key Interfaces

### MethodOutput (what every method returns)

```python
@dataclass
class MethodOutput:
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float              # time to first token
    total_time_ms: float        # end-to-end generation time
    decode_latency_ms: float    # total_time - ttft
    peak_kv_memory_mb: float    # peak KV cache memory
    metadata: dict              # method-specific extras
```

### BaseMethod (what every method implements)

```python
class BaseMethod(ABC):
    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None: ...
    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput: ...
    def teardown(self) -> None: ...  # optional
```

### Data Pipeline (what the data loader returns)

```python
@dataclass
class Sample:
    id: str                # unique identifier
    prompt: str            # formatted prompt ready for the model
    reference: str         # ground truth for evaluation
    dataset: str           # "mmlu", "cnn_dailymail", "longbench"
    subset: str            # e.g., MMLU subject name
    token_count: int       # pre-computed prompt token count
    metadata: dict         # dataset-specific fields
```

## Adding a New Method

1. Create `methods/your_method.py`
2. Subclass `BaseMethod`, implement `setup()` and `generate()`
3. Register it in `methods/__init__.py`
4. Run: `python scripts/run_experiment1.py --config configs/experiment1_short.yaml --method your_method`

See `CONTRIBUTING.md` for branching rules, GCP VM setup, and the full collaboration workflow.

## Infrastructure

- **Code:** GitHub (branch strategy in CONTRIBUTING.md)
- **Compute:** Individual GCP VMs per team member (GPU: A100 or L4)
- **Model weights:** Stored locally at `~/models/qwen2.5-7b` (not committed)
- **Results:** Local `results/` dir, synced to shared GCS bucket
- **Profiling:** PyTorch Profiler traces + Nsight `.nsys-rep` files
