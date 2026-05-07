# HPML Final Project: KV Cache Management for LLM Inference

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team 31
- **Members:**
  - Hung-Kai Huang (hh3164) — Full KV Cache baseline (HuggingFace)
  - Ting-Feng Huang (th3192) — StreamingLLM method + fairness analysis + serving benchmark
  - Sripad Karne (sk5695) — H2O method
  - Yipeng Wang (yw4623) — PagedAttention (vLLM)

## Submission

- **GitHub repository:** [https://github.com/Willkczy/HPML-KV-Cache-Mangement](https://github.com/Willkczy/HPML-KV-Cache-Mangement)
- **Final report:** [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML_Final_Presentation.pptx`](deliverables/HPML_Final_Presentation.pptx)
- **Experiment results dashboard:** [`results/dashboard/README.md`](results/dashboard/README.md)

---

## 1. Problem Statement

During LLM inference, the KV cache grows linearly with sequence length — a 7B model on a 16k-token input requires over 700 MB of KV cache per request, limiting serving throughput and increasing hardware cost. We benchmark four KV cache management strategies (Full Cache, H2O, StreamingLLM, PagedAttention) on Qwen2.5-7B, measuring the memory-quality trade-off across short-answer and long-form generation tasks, and under realistic concurrent serving workloads. The bottleneck we address is GPU memory capacity and bandwidth during inference.

---

## 2. Model/Application Description

- **Model architecture:** Qwen2.5-7B (28 layers, 28 query heads, 4 KV heads, GQA)
- **Framework:** PyTorch 2.6, HuggingFace Transformers 4.57, vLLM 0.8.5
- **Datasets:**
  - [MMLU](https://huggingface.co/datasets/cais/mmlu) — 5-shot multiple choice, 128–512 tokens
  - [LongBench v2](https://huggingface.co/datasets/THUDM/LongBench-v2) — long-context QA, 8k–16k tokens
  - [GovReport](https://huggingface.co/datasets/ccdv/govreport-summarization) — summarization, 4k–16k tokens
- **Custom modifications:**
  - StreamingLLM: attention-sink + sliding-window KV trim on HuggingFace DynamicCache
  - H2O: cumulative attention score eviction with sdpa prefill / eager decode
  - PagedAttention: vLLM block-based KV with fairness fixes (float16, no prefix cache, eager mode)
  - Concurrent serving: vLLM AsyncLLMEngine with Poisson arrival scheduling
- **Hardware:** NVIDIA RTX L40S / A6000 (48 GB), Insomnia HPC cluster, Columbia University

---

## 3. Final Results Summary

### Experiment 1 — Controlled Single-Request Benchmark

| Method | MMLU Acc. | LB Short Acc. | GovReport ROUGE-L | Decode KV (LB Short) |
|--------|-----------|---------------|-------------------|----------------------|
| Full Cache | 56.7% | 40.9% | 0.1859 | 732.8 MB |
| H2O (hh=64, r=64) | 56.7% | 40.9% | 0.0000 | **17.5 MB** |
| StreamingLLM (recent=1024) | 56.7% | 40.9% | 0.1292 | 56.2 MB |
| StreamingLLM (recent=4096) | 56.7% | 40.9% | **0.1832** | 224.2 MB |
| PagedAttention | 57.3% | 40.9% | 0.1850 | 732.9 MB |

### Experiment 2 — Concurrent Serving Benchmark (PagedAttention)

| Config | Throughput @ 4 req/s | P99 E2E @ 2 req/s | Peak KV Util |
|--------|---------------------|-------------------|--------------|
| v0 eager | **1.434 req/s** | **202.8 s** | 79.7% |
| v0 graph | 1.296 req/s | 230.5 s | 79.7% |
| v1 graph | 1.147 req/s | 274.5 s | N/A† |
| v1 eager | 1.142 req/s | 276.0 s | N/A† |

† vLLM v1 runs the engine in a subprocess; KV block utilization is not accessible in offline mode.

**Hardware:** NVIDIA RTX L40S 48 GB, CUDA 12.3, Python 3.11, vLLM 0.8.5, PyTorch 2.6

**Headline result:** StreamingLLM reduces decode-phase KV memory by 198× on LongBench (3.7 MB vs 732.8 MB) with zero quality loss on short-answer tasks, and recovers near-baseline summarization quality (ROUGE-L 0.1832 vs 0.1859) with 2.3× less memory using a 4096-token window. For concurrent serving, vLLM v0 eager outperforms v0 graph by 10% and v1 by 20% on heterogeneous mixed-length workloads.

---

## 4. Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── INSOMNIA_SETUP.md              # HPC cluster setup guide
├── STATIC_BENCHMARK_RESULTS.md   # Experiment 1 results & analysis
├── SERVING_RESULTS.md             # Experiment 2 serving results
├── configs/                       # YAML configs for all experiments
├── deliverables/                  # Final report and presentation
├── methods/                       # KV cache method implementations
│   ├── base.py
│   ├── full_cache.py
│   ├── h2o.py
│   ├── paged_attention.py
│   ├── streaming_llm.py
│   └── streaming_llm_vllm.py
├── data/
│   ├── pipeline.py                # Dataset loaders → list[Sample]
│   └── realistic_trace.py         # Poisson trace generator
├── scripts/
│   ├── run_experiment1.py         # Static single-request runner
│   ├── run_experiment_mixed_workload.py
│   ├── run_vllm_serving_workload.py
│   ├── sweep_streaming_llm.py
│   └── plot_results.py            # Generates figures/
├── slurm/                         # Insomnia HPC Slurm job scripts
└── results/
    └── dashboard/                 # Static experiment results dashboard
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
git clone https://github.com/Willkczy/HPML-KV-Cache-Mangement.git
cd HPML-KV-Cache-Mangement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**System requirements:** Python 3.11+, CUDA 12.3, ≥ 48 GB GPU memory for long-context experiments. See [INSOMNIA_SETUP.md](INSOMNIA_SETUP.md) for HPC cluster setup.

### B. Experiment Tracking Dashboard

All experiment results are stored as JSON files and summarized in a static dashboard:

> **📊 Dashboard:** [`results/dashboard/README.md`](results/dashboard/README.md)

The dashboard contains all reported metrics, hyperparameter settings, and per-run summaries generated from the raw JSON result files.

### C. Model Weights

Weights are not committed. Download Qwen2.5-7B:

```bash
# On Insomnia cluster
sbatch slurm/download_model.sbatch

# Local
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ~/models/qwen2.5-7b
```

### D. Run Experiment 1 (Static Benchmark)

```bash
# Smoke test
python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method streaming_llm --smoke_test

# Full run — all methods, all configs (Insomnia)
bash slurm/submit_all.sh streaming_llm l40s
bash slurm/submit_all.sh h2o l40s
bash slurm/submit_all.sh full_cache l40s
bash slurm/submit_all.sh paged_attention l40s
```

### E. Run Experiment 2 (Concurrent Serving)

```bash
# Generate trace
python -m data.realistic_trace --config configs/experiment_mixed_workload.yaml

# Run serving benchmark
sbatch --export=ALL,METHOD=paged_attention slurm/run_vllm_serving_workload.sbatch
```

### F. Generate Figures

```bash
python scripts/plot_results.py
# Saved to figures/
```

---

## 6. Results and Observations

- **Short-answer tasks:** All methods achieve identical accuracy. Trimming occurs but quality is unaffected because the answer token is generated from the full prefill before eviction begins. StreamingLLM reduces decode-phase KV by 198×; H2O reduces by 42×.
- **H2O throughput overhead:** 25% slower than Full Cache due to unfused eager attention kernel required for attention weight access at every decode step.
- **H2O long-form collapse:** ROUGE-L = 0.000 on GovReport at all tested budgets (128–2048 tokens). One-shot eviction at decode step 1 with no accumulated history fails for long documents.
- **StreamingLLM GovReport:** Window size is a tunable quality-memory knob. recent=4096 achieves ROUGE-L 0.1832 (−1.5% vs baseline) with 2.3× less KV memory.
- **Serving — CUDA graphs:** v0 eager outperforms v0 graph on mixed-length workloads. Variable request lengths (128–32k tokens) prevent effective graph reuse.
- **Serving — vLLM v1:** Underperforms v0 due to chunked prefill default (2048 tok/chunk) adding IPC overhead per scheduling step for long requests.

---

## 7. Notes

- Model weights on Insomnia cluster at `~/models/qwen2.5-7b`.
- Datasets downloaded at runtime via HuggingFace `datasets` library.
- Profiling traces and large result files stored on cluster; JSON summaries committed to `results/`.

### AI Use Disclosure

**Did your team use any AI tool in completing this project?**

- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude (Anthropic)

**Specific purpose:** Cluster setup scripting (Slurm, SSH), fairness analysis of benchmark methodology, KV eviction implementation debugging, serving benchmark architecture (AsyncLLMEngine), visualization code, result documentation.

**Sections affected:** `slurm/`, `scripts/plot_results.py`, `INSOMNIA_SETUP.md`, `STATIC_BENCHMARK_RESULTS.md`, `SERVING_RESULTS.md`, `README.md`

**How we verified correctness:** All reported numbers come from GPU experiments on Insomnia cluster. Figures generated directly from committed JSON result files. KV memory formulas verified analytically against tensor byte counts. paged_attention quality verified to match full_cache baseline (same full-cache strategy, expected identical output).

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

```bibtex
@misc{team312026hpml,
  title  = {KV Cache Management for LLM Inference},
  author = {Huang, Hung-Kai and Huang, Ting-Feng and Karne, Sripad and Wang, Yipeng},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/Willkczy/HPML-KV-Cache-Mangement}
}
```

---

*HPML Spring 2026 · Dr. Kaoutar El Maghraoui · Columbia University*