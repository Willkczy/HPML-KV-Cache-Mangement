# Contributing Guide

## Branching Strategy

```
main              ← stable, passing code only (merge via PR)
├── dev           ← integration branch, merge features here first
│   ├── feat/data-pipeline
│   ├── feat/full-cache
│   ├── feat/paged-attention
│   ├── feat/h2o
│   └── feat/streaming-llm
```

### Rules
1. **Never push directly to `main` or `dev`.** Always open a PR.
2. Branch off `dev` for your feature: `git checkout -b feat/your-feature dev`
3. Keep commits small and descriptive: `feat(data): add MMLU loader`
4. Rebase on `dev` before opening a PR: `git pull --rebase origin dev`
5. PRs require **1 approval** before merging into `dev`.
6. `dev → main` merges happen together before experiment runs.

## Ownership

| Component | Owner | Branch |
|-----------|-------|--------|
| Data pipeline + eval | (You) | `feat/data-pipeline` |
| Full KV cache (HF) | Hung-Kai | `feat/full-cache` |
| PagedAttention (vLLM) | Kane | `feat/paged-attention` |
| H2O | TBD | `feat/h2o` |
| StreamingLLM | Ting-Feng | `feat/streaming-llm` |

## Method Implementation Contract

Every method must subclass `methods.base.BaseMethod` and implement:
- `setup(model_name, **kwargs)` — load model/engine
- `generate(prompt, max_new_tokens, **kwargs) → MethodOutput`

This ensures the run script and eval code work identically across all methods.

## GCP VM Setup (Each Member)

Each team member runs their own GCP VM. GitHub is the single source of truth for code; VMs are disposable execution environments.

```
GitHub (code)                    Your GCP VM (execution)
┌──────────────┐    git pull    ┌──────────────────────┐
│ data/         │ ──────────►  │ ~/kv-cache-bench/      │
│ methods/      │               │ + GPU (A100/L4)        │
│ scripts/      │               │ + model weights        │
│ eval/         │               │ + results/             │
└──────────────┘                └──────────────────────┘
```

### 1. Create Your VM

```bash
gcloud compute instances create kvcache-<your-name> \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --boot-disk-size=100GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE
```

Adjust machine type / GPU based on your GCP quota and budget.

### 2. SSH and Clone

```bash
gcloud compute ssh kvcache-<your-name> --zone=us-central1-a

# Once on VM
git clone <repo-url> && cd kv-cache-bench
```

### 3. Download Model Weights (Once Per VM)

```bash
# Store weights outside the repo so they're never committed
mkdir -p ~/models
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ~/models/qwen2.5-7b
```

All experiment configs reference the model path via `configs/`, so set it to `~/models/qwen2.5-7b`.

### 4. Daily Workflow

```bash
# Pull latest code
cd ~/kv-cache-bench
git checkout dev && git pull origin dev
git checkout feat/your-branch && git rebase dev

# Run your experiment
python scripts/run_experiment1.py --config configs/experiment1_short.yaml --method <your-method>

# Results stay local or upload to shared GCS bucket
gsutil cp -r results/ gs://<your-bucket>/results/<your-name>/
```

### 5. Sharing Results

Do NOT commit large result files or profiling traces to GitHub. Use a shared GCS bucket:

```bash
# Upload
gsutil cp -r results/exp1_short/ gs://<your-bucket>/results/<your-name>/exp1_short/

# Download a teammate's results for comparison
gsutil cp -r gs://<your-bucket>/results/<teammate>/exp1_short/ results/<teammate>/
```

### 6. Shutting Down

**Always stop your VM when not running experiments to avoid charges:**

```bash
gcloud compute instances stop kvcache-<your-name> --zone=us-central1-a
```

## Local Environment Setup

For local development (editing code, no GPU needed):

```bash
# Clone and enter repo
git clone <repo-url> && cd kv-cache-bench

# Create environment (pick one)
python -m venv .venv && source .venv/bin/activate
# OR
conda create -n kvcache python=3.10 -y && conda activate kvcache

# Install dependencies
pip install -r requirements.txt
```

## Commit Message Format

```
<type>(<scope>): <short description>

type:  feat | fix | refactor | docs | test
scope: data | eval | full-cache | paged | h2o | streaming | scripts
```

## What NOT to Commit
- Model weights, checkpoints
- Large data files (use HuggingFace `datasets` to load on the fly)
- Profiling traces (add to `results/` which is gitignored except `.gitkeep`)
- Virtual env folders, `__pycache__`