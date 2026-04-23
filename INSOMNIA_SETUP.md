# Insomnia Cluster Setup Guide

Columbia University HPC cluster for HPML Final Project.

## Team & Branch Reference

| Method | Owner | Branch |
|--------|-------|--------|
| Full KV Cache (HuggingFace) | Hung-Kai Huang | `feat/full-cache` |
| PagedAttention (vLLM) | Kane Wang | `feat/paged-attention` |
| H2O | Sripad Karne | `feat/h2o` |
| StreamingLLM | Ting-Feng Huang | `feat/streaming-llm` |

---

## Step 1 — SSH In

Add to `~/.ssh/config` on your **local machine**:
```
Host insomnia
    HostName insomnia.rcs.columbia.edu
    User <YOUR_UNI>
```

Then connect:
```bash
ssh insomnia
```
Enter your Columbia password + Duo 2FA (type `1` for push notification).

---

## Step 2 — Clone Repo

```bash
mkdir -p /insomnia001/depts/edu/users/${USER}
cd /insomnia001/depts/edu/users/${USER}
git clone https://github.com/Willkczy/HPML-KV-Cache-Mangement.git
cd HPML-KV-Cache-Mangement
git checkout feat/<your-branch>
```

---

## Step 3 — Set Up Environment

```bash
bash scripts/setup_insomnia.sh
```

This creates a venv in scratch and installs all dependencies (~15 min).

> **Note:** If you are NOT on `feat/paged-attention`, skip vllm to save ~10GB disk:
> ```bash
> grep -v "^vllm" requirements.txt | pip install -r /dev/stdin
> ```

---

## Step 4 — Download Model Weights

```bash
sbatch --partition=short slurm/download_model.sbatch
squeue -u ${USER}  # wait until job disappears (~1 min)
```

Verify:
```bash
ls ~/models/qwen2.5-7b/   # should show 4 safetensor shards
```

---

## Step 5 — Smoke Test (always do this first)

```bash
source /insomnia001/depts/edu/users/${USER}/.venv/bin/activate
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=<your-method> \
    slurm/smoke_test.sbatch
tail -f slurm-*-smoke.out
```

If it completes with no OOM errors, you are ready to run full experiments.

---

## Step 6 — Run Experiments

### Option A — Submit all 4 benchmarks at once (recommended)

```bash
bash slurm/submit_all.sh <your-method>
```

### Option B — Submit a single benchmark manually

Use this when you want to run one specific method + benchmark combination.

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=<method>,CONFIG=<config> \
    slurm/run_experiment.sbatch
```

| Benchmark | CONFIG value |
|-----------|-------------|
| MMLU (short) | `configs/experiment1_short.yaml` |
| LongBench short answer | `configs/experiment2_long.yaml` |
| LongBench explain | `configs/experiment2_long_explain.yaml` |
| GovReport summarization | `configs/experiment2_summarization.yaml` |

Example — run H2O on LongBench only:
```bash
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=h2o,CONFIG=configs/experiment2_long.yaml \
    slurm/run_experiment.sbatch
```

For StreamingLLM, you can also override window sizes:
```bash
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=streaming_llm,CONFIG=configs/experiment2_long.yaml,START_SIZE=4,RECENT_SIZE=512 \
    slurm/run_experiment.sbatch
```

### Monitor jobs

```bash
squeue -u ${USER}
tail -f slurm-<JOBID>-*.out
```

Replace `<your-method>` with: `streaming_llm`, `h2o`, `full_cache`, or `paged_attention`.

---

## Step 7 — Upload Results to GCS

```bash
gsutil cp -r results/ gs://<shared-bucket>/results/${USER}/
```

**Results are not backed up on Insomnia — upload before logging out.**

---

## Cluster Reference

### Partitions & Account

- **Account:** `edu` (already set in all scripts)
- **Partition for all jobs:** `short` (12h limit)
- No `cpu`, `gpu`, or `edu1` GPU partition exists

### Available GPUs

| GPU | VRAM | Nodes | Use |
|-----|------|-------|-----|
| **A6000** | **48GB** | **ins[080-094]** | **Default** |
| H100 | 80GB | ins[048,050] | Faster, fewer nodes |
| L40S | 48GB | ins[038-039,056-061] | Fallback |
| L40 | 48GB | ins[035-037] | Fallback |

No A100s on Insomnia.

Check live GPU availability:
```bash
sinfo -p short -o "%n %G %C %t" | grep gpu
```

### Storage

| Path | Purpose |
|------|---------|
| `/insomnia001/home/${USER}/` | Home — small, for SSH keys/configs only |
| `/insomnia001/depts/edu/users/${USER}/` | Scratch — venv, model weights, results |

```bash
df -i /insomnia001/home/${USER}              # file count (150k limit)
du -sh /insomnia001/depts/edu/users/${USER}/ # disk usage
```

### Common Mistakes

| Mistake | Fix |
|---------|-----|
| `--partition=gpu` or `--partition=cpu` | Use `--partition=short` |
| `--gres=gpu:a100:1` | No A100s — use `--gres=gpu:A6000:1` |
| pip install without activating venv | `source .../.venv/bin/activate` first |
| Installing vllm on non-vllm branch | Skip it — saves ~10GB |