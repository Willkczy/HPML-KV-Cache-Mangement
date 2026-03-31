# GCP VM Setup Guide — Step by Step

This guide documents the **tested, working** procedure for provisioning a
GCP VM with GPU for the KV cache benchmark project.  It covers the specific
constraints of our school GCP organization (no external IPs, IAP-only SSH)
and the pitfalls we hit along the way so you don't have to.

> **Last verified:** March 2026 on `hpml-hw-proj`, NVIDIA L4, Ubuntu 24.04
> Deep Learning VM image.

---

## Prerequisites (Local Machine)

| Tool | Purpose |
|------|---------|
| `gcloud` CLI | VM management, SSH |
| GCP project access | `hpml-hw-proj` (or your project ID) |
| `gcloud auth login` | Authenticated to your school account |

---

## 1. Create the VM

Our GCP org **blocks external IPs**, so always add `--no-address`.

### Recommended: NVIDIA L4 (24 GB VRAM)

Sufficient for Qwen2.5-7B in FP16 (~14.5 GB) with room for KV cache.

```bash
gcloud compute instances create kvcache-<your-name> \
    --zone=us-east1-d \
    --project=hpml-hw-proj \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --boot-disk-size=200GB \
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --no-address
```

### Key details

| Parameter | Value | Why |
|-----------|-------|-----|
| `machine-type` | `g2-standard-8` | L4 requires G2 series (not N1) |
| `image-family` | `pytorch-2-7-cu128-ubuntu-2404-nvidia-570` | Latest DL VM with pre-installed PyTorch + CUDA + driver |
| `boot-disk-size` | `200GB` | Avoids GCP I/O performance warning at 100GB |
| `--no-address` | — | Required by our org policy (no external IPs) |

> **Finding available zones:** If `us-east1-d` has no capacity, check:
> ```bash
> # Using the provisioner
> uv run python gcp_gpu_provisioner.py --config config.json --list-zones
>
> # Or manually
> gcloud compute accelerator-types list --filter="name=nvidia-l4" \
>     --project=hpml-hw-proj
> ```

> **GPU sizing reference:**
> | GPU | VRAM | Fits Qwen2.5-7B FP16? |
> |-----|------|------------------------|
> | Tesla T4 | 15 GB | ✗ Too tight (~14.5 GB model alone) |
> | NVIDIA L4 | 23 GB | ✓ Recommended for development |
> | A100 40GB | 40 GB | ✓ Ideal for benchmark runs |

---

## 2. Network Setup (One-Time Per Project)

Since VMs have no external IP, you need **Cloud NAT** for outbound internet
(git clone, pip install, model downloads) and an **IAP firewall rule** for
SSH.

### 2a. IAP Firewall Rule

Allows Google IAP to reach port 22 on your VMs:

```bash
gcloud compute firewall-rules create allow-iap-ssh \
    --project=hpml-hw-proj \
    --direction=INGRESS \
    --action=ALLOW \
    --rules=tcp:22 \
    --source-ranges=35.235.240.0/20 \
    --network=default
```

> `35.235.240.0/20` is Google's IAP IP range. This is safe and documented
> by Google.

### 2b. Cloud NAT

Gives VMs outbound internet access without external IPs:

```bash
# Create a Cloud Router (required by NAT)
gcloud compute routers create nat-router \
    --region=us-east1 \
    --network=default \
    --project=hpml-hw-proj

# Create NAT config
gcloud compute routers nats create nat-config \
    --router=nat-router \
    --region=us-east1 \
    --nat-all-subnet-ip-ranges \
    --auto-allocate-nat-external-ips \
    --project=hpml-hw-proj
```

> **Wait 1-2 minutes** after creating NAT before testing connectivity on
> the VM.  You can verify with `curl -s --max-time 10 https://github.com`
> from inside the VM.

> **Note:** If your VMs are in a different region (e.g. `us-central1`),
> create a separate router + NAT for that region.

---

## 3. SSH into the VM

```bash
gcloud compute ssh kvcache-<your-name> \
    --zone=us-east1-d \
    --project=hpml-hw-proj \
    --tunnel-through-iap
```

If SSH fails immediately after VM creation, wait 2-3 minutes for the boot
process and GPU driver initialization to complete.

Verify GPU:

```bash
nvidia-smi
```

Expected: NVIDIA L4 with ~23 GB VRAM.

---

## 4. Environment Setup

### ⚠️ Important: Use `--system-site-packages`

The Deep Learning VM comes with PyTorch, CUDA, and NVIDIA drivers
pre-installed and matched to each other.  A plain `python3 -m venv .venv`
creates an isolated environment that **cannot see system PyTorch**, and
`pip install torch` inside it will pull a version that doesn't match the
VM's CUDA driver — causing `RuntimeError: The NVIDIA driver on your system
is too old`.

The fix is to create the venv with `--system-site-packages` so it inherits
the pre-installed PyTorch:

```bash
# Install venv support (Ubuntu 24.04 doesn't include it by default)
sudo apt update && sudo apt install python3.12-venv -y

# Create venv that inherits system packages (PyTorch, CUDA, etc.)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Verify system PyTorch is visible and CUDA works
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# Expected: True
```

### Clone and install

```bash
git clone <repo-url> && cd kv-cache-bench
git checkout feat/<your-branch>

# Install project dependencies (won't re-install PyTorch)
pip install -r requirements.txt

# Install huggingface CLI for model downloads
pip install huggingface_hub[cli]
```

---

## 5. Download Model Weights

```bash
mkdir -p ~/models
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ~/models/qwen2.5-7b
```

This downloads ~14 GB.  Weights are stored outside the repo and never
committed.

### Verify model loads correctly

```bash
python -c "
import torch
from methods.full_cache import FullCacheMethod

m = FullCacheMethod()
m.setup(model_name='$HOME/models/qwen2.5-7b', device='cuda')
print('dtype:', next(m.model.parameters()).dtype)
print('GPU mem (MB):', round(torch.cuda.memory_allocated() / 1024**2, 1))
m.teardown()
"
```

Expected output:

```
dtype: torch.float16
GPU mem (MB): ~14500
```

---

## 6. Run Experiments

### Phase 1 — Smoke test (3 samples, quick validation)

```bash
python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method <your-method> \
    --smoke_test
```

**What to check:**
- All samples produce output (no crashes or OOM)
- TTFT, decode latency, peak KV memory are positive numbers
- For MMLU: predicted answer is A/B/C/D (not empty `?`)
- JSON results file is saved to `results/exp1_short/`

### Phase 2 — Full experiment

```bash
python scripts/run_experiment1.py \
    --config configs/experiment1_short.yaml \
    --method <your-method>
```

Runs all MMLU samples (up to 250 across 5 subjects).

### Upload results (don't commit to Git)

```bash
gsutil cp -r results/exp1_short/ gs://<your-bucket>/results/<your-name>/exp1_short/
```

---

## 7. Shut Down When Done

**Always stop your VM to avoid charges:**

```bash
# From your local machine
gcloud compute instances stop kvcache-<your-name> \
    --zone=us-east1-d \
    --project=hpml-hw-proj
```

To restart later:

```bash
gcloud compute instances start kvcache-<your-name> \
    --zone=us-east1-d \
    --project=hpml-hw-proj
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| SSH `failed to connect to backend` | VM still booting or no IAP firewall rule | Wait 2-3 min; check firewall rule exists |
| `git clone` hangs or times out | No Cloud NAT configured | Set up Cloud NAT (Section 2b) |
| `externally-managed-environment` error | Ubuntu 24.04 PEP 668 | Use `python3 -m venv` (Section 4) |
| `NVIDIA driver too old` RuntimeError | venv PyTorch doesn't match VM driver | Recreate venv with `--system-site-packages` |
| `pip: command not found` in venv | Missing pip in venv | `python -m ensurepip` or reinstall venv |
| OOM during inference | GPU VRAM too small for model | Use L4 (23 GB) or A100; verify FP16 is active |
| `peak_kv_memory_mb` is 0 or negative | Memory measurement timing issue | Check `reset_peak_memory_stats` placement |
| All MMLU predictions are `?` | Model output format wrong | Inspect `generated_text` in results JSON |