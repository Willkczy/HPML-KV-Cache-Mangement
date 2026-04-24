```markdown
## Running the Realistic Workload

**Step 1 — Generate the trace (once, shared across all methods)**

```bash
python -m data.realistic_trace --config configs/experiment_mixed_workload.yaml
```

This writes `results/traces/mixed_workload_trace.jsonl` with Poisson arrival times at 2 req/s.

**Step 2 — Run a method against the trace**

```bash
python scripts/run_experiment_mixed_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method paged_attention
```

Replace `paged_attention` with `full_cache`, `h2o`, or `streaming_llm` to compare.

**Smoke test first (3 requests only)**

```bash
python scripts/run_experiment_mixed_workload.py \
    --config configs/experiment_mixed_workload.yaml \
    --method paged_attention \
    --smoke_test
```

**On Insomnia (Slurm)**

```bash
sbatch --partition=short --gres=gpu:A6000:1 \
    --export=ALL,METHOD=paged_attention,CONFIG=configs/experiment_mixed_workload.yaml \
    slurm/run_experiment.sbatch
```

Results are saved to `results/experiment_mixed_workload/mixed_<method>_full.json`.
Key metrics to compare across methods: `p95_queue_wait_ms` and `p99_e2e_latency_ms`.
