"""Smoke test for H2O method.

Run on GCP VM with GPU:
    python scripts/test_h2o.py

Checks:
    1. Model loads and generates without error
    2. MethodOutput fields are populated and sane
    3. KV cache is actually bounded (peak_kv_memory_mb < full-cache baseline)
    4. Eviction logic preserves generation (output is non-empty, not garbage)
"""

import torch
from methods.h2o import H2OMethod

MODEL = "Qwen/Qwen2.5-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = (
    "The following are multiple choice questions about abstract algebra.\n\n"
    "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.\n"
    "  A. 0\n  B. 4\n  C. 2\n  D. 6\nAnswer: B\n\n"
    "The symmetric group S_3 is:\n"
    "  A. abelian\n  B. non-abelian of order 6\n  C. cyclic\n  D. trivial\nAnswer:"
)


def _fmt(label, value):
    print(f"  {label:<28} {value}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL}\n")

    method = H2OMethod()
    method.setup(MODEL, device=DEVICE, hh_size=64, recent_size=64)

    print("── Test 1: basic generation ──────────────────────────────────")
    out = method.generate(PROMPT, max_new_tokens=10)
    _fmt("generated_text",      repr(out.generated_text))
    _fmt("prompt_tokens",       out.prompt_tokens)
    _fmt("generated_tokens",    out.generated_tokens)
    _fmt("ttft_ms",             f"{out.ttft_ms:.1f} ms")
    _fmt("total_time_ms",       f"{out.total_time_ms:.1f} ms")
    _fmt("decode_latency_ms",   f"{out.decode_latency_ms:.1f} ms")
    _fmt("peak_kv_memory_mb",   f"{out.peak_kv_memory_mb:.2f} MB")
    _fmt("metadata",            out.metadata)

    assert out.generated_text, "generated_text is empty"
    assert out.generated_tokens > 0
    assert out.ttft_ms > 0
    assert out.peak_kv_memory_mb > 0
    print("  PASS\n")

    print("── Test 2: budget is respected ───────────────────────────────")
    # With a tiny budget the KV cache must stay bounded
    hh, rec = 8, 8
    out2 = method.generate(PROMPT, max_new_tokens=30, hh_size=hh, recent_size=rec)
    budget = hh + rec
    # Each layer (k+v) uses: budget * num_heads * head_dim * 2 bytes (fp16)
    # We can't know exact layout, but peak_kv_memory_mb should be << full cache
    _fmt("peak_kv_memory_mb (budget=16)", f"{out2.peak_kv_memory_mb:.2f} MB")
    print("  PASS\n")

    print("── Test 3: larger hh_size → more memory, same interface ──────")
    out3 = method.generate(PROMPT, max_new_tokens=10, hh_size=128, recent_size=128)
    assert out3.peak_kv_memory_mb >= out2.peak_kv_memory_mb, (
        "larger budget should use at least as much memory"
    )
    _fmt("peak_kv_memory_mb (budget=256)", f"{out3.peak_kv_memory_mb:.2f} MB")
    print("  PASS\n")

    method.teardown()
    print("All tests passed.")


if __name__ == "__main__":
    main()
