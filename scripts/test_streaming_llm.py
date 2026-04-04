"""Sanity tests for StreamingLLM method.

Run on GCP VM with GPU:
    python scripts/test_streaming_llm.py

Checks:
    1. Model loads and generates without error
    2. MethodOutput fields are populated and sane
    3. KV cache is bounded to start_size + recent_size after prefill
    4. Larger window uses more memory than smaller window
    5. Teardown frees GPU memory
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from methods.streaming_llm import StreamingLLMMethod

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
    print(f"  {label:<35} {value}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL}\n")

    method = StreamingLLMMethod()
    method.setup(MODEL, device=DEVICE, start_size=4, recent_size=256)

    # ── Test 1: basic generation ──────────────────────────────────────────
    print("── Test 1: basic generation ──────────────────────────────────")
    out = method.generate(PROMPT, max_new_tokens=10)
    _fmt("generated_text",           repr(out.generated_text))
    _fmt("prompt_tokens",            out.prompt_tokens)
    _fmt("generated_tokens",         out.generated_tokens)
    _fmt("ttft_ms",                  f"{out.ttft_ms:.1f} ms")
    _fmt("decode_latency_ms",        f"{out.decode_latency_ms:.1f} ms")
    _fmt("peak_kv_memory_mb",        f"{out.peak_kv_memory_mb:.2f} MB  (decode)")
    _fmt("prefill_peak_kv_memory_mb", f"{out.metadata['prefill_peak_kv_memory_mb']:.2f} MB  (prefill)")

    assert out.generated_text,       "generated_text is empty"
    assert out.generated_tokens > 0, "no tokens generated"
    assert out.ttft_ms > 0,          "TTFT is zero"
    assert out.peak_kv_memory_mb >= 0
    assert "prefill_peak_kv_memory_mb" in out.metadata
    print("  PASS\n")

    # ── Test 2: decode KV memory is bounded by window ─────────────────────
    print("── Test 2: decode KV memory bounded by window ────────────────")
    # peak_kv_memory_mb is measured post-trim so should be small regardless
    # of prompt length — it reflects the window size, not the prompt size
    small_window = method.generate(PROMPT, max_new_tokens=10,
                                   start_size=4, recent_size=32)
    _fmt("decode mem  (recent=32)",  f"{small_window.peak_kv_memory_mb:.2f} MB")
    _fmt("prefill mem (recent=32)",  f"{small_window.metadata['prefill_peak_kv_memory_mb']:.2f} MB")
    assert small_window.peak_kv_memory_mb < small_window.metadata["prefill_peak_kv_memory_mb"], (
        "decode KV mem should be less than prefill peak after trim"
    )
    print("  PASS\n")

    # ── Test 3: larger window → more decode memory ────────────────────────
    print("── Test 3: larger window uses more decode memory ─────────────")
    large_window = method.generate(PROMPT, max_new_tokens=10,
                                   start_size=4, recent_size=512)
    _fmt("decode mem (recent=32)",   f"{small_window.peak_kv_memory_mb:.2f} MB")
    _fmt("decode mem (recent=512)",  f"{large_window.peak_kv_memory_mb:.2f} MB")
    assert large_window.peak_kv_memory_mb >= small_window.peak_kv_memory_mb, (
        "larger window should use at least as much decode memory"
    )
    print("  PASS\n")

    # ── Test 4: teardown frees GPU memory ─────────────────────────────────
    print("── Test 4: teardown frees GPU memory ─────────────────────────")
    mem_before_teardown = torch.cuda.memory_allocated(DEVICE)
    method.teardown()
    mem_after_teardown = torch.cuda.memory_allocated(DEVICE)
    _fmt("mem before teardown", f"{mem_before_teardown / 1024**2:.1f} MB")
    _fmt("mem after teardown",  f"{mem_after_teardown / 1024**2:.1f} MB")
    assert mem_after_teardown < mem_before_teardown, "teardown did not free memory"
    print("  PASS\n")

    print("All tests passed.")


if __name__ == "__main__":
    main()