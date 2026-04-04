"""Shared evaluation metrics for all KV cache experiments."""

import re
from dataclasses import dataclass
from typing import Optional

from methods.base import MethodOutput


@dataclass
class EvalResult:
    method: str
    num_samples: int
    accuracy: Optional[float]       # MMLU
    avg_ttft_ms: float
    avg_decode_latency_ms: float
    avg_throughput_tps: float
    avg_peak_kv_mb: float


def extract_mmlu_answer(text: str) -> Optional[str]:
    """Pull the first A/B/C/D letter out of generated text."""
    text = text.strip()
    m = re.search(r'\b([A-D])\b', text)
    return m.group(1) if m else None


def evaluate_mmlu(
    method_name: str,
    outputs: list[MethodOutput],
    references: list[str],
) -> EvalResult:
    correct = 0
    for out, ref in zip(outputs, references):
        pred = extract_mmlu_answer(out.generated_text)
        if pred == ref.strip():
            correct += 1

    n = len(outputs)
    return EvalResult(
        method=method_name,
        num_samples=n,
        accuracy=correct / n if n > 0 else 0.0,
        avg_ttft_ms=sum(o.ttft_ms for o in outputs) / n,
        avg_decode_latency_ms=sum(o.decode_latency_ms for o in outputs) / n,
        avg_throughput_tps=sum(
            o.generated_tokens / (o.total_time_ms / 1000.0) for o in outputs
        ) / n,
        avg_peak_kv_mb=sum(o.peak_kv_memory_mb for o in outputs) / n,
    )


def print_results(result: EvalResult) -> None:
    print("=" * 57)
    print(f"  Method:          {result.method}")
    print(f"  Samples:         {result.num_samples}")
    if result.accuracy is not None:
        print(f"  Accuracy:        {result.accuracy:.1%}  ({int(result.accuracy * result.num_samples)}/{result.num_samples})")
    print(f"  Avg TTFT:        {result.avg_ttft_ms:.1f} ms")
    print(f"  Avg decode:      {result.avg_decode_latency_ms:.1f} ms")
    print(f"  Avg throughput:  {result.avg_throughput_tps:.1f} tok/s")
    print(f"  Avg peak KV mem: {result.avg_peak_kv_mb:.1f} MB")
    print("=" * 57)
