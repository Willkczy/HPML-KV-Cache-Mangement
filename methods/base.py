"""Base interface and output dataclass for all KV cache methods.

Every method (full_cache, paged_attention, h2o, streaming_llm) subclasses
BaseMethod and returns MethodOutput from generate().  This keeps the
experiment runner and evaluation code method-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

def kv_memory_mb(past_key_values) -> float:
    """Sum bytes of all K and V tensors in the cache, converted to MB.

    Works with DynamicCache and any iterable of (key, value, ...) tuples.
    Use this across all methods to ensure consistent KV memory measurement.
    """
    if past_key_values is None:
        return 0.0
    total = 0
    for layer_kv in past_key_values:
        k, v = layer_kv[0], layer_kv[1]
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total / (1024 ** 2)

@dataclass
class MethodOutput:
    """Standardized output returned by every method's generate() call."""
    generated_text: str          # decoded model output
    prompt_tokens: int           # number of tokens in the prompt
    generated_tokens: int        # number of tokens generated
    ttft_ms: float               # time to first token (ms)
    total_time_ms: float         # end-to-end generation time (ms)
    decode_latency_ms: float     # total_time - ttft (ms)
    peak_kv_memory_mb: float     # peak KV cache GPU memory (MB)
    metadata: dict = field(default_factory=dict)  # method-specific extras


class BaseMethod(ABC):
    """Abstract base class that every KV cache strategy implements."""

    @abstractmethod
    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None:
        """Load model, tokenizer, and any method-specific state."""

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        """Run inference on a single prompt and return a MethodOutput."""

    def teardown(self) -> None:
        """Optional cleanup (free GPU memory, close handles, etc.)."""
