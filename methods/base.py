"""Base interface and output dataclass for all KV cache methods.

Every method (full_cache, paged_attention, h2o, streaming_llm) subclasses
BaseMethod and returns MethodOutput from generate().  This keeps the
experiment runner and evaluation code method-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
