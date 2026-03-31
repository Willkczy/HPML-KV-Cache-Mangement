"""Registry of all KV cache methods.

Import from here so the run script stays method-agnostic:

    from methods import METHODS
    method = METHODS["h2o"]()
    method.setup(model_name, device="cuda", hh_size=64, recent_size=64)
"""

from methods.base import BaseMethod, MethodOutput
from methods.h2o import H2OMethod

METHODS: dict[str, type[BaseMethod]] = {
    "h2o": H2OMethod,
    # filled in by other team members:
    # "full_cache":      FullCacheMethod,
    # "paged_attention": PagedAttentionMethod,
    # "streaming_llm":   StreamingLLMMethod,
}

__all__ = ["BaseMethod", "MethodOutput", "METHODS"]
