"""Registry of all KV cache methods.

Import from here so the run script stays method-agnostic:

    from methods import METHODS
    method = METHODS["h2o"]()
    method.setup(model_name, device="cuda", hh_size=64, recent_size=64)
"""

from methods.base import BaseMethod, MethodOutput
from methods.full_cache import FullCacheMethod
from methods.h2o import H2OMethod
from methods.streaming_llm import StreamingLLMMethod
from methods.paged_attention import PagedAttentionMethod

METHODS: dict[str, type[BaseMethod]] = {
    "full_cache": FullCacheMethod,
    "h2o": H2OMethod,
    "streaming_llm": StreamingLLMMethod,
    "paged_attention": PagedAttentionMethod,
}

__all__ = ["BaseMethod", "MethodOutput", "METHODS"]
