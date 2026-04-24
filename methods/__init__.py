"""Registry of all KV cache methods.

Import from here so the run script stays method-agnostic:

    from methods import METHODS
    method = METHODS["h2o"]()
    method.setup(model_name, device="cuda", hh_size=64, recent_size=64)
"""

import importlib
from typing import Callable

from methods.base import BaseMethod, MethodOutput


def _load_method(module_name: str, class_name: str):
    """Lazily import a method class so optional deps don't block other methods."""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


METHODS: dict[str, Callable[[], BaseMethod]] = {
    "full_cache": lambda: _load_method("methods.full_cache", "FullCacheMethod")(),
    "h2o": lambda: _load_method("methods.h2o", "H2OMethod")(),
    "streaming_llm": lambda: _load_method("methods.streaming_llm", "StreamingLLMMethod")(),
    "paged_attention": lambda: _load_method("methods.paged_attention", "PagedAttentionMethod")(),
}

__all__ = ["BaseMethod", "MethodOutput", "METHODS"]
