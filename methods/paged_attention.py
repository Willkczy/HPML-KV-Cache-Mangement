"""PagedAttention KV cache method using vLLM.

vLLM implements PagedAttention natively — KV cache is managed in fixed-size
blocks with a block table for non-contiguous memory allocation.  This removes
internal fragmentation and lets requests share physical memory pages.

To ensure fair benchmarking against HuggingFace-based methods, we disable
vLLM's implicit optimizations:
  - enforce_eager=True          → disables CUDA graph capture and torch.compile

Requires: pip install vllm
"""

import time

import torch
from vllm import LLM, SamplingParams

from methods.base import BaseMethod, MethodOutput


class PagedAttentionMethod(BaseMethod):
    """PagedAttention via vLLM's block-based KV cache manager."""

    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None:
        """Load the vLLM engine, which pre-allocates the KV block pool.

        Keyword args forwarded:
            gpu_memory_utilization (float): fraction of GPU for KV pool (default 0.90)
            max_model_len (int): max sequence length (default 4096)
            block_size (int): tokens per KV block (default 16)
            enforce_eager (bool): disable CUDA graphs (default True for fair benchmarking)
        """
        gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.90)
        max_model_len = kwargs.get("max_model_len", 4096)
        block_size = kwargs.get("block_size", 16)

        self.device = device
        self.block_size = block_size

        # Track memory attributed to model + KV pool
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            block_size=block_size,
            dtype="float16",
            # ── Disable implicit optimizations for fair benchmarking ──
            enforce_eager=kwargs.get("enforce_eager", True),
            enable_prefix_caching=False,
        )

        self.engine_memory_mb = (
            torch.cuda.memory_allocated(device) - mem_before
        ) / (1024 ** 2)

        self.tokenizer = self.llm.get_tokenizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_kv_memory_mb(self, total_tokens: int) -> float:
        """Compute the actual KV cache bytes for *total_tokens*.

        KV memory = 2 (K+V) × layers × tokens × kv_heads × head_dim × dtype_bytes

        vLLM pre-allocates its block pool at init time, so the per-request
        memory delta from ``torch.cuda`` is near-zero.  This calculation
        gives the true per-request KV footprint for apples-to-apples
        comparison with methods that allocate KV on the fly.
        """
        model_config = self.llm.model_config
        hf_config = model_config.hf_config

        num_layers = hf_config.num_hidden_layers
        num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        dtype_bytes = torch.tensor([], dtype=model_config.dtype).element_size()

        kv_bytes = (
            2 * num_layers * total_tokens * num_kv_heads * head_dim * dtype_bytes
        )
        return kv_bytes / (1024 ** 2)

    @staticmethod
    def _extract_ttft(request_output) -> float | None:
        """Try to pull TTFT from vLLM's RequestMetrics (version-dependent)."""
        metrics = getattr(request_output, "metrics", None)
        if metrics is None:
            return None
        first = getattr(metrics, "first_token_time", None)
        arrival = getattr(metrics, "arrival_time", None)
        if first is not None and arrival is not None:
            return (first - arrival) * 1000.0
        return None

    # ------------------------------------------------------------------
    # Public interface (BaseMethod)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        """Run a single-request inference through vLLM's PagedAttention engine."""
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_token_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_token_ids)

        # --- KV memory: prefill estimate (prompt-only, before decode) ---
        prefill_kv_mb = self._estimate_kv_memory_mb(prompt_tokens)

        # --- timing ---
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params)
        torch.cuda.synchronize(self.device)
        end = time.perf_counter()

        total_time_ms = (end - start) * 1000.0

        # --- unpack result ---
        request_output = outputs[0]
        completion = request_output.outputs[0]
        generated_text = completion.text
        generated_tokens = len(completion.token_ids)

        # --- TTFT ---
        ttft_ms = self._extract_ttft(request_output)
        if ttft_ms is None or ttft_ms <= 0.0:
            # Fallback: estimate TTFT proportional to prompt length.
            # In single-request mode, prefill dominates the prompt portion
            # and decode is roughly linear in generated tokens.
            if generated_tokens > 0:
                ttft_ms = total_time_ms * prompt_tokens / (prompt_tokens + generated_tokens)
            else:
                ttft_ms = total_time_ms

        decode_latency_ms = max(total_time_ms - ttft_ms, 0.0)

        # --- KV memory ---
        # Report total KV (prompt + generated) to match full_cache baseline.
        # Both full_cache and paged_attention are uncompressed full-cache methods
        # and should report the same KV footprint for the same input.
        total_tokens = prompt_tokens + generated_tokens
        total_kv_mb = self._estimate_kv_memory_mb(total_tokens)
        decode_peak_kv_mb = max(total_kv_mb - prefill_kv_mb, 0.0)

        prefill_blocks = -(-prompt_tokens // self.block_size)  # ceil division
        total_blocks = -(-total_tokens // self.block_size)  # ceil division
        decode_blocks = max(total_blocks - prefill_blocks, 0)

        return MethodOutput(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            decode_latency_ms=decode_latency_ms,
            peak_kv_memory_mb=total_kv_mb,
            metadata={
                "method": "paged_attention",
                "backend": "vllm",
                "block_size": self.block_size,
                "num_blocks_used": total_blocks,
                "decode_blocks_used": decode_blocks,
                "prefill_kv_memory_mb": round(prefill_kv_mb, 3),
                "decode_peak_kv_memory_mb": round(decode_peak_kv_mb, 3),
                "engine_memory_mb": round(self.engine_memory_mb, 2),
            },
        )

    def teardown(self) -> None:
        """Destroy the vLLM engine and release GPU memory."""
        if hasattr(self, "llm"):
            del self.llm
        torch.cuda.empty_cache()