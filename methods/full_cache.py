"""Full KV cache baseline using HuggingFace Transformers.

Stores all KV pairs without eviction — the standard approach.
Serves as the quality and latency reference for other methods.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from methods.base import BaseMethod, MethodOutput, kv_memory_mb


class FullCacheMethod(BaseMethod):
    """Baseline: standard HuggingFace generation with full KV cache."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None:
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float16,
        )
        self.model.eval()

        # Warm up CUDA to eliminate JIT/initialization overhead from TTFT timing
        _dummy = torch.ones(1, 1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.model(_dummy, use_cache=False)
        torch.cuda.synchronize()
        del _dummy

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        """Run standard HF generation and collect timing + memory stats."""
        # ── Tokenize ──────────────────────────────────────────────
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]

        # ── Memory baseline ───────────────────────────────────────
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)
        peak_kv_mb = 0.0

        # ── Prefill (measures TTFT) ───────────────────────────────
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs_prefill = self.model(input_ids, use_cache=True)
            past_key_values = outputs_prefill.past_key_values

        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        ttft_ms = (t_prefill - t_start) * 1000

    # ── Decode (generate remaining tokens) ────────────────────
        # Start from the last token of prefill
        next_token_id = outputs_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token_id]

        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out = self.model(
                    next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token_id)

            # Stop on EOS
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        # Final KV cache size (full cache only grows, so this is the peak)
        peak_kv_mb = kv_memory_mb(past_key_values)

        # ── Collect results ───────────────────────────────────────
        total_time_ms = (t_end - t_start) * 1000
        decode_latency_ms = total_time_ms - ttft_ms

        # GPU-level peak for reference (includes activations, logits, etc.)
        gpu_peak_mb = (torch.cuda.max_memory_allocated(self.device) - mem_before) / (1024 ** 2)

        all_token_ids = torch.cat(generated_ids, dim=-1)
        generated_text = self.tokenizer.decode(
            all_token_ids[0], skip_special_tokens=True
        )

        return MethodOutput(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            decode_latency_ms=decode_latency_ms,
            peak_kv_memory_mb=peak_kv_mb,
            metadata={
                "method": "full_cache",
                "gpu_peak_mb": round(gpu_peak_mb, 3),
            },
        )
    
    def teardown(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None