"""StreamingLLM KV cache method.

Implements the attention-sink + sliding-window eviction strategy from:
  "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023)

Key idea: keep the first `start_size` tokens (attention sinks) and the most
recent `recent_size` tokens.  Everything in between is evicted each step.

Compatibility note:
  The official streaming-llm repo (mit-han-lab) targets transformers ~4.37 where
  keys were stored WITHOUT RoPE applied.  In transformers >=5.x, Qwen2 applies
  RoPE *before* writing into DynamicCache, so absolute position is already baked
  into each cached key.  That means trimming the cache is positionally safe and
  no attention-forward monkey-patch is required.

  The one thing we must do: pass an explicit `cache_position` equal to the
  *absolute* token index in the decode loop.  If we let the model auto-compute
  it from `cache.get_seq_length()`, it will see the trimmed length instead of
  the real position and assign the wrong RoPE rotation to the new query.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from methods.base import BaseMethod, MethodOutput


def _trim_cache(past_key_values, start_size: int, recent_size: int) -> None:
    """Evict middle tokens from a DynamicCache in-place.

    Keeps:
      - tokens [0 : start_size]                      (attention sinks)
      - tokens [seq_len - recent_size : seq_len]      (recent window)

    Does nothing when seq_len <= start_size + recent_size.
    Operates directly on DynamicLayer.keys / .values tensors
    (shape: [batch, heads, seq_len, head_dim]).
    """
    cache_size = start_size + recent_size
    for layer in past_key_values.layers:
        if not layer.is_initialized:
            continue
        seq_len = layer.keys.shape[-2]
        if seq_len <= cache_size:
            continue
        layer.keys = torch.cat(
            [layer.keys[:, :, :start_size, :],
             layer.keys[:, :, seq_len - recent_size:, :]],
            dim=-2,
        )
        layer.values = torch.cat(
            [layer.values[:, :, :start_size, :],
             layer.values[:, :, seq_len - recent_size:, :]],
            dim=-2,
        )


class StreamingLLMMethod(BaseMethod):
    """StreamingLLM: attention sinks + sliding-window KV eviction."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.start_size = None
        self.recent_size = None

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(
        self,
        model_name: str,
        device: str = "cuda",
        start_size: int = 4,
        recent_size: int = 256,
        **kwargs,
    ) -> None:
        """Load model/tokenizer and store window parameters.

        Args:
            start_size:   Number of initial tokens to keep as attention sinks.
            recent_size:  Number of most-recent tokens to keep in the sliding window.
        """
        self.device = device
        self.start_size = start_size
        self.recent_size = recent_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float16,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        """Run StreamingLLM generation and collect timing + memory stats."""

        # ── Tokenize ──────────────────────────────────────────────────
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]

        # ── Memory baseline ───────────────────────────────────────────
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        # ── Prefill (measures TTFT) ───────────────────────────────────
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs_prefill = self.model(input_ids, use_cache=True)
            past_key_values = outputs_prefill.past_key_values

        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        ttft_ms = (t_prefill - t_start) * 1000

        # Trim after prefill: if prompt already exceeds the window, evict now.
        _trim_cache(past_key_values, self.start_size, self.recent_size)

        # Reset peak stats after trim so peak_kv_memory_mb reflects the
        # decode-phase KV cache size, not the transient prefill allocation.
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        # ── Decode (generate remaining tokens) ────────────────────────
        next_token_id = outputs_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = [next_token_id]

        for _ in range(max_new_tokens - 1):
            # absolute_pos = index of the token we are about to generate
            absolute_pos = prompt_tokens + len(generated_ids)
            cache_position = torch.tensor(
                [absolute_pos], dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                out = self.model(
                    next_token_id,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )

            past_key_values = out.past_key_values
            _trim_cache(past_key_values, self.start_size, self.recent_size)

            next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token_id)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        # ── Collect results ───────────────────────────────────────────
        total_time_ms = (t_end - t_start) * 1000
        decode_latency_ms = total_time_ms - ttft_ms

        mem_peak = torch.cuda.max_memory_allocated(self.device)
        peak_kv_memory_mb = (mem_peak - mem_before) / (1024 ** 2)

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
            peak_kv_memory_mb=peak_kv_memory_mb,
            metadata={
                "method": "streaming_llm",
                "start_size": self.start_size,
                "recent_size": self.recent_size,
            },
        )

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
