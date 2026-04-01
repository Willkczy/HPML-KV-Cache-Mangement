"""H2O (Heavy-Hitter Oracle) KV cache eviction method.

Reference:
    Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
    of Large Language Models," NeurIPS 2023.

Strategy:
    During generation, maintain a cumulative attention score for every token
    still in the KV cache.  When the cache would exceed the budget, evict the
    non-recent tokens with the lowest scores.

    Budget  =  hh_size  (heavy hitters)  +  recent_size  (local window)

    Heavy hitters are the tokens that have received the most cumulative
    attention across all heads and all decode steps — they are empirically
    shown to matter most for generation quality.  The local window is always
    kept to preserve immediate context.

Implementation notes:
    - We run the model with attn_implementation="eager" so that attention
      weights are materialised and returned when output_attentions=True.
      (Flash-Attention 2 fuses the computation and does not return weights.)
    - Generation is done token-by-token with an explicit loop so we can
      intercept past_key_values between steps and apply eviction.
    - Eviction is applied *after* the forward pass for a given step, so the
      new token is always present in the cache before any pruning.
"""

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from methods.base import BaseMethod, MethodOutput


# ── KV memory helper ────────────────────────────────────────────────────────

def _kv_memory_mb(cache) -> float:
    """Sum of bytes used by all K and V tensors, converted to MB."""
    if cache is None:
        return 0.0
    if isinstance(cache, DynamicCache):
        tensors = cache.key_cache + cache.value_cache
    else:
        tensors = [t for layer_kv in cache for t in layer_kv]
    return sum(t.nelement() * t.element_size() for t in tensors) / (1024 ** 2)


# ── Core eviction logic ─────────────────────────────────────────────────────

def _update_and_evict(
    past_key_values,
    attn_weights,           # tuple of [B, H, q_len, kv_len] per layer
    hh_scores: Optional[list],   # accumulated scores per layer (list of tensors)
    hh_size: int,
    recent_size: int,
):
    """Accumulate attention scores and evict over-budget tokens.

    Supports both legacy tuple-of-tuples and DynamicCache (transformers >= 4.38).

    Args:
        past_key_values: HuggingFace cache — either a tuple of (k, v) per layer
            or a DynamicCache object. Each k/v has shape [B, H, kv_len, head_dim].
        attn_weights: tuple of attention weight tensors per layer.
            Shape per layer: [B, H, q_len, kv_len].
        hh_scores: per-layer cumulative score tensors from the previous step,
            or None on the very first call.
        hh_size: number of heavy-hitter tokens to keep.
        recent_size: number of most-recent tokens to always keep.

    Returns:
        (new_past_key_values, new_hh_scores)
    """
    budget = hh_size + recent_size
    new_scores = []

    use_dynamic_cache = hasattr(past_key_values, "key_cache")

    if use_dynamic_cache:
        num_layers = len(past_key_values.key_cache)
        keys = past_key_values.key_cache
        vals = past_key_values.value_cache
    else:
        num_layers = len(past_key_values)
        keys = [layer_kv[0] for layer_kv in past_key_values]
        vals = [layer_kv[1] for layer_kv in past_key_values]

    for layer_idx in range(num_layers):
        k = keys[layer_idx]   # [B, H, kv_len, head_dim]
        v = vals[layer_idx]
        layer_attn = attn_weights[layer_idx]
        kv_len = k.shape[2]

        # Sum attention over batch, heads, and query positions → [kv_len]
        step_score = layer_attn.sum(dim=(0, 1, 2)).detach()

        score = step_score.clone()
        if hh_scores is not None and hh_scores[layer_idx] is not None:
            prev = hh_scores[layer_idx]
            score[: prev.shape[0]] += prev

        if kv_len > budget:
            non_recent_len = kv_len - recent_size

            if non_recent_len <= 0:
                keep_idx = torch.arange(kv_len - budget, kv_len, device=k.device)
            else:
                non_recent_scores = score[:non_recent_len]
                actual_hh = min(hh_size, non_recent_len)
                _, top_idx = non_recent_scores.topk(actual_hh, largest=True)
                top_idx = top_idx.sort().values
                recent_idx = torch.arange(
                    kv_len - recent_size, kv_len, device=k.device
                )
                keep_idx = torch.cat([top_idx, recent_idx])

            keys[layer_idx] = k[:, :, keep_idx, :]
            vals[layer_idx] = v[:, :, keep_idx, :]
            score = score[keep_idx]

        new_scores.append(score)

    # Always return a fresh DynamicCache so the model's get_seq_length()
    # and position embedding logic sees a consistent object.
    new_cache = DynamicCache()
    new_cache.key_cache = keys
    new_cache.value_cache = vals
    new_cache._seen_tokens = keys[0].shape[2] if keys else 0
    return new_cache, new_scores


# ── BaseMethod implementation ────────────────────────────────────────────────

class H2OMethod(BaseMethod):
    """H2O KV cache eviction.

    Keyword args for setup() / generate():
        hh_size     (int): heavy-hitter budget (default 64)
        recent_size (int): local-window budget  (default 64)
    """

    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None:
        self.device = device
        self.hh_size = int(kwargs.get("hh_size", 64))
        self.recent_size = int(kwargs.get("recent_size", 64))

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # eager attention is required to materialise attention weights
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        hh_size = int(kwargs.get("hh_size", self.hh_size))
        recent_size = int(kwargs.get("recent_size", self.recent_size))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id

        torch.cuda.reset_peak_memory_stats(self.device)

        generated_ids: list[int] = []
        past_key_values = None
        hh_scores = None
        peak_kv_mb = 0.0

        with torch.no_grad():
            # ── Prefill ──────────────────────────────────────────────────────
            t0 = time.perf_counter()

            outputs = self.model(
                input_ids=input_ids,
                past_key_values=None,
                output_attentions=True,
                use_cache=True,
            )

            t_first_token = time.perf_counter()
            ttft_ms = (t_first_token - t0) * 1000.0

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

            past_key_values, hh_scores = _update_and_evict(
                outputs.past_key_values,
                outputs.attentions,
                hh_scores,
                hh_size,
                recent_size,
            )
            peak_kv_mb = max(peak_kv_mb, _kv_memory_mb(past_key_values))

            if next_token.item() == eos_id or max_new_tokens <= 1:
                t_end = time.perf_counter()
                return self._make_output(
                    generated_ids, prompt_tokens,
                    ttft_ms, (t_end - t0) * 1000.0,
                    peak_kv_mb, hh_size, recent_size,
                )

            # ── Decode ───────────────────────────────────────────────────────
            for _ in range(max_new_tokens - 1):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    output_attentions=True,
                    use_cache=True,
                )

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())

                past_key_values, hh_scores = _update_and_evict(
                    outputs.past_key_values,
                    outputs.attentions,
                    hh_scores,
                    hh_size,
                    recent_size,
                )
                peak_kv_mb = max(peak_kv_mb, _kv_memory_mb(past_key_values))

                if next_token.item() == eos_id:
                    break

        t_end = time.perf_counter()
        return self._make_output(
            generated_ids, prompt_tokens,
            ttft_ms, (t_end - t0) * 1000.0,
            peak_kv_mb, hh_size, recent_size,
        )

    def teardown(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _make_output(
        self,
        generated_ids: list[int],
        prompt_tokens: int,
        ttft_ms: float,
        total_ms: float,
        peak_kv_mb: float,
        hh_size: int,
        recent_size: int,
    ) -> MethodOutput:
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return MethodOutput(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            decode_latency_ms=total_ms - ttft_ms,
            peak_kv_memory_mb=peak_kv_mb,
            metadata={"hh_size": hh_size, "recent_size": recent_size},
        )
