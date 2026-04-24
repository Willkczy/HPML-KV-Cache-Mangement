"""Full KV cache baseline using HuggingFace Transformers.

Stores all KV pairs without eviction — the standard approach.
Serves as the quality and latency reference for other methods.
"""

import re
import time
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RepetitionPenaltyLogitsProcessor,
)

from data.prompting import format_as_chat
from methods.base import BaseMethod, MethodOutput, kv_memory_mb


class FullCacheMethod(BaseMethod):
    """Baseline: standard HuggingFace generation with full KV cache."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.stop_token_ids: set[int] = set()

    def setup(self, model_name: str, device: str = "cuda", **kwargs) -> None:
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()
        self.stop_token_ids = self._resolve_stop_token_ids()

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        """Run standard HF generation and collect timing + memory stats."""
        temperature = float(kwargs.get("temperature", 0.0))
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))

        # ── Tokenize ──────────────────────────────────────────────
        formatted_prompt = format_as_chat(prompt, self.tokenizer)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]

        # ── Memory baseline ───────────────────────────────────────
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        # ── Prefill (measures TTFT) ───────────────────────────────
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        with torch.no_grad():
            outputs_prefill = self.model(input_ids, use_cache=True)
            past_key_values = outputs_prefill.past_key_values

        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
        ttft_ms = (t_prefill - t_start) * 1000
        prefill_kv_mb = kv_memory_mb(past_key_values)

        # ── Decode (generate remaining tokens) ────────────────────
        # Start from the last token of prefill
        next_token_id = self._select_next_token(
            outputs_prefill.logits[:, -1, :],
            generated_so_far=None,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        generated_ids = [next_token_id]
        generated_so_far = next_token_id

        if self._should_stop(next_token_id, generated_so_far, max_new_tokens):
            torch.cuda.synchronize()
            t_end = time.perf_counter()
            return self._make_output(
                generated_so_far=generated_so_far,
                prompt_tokens=prompt_tokens,
                ttft_ms=ttft_ms,
                total_time_ms=(t_end - t_start) * 1000,
                past_key_values=past_key_values,
                prefill_kv_mb=prefill_kv_mb,
                mem_before=mem_before,
            )

        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                out = self.model(
                    next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            next_token_id = self._select_next_token(
                out.logits[:, -1, :],
                generated_so_far=generated_so_far,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token_id)
            generated_so_far = torch.cat([generated_so_far, next_token_id], dim=-1)

            if self._should_stop(next_token_id, generated_so_far, max_new_tokens):
                break

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        return self._make_output(
            generated_so_far=generated_so_far,
            prompt_tokens=prompt_tokens,
            ttft_ms=ttft_ms,
            total_time_ms=(t_end - t_start) * 1000,
            past_key_values=past_key_values,
            prefill_kv_mb=prefill_kv_mb,
            mem_before=mem_before,
        )

    def teardown(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None

    def _resolve_stop_token_ids(self) -> set[int]:
        """Collect EOS-style token ids used by Qwen chat templates."""
        stop_ids: set[int] = set()

        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_id = self.tokenizer.unk_token_id
        if im_end_id is not None and im_end_id >= 0 and (unk_id is None or im_end_id != unk_id):
            stop_ids.add(im_end_id)

        return stop_ids

    def _select_next_token(
        self,
        logits: torch.Tensor,
        generated_so_far: Optional[torch.Tensor],
        temperature: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Apply decode-time hygiene before greedy/sample token selection."""
        scores = logits
        if generated_so_far is not None and generated_so_far.numel() > 0 and repetition_penalty != 1.0:
            processor = RepetitionPenaltyLogitsProcessor(repetition_penalty)
            scores = processor(generated_so_far, scores)

        if temperature > 0.0:
            probs = torch.softmax(scores / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        return scores.argmax(dim=-1, keepdim=True)

    def _should_stop(
        self,
        next_token_id: torch.Tensor,
        generated_so_far: torch.Tensor,
        max_new_tokens: int,
    ) -> bool:
        """Stop on EOS/chat terminators and early-exit clean short answers."""
        if next_token_id.item() in self.stop_token_ids:
            return True

        if max_new_tokens <= 10:
            text = self.tokenizer.decode(generated_so_far[0], skip_special_tokens=True)
            if re.match(r"^\s*[A-D]\b", text):
                return True

        return False

    def _make_output(
        self,
        *,
        generated_so_far: torch.Tensor,
        prompt_tokens: int,
        ttft_ms: float,
        total_time_ms: float,
        past_key_values,
        prefill_kv_mb: float,
        mem_before: int,
    ) -> MethodOutput:
        """Build the standardized output payload."""
        peak_kv_mb = kv_memory_mb(past_key_values)
        decode_latency_ms = total_time_ms - ttft_ms
        gpu_peak_mb = (torch.cuda.max_memory_allocated(self.device) - mem_before) / (1024 ** 2)
        generated_text = self.tokenizer.decode(
            generated_so_far[0], skip_special_tokens=True
        )

        return MethodOutput(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_so_far.shape[1],
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            decode_latency_ms=decode_latency_ms,
            peak_kv_memory_mb=peak_kv_mb,
            metadata={
                "method": "full_cache",
                "gpu_peak_mb": round(gpu_peak_mb, 3),
                "prefill_kv_memory_mb": round(prefill_kv_mb, 3),
            },
        )
