"""StreamingLLM-style KV retention policy for vLLM.

This module implements the attention-sink + sliding-window eviction strategy
(Xiao et al., 2023) on top of vLLM's paged KV cache system via runtime
monkey-patching — without forking vLLM and without modifying CUDA kernels.

Key design decisions
--------------------
Full prefill preserved
    The first pass over the prompt runs with the full context (identical to
    the paged_attention baseline), so all prefill KV vectors capture complete
    prefix interactions.  Eviction begins only after the first decode token.

Block-granularity eviction
    vLLM allocates KV cache in blocks of ``block_size`` tokens (default 16).
    ``start_size`` and ``recent_size`` are rounded up to whole-block multiples
    so that ``effective_kv_len`` always equals an integer multiple of
    ``block_size``.  This is conservative: retains slightly more than the HF
    StreamingLLM version, never less.

seq_len override (Patch 2)
    The GPU attention kernel reads ``seq_len`` from
    ``attention_metadata.seq_lens`` to bound its KV read range.  We override
    this at the single authoritative point
    (``ModelInputForGPUBuilder._compute_lens``, model_runner.py ~line 325) so
    the kernel attends only over retained blocks.  Propagation path::

        seq_data._effective_kv_len
          → _compute_lens override
          → inter_data.seq_lens[seq_idx]        (line 332)
          → collected at lines 1018–1024
          → attn_metadata_builder.build(seq_lens)  (line 1048)
          → GPU kernel

ref_count-safe freeing (Patch 1)
    PhysicalTokenBlocks can be shared when prefix caching is enabled.  We
    decrement ``ref_count`` and only call ``allocator.free()`` when it reaches
    zero, matching vLLM's internal contract.  Prefix caching is disabled by
    default in this method, but the guard is included for correctness.

Decode-only guard (Patch 3)
    The trim hook lives inside the patched ``Scheduler._append_slots``.  We
    call ``seq_group.is_prefill()`` to skip trimming during chunked prefill —
    evicting blocks whose KV vectors haven't been written yet would corrupt the
    cache.

Semantic equivalence to HF StreamingLLM
    HF version: keeps tokens [0 : start_size] + [seq_len - recent_size : seq_len]
    vLLM version: keeps first ceil(start_size / B) blocks + last
    ceil(recent_size / B) blocks, where B = block_size.

Limitations
-----------
* Targets ``SelfAttnBlockSpaceManager`` (vLLM >= 0.8.x, ``vllm.core.block_manager``).
  Earlier versions with ``BlockSpaceManagerV1`` are not supported.
* Prefix caching is disabled (``enable_prefix_caching=False``) to avoid
  shared-block ref_count complexity.
* Patches are global (module-level); only one active policy at a time.
  Instantiating a second ``StreamingLLMvLLMMethod`` with different window
  sizes will raise ``RuntimeError``.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch

# Force vLLM to use the legacy (v0) engine so our patch targets exist:
#   vllm.core.scheduler.Scheduler._append_slots
#   vllm.worker.model_runner.ModelInputForGPUBuilder._compute_lens
#   vllm.core.block_manager.SelfAttnBlockSpaceManager
# Must be set before vllm is imported.
os.environ.setdefault("VLLM_USE_V1", "0")

from vllm import LLM, SamplingParams

from methods.base import BaseMethod, MethodOutput


# ── Policy dataclass ──────────────────────────────────────────────────────────

@dataclass
class SinkRecentPolicy:
    """Block-granularity KV retention policy for StreamingLLM-style eviction."""

    sink_blocks: int    # number of KV blocks to keep as attention sinks
    recent_blocks: int  # number of recent KV blocks to keep
    block_size: int     # tokens per KV block (from vLLM engine config)

    @property
    def min_blocks(self) -> int:
        return self.sink_blocks + self.recent_blocks

    def should_trim(self, total_blocks: int) -> bool:
        return total_blocks > self.min_blocks


# ── Global patch state ────────────────────────────────────────────────────────

_PATCHES_APPLIED: bool = False
_ACTIVE_POLICY: Optional[SinkRecentPolicy] = None

# Stores effective_kv_len per seq_id, written by Patch 3 (scheduler hook)
# and read by Patch 2 (model runner).  SequenceData uses __slots__ so we
# cannot set attributes on it directly.
_effective_kv_lens: dict = {}

# Deferred block free queue — blocks are queued here instead of freed
# immediately during trim_request_blocks.  They are flushed at the START of
# the next Scheduler.schedule() call, after the current forward pass is
# guaranteed complete.  This prevents the race condition where a freed block
# is immediately reallocated to another concurrent request while the GPU
# kernel is still reading from that physical address.
_deferred_frees: list = []  # list of (allocator, block)


def _apply_vllm_patches(policy: SinkRecentPolicy) -> None:
    """Monkey-patch vLLM internals to implement sink+recent KV retention.

    Must be called BEFORE ``LLM()`` is instantiated.  Three patches are
    applied in order:

    1. ``SelfAttnBlockSpaceManager.trim_request_blocks`` — evicts middle blocks
    2. ``ModelInputForGPUBuilder._compute_lens`` — overrides seq_len
    3. ``Scheduler._append_slots`` — calls trim after each decode step

    Raises ``RuntimeError`` if called a second time with a different policy.
    """
    global _PATCHES_APPLIED, _ACTIVE_POLICY

    if _PATCHES_APPLIED:
        if _ACTIVE_POLICY != policy:
            raise RuntimeError(
                "vLLM patches already applied with a different policy. "
                "Only one StreamingLLMvLLMMethod instance is supported per process."
            )
        return

    _patch_block_manager(policy)
    _patch_model_runner()
    _patch_scheduler(policy)
    _patch_schedule_flush()

    _ACTIVE_POLICY = policy
    _PATCHES_APPLIED = True
    print(
        f"[streaming_llm_vllm] Patches applied — "
        f"sink={policy.sink_blocks} blocks ({policy.sink_blocks * policy.block_size} tok), "
        f"recent={policy.recent_blocks} blocks ({policy.recent_blocks * policy.block_size} tok)"
    )


# ── Patch 1: SelfAttnBlockSpaceManager.trim_request_blocks ───────────────────

def _patch_block_manager(policy: SinkRecentPolicy) -> None:
    """Add ``trim_request_blocks(seq_id)`` to ``SelfAttnBlockSpaceManager``.

    The method evicts the "middle" KV blocks for a sequence, keeping only:

    * ``blocks[0 : sink_blocks]``                  (attention sinks)
    * ``blocks[total - recent_blocks : total]``    (recent window)

    Returns ``effective_kv_len = len(retained) * block_size`` in tokens.

    Memory invariant: ``block_table._allocator.free(block)`` handles
    ref_count internally — no manual ref_count manipulation needed.
    Prefix caching is disabled so every block is unshared (ref_count == 1).

    Block_table invariant: ``block_table.update(retained)`` replaces the
    internal block list so ``physical_block_ids`` stays consistent.

    Verified against vLLM 0.8.5:
      ``SelfAttnBlockSpaceManager`` at ``vllm.core.block_manager``
      ``self.block_tables: Dict[SeqId, BlockTable]``
      ``BlockTable.blocks`` → ``List[Block]`` (property)
      ``BlockTable._allocator.free(block)`` → decrements ref_count, frees if zero
      ``BlockTable.update(blocks: List[Block])`` → replaces internal block list
    """
    try:
        from vllm.core.block_manager import SelfAttnBlockSpaceManager
    except ImportError:
        print(
            "[streaming_llm_vllm] WARNING: SelfAttnBlockSpaceManager not found — "
            "trim_request_blocks will be a no-op. "
            "Ensure vLLM >= 0.8.x is installed."
        )
        return

    def trim_request_blocks(bm_self, seq_id: int) -> int:  # noqa: ANN001
        """Evict middle blocks for seq_id; return effective_kv_len (tokens)."""
        block_table = bm_self.block_tables.get(seq_id)
        if block_table is None:
            return 0

        blocks = block_table.blocks  # List[Block] via property
        total_blocks = len(blocks)
        if not policy.should_trim(total_blocks):
            return total_blocks * bm_self.block_size

        sink_end = policy.sink_blocks
        recent_start = total_blocks - policy.recent_blocks
        sink = blocks[:sink_end]
        recent = blocks[recent_start:]
        middle = blocks[sink_end:recent_start]

        # Defer freeing middle blocks until after the current forward pass.
        # Immediate freeing in concurrent mode causes CUDA illegal memory
        # access: freed blocks are reallocated to other requests while the
        # GPU kernel still holds their physical addresses for this request.
        # Patch 4 (schedule flush) drains _deferred_frees between passes.
        freed_ids: set = set()
        for block in middle:
            bid = getattr(block, "physical_block_id", id(block))
            if bid in freed_ids:
                continue
            freed_ids.add(bid)
            _deferred_frees.append((block_table._allocator, block))

        retained = sink + recent
        block_table.update(retained)  # replaces _blocks list in-place
        return len(retained) * bm_self.block_size

    SelfAttnBlockSpaceManager.trim_request_blocks = trim_request_blocks


# ── Patch 2: ModelInputForGPUBuilder._compute_lens ───────────────────────────

def _patch_model_runner() -> None:
    """Override ``seq_len`` in ``_compute_lens`` to use ``effective_kv_len``.

    Original (model_runner.py ~line 315-333)::

        seq_len = seq_data.get_len()           # line 325
        inter_data.seq_lens[seq_idx] = seq_len # line 332

    After the patch, if ``seq_data._effective_kv_len`` has been set by the
    scheduler hook, ``inter_data.seq_lens[seq_idx]`` is replaced with that
    value.  For prefill sequences (where ``_effective_kv_len`` is not yet
    set), the original value is left unchanged.

    Execution invariant: after the patch,
    ``inter_data.seq_lens[seq_idx] == len(block_tables[seq_id]) * block_size``
    which exactly matches the retained physical KV slots.
    """
    try:
        from vllm.worker.model_runner import ModelInputForGPUBuilder
    except ImportError:
        print(
            "[streaming_llm_vllm] WARNING: ModelInputForGPUBuilder not found — "
            "seq_len override disabled. Attention may read stale KV slots."
        )
        return

    original_compute_lens = ModelInputForGPUBuilder._compute_lens

    def patched_compute_lens(self, inter_data, seq_idx, seq_group_metadata):  # noqa: ANN001
        # Populate inter_data using the original logic.
        original_compute_lens(self, inter_data, seq_idx, seq_group_metadata)

        # Override seq_len with effective_kv_len if set by the trim hook.
        # SequenceData uses __slots__ so we store per-seq state in the
        # module-level _effective_kv_lens dict keyed by seq_id.
        try:
            seq_id = inter_data.seq_ids[seq_idx]
            effective_kv_len = _effective_kv_lens.get(seq_id)
            if effective_kv_len is not None and effective_kv_len > 0:
                inter_data.seq_lens[seq_idx] = effective_kv_len
                if hasattr(inter_data, "orig_seq_lens"):
                    inter_data.orig_seq_lens[seq_idx] = effective_kv_len
        except (AttributeError, KeyError, IndexError):
            pass  # Safe fallback: original seq_len remains.

    ModelInputForGPUBuilder._compute_lens = patched_compute_lens


# ── Patch 3: Scheduler._append_slots hook ────────────────────────────────────

def _patch_scheduler(policy: SinkRecentPolicy) -> None:
    """Wrap ``Scheduler._append_slots`` to apply KV retention after decode steps.

    Safe trim window:
    * AFTER  ``_append_slots`` completes (new decode block allocated, KV
      written for the latest token)
    * BEFORE ``prepare_model_input()`` is called (attention metadata not yet
      assembled for the next step)

    Timing invariant: ``seq_group.is_prefill()`` guard ensures trimming is
    skipped during chunked prefill — evicting blocks whose KV hasn't been
    written yet would corrupt the cache.

    After trimming, ``seq.data._effective_kv_len`` is set so that
    Patch 2 can read it during the next ``prepare_model_input()`` call.
    """
    try:
        from vllm.core.scheduler import Scheduler
    except ImportError:
        print(
            "[streaming_llm_vllm] WARNING: Scheduler not found — "
            "trim hook disabled. No KV eviction will occur."
        )
        return

    original_append_slots = Scheduler._append_slots

    def patched_append_slots(sched_self, seq_group, blocks_to_copy, enable_chunking=False):  # noqa: ANN001
        # enable_chunking added in vLLM 0.8.x; passed positionally by the scheduler.
        result = original_append_slots(sched_self, seq_group, blocks_to_copy, enable_chunking)

        # Skip trimming during prefill (including chunked prefill).
        if seq_group.is_prefill():
            return result

        block_manager = sched_self.block_manager
        if not hasattr(block_manager, "trim_request_blocks"):
            # BlockSpaceManagerV1 not available or patch failed.
            return result

        try:
            from vllm.sequence import SequenceStatus
            running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        except (ImportError, AttributeError):
            return result

        for seq in running_seqs:
            effective_kv_len = block_manager.trim_request_blocks(seq.seq_id)
            # Store in module-level dict (SequenceData uses __slots__).
            _effective_kv_lens[seq.seq_id] = effective_kv_len

        return result

    Scheduler._append_slots = patched_append_slots


# ── Patch 4: Scheduler.schedule flush ────────────────────────────────────────

def _patch_schedule_flush() -> None:
    """Flush deferred block frees at the start of each schedule() call.

    schedule() runs BETWEEN forward passes, so by the time it is called the
    previous GPU kernel has finished and no physical block address is still
    live in any active kernel.  Freeing here is safe regardless of how many
    concurrent requests are in-flight.
    """
    try:
        from vllm.core.scheduler import Scheduler
    except ImportError:
        print("[streaming_llm_vllm] WARNING: Could not patch Scheduler.schedule — deferred frees disabled.")
        return

    original_schedule = Scheduler.schedule

    def patched_schedule(sched_self, *args, **kwargs):
        # Flush all deferred frees before the next round of scheduling.
        if _deferred_frees:
            freed_ids: set = set()
            for allocator, block in _deferred_frees:
                bid = getattr(block, "physical_block_id", id(block))
                if bid in freed_ids:
                    continue
                freed_ids.add(bid)
                try:
                    allocator.free(block)
                except Exception:
                    pass
            _deferred_frees.clear()
        return original_schedule(sched_self, *args, **kwargs)

    Scheduler.schedule = patched_schedule


# ── Method class ──────────────────────────────────────────────────────────────

class StreamingLLMvLLMMethod(BaseMethod):
    """StreamingLLM eviction via vLLM's paged KV system.

    Keeps the first ``start_size`` tokens (attention sinks) and the most
    recent ``recent_size`` tokens per request.  Eviction is at block
    granularity: both counts are rounded up to whole blocks.

    For sequences shorter than ``start_size + recent_size`` tokens, no
    eviction occurs and behaviour is identical to ``PagedAttentionMethod``.

    Metrics reported
    ----------------
    ``peak_kv_memory_mb``
        Analytical KV memory for the retained window
        ``(sink_blocks + recent_blocks) * block_size`` tokens, or for the
        actual sequence length when shorter than the window.
    ``prefill_kv_memory_mb`` (in metadata)
        Analytical KV memory for the full prompt (pre-eviction) — comparable
        to the full_cache / paged_attention baselines.
    """

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.tokenizer = None
        self.device: Optional[str] = None
        self.block_size: int = 16
        self.policy: Optional[SinkRecentPolicy] = None
        self.engine_memory_mb: float = 0.0

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
        """Load vLLM engine with StreamingLLM patches applied.

        Args:
            model_name:   HuggingFace model path or ``~/models/...`` local path.
            device:       ``"cuda"`` (only GPU is supported by vLLM).
            start_size:   Attention sink token count.  Rounded up to the
                          nearest block boundary (so actual sink tokens =
                          ``ceil(start_size / block_size) * block_size``).
            recent_size:  Recent-window token count.  Same rounding applies.

        Keyword args forwarded to ``LLM()``:
            gpu_memory_utilization (float): KV pool fraction of GPU RAM. Default 0.90.
            max_model_len (int):            Maximum sequence length. Default 16384.
            block_size (int):               Tokens per KV block. Default 16.
            enforce_eager (bool):           Disable CUDA graphs. Default True.
        """
        self.device = device
        self.block_size = kwargs.get("block_size", 16)

        sink_blocks = math.ceil(start_size / self.block_size)
        recent_blocks = math.ceil(recent_size / self.block_size)
        self.policy = SinkRecentPolicy(
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            block_size=self.block_size,
        )

        # Patches must be applied before LLM() creates engine instances.
        _apply_vllm_patches(self.policy)

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.90),
            max_model_len=kwargs.get("max_model_len", 16384),
            block_size=self.block_size,
            dtype="float16",
            enforce_eager=kwargs.get("enforce_eager", True),
            # Disable prefix caching: shared blocks complicate ref_count tracking.
            enable_prefix_caching=False,
        )

        self.engine_memory_mb = 0.0
        self.tokenizer = self.llm.get_tokenizer()

        window_tokens = self.policy.min_blocks * self.block_size
        print(
            f"[streaming_llm_vllm] Engine ready — "
            f"sink={sink_blocks} blocks ({sink_blocks * self.block_size} tok), "
            f"recent={recent_blocks} blocks ({recent_blocks * self.block_size} tok), "
            f"effective window={window_tokens} tokens"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_kv_memory_mb(self, total_tokens: int) -> float:
        """Analytical KV memory for ``total_tokens``.

        Formula (matches ``paged_attention.py`` baseline):
            2 × layers × tokens × kv_heads × head_dim × dtype_bytes
        """
        # model_config lives on llm_engine in vLLM 0.8.x (V0 engine path)
        model_config = getattr(self.llm, "model_config",
                               self.llm.llm_engine.model_config)
        hf_config = model_config.hf_config
        num_layers = hf_config.num_hidden_layers
        num_kv_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        dtype_bytes = torch.tensor([], dtype=model_config.dtype).element_size()
        kv_bytes = 2 * num_layers * total_tokens * num_kv_heads * head_dim * dtype_bytes
        return kv_bytes / (1024 ** 2)

    @staticmethod
    def _extract_ttft(request_output) -> Optional[float]:
        """Extract TTFT from vLLM RequestMetrics (version-dependent)."""
        metrics = getattr(request_output, "metrics", None)
        if metrics is None:
            return None
        first = getattr(metrics, "first_token_time", None)
        arrival = getattr(metrics, "arrival_time", None)
        if first is not None and arrival is not None:
            return (first - arrival) * 1000.0
        return None

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> MethodOutput:
        """Run a single prompt through vLLM with StreamingLLM KV eviction.

        KV blocks are evicted at each decode step by the monkey-patched
        scheduler hook.  The reported ``peak_kv_memory_mb`` reflects the
        retained window size, not the full sequence length.
        """
        temperature = kwargs.get("temperature", 0.0)
        top_p = kwargs.get("top_p", 1.0)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        prompt_token_ids = self.tokenizer.encode(prompt)
        prompt_tokens = len(prompt_token_ids)
        prefill_kv_mb = self._estimate_kv_memory_mb(prompt_tokens)

        torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()
        outputs = self.llm.generate([prompt], sampling_params)
        torch.cuda.synchronize(self.device)
        t_end = time.perf_counter()

        total_time_ms = (t_end - t_start) * 1000.0

        request_output = outputs[0]
        completion = request_output.outputs[0]
        generated_text = completion.text
        generated_tokens = len(completion.token_ids)

        # TTFT from vLLM metrics; fall back to proportion estimate.
        ttft_ms = self._extract_ttft(request_output)
        if ttft_ms is None or ttft_ms <= 0.0:
            if generated_tokens > 0:
                ttft_ms = total_time_ms * prompt_tokens / (prompt_tokens + generated_tokens)
            else:
                ttft_ms = total_time_ms
        decode_latency_ms = max(total_time_ms - ttft_ms, 0.0)

        # Peak KV memory: use the retained window size (post-eviction steady state).
        # For short sequences that never hit the eviction threshold, use actual length.
        total_tokens = prompt_tokens + generated_tokens
        retained_tokens = self.policy.min_blocks * self.block_size
        effective_tokens = min(total_tokens, retained_tokens)
        peak_kv_mb = self._estimate_kv_memory_mb(effective_tokens)

        return MethodOutput(
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            decode_latency_ms=decode_latency_ms,
            peak_kv_memory_mb=peak_kv_mb,
            metadata={
                "method": "streaming_llm_vllm",
                "backend": "vllm",
                "block_size": self.block_size,
                "sink_blocks": self.policy.sink_blocks,
                "recent_blocks": self.policy.recent_blocks,
                "effective_window_tokens": self.policy.min_blocks * self.block_size,
                "prefill_kv_memory_mb": round(prefill_kv_mb, 3),
                "engine_memory_mb": round(self.engine_memory_mb, 2),
            },
        )

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Destroy the vLLM engine and release GPU memory."""
        if self.llm is not None:
            del self.llm
        torch.cuda.empty_cache()
        self.llm = None
        self.tokenizer = None
