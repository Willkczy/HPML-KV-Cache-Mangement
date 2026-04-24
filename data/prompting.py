"""Shared prompt formatting helpers.

Single source of truth for how a raw prompt becomes the exact text fed to
a model. Used by:
  - data/pipeline.py (eventually) for honest token counting
  - methods/*.py (eventually) for tokenization / generation input

Kept as a scaffold: methods opt in on their own branches during rebase.
Nothing in the shared stack forces usage yet. See plan cutover step C1.
"""


def has_chat_template(tokenizer) -> bool:
    """True if the tokenizer ships a chat template (Instruct models).

    Base models like `Qwen/Qwen2.5-7B` return False and callers should
    fall back to the raw prompt.
    """
    return getattr(tokenizer, "chat_template", None) is not None


def format_as_chat(prompt: str, tokenizer) -> str:
    """Wrap `prompt` as a single-turn user message using the tokenizer's
    chat template, with the assistant generation prompt appended.

    Returns the formatted text. Caller tokenizes as needed.

    Falls back to the raw prompt if the tokenizer has no chat template,
    which keeps the helper safe to drop into code that may still run
    against a base model during the transition.
    """
    if not has_chat_template(tokenizer):
        return prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
