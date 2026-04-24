"""Shared evaluation metrics.

All experiment runners and sweep scripts import from this module so that
accuracy extraction and ROUGE-L scoring use the same implementation.
"""

import re

from rouge_score import rouge_scorer


# ── Multiple-choice answer extraction ───────────────────────────────────────

_ANSWER_PATTERNS = [
    r'[Aa]nswer\s*(?:is|:)\s*([A-D])',
    r'\b([A-D])\b\s*$',              # single letter at end
    r'^\s*([A-D])\b',                # single letter at start
]


def extract_answer(generated_text: str) -> str:
    """Extract A/B/C/D answer from generated text.

    Handles short answers ("B"), preambles ("The answer is B"),
    and chain-of-thought ("...therefore the answer is D").
    Prefers structured patterns; falls back to last standalone letter.
    Returns "" if no letter is found.
    """
    text = generated_text.strip()

    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()

    matches = re.findall(r'\b([A-D])\b', text)
    return matches[-1].upper() if matches else ""


# ── ROUGE-L for summarization ──────────────────────────────────────────────

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between hypothesis and reference."""
    if not hypothesis or not reference:
        return 0.0
    return _rouge_scorer.score(reference, hypothesis)["rougeL"].fmeasure
