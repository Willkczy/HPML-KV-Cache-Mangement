"""Shared data pipeline for all KV cache experiments.

Loads datasets from HuggingFace, formats prompts, filters by token count,
and returns a list of standardized Sample objects that every method consumes.

Usage:
    from data.pipeline import load_samples

    samples = load_samples("configs/experiment1_short.yaml")
    for s in samples:
        print(s.id, s.prompt, s.reference)
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class Sample:
    """Standardized sample that every method receives."""
    id: str                # unique identifier
    prompt: str            # formatted prompt ready for the model
    reference: str         # ground truth answer for evaluation
    dataset: str           # "mmlu", "cnn_dailymail", "longbench"
    subset: str            # e.g. MMLU subject name
    token_count: int       # prompt token count (from tokenizer)
    metadata: dict = field(default_factory=dict)


# ── MMLU formatting ────────────────────────────────────────────────────────

MMLU_CHOICES = ["A", "B", "C", "D"]


def _format_mmlu_question(example: dict) -> str:
    """Format a single MMLU question with lettered choices."""
    question = example["question"]
    choices = example["choices"]
    formatted = f"{question}\n"
    for letter, choice in zip(MMLU_CHOICES, choices):
        formatted += f"  {letter}. {choice}\n"
    formatted += "Answer:"
    return formatted


def _build_mmlu_fewshot_prefix(fewshot_examples: list) -> str:
    """Build the few-shot prefix from a list of examples."""
    prefix = ""
    for ex in fewshot_examples:
        q = _format_mmlu_question(ex)
        answer = MMLU_CHOICES[ex["answer"]]
        prefix += f"{q} {answer}\n\n"
    return prefix


def _load_mmlu(config: dict, tokenizer) -> list[Sample]:
    """Load MMLU samples filtered by token range."""
    data_cfg = config["data"]
    subjects = data_cfg.get("subjects", [])
    num_few_shot = data_cfg.get("num_few_shot", 5)
    cap = data_cfg.get("num_samples_per_subject", -1)
    tok_min = data_cfg["token_range"]["min"]
    tok_max = data_cfg["token_range"]["max"]

    # Determine which subjects to load
    if not subjects:
        ds = load_dataset("cais/mmlu", "all", split="test")
        subjects = list(set(ds["subject"]))

    samples = []

    for subject in subjects:
        # Load few-shot examples from dev split
        dev_split = load_dataset(
            "cais/mmlu", subject, split="dev"
        )
        fewshot_examples = list(dev_split.select(range(min(num_few_shot, len(dev_split)))))
        fewshot_prefix = _build_mmlu_fewshot_prefix(fewshot_examples)

        # Load test split
        test_split = load_dataset(
            "cais/mmlu", subject, split="test"
        )

        count = 0
        for i, example in enumerate(test_split):
            question = _format_mmlu_question(example)
            prompt = fewshot_prefix + question
            reference = MMLU_CHOICES[example["answer"]]

            # Token count filter
            token_count = len(tokenizer.encode(prompt))
            if token_count < tok_min or token_count > tok_max:
                continue

            samples.append(Sample(
                id=f"mmlu_{subject}_{i}",
                prompt=prompt,
                reference=reference,
                dataset="mmlu",
                subset=subject,
                token_count=token_count,
                metadata={
                    "num_few_shot": num_few_shot,
                    "answer_index": example["answer"],
                },
            ))

            count += 1
            if cap > 0 and count >= cap:
                break

    return samples


# ── Public API ──────────────────────────────────────────────────────────────

LOADERS = {
    "mmlu": _load_mmlu,
    # "cnn_dailymail": _load_cnn_dailymail,   # TODO: medium bucket
    # "longbench": _load_longbench,           # TODO: long bucket
}


def load_samples(config_path: str) -> list[Sample]:
    """Load samples from a config file. Entry point for all experiments.

    Args:
        config_path: path to a YAML config file.

    Returns:
        List of Sample objects, sorted by token_count ascending.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_name = config["data"]["dataset"]
    if dataset_name not in LOADERS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {list(LOADERS.keys())}"
        )

    model_name = config["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples = LOADERS[dataset_name](config, tokenizer)
    samples.sort(key=lambda s: s.token_count)

    print(f"[pipeline] Loaded {len(samples)} samples from '{dataset_name}' "
          f"({config['data']['token_range']['min']}-{config['data']['token_range']['max']} tokens)")

    return samples