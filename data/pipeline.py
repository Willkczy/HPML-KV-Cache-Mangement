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


# ── LongBench v2 formatting ───────────────────────────────────────────────

LONGBENCH_CHOICES = ["A", "B", "C", "D"]


def _format_longbench_question(example: dict, prompt_style: str = "short") -> str:
    """Format a LongBench v2 question with context and lettered choices.

    Args:
        prompt_style: "short" for just the answer letter,
                      "explain" to ask for reasoning before answering.
    """
    context = example["context"]
    question = example["question"]
    choices = [
        example["choice_A"],
        example["choice_B"],
        example["choice_C"],
        example["choice_D"],
    ]
    formatted = f"{context}\n\n{question}\n"
    for letter, choice in zip(LONGBENCH_CHOICES, choices):
        formatted += f"  {letter}. {choice}\n"
    if prompt_style == "explain":
        formatted += (
            "\nThink step by step to answer the question. "
            "After your reasoning, you MUST end your response with:\n"
            "Answer: X\n"
            "where X is A, B, C, or D."
        )
    else:
        formatted += "Answer:"
    return formatted


def _load_longbench(config: dict, tokenizer) -> list[Sample]:
    """Load LongBench v2 samples filtered by token range and domain."""
    data_cfg = config["data"]
    domains = data_cfg.get("domains", [])
    cap = data_cfg.get("num_samples_per_domain", -1)
    tok_min = data_cfg["token_range"]["min"]
    tok_max = data_cfg["token_range"]["max"]
    difficulty = data_cfg.get("difficulty", None)  # "easy", "hard", or None for all
    prompt_style = data_cfg.get("prompt_style", "short")  # "short" or "explain"

    ds = load_dataset("THUDM/LongBench-v2", split="train")

    # Group by domain for optional per-domain caps
    from collections import defaultdict
    domain_counts: dict[str, int] = defaultdict(int)

    samples = []
    for example in ds:
        # Filter by domain if specified
        if domains and example["domain"] not in domains:
            continue

        # Filter by difficulty if specified
        if difficulty and example["difficulty"] != difficulty:
            continue

        # Per-domain cap
        domain = example["domain"]
        if cap > 0 and domain_counts[domain] >= cap:
            continue

        prompt = _format_longbench_question(example, prompt_style=prompt_style)
        reference = example["answer"]

        # Token count filter
        token_count = len(tokenizer.encode(prompt))
        if token_count < tok_min or token_count > tok_max:
            continue

        samples.append(Sample(
            id=f"longbench_{example['_id']}",
            prompt=prompt,
            reference=reference,
            dataset="longbench",
            subset=example["sub_domain"],
            token_count=token_count,
            metadata={
                "domain": domain,
                "sub_domain": example["sub_domain"],
                "difficulty": example["difficulty"],
                "length_category": example["length"],
            },
        ))

        domain_counts[domain] += 1

    return samples


# ── GovReport summarization ───────────────────────────────────────────────

def _format_govreport_prompt(report: str) -> str:
    """Format a GovReport document for summarization."""
    return (
        f"Please read the following government report and write a concise summary.\n\n"
        f"{report}\n\n"
        f"Summary:"
    )


def _load_govreport(config: dict, tokenizer) -> list[Sample]:
    """Load GovReport summarization samples filtered by token range."""
    data_cfg = config["data"]
    split = data_cfg.get("split", "test")
    cap = data_cfg.get("num_samples", -1)
    tok_min = data_cfg["token_range"]["min"]
    tok_max = data_cfg["token_range"]["max"]

    ds = load_dataset("ccdv/govreport-summarization", split=split)

    samples = []
    for i, example in enumerate(ds):
        prompt = _format_govreport_prompt(example["report"])
        reference = example["summary"]

        token_count = len(tokenizer.encode(prompt))
        if token_count < tok_min or token_count > tok_max:
            continue

        samples.append(Sample(
            id=f"govreport_{i}",
            prompt=prompt,
            reference=reference,
            dataset="govreport",
            subset="govreport",
            token_count=token_count,
            metadata={
                "reference_token_count": len(tokenizer.encode(reference)),
            },
        ))

        if cap > 0 and len(samples) >= cap:
            break

    return samples


# ── Public API ──────────────────────────────────────────────────────────────

LOADERS = {
    "mmlu": _load_mmlu,
    "longbench": _load_longbench,
    "govreport": _load_govreport,
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