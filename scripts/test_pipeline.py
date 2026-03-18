"""Quick test for the data pipeline. No GPU needed."""

from data.pipeline import load_samples

config_path = "configs/experiment1_short.yaml"
samples = load_samples(config_path)

print(f"\nTotal samples: {len(samples)}")
print(f"Subjects: {set(s.subset for s in samples)}")
print(f"Token range: {min(s.token_count for s in samples)} - {max(s.token_count for s in samples)}")

# Show first sample
s = samples[0]
print(f"\n{'='*60}")
print(f"ID:          {s.id}")
print(f"Subject:     {s.subset}")
print(f"Tokens:      {s.token_count}")
print(f"Reference:   {s.reference}")
print(f"Prompt:\n{s.prompt[:500]}...")