"""Print DynamicCache attributes so we know the exact API on this transformers version."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B", dtype=torch.float16, device_map="cuda",
    trust_remote_code=True, attn_implementation="eager"
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)

ids = tok("Hello", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**ids, use_cache=True, output_attentions=True)

cache = out.past_key_values
print("type:", type(cache))
print("dir:", [a for a in dir(cache) if not a.startswith("__")])
print("len:", len(cache))

# Try iterating
try:
    for i, layer in enumerate(cache):
        print(f"layer {i} type: {type(layer)}, len: {len(layer)}")
        if hasattr(layer, '__len__'):
            k = layer[0]
            print(f"  k shape: {k.shape}")
        break
except Exception as e:
    print("iteration failed:", e)
