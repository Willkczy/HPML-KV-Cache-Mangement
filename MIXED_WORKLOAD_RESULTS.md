# Mixed Workload Results

Sequential mixed-length replay workload with Poisson arrival timestamps. E2E latency includes simulated queue wait, not true continuous batching.

| Run | Method | N | OOM | Avg TTFT ms | P95 E2E ms | P99 E2E ms | Avg KV MB | MCQ Acc | ROUGE-L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | full_cache | 500 | 0 | 624.534 | 1483158.806 | 1573911.715 | 200.347 | 0.5636 | 0.1945 |
| 500 | paged_attention | 500 | 0 | 569.557 | 1339594.494 | 1422948.695 | 200.402 | 0.5673 | 0.1948 |
| 500 | streaming_llm | 500 | 0 | 624.65 | 1048779.102 | 1110455.857 | 39.884 | 0.56 | 0.1414 |
| 500 | streaming_llm_vllm | 500 | 0 | 568.937 | 437917.175 | 462242.822 | 40.183 | 0.5273 | 0.0718 |
| 50 | full_cache | 50 | 0 | 664.3 | 121189.904 | 126584.303 | 213.668 | 0.6129 | 0.1549 |
| 50 | h2o | 50 | 0 | 668.37 | 57396.237 | 62709.497 | 7.0 | 0.6129 | 0.0 |
| 50 | paged_attention | 50 | 0 | 607.989 | 101412.821 | 106375.2 | 213.3 | 0.6129 | 0.1554 |
| 50 | streaming_llm | 50 | 0 | 667.648 | 96999.88 | 99906.082 | 38.0 | 0.6129 | 0.1308 |
| 50 | streaming_llm_vllm | 50 | 0 | 605.094 | 31460.137 | 32391.497 | 38.259 | 0.5806 | 0.0558 |

## Per-Bucket Breakdown


### full_cache (500 requests)
- short: count=250, oom=0, p95_total_ms=268.381, accuracy=0.576, rouge_l=None
- medium: count=150, oom=0, p95_total_ms=7215.538, accuracy=None, rouge_l=0.1905
- long: count=75, oom=0, p95_total_ms=16384.638, accuracy=None, rouge_l=0.2026
- very_long: count=25, oom=0, p95_total_ms=6906.448, accuracy=0.44, rouge_l=None

### paged_attention (500 requests)
- short: count=250, oom=0, p95_total_ms=240.246, accuracy=0.58, rouge_l=None
- medium: count=150, oom=0, p95_total_ms=6659.487, accuracy=None, rouge_l=0.1906
- long: count=75, oom=0, p95_total_ms=14518.365, accuracy=None, rouge_l=0.2032
- very_long: count=25, oom=0, p95_total_ms=6282.216, accuracy=0.44, rouge_l=None

### streaming_llm (500 requests)
- short: count=250, oom=0, p95_total_ms=268.862, accuracy=0.576, rouge_l=None
- medium: count=150, oom=0, p95_total_ms=6945.25, accuracy=None, rouge_l=0.1442
- long: count=75, oom=0, p95_total_ms=13731.578, accuracy=None, rouge_l=0.1358
- very_long: count=25, oom=0, p95_total_ms=6855.058, accuracy=0.4, rouge_l=None

### streaming_llm_vllm (500 requests)
- short: count=250, oom=0, p95_total_ms=240.132, accuracy=0.54, rouge_l=None
- medium: count=150, oom=0, p95_total_ms=6490.107, accuracy=None, rouge_l=0.0777
- long: count=75, oom=0, p95_total_ms=6131.59, accuracy=None, rouge_l=0.0599
- very_long: count=25, oom=0, p95_total_ms=6320.08, accuracy=0.4, rouge_l=None

### full_cache (50 requests)
- short: count=28, oom=0, p95_total_ms=266.755, accuracy=0.6429, rouge_l=None
- medium: count=10, oom=0, p95_total_ms=7151.338, accuracy=None, rouge_l=0.1504
- long: count=9, oom=0, p95_total_ms=12869.536, accuracy=None, rouge_l=0.16
- very_long: count=3, oom=0, p95_total_ms=5964.458, accuracy=0.3333, rouge_l=None

### h2o (50 requests)
- short: count=28, oom=0, p95_total_ms=194.121, accuracy=0.6429, rouge_l=None
- medium: count=10, oom=0, p95_total_ms=8917.068, accuracy=None, rouge_l=0.0
- long: count=9, oom=0, p95_total_ms=19035.248, accuracy=None, rouge_l=0.0
- very_long: count=3, oom=0, p95_total_ms=6014.733, accuracy=0.3333, rouge_l=None

### paged_attention (50 requests)
- short: count=28, oom=0, p95_total_ms=243.926, accuracy=0.6429, rouge_l=None
- medium: count=10, oom=0, p95_total_ms=6719.138, accuracy=None, rouge_l=0.1525
- long: count=9, oom=0, p95_total_ms=8616.656, accuracy=None, rouge_l=0.1587
- very_long: count=3, oom=0, p95_total_ms=5522.975, accuracy=0.3333, rouge_l=None

### streaming_llm (50 requests)
- short: count=28, oom=0, p95_total_ms=269.329, accuracy=0.6429, rouge_l=None
- medium: count=10, oom=0, p95_total_ms=5590.475, accuracy=None, rouge_l=0.1326
- long: count=9, oom=0, p95_total_ms=11539.771, accuracy=None, rouge_l=0.1288
- very_long: count=3, oom=0, p95_total_ms=5912.713, accuracy=0.3333, rouge_l=None

### streaming_llm_vllm (50 requests)
- short: count=28, oom=0, p95_total_ms=241.414, accuracy=0.6429, rouge_l=None
- medium: count=10, oom=0, p95_total_ms=5411.783, accuracy=None, rouge_l=0.0704
- long: count=9, oom=0, p95_total_ms=3419.274, accuracy=None, rouge_l=0.0395
- very_long: count=3, oom=0, p95_total_ms=5484.29, accuracy=0.0, rouge_l=None
