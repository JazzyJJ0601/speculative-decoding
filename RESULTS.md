# Speculative Decoding — Real Benchmark Results

**Status:** Measured on distilgpt2 (82M), not yet on Qwen. Speculative decoding was slower than plain decoding here: 60.5 vs 92.9 tokens/s (0.65x).
Autoregressive vs. heuristic speculative decoding on a small
language model (distilgpt2). The draft component proposes
3 tokens per step from the target's own top-3
logits, and the target verifies via rejection sampling.
All runs use seed 42 for reproducibility.
## Environment
| Field | Value |
|---|---|
| Model | distilgpt2 |
| Device | cuda |
| Parameters | 82M |
| Layers | 6 |
| Seed | 42 |
| Prompt length | 9 words |
| Generated tokens | 30 |
| Draft steps | 3 |

## Baseline (Autoregressive)
- **Latency**: 0.3228s
- **Throughput**: 92.9 tokens/s
- **Memory delta**: 290.6 MB
- **Perplexity**: 5.6124
- **Output tail**: `...s over the lazy dog.





























`

## Speculative Decoding
- **Latency**: 0.5124s
- **Throughput**: 60.5 tokens/s
- **Memory delta**: 298.1 MB
- **Perplexity**: 5.3775
- **Acceptance rate**: 46.7%
- **Avg draft lookahead**: 0.9 tokens
- **Output tail**: `...s over the lazy dog.





























`

## Comparison
- **Speedup** (speculative tps / baseline tps): 0.65x (slower)
- **Latency improvement**: -0.1895s
- **Throughput gain**: -32.4 tok/s (-35%)
- **Perplexity change**: -0.2349

Interpretation: when the draft model is a simple heuristic reusing the
target's own logits, the overhead of draft proposal + verification
can outweigh the gains from batching — especially for small models
on GPU where single-token forward passes are already fast. The
~47% acceptance rate shows the draft quality is moderate. A
dedicated draft model (e.g. a tiny n-gram or distilled transformer)
would reduce the overhead and likely yield a net speedup.

## Command
```
python3 results/run_real.py
```
