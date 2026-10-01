# Speculative decoding: measured results

Qwen3-8B, bf16, RTX 3090 Ti, greedy, 200 new tokens, prompt-lookup drafts (n <= 3, k = 8).
Command: `python results/run_real.py` (writes `results/real.json`).

| Task | Greedy tok/s | Speculative tok/s | Speedup | Target passes | Identical tokens |
|---|---|---|---|---|---|
| code_edit | 41.5 | 137.2 | 3.31x | 200 -> 54 | 200/200 |
| passage_qa | 41.4 | 79.8 | 1.93x | 200 -> 91 | 83/200 (exact bf16 tie at token 82, see divergence.json) |
| open_continuation | 41.4 | 48.4 | 1.17x | 200 -> 153 | 200/200 |

Earlier results in this file (distilgpt2, 0.65x) measured a broken implementation that ran the full model several times per token; they are withdrawn.
