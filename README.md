# Speculative Decoding

**Lossless greedy speculative decoding for Hugging Face models, with no draft model: up to 3.3× faster generation on Qwen3-8B, with identical output.**

A drafter guesses the next 8 tokens by looking the last few tokens up earlier in the text (prompt lookup). The full model checks all 8 guesses in **one** forward pass over its KV cache, keeps the longest run that matches what it would have generated anyway, and adds one token of its own. Rejected guesses are cropped from the cache. The output is the same as plain greedy decoding; only the number of full-model passes changes.

## Results (Qwen3-8B, bf16, one RTX 3090 Ti, 200 new tokens)

| Task | Plain greedy | Speculative | Speedup | Model passes | Output identical? |
|---|---|---|---|---|---|
| Add type hints to a code snippet | 41.5 tok/s | 137.2 tok/s | **3.31×** | 200 → 54 | 200/200 tokens |
| Copy the first sentences of a passage | 41.4 tok/s | 79.8 tok/s | **1.93×** | 200 → 91 | see below |
| Continue a Wikipedia passage freely | 41.4 tok/s | 48.4 tok/s | **1.17×** | 200 → 153 | 200/200 tokens |

The speedup tracks how much of the output already appears in the prompt: code edits and quoting repeat a lot, free continuation repeats little. It never made generation slower in these runs, because a failed guess costs almost nothing extra (verifying 9 tokens takes about as long as generating 1 on a GPU at this size).

**Passage-copy divergence, explained:** the outputs match for 82 tokens, then differ. At that step the model's top two choices (" rather" and " a") have exactly the same logit, 35.0 vs 35.0. Plain greedy and the multi-token verify pass run slightly different bf16 arithmetic, so they break the tie differently. Every later token follows from that. It is a numerical tie, not a logic error ([`results/divergence.json`](results/divergence.json)); in fp32, or with ties broken by token id, both paths would agree.

Full numbers: [`results/real.json`](results/real.json).

## Honest notes

- Prompt lookup drafting is a known idea (Saxena, 2023, "prompt lookup decoding"); this repo is a small, readable implementation of it, with correctness checked against plain greedy and real timings on an 8B model.
- **An earlier version of this repo was wrong.** Its "draft" was the full model itself, and each guess was checked with another uncached full pass, so it did up to 4× the work per token and ran at 0.65× on distilgpt2. That code (`src/spec_decode/core.py`) is kept only as a toy rejection-sampling example; the real implementation is `src/spec_decode/real.py`.
- Greedy decoding only (no sampling), batch size 1, one run per task.

## Usage

```python
from spec_decode.real import speculative, greedy

ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
tokens, passes, accepted = speculative(model, ids, max_new=200, k=8)
```

Reproduce (needs a local Qwen3-8B and about 17 GB of GPU memory):

```bash
python results/run_real.py
python results/check_divergence.py
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests/
```
