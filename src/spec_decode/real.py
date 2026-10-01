"""Lossless greedy speculative decoding for Hugging Face causal LMs.

A drafter guesses the next k tokens cheaply; the target model checks all k in
ONE forward pass over its KV cache, keeps the longest prefix that matches its
own greedy choice, and adds one token of its own. The output is identical to
plain greedy decoding; only the number of target forward passes changes.

Drafters:
  ngram_draft      prompt lookup: find the last n tokens earlier in the text and
                   propose what followed them (no extra model, no extra memory)
"""
import torch
from transformers import DynamicCache


def ngram_draft(ids: list[int], k: int, n_max: int = 3) -> list[int]:
    """Propose up to k tokens by matching the last n tokens earlier in ids."""
    for n in range(n_max, 0, -1):
        if len(ids) <= n:
            continue
        tail = ids[-n:]
        for start in range(len(ids) - n - 1, -1, -1):
            if ids[start:start + n] == tail:
                return ids[start + n:start + n + k]
    return []


@torch.no_grad()
def greedy(model, prompt_ids: torch.Tensor, max_new: int) -> tuple[list[int], int]:
    """Plain greedy decoding with a KV cache. Returns (new tokens, target passes)."""
    cache = DynamicCache()
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    tok = int(out.logits[0, -1].argmax())
    new, passes = [tok], 1
    while len(new) < max_new:
        out = model(torch.tensor([[tok]], device=prompt_ids.device), past_key_values=cache, use_cache=True)
        tok = int(out.logits[0, -1].argmax())
        new.append(tok)
        passes += 1
    return new, passes


@torch.no_grad()
def speculative(model, prompt_ids: torch.Tensor, max_new: int, k: int = 8, drafter=ngram_draft) -> tuple[list[int], int, int]:
    """Greedy speculative decoding. Returns (new tokens, target passes, drafted tokens accepted)."""
    dev = prompt_ids.device
    cache = DynamicCache()
    out = model(prompt_ids, past_key_values=cache, use_cache=True)
    ids = prompt_ids[0].tolist()
    nxt = int(out.logits[0, -1].argmax())  # target's next token, not yet in the cache
    new, passes, accepted = [], 1, 0
    while len(new) < max_new:
        draft = drafter(ids + [nxt], k)[: max(0, max_new - len(new) - 1)]
        block = [nxt] + draft
        out = model(torch.tensor([block], device=dev), past_key_values=cache, use_cache=True)
        passes += 1
        preds = out.logits[0].argmax(-1).tolist()  # preds[i] = target's choice after block[:i+1]
        n_ok = 0
        while n_ok < len(draft) and draft[n_ok] == preds[n_ok]:
            n_ok += 1
        keep = block[: 1 + n_ok]
        ids += keep
        new += keep
        accepted += n_ok
        cache.crop(len(ids))  # drop cache entries for rejected draft tokens
        nxt = preds[n_ok]
    return new[:max_new], passes, accepted
