#!/usr/bin/env python3
"""Real results: lossless greedy speculative decoding (prompt-lookup drafts) on Qwen3-8B.

For each task we generate MAX_NEW tokens with plain greedy decoding and with
speculative decoding, check the outputs are identical, and time both on the GPU.
Results go to results/real.json.
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from spec_decode.real import greedy, speculative  # noqa: E402

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
MAX_NEW = 200
OUT = HERE / "real.json"

CODE = '''def load_orders(path):
    orders = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            order_id, customer, amount, status = parts
            if status == "cancelled":
                continue
            orders.append({"id": order_id, "customer": customer, "amount": float(amount)})
    return orders


def total_by_customer(orders):
    totals = {}
    for o in orders:
        totals[o["customer"]] = totals.get(o["customer"], 0) + o["amount"]
    return totals
'''


def tasks(tok):
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    passage = text[5000:7000]

    def chat(msg):
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    return {
        "code_edit": chat("Add Python type hints to every function below. Reply with the full code only.\n\n" + CODE),
        "passage_qa": chat("Copy out, word for word, the first three sentences of this passage.\n\n" + passage),
        "open_continuation": passage,
    }


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.time()
    r = fn()
    torch.cuda.synchronize()
    return r, time.time() - t0


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, local_files_only=True, device_map="cuda").eval()
    greedy(model, tok("warm up", return_tensors="pt").input_ids.cuda(), 8)

    results = {"setup": {"model": "Qwen3-8B bf16, RTX 3090 Ti", "max_new_tokens": MAX_NEW,
                         "drafter": "prompt lookup, n<=3, k=8"}}
    for name, prompt in tasks(tok).items():
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        (g_out, g_passes), g_t = timed(lambda: greedy(model, ids, MAX_NEW))
        (s_out, s_passes, acc), s_t = timed(lambda: speculative(model, ids, MAX_NEW))
        same = sum(a == b for a, b in zip(g_out, s_out))
        first_diff = next((i for i, (a, b) in enumerate(zip(g_out, s_out)) if a != b), None)
        results[name] = {
            "prompt_tokens": ids.shape[1],
            "greedy_tok_per_s": round(MAX_NEW / g_t, 1),
            "spec_tok_per_s": round(MAX_NEW / s_t, 1),
            "speedup": round(g_t / s_t, 2),
            "target_passes": {"greedy": g_passes, "speculative": s_passes},
            "draft_tokens_accepted": acc,
            "identical_tokens": f"{same}/{MAX_NEW}",
            "first_difference_at": first_diff,
        }
        print(name, results[name], flush=True)
        OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
