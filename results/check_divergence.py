#!/usr/bin/env python3
"""Why did speculative and greedy outputs differ on passage_qa? Record the target's
top-2 logit margin at every greedy step, and at the first differing step compare the
logits the single-token pass and the multi-token verify pass produce for that position."""
import json, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
from run_real import MODEL_PATH, MAX_NEW, tasks
from spec_decode.real import greedy, speculative
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, local_files_only=True, device_map="cuda").eval()
ids = tok(tasks(tok)["passage_qa"], return_tensors="pt").input_ids.cuda()
g, _ = greedy(model, ids, MAX_NEW)
s, _, _ = speculative(model, ids, MAX_NEW)
d = next(i for i, (a, b) in enumerate(zip(g, s)) if a != b)
prefix = torch.cat([ids, torch.tensor([g[:d]], device="cuda")], 1)
with torch.no_grad():
    full = model(prefix).logits[0, -1].float()          # one pass over the whole prefix
    top = full.topk(2)
out = {"first_difference_at": d, "greedy_token": tok.decode([g[d]]), "spec_token": tok.decode([s[d]]),
       "logit_of_greedy_token": round(full[g[d]].item(), 4), "logit_of_spec_token": round(full[s[d]].item(), 4),
       "top2_margin_full_pass": round((top.values[0] - top.values[1]).item(), 4),
       "top2_tokens_full_pass": [tok.decode([t]) for t in top.indices.tolist()]}
(HERE / "divergence.json").write_text(json.dumps(out, indent=2)); print(out)
