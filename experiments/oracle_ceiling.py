"""Prompted tasks: how much does the frozen tokenizer cost, and how much do the generated coarse levels cost if every
finer level were perfect? No generator pass is needed (runs on four CPU cores in minutes).

1. Tokenizer ceiling. Each reference is cut to the release's target length (L - 1 tokens, then EOS and PAD, as the
   tokenizer saw text in training), encoded and decoded, and scored against the original reference. The cut alone
   (decoded without the tokenizer) separates the length cap from the quantization loss.
2. Perfect fine levels. Levels < k are the single pass's generated codes (results/cache/<task>_vcycle.pt); every level
   >= k takes the code the quantizer picks for the reference's residual after the reconstruction so far. k = 0 is the
   ceiling, k = K the single pass.

python experiments/oracle_ceiling.py --task wmt14
python experiments/oracle_ceiling.py --task xsum --n 3000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, batches, device, save_json
from vcycle.cond import Conditional
from vcycle.decoding import LevelDecoder
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="wmt14 or xsum")
ap.add_argument("--n", type=int, default=0, help="documents (0 = all)")
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm, cache_prefix=False)
task = Conditional(ncm, dec, ck, os.path.join(DATA, a.task), n=a.n, bs=64)
vq, K, levels, N = dec.vq, dec.K, dec.levels, len(task)
L, eos, pad = levels[-1], task.eos, task.pad
t0 = time.time()

raw = [task.tok.encode(r) for r in task.refs]
lens = np.array([len(r) for r in raw])
x_ref = torch.tensor([(r[:L - 1] + [eos] + [pad] * L)[:L] for r in raw], dtype=torch.long)
st = torch.load(os.path.join(RESULTS, "cache", f"{a.task}_vcycle.pt"), weights_only=False)
gen = [c[:N].long() for c in st["baseline_codes"]]
base_ids = st["variants"]["baseline"]["ids"][:N].long()
dev = ncm.device


def score(x):
    hyps = x if isinstance(x[0], str) else [task.text(r) for r in x.tolist()]
    refs = task.refs
    if task.metric == "bleu":
        from sacrebleu.metrics import BLEU, CHRF
        b = BLEU().corpus_score(hyps, [refs])
        return {"bleu": round(b.score, 2), "bleu_lc": round(BLEU(lowercase=True).corpus_score(hyps, [refs]).score, 2),
                "chrf": round(CHRF().corpus_score(hyps, [refs]).score, 2), "bp": round(b.bp, 4),
                "prec": [round(p, 1) for p in b.precisions],
                "len_ratio": round(sum(len(h.split()) for h in hyps) / sum(len(r.split()) for r in refs), 4)}
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    acc = {k: 0.0 for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(hyps, refs):
        v = sc.score(r, h)
        for k in acc:
            acc[k] += v[k].fmeasure
    out = {k: round(100 * v / len(hyps), 2) for k, v in acc.items()}
    out["words"] = round(float(np.mean([len(h.split()) for h in hyps])), 2)
    return out


def decode(codes, released=True):
    out = []
    for sl in batches(N, 128):
        z = dec.z_from_codes([c[sl].to(dev) for c in codes])
        out.append((task.pick(z) if released else ncm.vqvae.decoder(z).argmax(-1)).cpu())
    return torch.cat(out)


def corrector_codes(k):
    """Generated codes for levels < k, then the quantized reference residual for every level >= k."""
    out = [[] for _ in range(K)]
    for sl in batches(N, 128):
        z = ncm.vqvae.encode_z(x_ref[sl].to(dev)).float()
        B, _, C = z.shape
        z_acc = torch.zeros_like(z)
        for j, l in enumerate(levels):
            if j < k:
                c = gen[j][sl].to(dev)
            else:
                rest = z - z_acc
                flat = (vq._down_for_quant(rest, j, l) if j != K - 1 else rest).reshape(-1, C)
                c = torch.argmin(vq._distances(flat, vq.embeddings[j].weight), dim=1).view(B, l)
            out[j].append(c.cpu())
            z_acc = z_acc + dec.recon(j, c)
    return [torch.cat(o) for o in out]


res = {"task": a.task, "N": N, "levels": levels, "target_len": L, "ref_tokens_mean": round(float(lens.mean()), 2),
       "ref_truncated_pct": round(float(100 * (lens > L - 1).mean()), 2)}
res["single_pass"] = score(base_ids)
res["truncation_only"] = score(x_ref)
gold = []
for sl in batches(N, 128):
    gold.append([g.cpu() for g in dec.encode(x_ref[sl].to(dev))])
gold = [torch.cat([g[k] for g in gold]) for k in range(K)]
ceil_ids = decode(gold)
res["ceiling"] = score(ceil_ids)
res["ceiling_argmax"] = score(decode(gold, released=False))
cut = lambda t: torch.where((t == eos).long().cumsum(1) > 0, torch.full_like(t, pad), t)
res["ceiling_token_acc"] = round(float((cut(ceil_ids) == cut(x_ref)).float().mean()), 4)
res["ceiling_exact_pct"] = round(float(100 * (cut(ceil_ids) == cut(x_ref)).all(1).float().mean()), 2)
res["corrector_matches_encode"] = float(np.mean([(u == v).float().mean().item() for u, v in zip(corrector_codes(0), gold)]))
res["perfect_fine_levels"] = []
for k in range(K + 1):
    s = score(decode(corrector_codes(k)))
    s["k_generated_levels"] = k
    res["perfect_fine_levels"].append(s)
    print(k, s, f"{time.time() - t0:.0f}s", flush=True)
save_json(res, f"oracle_ceiling_{a.task}.json")
