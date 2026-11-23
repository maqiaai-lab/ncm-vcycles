"""Prompted tasks: search error or model error? Score whole plans under NCM's own generator, teacher-forced level by
level: log p(plan) = sum_k sum_positions log p(r_k | z_hat(plan_{<k}), prompt), at the released temperature (1).
Plans: own (NCM's greedy single pass, regenerated here), gold (encode of the reference), and every external plan that
experiments/oracle_plans.py saved for the same data. If NCM scored a better plan above its own greedy plan, greedy
decoding would be making search errors; if it does not, the model prefers its own plan (model error; Stahlberg &
Byrne, 2019). Also: where each plan's codes rank in NCM's distribution at levels 0-4, and whether the score gap
predicts, per document, that the external plan's levels 0-4 help (WMT14 sentence chrF, XSum ROUGE-1 of the k = 5 run).

python experiments/plan_scores.py --task wmt14            # also --dev (newstest2013)
python experiments/plan_scores.py --task xsum --n 3000 --bs 16
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, batches, device, save_json
from vcycle.cond import Conditional
from vcycle.decoding import LevelDecoder
from vcycle.devsets import wmt14_dev
from vcycle.lm1b import auroc
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="wmt14 or xsum")
ap.add_argument("--dev", action="store_true", help="WMT14 only: newstest2013 instead of newstest2014")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=dev)
dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
task = Conditional(ncm, dec, ck, wmt14_dev() if a.dev else os.path.join(DATA, a.task), n=a.n, bs=a.bs)
K, levels, N = dec.K, dec.levels, len(task)
L, eos, pad = levels[-1], task.eos, task.pad
COARSE = 5
tag = "_nt2013" if a.dev else ""
t0 = time.time()
stamp = lambda *s: print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)


def canonical_from_text(texts):
    rows = [task.tok.encode(t) for t in texts]
    return torch.tensor([(r[:L - 1] + [eos] + [pad] * L)[:L] for r in rows], dtype=torch.long)


def encode_all(x):
    out = []
    for sl in batches(N, 128):
        out.append([g.cpu() for g in dec.encode(x[sl].to(dev))])
    return [torch.cat([g[k] for g in out]) for k in range(K)]


own = []
for sl in batches(N, a.bs):
    c, _ = task._run(dec.prefix(task.prefix_tokens(list(range(sl.start, sl.stop)))))
    own.append([ci.cpu() for ci in c])
plans = {"own": [torch.cat([o[k] for o in own]) for k in range(K)], "gold": encode_all(canonical_from_text(task.refs))}
prev_path = os.path.join(RESULTS, "cache", f"oracle_plans_{a.task}{tag}.pt")
prev = torch.load(prev_path, weights_only=False) if os.path.exists(prev_path) else {}
for e in [k[4:] for k in prev if k.startswith("ext:") and "|" not in k]:
    plans[e] = encode_all(canonical_from_text(prev[f"ext:{e}"]))
stamp("plans", list(plans))

ll = {p: torch.zeros(N, K, dtype=torch.float64) for p in plans}
rk = {p: [[] for _ in range(COARSE)] for p in plans}
for sl in batches(N, a.bs):
    prefix = dec.prefix(task.prefix_tokens(list(range(sl.start, sl.stop))))
    for p, codes in plans.items():
        z = torch.zeros(sl.stop - sl.start, L, dec.vq.embed_dim, device=dev)
        for k in range(K):
            c = codes[k][sl].to(dev)
            lg = dec.logits(k, prefix, z).float()
            ll[p][sl, k] = torch.log_softmax(lg, -1).gather(-1, c[..., None]).squeeze(-1).sum(1).double().cpu()
            if k < COARSE:
                rk[p][k].append((lg > lg.gather(-1, c[..., None])).sum(-1).cpu())
            z = z + dec.recon(k, c)
stamp("scored")

res = {"task": a.task, "data": "newstest2013" if a.dev else "test", "N": N, "levels": levels,
       "coarse_levels": list(range(COARSE)), "plans": list(plans),
       "mean_ll_per_level": {p: [round(float(v), 2) for v in ll[p].mean(0)] for p in plans},
       "own_rank0_share_coarse": [round(float((torch.cat(rk["own"][k]) == 0).float().mean()), 4) for k in range(COARSE)]}


def sent_metric(texts):
    if task.metric == "bleu":
        from sacrebleu.metrics import CHRF
        ch = CHRF()
        return np.array([ch.sentence_score(h, [r]).score for h, r in zip(texts, task.refs)])
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    return np.array([100 * sc.score(r, h)["rouge1"].fmeasure for h, r in zip(texts, task.refs)])


m_single = sent_metric(prev["single_pass"]) if "single_pass" in prev else None
for p in plans:
    if p == "own":
        continue
    co, al = ll[p][:, :COARSE].sum(1) - ll["own"][:, :COARSE].sum(1), ll[p].sum(1) - ll["own"].sum(1)
    r = [torch.cat(rk[p][k]).flatten().float() for k in range(COARSE)]
    row = {"share_scores_above_own_coarse": round(float((co > 0).double().mean()), 4),
           "share_scores_above_own_all": round(float((al > 0).double().mean()), 4),
           "median_gap_coarse_nats": round(float(co.median()), 2), "median_gap_all_nats": round(float(al.median()), 2),
           "coarse_rank": [{"level": k, "top1": round(float((x == 0).float().mean()), 3),
                            "top5": round(float((x < 5).float().mean()), 3), "top20": round(float((x < 20).float().mean()), 3),
                            "median": float(x.median())} for k, x in enumerate(r)]}
    key = f"ext:{p}|k=5"
    if m_single is not None and key in prev:
        m_t = sent_metric(prev[key])
        helps = (m_t > m_single).astype(int)
        row["k5_helps_share"] = round(float(helps.mean()), 4)
        row["auroc_gap_predicts_help"] = round(float(auroc(co.numpy(), helps)), 4)
        row["metric_single"], row["metric_k5_always"] = round(float(m_single.mean()), 2), round(float(m_t.mean()), 2)
        row["metric_k5_where_ncm_prefers"] = round(float(np.where(co.numpy() > 0, m_t, m_single).mean()), 2)
    res[p] = row
    stamp(p, json.dumps(row)[:300])
save_json(res, f"plan_scores_{a.task}{tag}.json")
torch.save(ll, os.path.join(RESULTS, "cache", f"plan_scores_{a.task}{tag}_ll.pt"))
stamp("done")
