"""WMT14: variants of soft decisions, all from NCM's own distribution (no reference, no other model), 0 extra NFE.
Every level k is decided by one rule, from the generator's logits l_k and the codebook e_k:
  hard        the argmax code (the released greedy pass);
  soft        the posterior-mean vector sum_c softmax(l_k / Ts)_c e_k(c) (Section 1.4);
  ntm         nearest to mean: the code the quantizer picks for that posterior-mean vector (a hard, on-codebook
              version of soft), on levels 0-4, 0..K-2 or all;
  gate        commit only when sure: the argmax code where the top probability (T = 1) is at least tau, soft elsewhere;
  topk        the posterior mean over the top-k codes only, renormalized at Ts.

--stage tune (newstest2014, all arms): documents split at random (seed 1, as cond_guide_soft.py) into a tuning half A
and a report half B; the best arm of each family on A (cased BLEU) is reported on B against the single pass and
against plain soft decisions (Ts = 1), by paired bootstrap over sentences.
--stage confirm (fixed arms: single pass, soft Ts=1, top8 Ts=2 and top16 Ts=1, the two best top-k arms on A): deltas
over all sentences, on newstest2014 or, with --dev, newstest2013; texts are saved for experiments/comet_oracle.py.

python experiments/soft_ablations.py --stage tune
python experiments/soft_ablations.py --stage confirm            # and --stage confirm --dev
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
from vcycle.cond import Conditional, bleu_from_stats
from vcycle.decoding import LevelDecoder
from vcycle.devsets import wmt14_dev
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["tune", "confirm"])
ap.add_argument("--dev", action="store_true", help="confirm on newstest2013 instead of newstest2014")
ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
assert not (a.dev and a.stage == "tune"), "the tuning stage runs on newstest2014's half A"

torch.set_grad_enabled(False)
dev = device()
ck = os.path.join(ARTIFACTS, "wmt14")
ncm = load(ck, device=dev)
dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
task = Conditional(ncm, dec, ck, wmt14_dev() if a.dev else os.path.join(DATA, "wmt14"), n=a.n, bs=32)
K, N, vq, L = dec.K, len(task), dec.vq, dec.levels[-1]
perm = np.random.default_rng(1).permutation(N)
half = {"A": np.sort(perm[:N // 2]), "B": np.sort(perm[N // 2:]), "all": np.arange(N)}
cpath = os.path.join(RESULTS, "cache", "wmt14_vcycle.pt")
cached = None
if not a.dev and os.path.exists(cpath):
    cached = [task.text(r) for r in torch.load(cpath, weights_only=False)["variants"]["baseline"]["ids"][:N].long().tolist()]
name = f"soft_ablations_wmt14{'_nt2013' if a.dev else ''}_{a.stage}"
t0 = time.time()
stamp = lambda *s: print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)


def to_full(k, q):
    """A level-k code-space vector (B, l_k, D) -> its contribution to z_hat, as recon() and soft_recon() compute it."""
    if k != K - 1:
        q = vq._up_to_full(q, k, L)
    return vq._apply_quant_resi(q, k)


def decode(mode="hard", ts=1.0, levels=None, tau=None, topk=None):
    """Texts, and per level: the share of positions whose code differs from the argmax (ntm) or is committed (gate)."""
    texts, count, total = [], np.zeros(K), np.zeros(K)
    for sl in batches(N, task.bs):
        prefix = dec.prefix(task.prefix_tokens(list(range(sl.start, sl.stop))))
        z = torch.zeros(sl.stop - sl.start, L, vq.embed_dim, device=dev)
        for k in range(K):
            lg = dec.logits(k, prefix, z).float()
            E = vq.embeddings[k].weight.float()
            c = lg.argmax(-1)
            on = levels is None or k in levels
            if mode == "hard" or not on:
                q = E[c]
            elif mode == "soft":
                q = (lg / ts).softmax(-1) @ E
            elif mode == "ntm":
                mu = (lg / ts).softmax(-1) @ E
                B, l, D = mu.shape
                c2 = torch.argmin(vq._distances(mu.reshape(-1, D), E), dim=1).view(B, l)
                count[k] += float((c2 != c).sum())
                total[k] += c.numel()
                q = E[c2]
            elif mode == "gate":
                sure = lg.softmax(-1).amax(-1) >= tau
                count[k] += float(sure.sum())
                total[k] += sure.numel()
                q = torch.where(sure[..., None], E[c], (lg / ts).softmax(-1) @ E)
            else:
                v, i = lg.topk(topk, dim=-1)
                q = ((v / ts).softmax(-1)[..., None] * E[i]).sum(-2)
            z = z + to_full(k, q)
        texts += [task.text(r) for r in task.pick(z).cpu().tolist()]
    with np.errstate(invalid="ignore"):
        share = (count / total).round(4).tolist() if total.any() else None
    return texts, share


def stats(texts):
    from sacrebleu.metrics import BLEU
    bleu, rows = BLEU(), []
    for h, r in zip(texts, task.refs):
        s = bleu.corpus_score([h], [[r]])
        rows.append([s.sys_len, s.ref_len] + list(s.counts) + list(s.totals))
    return np.array(rows, dtype=np.int64)


def corpus(texts):
    from sacrebleu.metrics import BLEU, CHRF
    out = {}
    for h, ix in half.items():
        hy, rf = [texts[i] for i in ix], [task.refs[i] for i in ix]
        b = BLEU().corpus_score(hy, [rf])
        out[h] = {"bleu": round(b.score, 2), "bp": round(b.bp, 4), "prec": [round(p, 1) for p in b.precisions],
                  "chrf": round(CHRF().corpus_score(hy, [rf]).score, 2)}
    out["all"]["bleu_lc"] = round(BLEU(lowercase=True).corpus_score(texts, [task.refs]).score, 2)
    out["all"]["len_ratio"] = round(sum(len(t.split()) for t in texts) / sum(len(r.split()) for r in task.refs), 4)
    return out


boots = {h: np.random.default_rng(0).integers(0, len(ix), size=(1000, len(ix))) for h, ix in half.items()}


def delta(sa, sb, h):
    boot, sa, sb = boots[h], sa[half[h]], sb[half[h]]
    d = np.array([bleu_from_stats(x) - bleu_from_stats(y) for x, y in zip(sa[boot].sum(1), sb[boot].sum(1))])
    return [round(bleu_from_stats(sa.sum(0)) - bleu_from_stats(sb.sum(0)), 2), round(float(np.percentile(d, 2.5)), 2),
            round(float(np.percentile(d, 97.5)), 2)]


arms = {"single pass": dict(mode="hard"), "soft all Ts=1": dict(mode="soft", ts=1.0)}
if a.stage == "tune":
    arms["soft coarse Ts=1"] = dict(mode="soft", ts=1.0, levels=set(range(5)))
    for band, lv in (("coarse", set(range(5))), ("not final", set(range(K - 1))), ("all", None)):
        for ts in (0.5, 1.0, 2.0):
            arms[f"ntm {band} Ts={ts:g}"] = dict(mode="ntm", ts=ts, levels=lv)
    for tau, ts in ((0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0), (0.5, 2.0), (0.9, 2.0)):
        arms[f"gate tau={tau:g} Ts={ts:g}"] = dict(mode="gate", tau=tau, ts=ts)
    for ts in (1.0, 2.0):
        for kk in (2, 4, 8, 16):
            arms[f"top{kk} Ts={ts:g}"] = dict(mode="topk", topk=kk, ts=ts)
else:
    arms["top8 Ts=2"] = dict(mode="topk", topk=8, ts=2.0)
    arms["top16 Ts=1"] = dict(mode="topk", topk=16, ts=1.0)
hs = ("B",) if a.stage == "tune" else ("all", "B")

res = {"task": "wmt14", "data": "newstest2013" if a.dev else "newstest2014", "stage": a.stage, "N": N,
       "split": {"A": len(half["A"]), "B": len(half["B"]), "seed": 1}, "arms": {}}
arm_stats, texts_out = {}, {}
for nm, kw in arms.items():
    texts, share = decode(**kw)
    texts_out[nm], arm_stats[nm] = texts, stats(texts)
    row = corpus(texts)
    if nm == "single pass":
        if cached is not None:
            row["matches_cached"] = float(np.mean([x == y for x, y in zip(texts, cached)]))
    else:
        for h in hs:
            row[f"delta_{h}_vs_single"] = delta(arm_stats[nm], arm_stats["single pass"], h)
            if nm != "soft all Ts=1":
                row[f"delta_{h}_vs_soft"] = delta(arm_stats[nm], arm_stats["soft all Ts=1"], h)
    if share is not None:
        row["changed_share_per_level" if kw["mode"] == "ntm" else "committed_share_per_level"] = share
    res["arms"][nm] = row
    stamp(nm, row["all"], {k: v for k, v in row.items() if k.startswith("delta")})
    save_json(res, name + ".json")

if a.stage == "tune":
    for fam in ("ntm", "gate", "top"):
        best = max([n for n in arms if n.startswith(fam)], key=lambda n: res["arms"][n]["A"]["bleu"])
        res[f"selected_on_A_{fam}"] = {"arm": best, "B_vs_single": res["arms"][best]["delta_B_vs_single"],
                                       "B_vs_soft": res["arms"][best]["delta_B_vs_soft"]}
        stamp("selected on A:", res[f"selected_on_A_{fam}"])
    save_json(res, name + ".json")
else:
    srcs = [it if isinstance(it, str) else "" for it in task.items]
    json.dump({"sources": srcs, "refs": list(task.refs), "systems": texts_out},
              open(os.path.join(RESULTS, "cache", name + "_texts.json"), "w"))
stamp("done")
