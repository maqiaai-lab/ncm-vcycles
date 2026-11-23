"""Prompted tasks (WMT14, XSum): the two changes together. V-cycles started from the soft-decision draft.

Arms (greedy decoding throughout, as released):
  single pass                       the released pass (hard decisions)
  soft                              soft decisions at every level (Ts = 1 on WMT14, 0.5 on XSum: cond_soft_confirm.py)
  soft + vcycle m=M c=C             C V-cycles from the soft draft: re-encode the output in canonical form (text, EOS,
                                    PAD, as cond_vcycle.py), keep the re-encoded levels < M as hard codes, regenerate
                                    levels >= M with soft decisions (same Ts)
  soft draft, hard vcycle m=M c=C   the same, but the regenerated levels use hard decisions
  soft + no-restrict m=M            control without re-encoding: levels < M keep the soft pass's own chosen codes (hard),
                                    levels >= M are regenerated with soft decisions. Under hard decisions this control
                                    reproduces the single pass; under soft decisions it does not, because the soft pass
                                    fed posterior means, not codes, at the coarse levels. It separates "re-encoding" from
                                    "hardening the coarse levels".
  vcycle m=M0 c=1..3                sanity arm: the hard V-cycle from the hard draft, as cond_vcycle.py (M0 = 3 on WMT14,
                                    5 on XSum); must reproduce the stored numbers up to hardware near-ties

Metrics as cond_vcycle.py / cond_soft_confirm.py (WMT14: cased and lowercased BLEU, chrF, length ratio; XSum:
ROUGE-1/2/L, GPT-2 generative perplexity, words). Paired bootstrap over documents of every arm against `soft` and
against `single pass`. WMT14: --dev runs newstest2013 (for choosing M and C), without it newstest2014. XSum has no
development split: every arm runs on all test articles, and the scores are reported separately for the 6,000 tuning
articles of cond_guide_soft.py and the 5,334 held-out articles of cond_soft_confirm.py (same split, seed 0).

Writes results/cond_cycle_soft_<task>[_dev].json and results/cache/cond_cycle_soft_<task>[_dev].pt. Resumable.

python experiments/cond_cycle_soft.py --task wmt14 --dev
python experiments/cond_cycle_soft.py --task wmt14
python experiments/cond_cycle_soft.py --task xsum
"""
import argparse
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, batches, device, save_json
from vcycle.cond import Conditional, bleu_from_stats
from vcycle.decoding import LevelDecoder
from vcycle.devsets import wmt14_dev
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="wmt14 or xsum")
ap.add_argument("--dev", action="store_true", help="WMT14: newstest2013 instead of newstest2014")
ap.add_argument("--n", type=int, default=0, help="first n documents only (smoke tests)")
ap.add_argument("--bs", type=int, default=0, help="batch size (0: 64 for WMT14, 32 for XSum)")
ap.add_argument("--levels", default="3,5", help="restart levels M")
ap.add_argument("--cycles", type=int, default=3)
ap.add_argument("--ts", type=float, default=0.0, help="soft-decision temperature (0: 1 on WMT14, 0.5 on XSum)")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--tag", default="")
a = ap.parse_args()
assert a.task in ("wmt14", "xsum") and not (a.dev and a.task != "wmt14"), "--dev exists for wmt14 only"

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
K = dec.K
bs = a.bs or (64 if a.task == "wmt14" else 32)
TS = a.ts or (1.0 if a.task == "wmt14" else 0.5)
data_dir = wmt14_dev() if a.dev else os.path.join(DATA, a.task)
task = Conditional(ncm, dec, ck, data_dir, bs=bs, n=a.n)
assert task.top_k == 1 or task.temp == 0, "this script assumes the released greedy decoding"
idxs = list(range(len(task)))
N = len(idxs)
levels = [int(v) for v in a.levels.split(",") if v]
M0 = 3 if a.task == "wmt14" else 5
name = f"cond_cycle_soft_{a.task}{'_dev' if a.dev else ''}{a.tag}"
state_path = os.path.join(RESULTS, "cache", name + ".pt")
os.makedirs(os.path.dirname(state_path), exist_ok=True)
st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {"arms": {}}
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
out = st.get("out") or {"task": a.task, "set": "newstest2013" if a.dev else ("newstest2014" if a.task == "wmt14" else "xsum test"),
                        "n": N, "levels": dec.levels, "K": K, "soft_ts": TS, "device": device(), "gpu": gpu, "bs": bs,
                        "restriction": "canonical: text, EOS, PAD", "arms": {}}
st["out"] = out
seed_of = lambda key: zlib.crc32(f"{name}|{key}".encode())

# Subsets on which the scores are reported.
subsets = {"all": np.arange(N)}
if a.task == "xsum":
    full = len(Conditional(ncm, dec, ck, data_dir, bs=bs)) if a.n else N       # the split is defined on the full test set
    seen = set(np.random.default_rng(0).permutation(full)[:6000].tolist())     # cond_guide_soft.py --n 6000
    subsets["tuning"] = np.array([i for i in idxs if i in seen], dtype=int)
    subsets["held_out"] = np.array([i for i in idxs if i not in seen], dtype=int)
    out["subsets"] = {k: int(len(v)) for k, v in subsets.items()}


def save():
    torch.save(st, state_path)
    save_json(out, name + ".json")


def restart(ids, m, soft=None, codes=None):
    """One cycle from level m. codes=None: keep the re-encoded levels < m of the canonical output (a V-cycle);
    codes given: keep those codes instead (no re-encoding). soft: {level: Ts} for the regenerated levels, or None."""
    out_c, out_i = [], []
    for sl in batches(N, task.bs):
        base = dec.encode(task.canonical(ids[sl]).to(task.dev)) if codes is None else [c[sl].to(task.dev) for c in codes]
        c2, z = task._run(dec.prefix(task.prefix_tokens(idxs[sl])), start=m, codes=base, soft=soft)
        out_c.append([c.cpu() for c in c2])
        out_i.append(task.pick(z).cpu())
    return [torch.cat([c[k] for c in out_c]) for k in range(K)], torch.cat(out_i)


def metric(per, ix):
    if a.task == "wmt14":
        s = per["bleu_stats"][ix].sum(0)
        return {"bleu": bleu_from_stats(s), "chrf_sentence": float(per["chrf"][ix].mean()), "len_ratio_tokens": float(s[0] / s[1])}
    r = {k: 100 * float(per[k][ix].mean()) for k in ("rouge1", "rouge2", "rougeL")}
    r["gen_ppl"] = float(np.exp(per["nll"][ix].sum() / max(per["ntok"][ix].sum(), 1)))
    r["words"] = float(per["words"][ix].mean())
    return r


def paired(key, base):
    """Paired bootstrap over documents, key - base, on every subset: [difference, lo, hi] per metric."""
    pv, pb = st["arms"][key]["per"], st["arms"][base]["per"]
    res = {}
    for sub, ix in subsets.items():
        if len(ix) == 0:
            continue
        rng = np.random.default_rng(seed_of(f"{key}|{base}|{sub}"))
        bx = ix[rng.integers(0, len(ix), size=(a.boot, len(ix)))]
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        r = {}
        if a.task == "wmt14":
            d = lambda i: bleu_from_stats(pv["bleu_stats"][i].sum(0)) - bleu_from_stats(pb["bleu_stats"][i].sum(0))
            r["bleu"] = [d(ix)] + ci([d(b) for b in bx])
            dc = pv["chrf"] - pb["chrf"]
            r["chrf_sentence"] = [float(dc[ix].mean())] + ci(dc[bx].mean(1))
        else:
            for k in ("rouge1", "rouge2", "rougeL"):
                dk = 100 * (pv[k] - pb[k])
                r[k] = [float(dk[ix].mean())] + ci(dk[bx].mean(1))
            pp = lambda p, i: np.exp(p["nll"][i].sum(-1) / p["ntok"][i].sum(-1))
            r["gen_ppl_ratio"] = [float(pp(pv, ix) / pp(pb, ix))] + ci(pp(pv, bx) / pp(pb, bx))
            dw = pv["words"] - pb["words"]
            r["words"] = [float(dw[ix].mean())] + ci(dw[bx].mean(1))
        res[sub] = r
    return res


def record(key, codes, ids, nfe, seconds, prev_ids=None, keep_codes=False):
    per, corpus, hyps = task.score(ids)
    st["arms"][key] = {"ids": ids.to(torch.int32), "per": per, "hyps": hyps}
    if keep_codes:
        st["arms"][key]["codes"] = [c.to(torch.int32) for c in codes]
    r = dict(corpus, nfe=nfe, seconds=seconds)
    if len(subsets) > 1:
        r["by_subset"] = {sub: metric(per, ix) for sub, ix in subsets.items() if len(ix)}
    if prev_ids is not None:
        r["changed_vs_prev"] = float((ids != prev_ids).any(1).float().mean())
    for base in ("single pass", "soft"):
        if base in st["arms"] and key != base:
            r["vs_" + base] = paired(key, base)
    out["arms"][key] = r
    print(f"{key:36s} nfe {nfe:3d} {seconds:6.0f}s " + " ".join(
        f"{k} {v:.2f}" for k, v in corpus.items() if isinstance(v, float) and k != "bleu_from_stats"), flush=True)
    save()


def run(key, fn, nfe, prev_key=None, keep_codes=False):
    """Run one arm unless it is cached; returns its ids."""
    if key not in st["arms"]:
        torch.manual_seed(seed_of(key))
        with Timer() as t:
            codes, ids = fn()
        record(key, codes, ids, nfe, t.s, None if prev_key is None else st["arms"][prev_key]["ids"].long(), keep_codes)
    return st["arms"][key]["ids"].long()


soft_all = {k: TS for k in range(K)}
hard_ids = run("single pass", lambda: task.generate(idxs), K, keep_codes=True)
soft_ids = run("soft", lambda: task.generate(idxs, soft=soft_all), K, keep_codes=True)
soft_codes = [c.long() for c in st["arms"]["soft"]["codes"]]

for m in levels:
    soft_m = {k: TS for k in range(m, K)}
    run(f"soft + no-restrict m={m}", lambda: restart(soft_ids, m, soft=soft_m, codes=soft_codes), 2 * K - m, "soft")
    for label, sf in ((f"soft + vcycle m={m}", soft_m), (f"soft draft, hard vcycle m={m}", None)):
        cur, prev = soft_ids, "soft"
        for c in range(1, a.cycles + 1):
            key = f"{label} c={c}"
            cur = run(key, lambda: restart(cur, m, soft=sf), K + c * (K - m), prev)
            prev = key

# Sanity arm: the hard V-cycle from the hard draft, as cond_vcycle.py.
hard_codes = [c.long() for c in st["arms"]["single pass"]["codes"]]
cc, cur, prev = hard_codes, hard_ids, "single pass"
for c in range(1, a.cycles + 1):
    key = f"vcycle m={M0} c={c}"
    if key not in st["arms"]:
        torch.manual_seed(seed_of(key))
        with Timer() as t:
            cc, new = task.restart(cc, cur, idxs, M0, reencode=True)
        record(key, cc, new, K + c * (K - M0), t.s, cur)
    cur = st["arms"][key]["ids"].long()
    prev = key

out["done"] = True
save()
print("wrote", os.path.join(RESULTS, name + ".json"), flush=True)
