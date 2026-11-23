"""Confirmation run for soft decisions on the prompted tasks, with every setting fixed before the test data is seen.

WMT14: the soft-decision temperature Ts and the guidance scale are chosen on newstest2013, the standard development
set (3,000 sentences, fetched with sacrebleu), by cased BLEU; the chosen arms then run once on all of newstest2014
(3,003 sentences). Arms on the development set: the single pass, soft decisions at every level for each Ts in TS_DEV,
the released guidance ramp for each scale, and soft (chosen Ts) + guidance for each scale.
XSum: no development split is released, so the confirmation uses the 5,334 test articles that
experiments/cond_guide_soft.py never saw (its 6,000 were drawn with seed 0). Arms fixed in advance: the single pass,
Ts = 0.5 (the setting that kept recall and fluency there), Ts = 1 (its pre-registered pick) and the paper's guidance
(s = 0.5). Full-test-set scores pool these articles with the earlier 6,000 (per-document ROUGE is additive).

Writes results/cond_soft_confirm_<task>.json, results/cache/cond_soft_confirm_<task>.pt (ids and per-document scores;
the judge reads it with --cache) and, for WMT14, results/cache/cond_soft_confirm_wmt14_texts.json for
experiments/comet_score.py. Resumable.

python experiments/cond_soft_confirm.py --task wmt14
python experiments/cond_soft_confirm.py --task xsum
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json
from vcycle.cond import Conditional, bleu_from_stats
from vcycle.decoding import LevelDecoder
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="wmt14 or xsum")
ap.add_argument("--bs", type=int, default=0, help="batch size (0: 64 for WMT14, 32 for XSum)")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--n", type=int, default=0, help="first n documents of each set only (smoke tests)")
ap.add_argument("--dev_dir", default="", help="WMT14 development data (default: data/wmt14_dev, fetched if missing)")
ap.add_argument("--tag", default="")
a = ap.parse_args()

TS_DEV = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SCALES = [0.5, 1.0, 2.0]
torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
K = dec.K
bs = a.bs or (64 if a.task == "wmt14" else 32)
ALL = list(range(K))
name = f"cond_soft_confirm_{a.task}{a.tag}"
state_path = os.path.join(RESULTS, "cache", name + ".pt")
st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {"arms": {}, "dev": {}}
out = st.get("out") or {"task": a.task, "levels": dec.levels, "K": K, "device": device(), "bs": bs, "arms": {},
                        "dev": {}}
st["out"] = out
seed_of = lambda key: zlib.crc32(f"{name}|{key}".encode())


def save():
    torch.save(st, state_path)
    save_json(out, name + ".json")


def wmt13_dev():
    """newstest2013 de-en as the release's pickles (lists of strings), fetched once with sacrebleu."""
    d = a.dev_dir or os.path.join(DATA, "wmt14_dev")
    if not os.path.exists(os.path.join(d, "sources.pickle")):
        os.makedirs(d, exist_ok=True)
        echo = lambda side: subprocess.run([sys.executable, "-m", "sacrebleu", "-t", "wmt13", "-l", "de-en", "--echo", side],
                                           check=True, capture_output=True, text=True, encoding="utf-8").stdout
        src, ref = echo("src").rstrip("\n").split("\n"), echo("ref").rstrip("\n").split("\n")
        assert len(src) == len(ref) == 3000, (len(src), len(ref))
        pickle.dump(src, open(os.path.join(d, "sources.pickle"), "wb"))
        pickle.dump(ref, open(os.path.join(d, "reference_translations.pickle"), "wb"))
    return d


def metric(per, ix=None):
    ix = np.arange(len(next(iter(per.values())))) if ix is None else ix
    if a.task == "wmt14":
        s = per["bleu_stats"][ix].sum(0)
        return {"bleu": bleu_from_stats(s), "chrf": float(per["chrf"][ix].mean()), "len_ratio": float(s[0] / s[1])}
    r = {k: 100 * float(per[k][ix].mean()) for k in ("rouge1", "rouge2", "rougeL")}
    r["gen_ppl"] = float(np.exp(per["nll"][ix].sum() / per["ntok"][ix].sum()))
    r["words"] = float(per["words"][ix].mean())
    return r


def ci_vs(per, base, key, n):
    rng = np.random.default_rng(seed_of(key))
    vals = []
    for _ in range(a.boot):
        b = rng.integers(0, n, n)
        vals.append(metric(per, b)[key] - metric(base, b)[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(task, slot, key, guide_w=None, soft=None, extra=None):
    """One arm on all documents of task; slot is st['arms'] (test) or st['dev']."""
    if key in st[slot]:
        return
    torch.manual_seed(seed_of(f"{slot}|{key}"))
    with Timer() as t:
        _, ids = task.generate(list(range(len(task))), guide_w=guide_w, soft=soft)
    per, corpus, hyps = task.score(ids)
    st[slot][key] = {"ids": ids.to(torch.int32), "per": per, "hyps": hyps}
    nfe = K + sum(1 for w in (guide_w or []) if w > 0)
    r = dict(corpus, nfe=nfe, seconds=t.s, guide_w=guide_w, soft_ts=None if soft is None else soft[0], **(extra or {}))
    out[slot][key] = r
    base = st[slot].get("single pass")
    if base is not None and key != "single pass":
        r["delta"] = {k: v - metric(base["per"])[k] for k, v in metric(per).items()}
        if slot == "arms":
            keys = ["bleu", "chrf"] if a.task == "wmt14" else ["rouge1", "rouge2", "rougeL"]
            r["ci"] = {k: ci_vs(per, base["per"], k, len(task)) for k in keys}
    print(f"[{slot}] {key:32s} nfe {nfe:2d} {t.s:5.0f}s " + " ".join(
        f"{k} {v:.2f}" for k, v in corpus.items() if isinstance(v, float) and k != "bleu_from_stats") +
        (" | delta " + " ".join(f"{k} {v:+.2f}" for k, v in r["delta"].items()) if "delta" in r else ""), flush=True)
    save()


soft = lambda ts: {k: ts for k in ALL}
ramp = lambda s: dec.released_ramp(s, K)

if a.task == "wmt14":
    dev = Conditional(ncm, dec, ck, wmt13_dev(), bs=bs, n=a.n)
    test = Conditional(ncm, dec, ck, os.path.join(DATA, "wmt14"), bs=bs, n=a.n)
    out["dev_docs"], out["test_docs"] = len(dev), len(test)
    # 1. Development set: choose Ts, then the guidance scale alone and on top of soft decisions.
    run(dev, "dev", "single pass")
    for ts in TS_DEV:
        run(dev, "dev", f"soft Ts={ts:g}", soft=soft(ts))
    for s in SCALES:
        run(dev, "dev", f"cfg ramp s={s:g}", guide_w=ramp(s))
    best = lambda prefix, combo=False: max((k for k in out["dev"] if k.startswith(prefix) and ("+" in k) == combo),
                                           key=lambda k: out["dev"][k]["bleu"])
    ts = out["dev"][best("soft Ts=")]["soft_ts"]
    for s in SCALES:
        run(dev, "dev", f"soft Ts={ts:g} + cfg ramp s={s:g}", guide_w=ramp(s), soft=soft(ts))
    s_cfg = best("cfg ramp s=")
    s_both = best(f"soft Ts={ts:g} + cfg", combo=True)
    out["choice"] = {"soft": f"soft Ts={ts:g}", "cfg": s_cfg, "soft + cfg": s_both, "by": "cased BLEU on newstest2013"}
    print("chosen on the development set:", out["choice"], flush=True)
    save()
    # 2. Test set, once per chosen arm (plus Ts = 1, the earlier run's setting, if it was not chosen).
    run(test, "arms", "single pass")
    run(test, "arms", f"soft Ts={ts:g}", soft=soft(ts))
    if ts != 1.0:
        run(test, "arms", "soft Ts=1", soft=soft(1.0))
    run(test, "arms", s_cfg, guide_w=out["dev"][s_cfg]["guide_w"])
    run(test, "arms", s_both, guide_w=out["dev"][s_both]["guide_w"], soft=soft(ts))
    out["docs"] = "all"
    # Texts for COMET: source, reference and every test arm's hypotheses.
    with open(os.path.join(RESULTS, "cache", name + "_texts.json"), "w", encoding="utf-8") as f:
        json.dump({"src": [s.strip() for s in test.items], "ref": test.refs,
                   "hyps": {k: v["hyps"] for k, v in st["arms"].items()}}, f, ensure_ascii=False)
else:
    test = Conditional(ncm, dec, ck, os.path.join(DATA, "xsum"), bs=bs)
    seen = set(np.random.default_rng(0).permutation(len(test))[:6000].tolist())  # cond_guide_soft.py --n 6000
    docs = [i for i in range(len(test)) if i not in seen][:a.n or None]
    test.items, test.refs = [test.items[i] for i in docs], [test.refs[i] for i in docs]
    out["docs"] = docs
    arms = {"single pass": (None, None), "soft Ts=0.5": (None, soft(0.5)), "soft Ts=1": (None, soft(1.0)),
            "cfg ramp s=0.5": (ramp(0.5), None)}
    for key, (w, sf) in arms.items():
        run(test, "arms", key, guide_w=w, soft=sf)
    # Full test set: pool with cond_guide_soft.py's 6,000 articles (same arms, there named "soft all Ts=...").
    prev_path = os.path.join(RESULTS, "cache", "cond_guide_soft_xsum.pt")
    if os.path.exists(prev_path) and not a.n:
        prev = torch.load(prev_path, weights_only=False)["arms"]
        old = {"single pass": "single pass", "soft Ts=0.5": "soft all Ts=0.5", "soft Ts=1": "soft all Ts=1",
               "cfg ramp s=0.5": "cfg ramp s=0.5"}
        pooled = {k: {m: np.concatenate([prev[old[k]]["per"][m], st["arms"][k]["per"][m]])
                      for m in ("rouge1", "rouge2", "rougeL", "nll", "ntok", "words")} for k in arms}
        out["full_test_set"] = {"docs": len(pooled["single pass"]["rouge1"]),
                                "arms": {k: metric(v) for k, v in pooled.items()}}
        print("full test set:", json.dumps(out["full_test_set"], indent=1), flush=True)
out["done"] = True
save()
