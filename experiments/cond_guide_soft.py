"""Prompted tasks (WMT14, XSum): soft decisions and level-banded classifier-free guidance, in one greedy pass.

Soft decisions: at the chosen levels the generator's next level and the decoder read the posterior-mean code vector
sum_c softmax(logits / Ts)_c e(c) instead of the argmax code's (0 extra NFE). Guidance: (1 + w_k) cond - w_k uncond at
level k against the released null prompt, either with the released ramp w_k = s k / (K - 1) (generate(cfg_scale=s);
the paper's rows use s = 1.0 on WMT14 and 0.5 on XSum), or with a constant w on the coarse levels only, or on the fine
levels only. A guided level costs 2 NFE (the paper's count); compute is also reported as generator token positions
(prompt + levels, the null prompt embedded once per run) relative to the single pass, and as wall-clock seconds.

Design, fixed before the run: documents are split at random into a tuning half (A) and a report half (B). Stage 1 runs
every single-factor arm. Stage 2 takes the best soft arm, the best coarse-band arm and the best ramp arm on half A
(WMT14: cased BLEU; XSum: mean of ROUGE-1/2/L) and runs the two combinations soft + coarse band and soft + ramp. Every
arm is reported on half B against the single pass (paired bootstrap over documents), and on all documents.
Checks: the single pass reproduces the cached single pass of experiments/cond_vcycle.py, and guided decoding with the
released ramp reproduces generate(cfg_scale=s) on the first documents. Resumable.

python experiments/cond_guide_soft.py --task wmt14
python experiments/cond_guide_soft.py --task xsum --n 6000
"""
import argparse
import os
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
ap.add_argument("--task", required=True, help="xsum or wmt14")
ap.add_argument("--n", type=int, default=0, help="documents, drawn at random (0 = all)")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--check", type=int, default=64, help="documents to compare against generate(cfg_scale=s)")
ap.add_argument("--ts", default="0.25,0.5,1,2", help="soft-decision temperatures")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--tag", default="")
a = ap.parse_args()

TASK = {"wmt14": {"scales": [0.5, 1.0, 2.0], "paper": 1.0, "coarse": 5},
        "xsum": {"scales": [0.25, 0.5, 1.0], "paper": 0.5, "coarse": 4}}[a.task]

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=device())
dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
task = Conditional(ncm, dec, ck, os.path.join(DATA, a.task), bs=a.bs)
K, levels = dec.K, dec.levels
docs = np.arange(len(task))
if a.n and a.n < len(task):
    docs = np.sort(np.random.default_rng(0).permutation(len(task))[:a.n])
task.items, task.refs = [task.items[i] for i in docs], [task.refs[i] for i in docs]
N = len(docs)
idxs = list(range(N))
perm = np.random.default_rng(1).permutation(N)
half = {"A": np.sort(perm[:N // 2]), "B": np.sort(perm[N // 2:]), "all": np.arange(N)}
COARSE = list(range(TASK["coarse"]))
FINE = list(range(TASK["coarse"], K))

name = f"cond_guide_soft_{a.task}{a.tag}"
state_path = os.path.join(RESULTS, "cache", name + ".pt")
st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {"arms": {}}
out = st.get("out") or {"task": a.task, "n": N, "docs": docs.tolist() if a.n else "all", "levels": levels, "K": K,
                        "device": device(), "bs": a.bs, "coarse_levels": COARSE, "fine_levels": FINE,
                        "split": {"A": len(half["A"]), "B": len(half["B"]), "seed": 1},
                        "selection": "best on half A by " + ("cased BLEU" if a.task == "wmt14" else "mean ROUGE-1/2/L"),
                        "arms": {}}
st["out"] = out
seed_of = lambda key: zlib.crc32(f"{name}|{key}".encode())


def save():
    torch.save(st, state_path)
    save_json(out, name + ".json")


# ---- compute accounting --------------------------------------------------------------------------------------------
P = task.P
ONE = P + sum(levels)  # generator token positions of one single pass (prompt cached once per document)


def cost(w):
    w = w or [0.0] * K
    guided = [k for k in range(K) if w[k] > 0]
    return {"nfe": K + len(guided), "positions_ratio": (ONE + sum(levels[k] for k in guided)) / ONE}


# ---- metrics -------------------------------------------------------------------------------------------------------
def metric(per, ix):
    """The selection metric on documents ix (and the reported scores)."""
    if a.task == "wmt14":
        s = per["bleu_stats"][ix].sum(0)
        return {"bleu": bleu_from_stats(s), "chrf": float(per["chrf"][ix].mean()),
                "len_ratio": float(s[0] / s[1])}
    r = {k: 100 * float(per[k][ix].mean()) for k in ("rouge1", "rouge2", "rougeL")}
    r["rouge_mean"] = (r["rouge1"] + r["rouge2"] + r["rougeL"]) / 3
    r["gen_ppl"] = float(np.exp(per["nll"][ix].sum() / per["ntok"][ix].sum()))
    r["words"] = float(per["words"][ix].mean())
    return r


SEL = "bleu" if a.task == "wmt14" else "rouge_mean"


def paired(per, base, ix, key):
    """Bootstrap CI over documents ix of (arm - single pass) for metric key."""
    rng = np.random.default_rng(seed_of(key))
    vals = []
    for _ in range(a.boot):
        b = ix[rng.integers(0, len(ix), len(ix))]
        vals.append(metric(per, b)[key] - metric(base, b)[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def report(key):
    per, base = st["arms"][key]["per"], st["arms"]["single pass"]["per"]
    r = out["arms"][key]
    for h, ix in half.items():
        m, mb = metric(per, ix), metric(base, ix)
        r[h] = {k: v for k, v in m.items()}
        if key != "single pass":
            r[h]["delta"] = {k: m[k] - mb[k] for k in m}
            if h != "A":
                keys = ["bleu", "chrf"] if a.task == "wmt14" else ["rouge1", "rouge2", "rougeL", "rouge_mean"]
                r[h]["ci"] = {k: paired(per, base, ix, k) for k in keys}
    save()


def arm(key, guide_w=None, soft=None):
    if key in st["arms"]:
        return
    torch.manual_seed(seed_of(key))
    with Timer() as t:
        codes, ids = task.generate(idxs, guide_w=guide_w, soft=soft)
    per, corpus, _ = task.score(ids)
    base = st["arms"].get("single pass")
    changed = float((ids != base["ids"]).any(1).float().mean()) if base is not None else 0.0
    out["arms"][key] = dict(cost(guide_w), seconds=t.s, changed_vs_single_pass=changed,
                            guide_w=guide_w, soft={str(k): v for k, v in (soft or {}).items()})
    st["arms"][key] = {"ids": ids.to(torch.int32), "per": per}
    report(key)
    r = out["arms"][key]
    d = r["B"].get("delta", {})
    print(f"{key:34s} nfe {r['nfe']:2d} pos x{r['positions_ratio']:.2f} {t.s:6.0f}s changed {changed:.2f} | B: "
          + " ".join(f"{k} {v:.2f}" + (f" ({d[k]:+.2f})" if k in d else "") for k, v in r["B"].items()
                     if isinstance(v, float)), flush=True)


# ---- 1. checks -----------------------------------------------------------------------------------------------------
arm("single pass")
if "check_cached_single_pass" not in out:
    old = torch.load(os.path.join(RESULTS, "cache", f"{a.task}_vcycle.pt"), weights_only=False)
    ref = old["variants"]["baseline"]["ids"][torch.as_tensor(docs)].long()
    mine = st["arms"]["single pass"]["ids"].long()
    out["check_cached_single_pass"] = {"docs": N, "differ": float((ref != mine).any(1).float().mean())}
    print("single pass vs cached single pass:", out["check_cached_single_pass"], flush=True)
    save()
if "check_cfg" not in out:
    chk = idxs[:min(a.check, N)]
    torch.manual_seed(0)
    bad, ties = [], []
    for s in range(0, len(chk), 16):  # generate() runs the doubled batch uncached: XSum's 512-token prompts need chunks
        part = chk[s:s + 16]
        b = task.check(part, cfg_scale=TASK["paper"])
        if b.any():
            ties += [dict(t, doc=t["doc"] + s) for t in
                     task.tie_margins(part, task.last_check[0], dec.released_ramp(TASK["paper"], K))]
        bad.append(b)
    bad = torch.cat(bad)
    out["check_cfg"] = {"docs": len(chk), "scale": TASK["paper"], "mismatch": int(bad.sum()), "first_differences": ties}
    print(f"guided LevelDecoder vs generate(cfg_scale={TASK['paper']}) on {len(chk)} documents: {int(bad.sum())} differ",
          ties, flush=True)
    save()
    # A mismatch is acceptable only at a near-tie: the two codes' guided logits within 0.05 (guidance multiplies the
    # float noise of both branches by up to 1 + 2w; CPU runs differed by 0.006 at w = 0.8).
    assert all(abs(t["logit_gap"]) < 0.05 for t in ties), "guided decoding differs from generate(cfg_scale=...)"

# ---- 2. stage 1: single-factor arms --------------------------------------------------------------------------------
TS = [float(v) for v in a.ts.split(",")]
soft_arms, coarse_arms, ramp_arms = {}, {}, {}
for ts in TS:
    for mode, lv in (("all", range(K)), ("not final", range(K - 1)), ("coarse", COARSE)):
        key = f"soft {mode} Ts={ts:g}"
        soft_arms[key] = {k: ts for k in lv}
        arm(key, soft=soft_arms[key])
for s in TASK["scales"]:
    key = f"cfg ramp s={s:g}"
    ramp_arms[key] = dec.released_ramp(s, K)
    arm(key, guide_w=ramp_arms[key])
for s in TASK["scales"]:
    key = f"cfg coarse w={s:g}"
    coarse_arms[key] = [s if k in COARSE else 0.0 for k in range(K)]
    arm(key, guide_w=coarse_arms[key])
s = TASK["paper"]
arm(f"cfg fine w={s:g}", guide_w=[s if k in FINE else 0.0 for k in range(K)])

# ---- 3. stage 2: combinations of the best arms on half A -----------------------------------------------------------
best = lambda arms: max(arms, key=lambda k: out["arms"][k]["A"][SEL])
bs_, bc, br = best(soft_arms), best(coarse_arms), best(ramp_arms)
out["stage2_choice"] = {"soft": bs_, "coarse": bc, "ramp": br}
save()
print("stage 2 uses", out["stage2_choice"], flush=True)
for g, w in ((bc, coarse_arms[bc]), (br, ramp_arms[br])):
    arm(f"{bs_} + {g}", guide_w=w, soft=soft_arms[bs_])

# ---- 4. summary ----------------------------------------------------------------------------------------------------
rows = []
for key, r in out["arms"].items():
    d, ci = r["B"].get("delta", {}), r["B"].get("ci", {})
    rows.append((key, r["nfe"], r["positions_ratio"], r["B"][SEL], d.get(SEL), ci.get(SEL)))
print(f"\nhalf B ({len(half['B'])} documents), {SEL}:")
for key, nfe, pr, v, d, ci in rows:
    print(f"  {key:34s} nfe {nfe:2d} pos x{pr:.2f}  {v:.2f}" +
          (f"  {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if d is not None else ""))
out["done"] = True
save()
