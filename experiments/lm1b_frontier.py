"""Unconditional text (LM1B, TinyStories): the V-cycle against sampling colder, soft decisions under sampling, and the
two changes together. One job per model; every arm is scored on the same seeds with the same scorer.

Arms, per seed (n samples each; level 0 is always drawn from the released unigram and passed as a hard code):
  baseline               the released sampler (levels 1..K-1 at the released temperature), as lm1b_diagnostic.py
  vcycle|m=M             one V-cycle from level M on the baseline samples (re-encode, keep levels < M, resample the rest)
  no-restrict|m=M        its equal-compute control (keep the generated levels < M, resample the rest)
  T=t                    the released sampler at t times the released temperature (the temperature curve)
  soft fine              levels >= M pass the posterior-mean code vector (Ts = 1); levels < M are sampled and passed hard
  soft coarse            levels 1..M-1 pass the posterior mean; levels >= M are sampled and passed hard
  soft all               every generated level passes the posterior mean
  vcycle|m=M soft        the V-cycle whose regenerated levels (>= M) use soft decisions (Ts = 1)
  no-restrict|m=M soft   its control: the generated levels < M kept, levels >= M regenerated softly

What the output tokens are. LevelDecoder.run(soft={k: Ts}) still draws a code at a soft level (so the random stream is
consumed exactly as in the baseline), but the reconstruction, which is all that finer levels and the decoder read, adds
the posterior-mean code vector softmax(logits / Ts) @ codebook instead of that code's vector. The output tokens are
picked from the decoder's logits on this reconstruction, with the released penalties. Consequences under sampling:
  - `soft fine`: the coarse codes are the baseline's own (same seed, same stream), and everything from level M on is a
    deterministic function of them. `no-restrict|m=M soft` computes the same thing from the stored baseline codes, so
    the two agree up to numerical near-ties (the share of identical samples is recorded).
  - `soft all`: the sample is a deterministic function of its level-0 codes alone.
  - `vcycle|m=M soft`: a deterministic function of the re-encoded coarse levels.

Metrics per arm, per seed and pooled over seeds: corpus generative perplexity under GPT-2 large with the protocol of
lm1b_diagnostic.py (LM1B.nll), mean per-sample unigram entropy, per-sample and corpus distinct-2/3, adjacent repeats,
seconds of generation, and the cost in generator evaluations and in code positions processed (one full pass = 1).
Paired bootstrap over samples (1,000 resamples, seeds pooled) for the perplexity ratio of every arm to `baseline` and
of each V-cycle to its control. Frontier: the perplexity of the temperature curve (log perplexity linear in corpus
distinct-2 between neighbouring temperature points, the baseline included) at an arm's corpus distinct-2, and the ratio
arm / curve, with a bootstrap that resamples the samples of every arm (within seeds). Corpus distinct-n is computed
per seed (n samples) and averaged over seeds, as in ed_sweep.py, because it depends on the size of the set.

Resumable: arms already in results/cache/lm1b_frontier_<tag>.pt are not regenerated.

python experiments/lm1b_frontier.py --size small --seeds 0,1,42
python experiments/lm1b_frontier.py --task tinystories --seeds 0,1,42
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, RESULTS, Timer, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.lm1b import LM1B, _body, corpus_distinct
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="lm1b", help="lm1b (with --size) or tinystories")
ap.add_argument("--size", default="small")
ap.add_argument("--seeds", default="0,1,42")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--m", type=int, default=6, help="restart level of the V-cycle and first soft level of `soft fine`")
ap.add_argument("--temps", default="0.95,0.9,0.85,0.8", help="temperature curve, as multiples of the released temperature")
ap.add_argument("--ts", type=float, default=1.0, help="soft-decision temperature")
ap.add_argument("--ref", default="openai-community/gpt2-large")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--check", type=int, default=8, help="samples to compare against net.generate")
ap.add_argument("--tag", default="")
a = ap.parse_args()

torch.set_grad_enabled(False)
ck = os.path.join(ARTIFACTS, "lm1b", a.size) if a.task == "lm1b" else os.path.join(ARTIFACTS, a.task)
tag = (f"lm1b_{a.size}" if a.task == "lm1b" else a.task) + a.tag
ncm = load(ck, device=device())
dec = LevelDecoder(ncm)
lm = LM1B(ncm, dec, ck, ref=a.ref, bs=a.bs)
K, M, TS = dec.K, a.m, a.ts
seeds = [int(s) for s in a.seeds.split(",") if s != ""]
temps = [float(t) for t in a.temps.split(",") if t]

# The released sampler path is untouched: without `soft`, LevelDecoder must still reproduce generate() exactly.
if a.check:
    torch.manual_seed(1234)
    ref = ncm.net.generate(lm.sample_l0(a.check), lm.prompt(a.check), temperature=lm.temp, top_k=0, top_p=1.0,
                           cfg_scale=0.0, cfg_null_eos=lm.eos)
    torch.manual_seed(1234)
    l0 = lm.sample_l0(a.check)
    mine = torch.cat(dec.run(dec.prefix(lm.prompt(a.check)), start=1, codes=[l0], temperature=lm.temp)[0], 1)
    assert torch.equal(ref, mine), f"LevelDecoder differs from generate() on {tag}"
    print(f"LevelDecoder == generate() on {a.check} {tag} samples (temperature {lm.temp})", flush=True)

FULL = sum(dec.levels[1:])                                   # code positions of one pass (level 0 is not generated)
SOFT = {"soft fine": {k: TS for k in range(M, K)}, "soft coarse": {k: TS for k in range(1, M)},
        "soft all": {k: TS for k in range(1, K)}}
VC, NR = f"vcycle|m={M}", f"no-restrict|m={M}"
T_ARMS = [f"T={t:g}" for t in temps]
ARMS = ["baseline", VC, NR] + T_ARMS + list(SOFT) + [VC + " soft", NR + " soft"]
cost = lambda levels: {"nfe": len(levels), "positions": sum(dec.levels[k] for k in levels),
                       "positions_fraction_of_pass": sum(dec.levels[k] for k in levels) / FULL}
COST = {arm: cost(range(1, K)) for arm in ["baseline"] + T_ARMS + list(SOFT)}
for arm in (VC, NR, VC + " soft", NR + " soft"):             # extra cost on top of the pass that made the draft
    COST[arm] = {"extra_" + k: v for k, v in cost(range(M, K)).items()}


def generate_soft(seed, soft):
    """LM1B.generate with soft decisions: same seed, same random stream as the baseline."""
    torch.manual_seed(seed)
    ids = []
    for sl in batches(a.n, lm.bs):
        B = sl.stop - sl.start
        _, z = dec.run(dec.prefix(lm.prompt(B)), start=1, codes=[lm.sample_l0(B)], temperature=lm.temp, soft=soft)
        ids.append(lm.pick_tokens(ncm.vqvae.decoder(z)).cpu())
    return torch.cat(ids)


def restart_soft(codes, ids, reencode):
    """LM1B.restart from level M with soft decisions at the regenerated levels."""
    out = []
    soft = {k: TS for k in range(M, K)}
    for sl in batches(len(ids), lm.bs):
        B = sl.stop - sl.start
        c = [ck_[sl].to(lm.dev) for ck_ in codes]
        base = dec.encode(ids[sl].to(lm.dev)) if reencode else c
        _, z = dec.run(dec.prefix(lm.prompt(B)), start=M, codes=base, temperature=lm.temp, soft=soft)
        out.append(lm.pick_tokens(ncm.vqvae.decoder(z)).cpu())
    return torch.cat(out)


def make(arm, seed, base):
    """Token ids of one arm for one seed; base = (codes, ids) of the baseline samples of that seed."""
    if arm == "baseline":
        return lm.generate(a.n, seed)
    if arm in T_ARMS:
        return lm.generate(a.n, seed, temperature=float(arm[2:]) * lm.temp)[1]
    if arm in SOFT:
        return generate_soft(seed, SOFT[arm])
    torch.manual_seed(seed + 1000 * M)                       # as lm1b_diagnostic.py, for the V-cycle and its control
    if arm in (VC, NR):
        return lm.restart(base[0], base[1], M, reencode=arm == VC)[1]
    return restart_soft(base[0], base[1], reencode=arm == VC + " soft")


def summarize(ids, nl, nt):
    r = lm.corpus(nl, nt, lm.quality(ids))
    r["corpus_distinct2"] = corpus_distinct(ids, lm.pad, 2, lm.cut_eos)
    r["corpus_distinct3"] = corpus_distinct(ids, lm.pad, 3, lm.cut_eos)
    return r


# ---- 1. generate and score (resumable) -----------------------------------------------------------------------------
cache_path = os.path.join(RESULTS, "cache", f"lm1b_frontier_{tag}.pt")
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
st = torch.load(cache_path, weights_only=False) if os.path.exists(cache_path) else {"arms": {}, "codes": {}, "seconds": {}}
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
for seed in seeds:
    for arm in ARMS:
        if seed in st["arms"].get(arm, {}):
            continue
        base = None if arm == "baseline" else ([c.long() for c in st["codes"][seed]], st["arms"]["baseline"][seed]["ids"].long())
        with Timer() as t:
            got = make(arm, seed, base)
        if arm == "baseline":
            st["codes"][seed], ids = [c.to(torch.int32) for c in got[0]], got[1]
        else:
            ids = got
        with Timer() as t2:
            nl, nt = lm.nll(ids)
        st["arms"].setdefault(arm, {})[seed] = {"ids": ids.to(torch.int32), "nll": nl, "ntok": nt}
        st["seconds"].setdefault(arm, {})[seed] = {"generate": t.s, "score": t2.s}
        r = summarize(ids, nl, nt)
        print(f"[{tag} s{seed}] {arm:22s} ppl {r['gen_ppl']:8.2f} cd2 {r['corpus_distinct2']:.4f} H {r['entropy']:.3f} "
              f"rep {r['adjacent_repeat']:.4f} ({t.s:.0f}s + {t2.s:.0f}s)", flush=True)
        torch.save(st, cache_path)

# ---- 2. analysis (no GPU needed: everything below reads the cache) --------------------------------------------------
N = a.n
ids_of = lambda arm, s: st["arms"][arm][s]["ids"].long()
_cat = {}


def cat(arm, key):
    """Per-sample values of an arm, seeds concatenated in the order of `seeds`."""
    if (arm, key) not in _cat:
        _cat[arm, key] = np.concatenate([np.asarray(st["arms"][arm][s][key], dtype=np.float64) for s in seeds])
    return _cat[arm, key]


logppl = lambda arm, ix=None: (cat(arm, "nll") if ix is None else cat(arm, "nll")[ix]).sum() / (
    cat(arm, "ntok") if ix is None else cat(arm, "ntok")[ix]).sum()
rng = np.random.default_rng(0)
boot = np.stack([np.concatenate([j * N + rng.integers(0, N, N) for j in range(len(seeds))]) for _ in range(a.boot)])
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def ratio(arm, ref):
    """Pooled perplexity ratio arm / ref with a paired bootstrap over samples (same resampled indices in both)."""
    na, ta, nb, tb = cat(arm, "nll"), cat(arm, "ntok"), cat(ref, "nll"), cat(ref, "ntok")
    f = lambda ix: math.exp(na[ix].sum() / ta[ix].sum() - nb[ix].sum() / tb[ix].sum())
    return [f(np.arange(len(na)))] + ci([f(ix) for ix in boot])


def incidence(arm, s, n=2):
    """(samples x distinct n-grams) boolean matrix and the n-gram count per sample, for corpus distinct-n under
    resampling: distinct n-grams of a resample = columns with any selected row."""
    rows = [list(zip(*[b[i:] for i in range(n)])) for b in (_body(r, lm.pad, lm.cut_eos) for r in ids_of(arm, s).tolist())]
    vocab = {g: j for j, g in enumerate(sorted(set(g for r in rows for g in r)))}
    inc = np.zeros((len(rows), max(len(vocab), 1)), dtype=bool)
    for i, r in enumerate(rows):
        inc[i, [vocab[g] for g in set(r)]] = True
    return inc, np.array([len(r) for r in rows], dtype=np.float64)


out = {"task": a.task, "size": a.size if a.task == "lm1b" else a.task, "tag": tag, "n": N, "seeds": seeds, "levels": dec.levels,
       "K": K, "m": M, "soft_ts": TS, "released_temperature": lm.temp, "scorer": a.ref, "device": device(), "gpu": gpu,
       "positions_full_pass": FULL, "boot": a.boot, "arms": {}, "comparisons": {}, "frontier": {}}
per_seed = {arm: {s: summarize(ids_of(arm, s), st["arms"][arm][s]["nll"], st["arms"][arm][s]["ntok"]) for s in seeds} for arm in ARMS}
for arm in ARMS:
    ps = per_seed[arm]
    pooled = {k: float(np.mean([ps[s][k] for s in seeds])) for k in ps[seeds[0]]}        # means over seeds
    pooled["gen_ppl"] = float(math.exp(logppl(arm)))                                     # corpus perplexity, seeds pooled
    r = {"pooled": pooled, "per_seed": {str(s): ps[s] for s in seeds}, **COST[arm],
         "seconds_generate": float(sum(st["seconds"][arm][s]["generate"] for s in seeds if s in st["seconds"].get(arm, {}))),
         "seconds_score": float(sum(st["seconds"][arm][s]["score"] for s in seeds if s in st["seconds"].get(arm, {})))}
    if arm != "baseline":
        r["ppl_ratio_vs_baseline"] = ratio(arm, "baseline")
        r["corpus_d2_change"] = float(np.mean([ps[s]["corpus_distinct2"] / per_seed["baseline"][s]["corpus_distinct2"] - 1 for s in seeds]))
    out["arms"][arm] = r
for x, y in ((VC, NR), (VC + " soft", NR + " soft"), (VC + " soft", VC), ("soft fine", VC), (VC + " soft", "soft fine")):
    out["comparisons"][f"{x} / {y}"] = ratio(x, y)
same = [float((ids_of(NR + " soft", s) == ids_of("soft fine", s)).all(1).float().mean()) for s in seeds]
out["no_restrict_soft_identical_to_soft_fine"] = float(np.mean(same))
l0 = [float(np.mean([len(set(map(tuple, ids_of("soft all", s).tolist()))) / N])) for s in seeds]
out["soft_all_distinct_samples_fraction"] = float(np.mean(l0))

# Frontier: log perplexity of the temperature curve at an arm's corpus distinct-2 (clamped at the ends of the curve).
curve_arms = ["baseline"] + T_ARMS
inc = {arm: {s: incidence(arm, s) for s in seeds} for arm in ARMS}


def cd2(arm, ix=None):
    """Mean over seeds of the corpus distinct-2 of the (resampled) n samples of each seed."""
    vals = []
    for j, s in enumerate(seeds):
        m_, cnt = inc[arm][s]
        rows = np.arange(N) if ix is None else ix[j * N:(j + 1) * N] - j * N
        vals.append(m_[rows].any(0).sum() / max(cnt[rows].sum(), 1.0))
    return float(np.mean(vals))


def frontier(ix=None):
    pts = sorted((cd2(c, ix), logppl(c, ix)) for c in curve_arms)
    xs, ys = [p_[0] for p_ in pts], [p_[1] for p_ in pts]
    res = {}
    for arm in ARMS:
        if arm in curve_arms:
            continue
        d = cd2(arm, ix)
        res[arm] = (math.exp(logppl(arm, ix) - float(np.interp(d, xs, ys))), math.exp(float(np.interp(d, xs, ys))), d,
                    bool(d < xs[0] or d > xs[-1]))
    return res


point = frontier()
boots = [frontier(ix) for ix in boot]
for arm, (r, curve_ppl, d, outside) in point.items():
    out["frontier"][arm] = {"ppl_ratio_to_temperature_curve": [r] + ci([b[arm][0] for b in boots]),
                            "curve_ppl_at_same_corpus_distinct2": curve_ppl, "corpus_distinct2": d,
                            "outside_curve_range": outside,
                            "share_of_resamples_outside_range": float(np.mean([b[arm][3] for b in boots]))}
out["temperature_curve"] = [{"arm": c, "corpus_distinct2": cd2(c), "gen_ppl": math.exp(logppl(c))} for c in curve_arms]
save_json(out, f"lm1b_frontier_{tag}.json")
for arm in ARMS:
    r = out["arms"][arm]
    fr = out["frontier"].get(arm, {}).get("ppl_ratio_to_temperature_curve")
    print(f"{arm:22s} ppl {r['pooled']['gen_ppl']:8.2f} cd2 {r['pooled']['corpus_distinct2']:.4f} "
          + (f"vs baseline {r['ppl_ratio_vs_baseline'][0]:.3f} [{r['ppl_ratio_vs_baseline'][1]:.3f}, {r['ppl_ratio_vs_baseline'][2]:.3f}] "
             if arm != "baseline" else "")
          + (f"vs curve {fr[0]:.3f} [{fr[1]:.3f}, {fr[2]:.3f}]" if fr else ""), flush=True)
print("wrote", os.path.join(RESULTS, f"lm1b_frontier_{tag}.json"), flush=True)
