"""WMT14: minimum-Bayes-risk decoding of the released checkpoint, with hard and with soft decisions.

The NCM paper draws K candidates and keeps the one with the highest average similarity to the rest (K = 12 on WMT14,
132 generator evaluations); it does not state the similarity or the sampling temperature, and the public code has no
MBR script. This is our own run of that recipe. Candidate sets, K per source, seeded:
  sampled                  every level drawn from the released sampler at temperature tau (top_k = 0, top_p = 1)
  sampled coarse + soft    levels < C (default 5) drawn at temperature tau and passed as hard codes; levels >= C pass
                           the posterior-mean code vector (soft decisions, Ts = 1). Under a soft decision the drawn
                           code of a level is not read by anything downstream, so the candidate is a deterministic
                           function of its sampled coarse levels: K candidates = K sampled plans, each completed softly.
Selection: the candidate with the highest mean utility against the other candidates, for two utilities (sentence chrF
and sentence BLEU with effective order, sacreBLEU), also with only the first 4 and the first 8 candidates.

Reported per (candidate set, tau): the MBR pick for each utility and K (cased and lowercased BLEU, chrF, length ratio,
paired bootstrap against the greedy single pass and the greedy soft pass, both run here); the mean score of single
candidates; and the oracle (the candidate with the highest sentence chrF against the reference).
--dev runs newstest2013, on which (set, tau, utility) is to be chosen; without it newstest2014.

Writes results/cond_mbr_wmt14[_dev].json and results/cache/cond_mbr_wmt14[_dev].pt (candidate ids, utilities).
Resumable per candidate. The utilities are computed on the CPU (--workers processes).

python experiments/cond_mbr.py --task wmt14 --dev --k 12
python experiments/cond_mbr.py --task wmt14 --k 12
"""
import argparse
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_METRICS = None


def utilities(job):
    """One sentence: (texts of the K candidates, reference) -> (chrF [K, K], BLEU [K, K], chrF against the reference
    [K]). U[i, j] scores candidate i as the hypothesis against candidate j as the reference. Identical candidates are
    scored once."""
    global _METRICS
    if _METRICS is None:
        from sacrebleu.metrics import BLEU, CHRF
        _METRICS = (CHRF(), BLEU(effective_order=True))
    chrf, bleu = _METRICS
    texts, ref = job
    uniq = list(dict.fromkeys(texts))
    pos = [uniq.index(t) for t in texts]
    u = len(uniq)
    uc, ub = np.zeros((u, u), dtype=np.float32), np.zeros((u, u), dtype=np.float32)
    for i in range(u):
        for j in range(u):
            uc[i, j] = chrf.sentence_score(uniq[i], [uniq[j]]).score
            ub[i, j] = bleu.sentence_score(uniq[i], [uniq[j]]).score
    rc = np.array([chrf.sentence_score(t, [ref]).score for t in uniq], dtype=np.float32)
    ix = np.array(pos)
    return uc[np.ix_(ix, ix)], ub[np.ix_(ix, ix)], rc[ix], u


def main():
    import torch

    from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json
    from vcycle.cond import Conditional, bleu_from_stats
    from vcycle.decoding import LevelDecoder
    from vcycle.devsets import wmt14_dev
    from vcycle.loading import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="wmt14")
    ap.add_argument("--dev", action="store_true", help="newstest2013 instead of newstest2014")
    ap.add_argument("--k", type=int, default=12, help="candidates per source")
    ap.add_argument("--taus", default="1.0,0.7", help="sampling temperatures of the candidates")
    ap.add_argument("--coarse", type=int, default=5, help="`sampled coarse + soft`: levels below this are sampled hard")
    ap.add_argument("--ts", type=float, default=1.0, help="soft-decision temperature")
    ap.add_argument("--n", type=int, default=0, help="first n sentences only (smoke tests)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=0, help="processes for the utilities (0: serial)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert a.task == "wmt14", "only WMT14 is set up (the XSum recipe would need ROUGE utilities)"

    torch.set_grad_enabled(False)
    ck = os.path.join(ARTIFACTS, a.task)
    ncm = load(ck, device=device())
    dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
    K = dec.K
    task = Conditional(ncm, dec, ck, wmt14_dev() if a.dev else os.path.join(DATA, a.task), bs=a.bs, n=a.n)
    greedy_cfg = (task.temp, task.top_k, task.top_p)
    assert task.top_k == 1 or task.temp == 0, "the released WMT14 config decodes greedily"
    idxs = list(range(len(task)))
    N = len(idxs)
    taus = [float(t) for t in a.taus.split(",") if t]
    ks = sorted({k for k in (4, 8, a.k) if k <= a.k})
    name = f"cond_mbr_{a.task}{'_dev' if a.dev else ''}{a.tag}"
    state_path = os.path.join(RESULTS, "cache", name + ".pt")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    st = torch.load(state_path, weights_only=False) if os.path.exists(state_path) else {"greedy": {}, "cands": {}, "util": {}}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out = {"task": a.task, "set": "newstest2013" if a.dev else "newstest2014", "n": N, "k": a.k, "levels": dec.levels, "K": K,
           "taus": taus, "coarse_levels_sampled": a.coarse, "soft_ts": a.ts, "device": device(), "gpu": gpu,
           "nfe_per_candidate": K, "greedy": {}, "configs": {}}
    seed_of = lambda key: zlib.crc32(f"{name}|{key}".encode())
    SETS = {"sampled": None, "sampled coarse + soft": {k: a.ts for k in range(a.coarse, K)}}

    def save():
        torch.save(st, state_path)
        save_json(out, name + ".json")

    def generate(tau=None, soft=None):
        """Token ids of one pass over all sentences: greedy as released (tau None), or sampled at temperature tau."""
        task.temp, task.top_k, task.top_p = greedy_cfg if tau is None else (tau, 0, 1.0)
        try:
            return task.generate(idxs, soft=soft)[1]
        finally:
            task.temp, task.top_k, task.top_p = greedy_cfg

    def paired(per, base):
        """Paired bootstrap over sentences against a greedy arm: [difference, lo, hi] for BLEU and sentence chrF."""
        pb = st["greedy"][base]["per"]
        bx = np.random.default_rng(seed_of(f"boot|{base}")).integers(0, N, size=(a.boot, N))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        d = lambda i: bleu_from_stats(per["bleu_stats"][i].sum(0)) - bleu_from_stats(pb["bleu_stats"][i].sum(0))
        dc = per["chrf"] - pb["chrf"]
        return {"bleu": [d(np.arange(N))] + ci([d(b) for b in bx]), "chrf_sentence": [float(dc.mean())] + ci(dc[bx].mean(1))}

    def scored(ids, compare=True):
        per, corpus, _ = task.score(ids)
        r = {k: v for k, v in corpus.items() if k != "bleu_from_stats"}
        if compare:
            for base in ("single pass", "soft"):
                r["vs_" + base] = paired(per, base)
        return per, r

    # ---- 1. greedy arms ---------------------------------------------------------------------------------------------
    for key, soft in (("single pass", None), ("soft", {k: a.ts for k in range(K)})):
        if key not in st["greedy"]:
            torch.manual_seed(seed_of(key))
            with Timer() as t:
                ids = generate(soft=soft)
            per, _ = scored(ids, compare=False)
            st["greedy"][key] = {"ids": ids.to(torch.int32), "per": per, "seconds": t.s}
            torch.save(st, state_path)
        per, r = scored(st["greedy"][key]["ids"].long(), compare=False)
        out["greedy"][key] = dict(r, nfe=K, seconds=st["greedy"][key]["seconds"])
        print(f"{key:28s} " + " ".join(f"{k} {v:.2f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    out["greedy"]["soft"]["vs_single pass"] = paired(st["greedy"]["soft"]["per"], "single pass")
    save()

    # ---- 2. candidates (GPU) ----------------------------------------------------------------------------------------
    configs = [(s, tau) for s in SETS for tau in taus]
    cname = lambda s, tau: f"{s}|tau={tau:g}"
    for s, tau in configs:
        c = st["cands"].setdefault(cname(s, tau), {"ids": [], "seconds": 0.0})
        while len(c["ids"]) < a.k:
            j = len(c["ids"])
            torch.manual_seed(seed_of(f"{cname(s, tau)}|{j}"))
            with Timer() as t:
                ids = generate(tau, SETS[s])
            c["ids"].append(ids.to(torch.int32))
            c["seconds"] += t.s
            print(f"[{cname(s, tau)}] candidate {j + 1}/{a.k} ({t.s:.0f}s)", flush=True)
            torch.save(st, state_path)

    # ---- 3. utilities (CPU) and selection ---------------------------------------------------------------------------
    pool = None
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers)
    for s, tau in configs:
        key = cname(s, tau)
        cand = torch.stack([c.long() for c in st["cands"][key]["ids"][:a.k]])            # [k, N, L]
        texts = [[task.text(cand[j, n].tolist()) for j in range(a.k)] for n in range(N)]
        if key not in st["util"] or st["util"][key]["chrf"].shape[1] != a.k:
            jobs = [(texts[n], task.refs[n]) for n in range(N)]
            with Timer() as t:
                res = pool.map(utilities, jobs, chunksize=16) if pool else [utilities(j) for j in jobs]
            st["util"][key] = {"chrf": np.stack([r[0] for r in res]), "bleu": np.stack([r[1] for r in res]),
                               "ref_chrf": np.stack([r[2] for r in res]), "unique": np.array([r[3] for r in res]),
                               "seconds": t.s}
            torch.save(st, state_path)
        u = st["util"][key]
        r = {"nfe": a.k * K, "seconds_generate": st["cands"][key]["seconds"], "seconds_utilities": u["seconds"],
             "unique_candidates_mean": float(u["unique"].mean()), "single": {}, "mbr": {}}
        # single candidates as systems (corpus scores), and the oracle by sentence chrF against the reference
        from sacrebleu.metrics import BLEU, CHRF
        refs = task.refs[:N]
        sing = {"bleu": [], "bleu_lc": [], "chrf": []}
        for j in range(a.k):
            hyps = [texts[n][j] for n in range(N)]
            sing["bleu"].append(BLEU().corpus_score(hyps, [refs]).score)
            sing["bleu_lc"].append(BLEU(lowercase=True).corpus_score(hyps, [refs]).score)
            sing["chrf"].append(CHRF().corpus_score(hyps, [refs]).score)
        r["single"] = {m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for m, v in sing.items()}
        ar = torch.arange(N)
        pick = lambda ix: cand[torch.as_tensor(ix), ar]
        r["oracle_by_chrf"] = scored(pick(u["ref_chrf"].argmax(1)))[1]
        for util in ("chrf", "bleu"):
            r["mbr"][util] = {}
            for k in ks:
                m = u[util][:, :k, :k].astype(np.float64).copy()
                m[:, np.arange(k), np.arange(k)] = 0.0                                   # a candidate does not vote for itself
                best = (m.sum(2) / max(k - 1, 1)).argmax(1)
                per, sc = scored(pick(best))
                sc["nfe"] = k * K
                sc["picked_first_candidate_share"] = float((best == 0).mean())
                r["mbr"][util][f"k={k}"] = sc
                print(f"[{key}] MBR {util:4s} k={k:2d}: bleu {sc['bleu']:.2f} lc {sc['bleu_lc']:.2f} chrf {sc['chrf']:.2f} "
                      f"| vs single pass {sc['vs_single pass']['bleu'][0]:+.2f} vs soft {sc['vs_soft']['bleu'][0]:+.2f}", flush=True)
        out["configs"][key] = r
        save()
    if pool:
        pool.close()
    out["done"] = True
    save()
    print("wrote", os.path.join(RESULTS, name + ".json"), flush=True)


if __name__ == "__main__":
    main()
