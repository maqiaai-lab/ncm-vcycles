"""Prompted tasks: whose plan is NCM missing? Keep the first k levels of a plan and let NCM generate levels k..K-1
greedily (released settings, same prompt), for every k. Plans:
  gold     - encode(reference): the teacher-forced oracle, which also leaks the reference's own wording;
  self     - encode(NCM's own single-pass output): the V-cycle restricted at m = k, the no-information control;
  ext:NAME - encode(an off-the-shelf model's output), given with --ext NAME=model_id[:beams] (beams 0 = the model's own
             generation settings): a good plan that is not the reference.
k = 0 is the single pass for every plan (checked against results/cache/<task>_vcycle.pt when it exists). Scores per k:
WMT14 BLEU, lowercased BLEU, chrF, brevity penalty, n-gram precisions and length ratio; XSum ROUGE-1/2/L and words.
Every k is compared with the single pass by a paired bootstrap over documents. Texts go to results/cache/ (COMET,
experiments/oracle_report.py).

python experiments/oracle_plans.py --task wmt14 --plans gold,self,ext \
    --ext marian=Helsinki-NLP/opus-mt-de-en:4,fsmt=facebook/wmt19-de-en:5
python experiments/oracle_plans.py --task wmt14 --dev --plans ext \
    --ext marian=Helsinki-NLP/opus-mt-de-en:4,fsmt=facebook/wmt19-de-en:5        # newstest2013
python experiments/oracle_plans.py --task xsum --n 3000 --bs 16 --plans gold,self,ext --ext bart=facebook/bart-large-xsum:0
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.cond import Conditional, bleu_from_stats
from vcycle.common import ARTIFACTS, DATA, RESULTS, batches, device, save_json
from vcycle.decoding import LevelDecoder
from vcycle.devsets import wmt14_dev
from vcycle.loading import load

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="wmt14 or xsum")
ap.add_argument("--dev", action="store_true", help="WMT14 only: newstest2013 instead of newstest2014")
ap.add_argument("--n", type=int, default=0, help="documents (0 = all)")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--plans", default="gold,self,ext")
ap.add_argument("--ext", default="", help="NAME=model_id[:beams],...")
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()

torch.set_grad_enabled(False)
dev = device()
ck = os.path.join(ARTIFACTS, a.task)
ncm = load(ck, device=dev)
dec = LevelDecoder(ncm, cache_prefix=True, cache_padded=True)
task = Conditional(ncm, dec, ck, wmt14_dev() if a.dev else os.path.join(DATA, a.task), n=a.n, bs=a.bs)
K, levels, N = dec.K, dec.levels, len(task)
L, eos, pad = levels[-1], task.eos, task.pad
plans = a.plans.split(",")
name = f"oracle_plans_{a.task}{'_nt2013' if a.dev else ''}"
os.makedirs(os.path.join(RESULTS, "cache"), exist_ok=True)
t0 = time.time()
stamp = lambda *s: print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)
stamp(a.task, "newstest2013" if a.dev else "test", "N", N, "levels", levels, "plans", plans, "ext", a.ext)


def canonical_from_text(texts):
    """Text -> pre-tokenizer ids in the training form: up to L - 1 tokens, EOS, PAD."""
    rows = [task.tok.encode(t) for t in texts]
    return torch.tensor([(r[:L - 1] + [eos] + [pad] * L)[:L] for r in rows], dtype=torch.long)


def encode_all(x):
    out = []
    for sl in batches(N, 128):
        out.append([g.cpu() for g in dec.encode(x[sl].to(dev))])
    return [torch.cat([g[k] for g in out]) for k in range(K)]


def source_text(i):
    it = task.items[i]
    return it if isinstance(it, str) else task.tok.decode([t for t in it if t not in (eos, pad)], skip_special_tokens=True)


def per_doc(texts):
    refs = task.refs[:len(texts)]
    if task.metric == "bleu":
        from sacrebleu.metrics import BLEU
        bleu, rows = BLEU(), []
        for h, r in zip(texts, refs):
            s = bleu.corpus_score([h], [[r]])
            rows.append([s.sys_len, s.ref_len] + list(s.counts) + list(s.totals))
        return {"bleu_stats": np.array(rows, dtype=np.int64)}
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    per = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(texts, refs):
        v = sc.score(r, h)
        for k in per:
            per[k].append(100 * v[k].fmeasure)
    return {k: np.array(v) for k, v in per.items()}


def corpus(texts):
    refs = task.refs[:len(texts)]
    if task.metric == "bleu":
        from sacrebleu.metrics import BLEU, CHRF
        b = BLEU().corpus_score(texts, [refs])
        return {"bleu": round(b.score, 2), "bleu_lc": round(BLEU(lowercase=True).corpus_score(texts, [refs]).score, 2),
                "chrf": round(CHRF().corpus_score(texts, [refs]).score, 2), "bp": round(b.bp, 4),
                "prec": [round(p, 1) for p in b.precisions],
                "len_ratio": round(sum(len(h.split()) for h in texts) / sum(len(r.split()) for r in refs), 4)}
    per = per_doc(texts)
    out = {k: round(float(v.mean()), 2) for k, v in per.items()}
    out["words"] = round(float(np.mean([len(h.split()) for h in texts])), 2)
    return out


boot_idx = np.random.default_rng(0).integers(0, N, size=(a.boot, N))


def delta_ci(pa, pb):
    """Paired bootstrap of (a - b) over documents: corpus BLEU from summed statistics, or mean ROUGE."""
    if "bleu_stats" in pa:
        sa, sb = pa["bleu_stats"], pb["bleu_stats"]
        d = []
        for i in range(0, a.boot, 100):
            ix = boot_idx[i:i + 100]
            d += [bleu_from_stats(x) - bleu_from_stats(y) for x, y in zip(sa[ix].sum(1), sb[ix].sum(1))]
        point = bleu_from_stats(sa.sum(0)) - bleu_from_stats(sb.sum(0))
        return {"bleu": [round(point, 2), round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]}
    out = {}
    for k in ("rouge1", "rouge2", "rougeL"):
        diff = pa[k] - pb[k]
        bs = diff[boot_idx].mean(1)
        out[k] = [round(float(diff.mean()), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
    return out


def texts_of(ids):
    return [task.text(r) for r in ids.tolist()]


def run_plan(codes, k, want_acc=False):
    """Ids of plan_{<k} + generated levels k..K-1; with want_acc, also the share of level-k codes equal to the plan's
    (teacher-forced accuracy at level k)."""
    ids, hit = [], []
    for sl in batches(N, a.bs):
        prefix = dec.prefix(task.prefix_tokens(list(range(sl.start, sl.stop))))
        c, z = task._run(prefix, start=k, codes=None if codes is None else [g[sl].to(dev) for g in codes])
        ids.append(task.pick(z).cpu())
        if want_acc and k < K:
            hit.append((c[k].cpu() == codes[k][sl]).float().mean(1))
    return torch.cat(ids), (round(float(torch.cat(hit).mean()), 4) if hit else None)


res = {"task": a.task, "data": "newstest2013" if a.dev else "test", "N": N, "levels": levels, "plans": {}, "notes": {}}
extra = {"sources": [source_text(i) for i in range(N)], "refs": list(task.refs)}


def save():
    save_json(res, name + ".json")
    torch.save(extra, os.path.join(RESULTS, "cache", name + ".pt"))


# k = 0: the single pass, the same for every plan.
ids0, _ = run_plan(None, 0)
t_single = texts_of(ids0)
p_single = per_doc(t_single)
res["single_pass"] = corpus(t_single)
cache = os.path.join(RESULTS, "cache", f"{a.task}_vcycle.pt")
if not a.dev and os.path.exists(cache):
    base = torch.load(cache, weights_only=False)["variants"]["baseline"]["ids"][:N].long()
    res["single_pass"]["matches_cached_text"] = float(np.mean([x == y for x, y in zip(t_single, texts_of(base))]))
extra["single_pass"] = t_single
stamp("single pass", res["single_pass"])
save()

plan_codes = {}
if "gold" in plans:
    plan_codes["gold"] = encode_all(canonical_from_text(task.refs))
if "self" in plans:
    plan_codes["self"] = encode_all(task.canonical(ids0))
if "ext" in plans:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    srcs = extra["sources"]
    for spec in [s for s in a.ext.split(",") if s]:
        nm, mid = spec.split("=", 1)
        mid, beams = mid.rsplit(":", 1) if ":" in mid else (mid, "0")
        beams = int(beams)
        mtok = AutoTokenizer.from_pretrained(mid)
        mt = AutoModelForSeq2SeqLM.from_pretrained(mid).to(dev).eval()
        gen_kw = {"num_beams": beams, "max_new_tokens": 256} if beams > 0 else {}
        hyps = []
        for sl in batches(N, 32 if a.task == "wmt14" else 8):
            enc = mtok(srcs[sl], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
            hyps += [h.strip() for h in mtok.batch_decode(mt.generate(**enc, **gen_kw), skip_special_tokens=True)]
        del mt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        extra[f"ext:{nm}"] = hyps
        res["notes"][nm] = {"model": mid, "beams": beams or "model default",
                            "cut_pct": float(100 * np.mean([len(task.tok.encode(h)) > L - 1 for h in hyps]))}
        res[f"{nm}_alone"] = dict(corpus(hyps), delta_vs_single=delta_ci(per_doc(hyps), p_single))
        plan_codes[f"ext:{nm}"] = encode_all(canonical_from_text(hyps))
        stamp(nm, "alone", res[f"{nm}_alone"])
        save()

for plan, codes in plan_codes.items():
    curve = []
    for k in range(1, K + 1):
        ids, acc = run_plan(codes, k, want_acc=(plan == "gold"))
        t = texts_of(ids)
        row = dict(corpus(t), k=k, delta_vs_single=delta_ci(per_doc(t), p_single))
        if acc is not None:
            row["teacher_forced_acc_level_k"] = acc
        curve.append(row)
        extra[f"{plan}|k={k}"] = t
        stamp(plan, k, row)
        res["plans"][plan] = curve
        save()
stamp("done")
