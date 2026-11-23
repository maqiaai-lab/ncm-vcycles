"""Checks on the external plans of experiments/oracle_plans.py, from its text caches (no GPU):
1. Memorization: how often each external system reproduces the reference exactly or nearly (WMT14: sentence BLEU
   >= 90; XSum: ROUGE-1 >= 0.9). A system that saw the test references in training leaks them into its plan.
2. XSum by subset: the plan-swap gain at k = 1, 3, 5 on all articles and on those where the external summarizer is
   far from the reference (its own ROUGE-1 below 90 and below 50), with the share of the summarizer's lead it recovers.
Output: results/oracle_report.json.

python experiments/oracle_report.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import RESULTS, save_json

res = {"memorization": {}}
for name in ("oracle_plans_wmt14", "oracle_plans_wmt14_nt2013", "oracle_plans_xsum"):
    p = os.path.join(RESULTS, "cache", name + ".pt")
    if not os.path.exists(p):
        continue
    x = torch.load(p, weights_only=False)
    refs = x["refs"]
    if "xsum" in name:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        sent = lambda hyps: np.array([100 * sc.score(r, h)["rouge1"].fmeasure for h, r in zip(hyps, refs)])
    else:
        from sacrebleu.metrics import BLEU
        b = BLEU(effective_order=True)
        sent = lambda hyps: np.array([b.sentence_score(h, [r]).score for h, r in zip(hyps, refs)])
    systems = {"ncm single pass": x["single_pass"]}
    systems.update({k[4:]: v for k, v in x.items() if k.startswith("ext:") and "|" not in k})
    rows = {}
    for nm, hyps in systems.items():
        s = sent(hyps)
        rows[nm] = {"exact_match_pct": round(100 * float(np.mean([h.strip().lower() == r.strip().lower()
                                                                   for h, r in zip(hyps, refs)])), 2),
                    "near_copy_pct": round(100 * float((s >= 90).mean()), 2), "median_sentence_score": round(float(np.median(s)), 2)}
    res["memorization"][name.replace("oracle_plans_", "")] = rows

    if "xsum" in name and "ext:bart" in x:
        bart, single = sent(x["ext:bart"]), sent(x["single_pass"])
        sub = {}
        for label, mask in (("all", bart >= 0), ("bart_rouge1_below_90", bart < 90), ("bart_rouge1_below_50", bart < 50)):
            row = {"n": int(mask.sum()), "bart_alone_rouge1": round(float(bart[mask].mean()), 2),
                   "single_pass_rouge1": round(float(single[mask].mean()), 2)}
            for k in (1, 3, 5):
                gain = sent(x[f"ext:bart|k={k}"])[mask] - single[mask]
                row[f"k={k}"] = {"rouge1_gain": round(float(gain.mean()), 2),
                                 "share_of_lead": round(float(gain.mean() / (bart[mask] - single[mask]).mean()), 3)}
            sub[label] = row
        res["xsum_bart_subsets"] = sub
save_json(res, "oracle_report.json")
print(res)
