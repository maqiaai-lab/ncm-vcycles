"""COMET-22 (Unbabel/wmt22-comet-da) for the WMT14 outputs of the plan swaps and the soft-decision confirmation, on
newstest2014 and newstest2013: single pass, plain soft decisions, top8 Ts=2 and top16 Ts=1
(experiments/soft_ablations.py --stage confirm), the external systems alone and NCM with their levels < k for
k = 1, 3, 5, 9 (experiments/oracle_plans.py). Paired bootstrap over sentences against the single pass, and against
plain soft decisions for the top-k arms. Output: results/comet_oracle.json, per-sentence scores in results/cache/.
Runs in its own environment: pip install unbabel-comet "setuptools<80".

python experiments/comet_oracle.py
"""
import json
import os
import sys

import numpy as np
import torch
from comet import download_model, load_from_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
rng = np.random.default_rng(0)
res = {}
for setname, tag in (("newstest2014", ""), ("newstest2013", "_nt2013")):
    svp = os.path.join(RES, "cache", f"soft_ablations_wmt14{tag}_confirm_texts.json")
    opp = os.path.join(RES, "cache", f"oracle_plans_wmt14{tag}.pt")
    if not os.path.exists(svp):
        print("missing", svp, file=sys.stderr)
        continue
    sv = json.load(open(svp, encoding="utf-8"))
    src, ref, systems = sv["sources"], sv["refs"], dict(sv["systems"])
    if os.path.exists(opp):
        op = torch.load(opp, weights_only=False)
        res[f"{setname}_single_pass_agreement"] = float(np.mean([x == y for x, y in zip(op["single_pass"], systems["single pass"])]))
        for e in ("marian", "fsmt"):
            if f"ext:{e}" in op:
                systems[f"{e} alone"] = op[f"ext:{e}"]
            for k in (1, 3, 5, 9):
                if f"ext:{e}|k={k}" in op:
                    systems[f"{e} plan k={k}"] = op[f"ext:{e}|k={k}"]
    scores = {}
    for nm, hyps in systems.items():
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyps, ref)]
        scores[nm] = np.array(model.predict(data, batch_size=64, gpus=1, progress_bar=False).scores)
        print(setname, nm, round(100 * float(scores[nm].mean()), 2), flush=True)
    idx = rng.integers(0, len(src), size=(1000, len(src)))

    def ci(d):
        bs = d[idx].mean(1)
        return [round(100 * float(d.mean()), 2), round(100 * float(np.percentile(bs, 2.5)), 2),
                round(100 * float(np.percentile(bs, 97.5)), 2)]

    rows = {}
    for nm, s in scores.items():
        rows[nm] = {"comet": round(100 * float(s.mean()), 2), "delta_vs_single": ci(s - scores["single pass"])}
        if nm.startswith("top"):
            rows[nm]["delta_vs_soft"] = ci(s - scores["soft all Ts=1"])
    res[setname] = rows
    json.dump(res, open(os.path.join(RES, "comet_oracle.json"), "w"), indent=1)
    np.savez(os.path.join(RES, "cache", f"comet_oracle_{setname}.npz"), **{k.replace(" ", "_"): v for k, v in scores.items()})
