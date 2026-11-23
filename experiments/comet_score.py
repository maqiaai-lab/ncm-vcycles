"""COMET (Unbabel/wmt22-comet-da) for the WMT14 arms of experiments/cond_soft_confirm.py: a learned MT metric that
does not count n-grams, as a second opinion on BLEU and chrF. Runs in its own environment (unbabel-comet pins its own
dependencies):  python -m venv .venv_comet && .venv_comet/bin/pip install unbabel-comet

Reads results/cache/cond_soft_confirm_wmt14_texts.json; writes results/comet_wmt14.json (system scores, and every arm
minus the single pass with a paired bootstrap CI over sentences) and the per-sentence scores to results/cache.

python experiments/comet_score.py
"""
import json
import os
import sys
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
name = sys.argv[1] if len(sys.argv) > 1 else "cond_soft_confirm_wmt14"
texts = json.load(open(os.path.join(RESULTS, "cache", name + "_texts.json"), encoding="utf-8"))

from comet import download_model, load_from_checkpoint  # noqa: E402

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
scores = {}
for arm, hyps in texts["hyps"].items():
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(texts["src"], hyps, texts["ref"])]
    scores[arm] = np.array(model.predict(data, batch_size=64, gpus=1, progress_bar=False).scores)
    print(f"{arm:32s} COMET {100 * scores[arm].mean():.2f}", flush=True)

base = scores["single pass"]
n = len(base)
res = {"model": "Unbabel/wmt22-comet-da", "sentences": n, "arms": {}}
for arm, s in scores.items():
    r = {"comet": 100 * float(s.mean())}
    if arm != "single pass":
        d = 100 * (s - base)
        idx = np.random.default_rng(zlib.crc32(arm.encode())).integers(0, n, (1000, n))
        r["delta"] = float(d.mean())
        r["ci"] = [float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))]
        r["share_better"] = float((s > base).mean())
    res["arms"][arm] = r
    print(arm, r, flush=True)
with open(os.path.join(RESULTS, "comet_wmt14.json"), "w") as f:
    json.dump(res, f, indent=2)
np.savez(os.path.join(RESULTS, "cache", "comet_wmt14.npz"), **{k.replace(" ", "_"): v for k, v in scores.items()})
