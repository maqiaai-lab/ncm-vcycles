"""Held-out data not shipped with the release."""
import os
import pickle
import subprocess
import sys

from vcycle.common import DATA


def wmt14_dev(d=None):
    """newstest2013 de-en (3,000 sentences) as the release's WMT14 pickles (lists of strings), fetched once with
    sacrebleu into data/wmt14_dev (the folder experiments/cond_soft_confirm.py uses)."""
    d = d or os.path.join(DATA, "wmt14_dev")
    if not os.path.exists(os.path.join(d, "sources.pickle")):
        os.makedirs(d, exist_ok=True)
        echo = lambda side: subprocess.run([sys.executable, "-m", "sacrebleu", "-t", "wmt13", "-l", "de-en", "--echo", side],
                                           check=True, capture_output=True, text=True, encoding="utf-8").stdout
        src, ref = echo("src").rstrip("\n").split("\n"), echo("ref").rstrip("\n").split("\n")
        assert len(src) == len(ref) == 3000, (len(src), len(ref))
        pickle.dump(src, open(os.path.join(d, "sources.pickle"), "wb"))
        pickle.dump(ref, open(os.path.join(d, "reference_translations.pickle"), "wb"))
    return d
