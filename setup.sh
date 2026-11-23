#!/usr/bin/env bash
# Fetch the NCM code (pinned commit), the released checkpoints and the evaluation data (pinned revisions), and the
# Python packages. Nothing is trained here: every experiment runs the released checkpoints unchanged.
#   ./setup.sh                    everything
#   ./setup.sh sudoku wmt14       only the named tasks (any of: sudoku wmt14 xsum lm1b tinystories)
set -euo pipefail
cd "$(dirname "$0")"
NCM_COMMIT=c70274273f489abc73832b2b98584a8483b6b203
CKPT_REV=d342ce8aaea58830eef3b8ad473664673e41f692
DATA_REV=6580ba41a4b0a976dde876fbe84d25327617b0e6

if [ $# -gt 0 ]; then TASKS=("$@"); else TASKS=(sudoku wmt14 xsum lm1b tinystories); fi

if [ ! -d third_party/ncm ]; then
  git clone https://github.com/HazyResearch/ncm.git third_party/ncm
fi
git -C third_party/ncm checkout -q "$NCM_COMMIT"

pip install -r requirements.txt
# One download per task, with a single --include pattern each (several patterns in one call failed in an earlier run).
for task in "${TASKS[@]}"; do
  hf download hazyresearch/ncm-checkpoints --revision "$CKPT_REV" --include "$task/*" --local-dir artifacts
  case "$task" in
    tinystories) ;;   # unconditional generation: no evaluation data
    *) hf download hazyresearch/ncm-datasets --revision "$DATA_REV" --repo-type dataset --include "$task/*" --local-dir data ;;
  esac
done
