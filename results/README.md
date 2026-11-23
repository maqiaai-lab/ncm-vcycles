# Result files

Every file here was written by the script named in the table, in the runs behind the paper, and is shipped unchanged.
`SUMMARY.md` and `summary.json` are rebuilt from them by `python experiments/analyze.py`.

## Soft decisions and the diagnosis (prompted tasks)

| File | Written by | Contents |
|---|---|---|
| `cond_guide_soft_{wmt14,xsum}.json` | `experiments/cond_guide_soft.py` | first run: soft decisions by temperature and level subset, guidance bands; tuning half A, report half B |
| `cond_soft_confirm_wmt14.json` | `experiments/cond_soft_confirm.py` | newstest2013 sweep (`dev`), the chosen settings on newstest2014 (`arms`) |
| `cond_soft_confirm_xsum.json` | `experiments/cond_soft_confirm.py` | the 5,334 XSum articles not used by the first run (`arms`), and the full test set |
| `comet_wmt14.json`, `cache/comet_wmt14.npz` | `experiments/comet_score.py` | COMET per decoder, and per sentence |
| `cache/cond_soft_confirm_wmt14_texts.json` | `experiments/cond_soft_confirm.py` | newstest2014 references and the output of every decoder |
| `soft_ablations_wmt14_{tune,confirm}.json`, `soft_ablations_wmt14_nt2013_confirm.json` | `experiments/soft_ablations.py` | variants of soft decisions: nearest code, confidence gate, top-k blend |
| `oracle_ceiling_{wmt14,xsum}.json` | `experiments/oracle_ceiling.py` | tokenizer ceiling |
| `oracle_plans_wmt14.json`, `oracle_plans_wmt14_nt2013.json`, `oracle_plans_xsum.json` | `experiments/oracle_plans.py` | plan swaps: the first k levels from the reference (`gold`), from NCM's own re-encoded output (`self`), or from another system (`ext:*`) |
| `plan_scores_*.json` | `experiments/plan_scores.py` | a plan's levels under NCM's own generator |
| `oracle_report.json` | `experiments/oracle_report.py` | memorization check of the external systems |
| `comet_oracle.json`, `cache/comet_oracle_newstest201{3,4}.npz` | `experiments/comet_oracle.py` | COMET for the plan swaps and the variants |

## V-cycles

| File | Written by | Contents |
|---|---|---|
| `lm1b_{small,medium,large}_s<seed>.json`, `tinystories_s<seed>.json`, `cache/` files of the same names | `experiments/lm1b_diagnostic.py` | unconditional text, 256 samples per seed: the single pass, one V-cycle from each restart level and its equal-compute control; round-trip residuals. The caches hold the samples (token ids and codes) and the per-sample scores |
| `lm1b_sweep_small_s42.json` (+ cache) | `experiments/lm1b_sweep.py` | restart levels, repeated cycles, lower sampling temperatures |
| `lm1b_converge_{small,medium}_s42.json` (+ cache) | `experiments/lm1b_converge.py` | iterated V-cycles |
| `lm1b_quality_medium.json` (+ cache), `lm1b_mauve_medium.json` | `experiments/lm1b_quality.py`, `experiments/lm1b_mauve.py` | LM1B-medium, 20 seeds pooled: human anchor, a second perplexity scorer, MAUVE with paired subsampling |
| `{wmt14,xsum}_vcycle.json`, `{wmt14,xsum}_vcycle_raw.json` | `experiments/cond_vcycle.py` | prompted tasks: V-cycles by restart level and number of cycles (`_raw`: the output is re-encoded as picked, not in canonical form) |
| `phase0_sudoku_*.json`, `cache/sudoku_*_greedy_n2000.pt` | `experiments/phase0_baseline.py` | Sudoku: the released single pass, and the check of the decoder against `generate()` |
| `phase1_sudoku_*.json`, `cache/phase1_scores_*.pt`, `phase1b_localization_*.json` | `experiments/phase1_fixed_point.py`, `experiments/phase1b_localization.py` | Sudoku: the round-trip residual as an error detector, and where it points |
| `phase2_sudoku_hard_n1000_c3.json` (+ cache) | `experiments/phase2_vcycle.py` | Sudoku: greedy V-cycles without a verifier |
| `phase3_sudoku_hard_n2000_T1.0_s<seed>.json` (+ caches) | `experiments/phase3_controls.py` | Sudoku: checker-gated retries at matched compute, fixed frame |
| `sudoku_stack_{easy,medium,hard}.json`, `sudoku_stack_*_nosingles.json`, `sudoku_stack_*_a10g.json`, `cache/sudoku_stack_*_perboard.pt` | `experiments/sudoku_stack.py` | Sudoku: the one-pass soft-decision grid (`greedy_first_pass`, `soft`, `ts_star`) and checker-gated retries with symmetry frames and soft decisions; with and without singles propagation; `_a10g`: the replication on a second GPU |
| `ed_stage1.json`, `ed_stage2.json` | `experiments/ed_sweep.py` | error diffusion within a level (a negative result) |
| `judge_*.json`, `cache/judge_*.pt` | `experiments/judge.py` | pairwise LLM judge, and its per-item preferences |

Not included: the run logs, and the larger resumable caches of the runs (token ids and codes of every prompted-task
decoder, MAUVE features), which the scripts rebuild. `.gitignore` lists them.
