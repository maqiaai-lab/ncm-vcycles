# Pinned inputs

Everything this repository downloads, at the exact version behind the reported results. Every Hugging Face revision
below was last modified before the experiments (October 2026), so it is the version that was used; the one exception is
marked.

## Code

| Input | Used by | Version |
|---|---|---|
| [HazyResearch/ncm](https://github.com/HazyResearch/ncm) | `setup.sh` (cloned into `third_party/ncm`) | commit `c70274273f489abc73832b2b98584a8483b6b203` |

## Checkpoints and data (Hugging Face Hub)

| Repository | Files | Revision | Last modified |
|---|---|---|---|
| `hazyresearch/ncm-checkpoints` (model) | `sudoku/*`, `wmt14/*`, `xsum/*`, `lm1b/*` (tokenizer; small, medium, large), `tinystories/*` | `d342ce8aaea58830eef3b8ad473664673e41f692` | 2026-10-16 (model card only since the 2026-10-14 release) |
| `hazyresearch/ncm-datasets` (dataset) | `sudoku/*`, `wmt14/*`, `xsum/*`, `lm1b/*` (held-out validation text) | `6580ba41a4b0a976dde876fbe84d25327617b0e6` | 2026-10-16 (card only since the release) |

newstest2013, the WMT14 development set, is fetched by `vcycle/devsets.py` with sacrebleu (`sacrebleu -t wmt13 -l de-en`).

## Tokenizers named by the released configurations

| Repository | Use | Revision | Last modified |
|---|---|---|---|
| `google-bert/bert-base-uncased` (loaded as `bert-base-uncased`) | LM1B pre-tokenizer | `86b5e0934494bd15c9632b12f734a8a67f723594` | 2024-02-19 |
| `openai-community/gpt2` (loaded as `gpt2`) | XSum pre-tokenizer | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 2024-02-19 |
| `Qwen/Qwen3-0.6B` (tokenizer files only) | WMT14 pre-tokenizer | `c1899de289a04d12100db370d81485cdf75e47ca` | not recorded: this is the revision in the cache of the machine that ran the experiments |

The TinyStories tokenizer ships with its checkpoint.

## Models used only for evaluation

| Repository | Use | Revision | Last modified |
|---|---|---|---|
| `openai-community/gpt2-large` | generative perplexity of unconditional text; MAUVE features | `32b71b12589c2f8d625668d2335a01cac3249519` | 2024-02-19 |
| `openai-community/gpt2` | perplexity of XSum outputs (`vcycle/cond.py`) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 2024-02-19 |
| `Qwen/Qwen2.5-1.5B` | second perplexity scorer (`experiments/lm1b_quality.py`) | `8faed761d45a263340a0528343f099c05c9a4323` | 2024-10-08 |
| `Qwen/Qwen2.5-7B-Instruct` | LLM judge | `a09a35458c702b33eeacc393d103063234e8bc28` | 2025-01-12 |
| `Unbabel/wmt22-comet-da` | COMET | `2760a223ac957f30acfb18c8aa649b01cf1d75f2` | 2025-02-26 |
| `Helsinki-NLP/opus-mt-de-en` | plan swaps | `1a922f3b32a8e809e17a47d4b32142d8105924e5` | 2023-08-16 |
| `facebook/wmt19-de-en` | plan swaps | `bb316e37ea06b2267369c09fe6dcd30a308d2bbd` | 2025-07-29 |
| `facebook/bart-large-xsum` | plan swaps | `2179ab81d3f133e639f2957aec5380e9d56b2783` | 2023-01-24 |

The scripts load these by name (the default branch). To force the exact revisions, download them first, for example
`hf download openai-community/gpt2-large --revision 32b71b12589c2f8d625668d2335a01cac3249519`, and run with
`HF_HUB_OFFLINE=1`.

## Python environment

Python 3.12.10, torch 2.11.0+cu128. `requirements-lock.txt` is the full `pip freeze` of the environment the experiments
ran in (it also lists packages this repository does not need). COMET ran in its own environment
(`pip install unbabel-comet "setuptools<80"`).

Sudoku phases 0–3 and the first LM1B runs used four CPU cores. Later runs used an NVIDIA RTX 4050 laptop GPU (6 GB) and,
for the LLM judge, the error-diffusion sweep, the soft-decision runs and one replication of the Sudoku retries (the
`*_a10g` result files), one NVIDIA A10G or L4 GPU. Bit-identical outputs are expected only with the same hardware and
software; elsewhere, near-tie decisions can differ.
