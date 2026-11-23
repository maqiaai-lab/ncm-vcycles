# Third-party notices

## Next-Concept Modeling (NCM)

The code here runs on top of the public NCM release, [HazyResearch/ncm](https://github.com/HazyResearch/ncm), which
`setup.sh` fetches at a pinned commit; it is not copied into this repository. The NCM checkpoints and datasets are
downloaded from the [NCM Hugging Face collection](https://huggingface.co/collections/hazyresearch/ncm) and keep their own
terms. NCM is released under the following license:

```
MIT License

Copyright (c) 2025 hermann kumbong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data

**WMT14 (newstest2013, newstest2014), XSum, LM1B, TinyStories and the Sudoku boards** are used through the NCM release's
checkpoints and evaluation data and through sacrebleu; they are not redistributed here, except for model outputs and
scores under `results/`. Two kinds of result files hold text from the evaluation sets next to the system outputs:
`results/cache/cond_soft_confirm_wmt14_texts.json` (the newstest2014 sources and references, as needed to recompute
BLEU), and the `examples` field of `results/wmt14_vcycle*.json` and `results/xsum_vcycle*.json` (three source documents
each, with their references).

## Models used for evaluation

GPT-2 and GPT-2 large, Qwen2.5-1.5B, Qwen2.5-7B-Instruct, Unbabel COMET (`wmt22-comet-da`), OPUS-MT
`Helsinki-NLP/opus-mt-de-en`, `facebook/wmt19-de-en` and `facebook/bart-large-xsum`, and the tokenizers
`bert-base-uncased`, `gpt2` and `Qwen/Qwen3-0.6B` named by the released configurations, are downloaded from the Hugging
Face Hub by the scripts and keep their own licenses. MAUVE is computed with the `mauve-text` package.
