# MemCode

[![NeurIPS 2026 Poster](https://img.shields.io/badge/NeurIPS-2026_Poster-8A2BE2)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.19472-b31b1b.svg)](https://arxiv.org/abs/2610.19472)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official research code for **MemCode: Discrete Semantic Representations for Long-Term Agent Memory**, accepted at **NeurIPS 2026 (Poster)**.

[Paper](https://arxiv.org/abs/2610.19472) | [PDF](https://arxiv.org/pdf/2610.19472) | [中文说明](README_zh.md)

MemCode extracts source-grounded atomic memory units (AMUs), encodes them with
a T5 VQ-VAE, and retrieves evidence through dense vectors and codebook
co-occurrence. Optional consolidation retains links to the source dialogue.

## Setup

Use Python 3.10 or 3.11 and run commands from this directory:

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
```

1. Prepare the JSON files described in [data/README.md](data/README.md).
2. Configure the local model assets described in [models/README.md](models/README.md).
3. Copy `local_config.example.py` to `local_config.py`. Set `API_KEY`,
   `API_BASE_URL`, model paths and `DEVICE`. The endpoint must serve the selected
   working model and `JUDGE_MODEL`. Private configuration is excluded from Git.
4. Set `MAX_ROWS = 1` for an initial run, or `None` for all rows.
   Use a new `RUN_LABEL` for each independent run.
5. Run `python check_environment.py`, then choose an entry below.

Parameters are explicit variables in `settings.py` and `local_config.py`;
experiment selection does not require command-line options.

## Run configurations

| Dataset | Working model | Without consolidation | With consolidation |
|---|---|---|---|
| LoCoMo | GPT | `run_locomo_gpt_baseline.py` | `run_locomo_gpt_consolidated.py` |
| LoCoMo | Qwen | `run_locomo_qwen_baseline.py` | `run_locomo_qwen_consolidated.py` |
| LongMemEval | GPT | `run_longmemeval_gpt_baseline.py` | `run_longmemeval_gpt_consolidated.py` |
| LongMemEval | Qwen | `run_longmemeval_qwen_baseline.py` | `run_longmemeval_qwen_consolidated.py` |

For example:

```bash
python run_locomo_gpt_baseline.py
```

`baseline` selects MemCode without consolidation. Memory construction,
consolidation and answer generation use the selected working model.
Configure the evaluator model through `JUDGE_MODEL`.

Outputs are stored in `runs/<profile>/<RUN_LABEL>/`, including QA traces,
aggregate results, performance counters and memory stores. A run manifest
records the configuration and dataset/checkpoint hashes. LoCoMo requires a
fresh output directory; LongMemEval can resume a matching configuration.
LoCoMo category 5 is excluded by the runner configuration.

## Existing-memory workflow

Set `EXISTING_MEMORY_SOURCE_DIR` and `EXISTING_MEMORY_PROFILE`, then run:

```bash
python run_longmemeval_from_existing_memory.py
```

Each source `row_<index>_data/` must contain `metadata.db`,
`metadata.db.stats.json`, `vector_stores/atomic_memory.faiss` and
`vector_stores/atomic_memory_map.json`. Use the same dataset ordering,
embedding model, checkpoint and working model. The workflow copies each
store to a separate output directory before consolidation.

## Optional retrieval variants

For a baseline profile, set `RETRIEVER_OVERRIDE` to a supported module:

- `src.retrieval.bm25_baseline_retriever`
- `src.retrieval.ablation_vector_top40_retriever`
- `src.retrieval.ablation_codebook_expansion_retriever` (or `_v2`, `_v3`)
- `src.retrieval.idf_retriever`
- `src.retrieval.codebook_expansion_retriever`

## Training and tests

Prepare local DailyDialog JSON as described in `data/README.md`, then run:

```bash
python -m pip install -r requirements-analysis.txt
python -m vqvae_training.train_vqvae
```

Checkpoints are saved under `models/T5/`. Training and inference share
`src.utils.vqvae_t5_v10.VQVAE_T5`.

```bash
python -m unittest discover -s tests -v
python tools/validate_release.py
```

## Layout

```text
settings.py              Shared configuration and named profiles
local_config.example.py  Private configuration template
run_*.py                 Experiment entry points
experiments/             Evaluation loops and existing-memory workflow
src/core/                Memory structures, SQLite metadata and FAISS vectors
src/processing/          Asynchronous extraction and consolidation
src/retrieval/           Retrieval implementations
src/utils/               Embedding, LLM, evaluator and VQ-VAE implementations
vqvae_training/          Training and codebook visualization helpers
data/                    Data preparation instructions and downloader
models/                  Local model configuration instructions
tests/                   Offline tests
tools/                   Source checking and packaging
.github/                 CI workflow
```

中文运行说明见 [README_zh.md](README_zh.md).

## Citation

If you find MemCode useful, please cite our paper:

```bibtex
@article{li2026memcode,
  title   = {MemCode: Discrete Semantic Representations for Long-Term Agent Memory},
  author  = {Li, Xin and Hu, Liang and Miao, Duoqian and Peng, Ming and Zhang, Qi},
  journal = {arXiv preprint arXiv:2610.19472},
  year    = {2026}
}
```
