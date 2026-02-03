<div align="center">

# When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification

[![Paper](https://img.shields.io/badge/Paper-PDF-blue?logo=adobeacrobatreader&logoColor=white)](paper.pdf)
[![HuggingFace (Bench)](https://img.shields.io/badge/HuggingFace-askbench__bench-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/jialeuuz/askbench_bench)
[![HuggingFace (Train)](https://img.shields.io/badge/HuggingFace-askbench__train-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/jialeuuz/askbench_train)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

English | [中文](readme_zh.md) | [LLM Guide](readme_for_ai.md)

</div>

This repository contains the code and assets for the paper **“When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification”**. The arXiv version is under review; you can read the current PDF here: 🔗 [paper.pdf](paper.pdf).

Large language models often respond confidently even when a prompt is underspecified or contains misleading premises. This project studies **when** a model should ask for clarification and **what** it should ask, and provides:

- **AskBench**: an interactive benchmark that converts standard QA pairs into multi-turn interactions with explicit checkpoints.
- A **unified judge loop** that (1) evaluates final answers and (2) simulates user replies when the model asks questions.
- Two core settings:
  - **AskMind**: intent-deficient / missing-information queries that require clarification.
  - **AskOverconfidence**: queries with false premises that must be identified and corrected before answering.

For a concise, LLM-oriented guide to the codebase structure and key entry points (useful when debugging/modifying the repo with an LLM), see `readme_for_ai.md` (Chinese: `readme_for_ai_zh.md`).

## 📌 Table of contents

- 🚀 Evaluation: [run evaluation](#evaluation)
- 🎯 Training: [RLVR reward + VERL integration](#training)
- 🧪 Data pipeline: [build AskBench-style data](#data-pipeline)
- 🛠️ Tools: [checkpoint merge + OpenAI-compatible serving](#tools)
- 📦 Datasets: [Hugging Face links](#datasets)

## ✨ AskBench at a glance

AskBench evaluates clarification as an *interactive* skill. Each example is run with:

- a **tested model** (the assistant under evaluation), and
- a **judge model** that plays multiple roles:
  - **simulated user** (provides follow-up information when the assistant asks), and
  - **grader** (judges whether the final answer is correct and whether required points were properly covered).

The tested model may ask clarification questions; the judge loop may simulate user replies as needed; and the evaluation ends with a final answer and a judge decision.

## 🔎 Why AskBench?

Many real user prompts are **underspecified** or contain **misleading premises**. Traditional single-turn QA benchmarks mostly measure “final answering”, but they do not directly measure:

- whether a model decides to ask a follow-up question at the right time, or
- whether the follow-up question targets the *right missing/misleading points*.

AskBench is designed to make clarification **measurable and scalable**:

- **Interactive + automatable**: the judge loop simulates user replies only when the candidate explicitly asks, and grades the final answer end-to-end.
- **Fine-grained + interpretable**: checkpoint/rubric items turn “clarification quality” into actionable diagnostics (e.g., checkpoint coverage).
- **Extensible**: standard QA can be adapted by generating a “variant question” (degraded or misleading) plus a checklist.
- **Easy to adopt**: the evaluation pipeline only requires OpenAI-compatible API endpoints (candidate + judge), which can be served locally (e.g., via vLLM).

## 📈 Results

In the paper, rubric-guided RLVR improves AskBench multi-turn clarification performance while preserving (and often improving) broad QA capabilities.

### AskBench multi-turn clarification (Standard protocol, Table 4)

Metrics:

- `acc` (accuracy): whether the final answer is correct (judge-graded).
- `cov` (checkpoint coverage): how much of the checklist is explicitly covered before answering (`required_points` for AskMind; `misleading_points` for AskOverconfidence).

| Model | AskMind `acc` | AskMind `cov` | AskOverconfidence `acc` | AskOverconfidence `cov` |
| --- | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 0.567 | 0.124 | 0.840 | 0.749 |
| GPT-4.1 | 0.495 | 0.118 | 0.730 | 0.602 |
| Qwen2.5-7B-Instruct | 0.332 | 0.214 | 0.443 | 0.188 |
| OursI | 0.615 | 0.679 | 0.628 | 0.641 |
| OursO | 0.617 | 0.807 | 0.548 | 0.894 |

### Strict two-turn protocol (“Hard”, Table 5)

Under the strict two-turn protocol, turn 1 must clarify/correct; turn 2 must answer directly (no more follow-ups).

| Model | AskMind `acc` | AskMind `cov` | AskOverconfidence `acc` | AskOverconfidence `cov` |
| --- | :---: | :---: | :---: | :---: |
| Gemini-2.5-Pro | 0.0551 | 0.2206 | 0.0100 | 0.7350 |
| GPT-4.1 | 0.0352 | 0.2035 | 0.0000 | 0.5865 |
| Qwen2.5-7B-Instruct | 0.0176 | 0.1288 | 0.0050 | 0.1955 |
| OursI | 0.2714 | 0.5013 | 0.1975 | 0.5065 |
| OursO | 0.1965 | 0.4235 | 0.2600 | 0.7778 |

Note: the paper abbreviates Gemini-2.5-Pro as *Gemini*, GPT-4.1 as *GPT*, and Qwen2.5-7B-Instruct as *Qwen*.
OursI and OursO are our rubric-trained models for AskMind and AskOverconfidence, respectively.

### Single-turn QA + HealthBench (Table 3)

| Model | Math500 | MedQA | HealthBench | GPQA-d | BBH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 0.760 | 0.653 | 0.526 | 0.309 | 0.506 |
| OursI | 0.780 | 0.936 | 0.606 | 0.497 | 0.758 |
| OursO | 0.720 | 0.992 | 0.559 | 0.781 | 0.760 |

Note: Some benchmarks here (e.g., HealthBench) are LLM-judge-based. To reduce cost and improve reproducibility, we use an open-source judge (e.g., Qwen3-30B-A3B-Instruct-2507 in the paper) instead of a proprietary GPT-based judge, so absolute scores may differ from official numbers while the overall ranking trends remain consistent.

## 🧩 Repository layout

- `ask_eval/`: evaluation pipeline (single-turn + AskBench-style multi-turn).
  - User guide: `ask_eval/README.md`
  - Implementation notes: `ask_eval/readme_for_ai.md`
  - Entry script: `ask_eval/run.sh`
- `data_pipeline/`: data construction pipeline for building AskBench-style data for **training and evaluation** (e.g., adapting standard QA into AskMind/AskOverconfidence-style variants + checklists).
  - User guide: `data_pipeline/README.md`
  - Implementation notes: `data_pipeline/readme_for_ai.md`
  - Entry script: `data_pipeline/main.py`
- `reward/`: rubric-guided reward function / training helpers (for RLVR-style training).
- `tools/`: helper scripts for (1) converting training checkpoints into an inference-ready HuggingFace model dir, and (2) serving a model as an OpenAI-compatible API (vLLM).
- `readme_for_ai.md`: LLM-oriented repository guide (architecture + key entry points).
- `paper.pdf`: paper PDF (anonymous submission build).

Chinese copies of the original documentation are preserved with a `_zh` suffix (e.g., `readme_zh.md`, `ask_eval/README_zh.md`).

## ⚙️ Setup

Recommended: Python 3.10+ in a conda environment.

### Install `ask_eval`

```bash
conda create -n askq python=3.10 -y
conda activate askq

pip install -e ./ask_eval
```

### Install `data_pipeline` dependencies

```bash
pip install -r data_pipeline/requirements.txt
```

<a id="evaluation"></a>
## 🚀 Quickstart: run evaluation (AskBench + standard QA)

`ask_eval` expects an **OpenAI-compatible** chat-completions API for:

- the **tested model** (candidate), and
- the **judge model** (used for grading; and for AskBench, also for user simulation).

1) Configure your model endpoints and tokens in `ask_eval/config/base.ini` (and/or per-task overrides under `ask_eval/config/common/`).
2) Run:

```bash
cd ask_eval
python scripts/main.py --config config/base.ini
```

For a convenience wrapper that overrides config fields via shell variables, see `ask_eval/run.sh`.

Notes:

- AskBench-style tasks run a judge-driven multi-turn protocol via `ask_eval/scripts/run_ask.py`.
- You can enable a stricter two-turn AskBench protocol via `STRICT_MODE=1` in `ask_eval/run.sh`.
- Evaluation outputs are written under `ask_eval/results/<task>/<task_name>/`, and an aggregated line is appended to `ask_eval/results/final_result.txt`.

<a id="tools"></a>
## 🛠️ Tools: checkpoint conversion + OpenAI-compatible serving

`ask_eval` calls models via an OpenAI-compatible chat-completions API. If your workflow is API-based, the two scripts under `tools/` are intended to cover a common flow:

1) (Optional) **Convert a training checkpoint** into an inference-ready HuggingFace model directory: `tools/merge.sh`.
2) **Serve the model as an OpenAI-compatible API** using vLLM: `tools/vllm.sh`.

### Convert (merge) a trained checkpoint for inference (`tools/merge.sh`)

Some training runs (e.g., sharded checkpoints from VERL/RLVR training) are not directly loadable by vLLM. In that case, run the conversion step to export a standard HuggingFace model folder.

1) Edit `tools/merge.sh` to set:
   - `CHECKPOINT_DIR`: the training checkpoint directory (often an `.../actor` folder)
   - `OUTPUT_DIR`: where to write the merged/exported model
   - `WORLD_SIZE`: number of checkpoint shards (typically your training world size)
   - `MERGE_SCRIPT_PATH`: path to the `merge_verl.py` conversion script in your environment
2) Run:

```bash
bash tools/merge.sh
```

After success, point `MODEL_PATH` in `tools/vllm.sh` to the exported `OUTPUT_DIR`.

### Serve a model as an OpenAI-compatible API (`tools/vllm.sh`)

This script launches vLLM’s OpenAI-compatible server (`vllm.entrypoints.openai.api_server`).

1) Edit `tools/vllm.sh` to set:
   - `MODEL_PATH`: a HuggingFace model directory (base model, or the `OUTPUT_DIR` produced by `tools/merge.sh`)
   - `CUDA_DEVICES` and `TP`: should match the number of GPUs used for tensor-parallelism
   - `PORT`: server port
2) Run:

```bash
bash tools/vllm.sh
```

Then configure `ask_eval/config/base.ini` (or `ask_eval/run.sh`) to point at the server, e.g.:

- `[model] api_url = http://<host>:<port>/v1`
- `[model] model_name = default` (must match `--served-model-name` in `tools/vllm.sh`)

<a id="datasets"></a>
## 📦 Datasets

- **Hugging Face (recommended download links)**:
  - 🤗 AskBench evaluation data: [jialeuuz/askbench_bench](https://huggingface.co/datasets/jialeuuz/askbench_bench)
  - 🤗 AskMind/AskOverconfidence training trajectories: [jialeuuz/askbench_train](https://huggingface.co/datasets/jialeuuz/askbench_train)
- **Evaluation data (tracked in this repo)**: under `ask_eval/data/` (AskBench subsets + standard benchmarks used by the pipeline).
- **Optional training / intermediate data (not tracked)**: you can place large local files under `data/` (this repo’s `.gitignore` ignores `data/` by default).

## 📝 Outputs (what gets written)

Depending on the task type, `ask_eval` writes a combination of:

- `results.txt`: human-readable summary (metrics + timing).
- `summary_results.json`: per-example outputs for single-turn tasks.
- `askbench_detailed_results.json`: turn-by-turn traces and judge decisions for AskBench-style tasks.

## 🧱 Build (or rebuild) combined AskBench eval sets

The AskBench “main” tasks are small mixtures built from multiple subsets (e.g., 100 per source benchmark).

```bash
python ask_eval/data/ask_bench/ask_mind/build_combined_eval.py
python ask_eval/data/ask_bench/ask_overconfidence/build_combined_eval.py
```

<a id="data-pipeline"></a>
## 🧪 Quickstart: build AskBench-style data (training + evaluation)

The data construction pipeline can generate AskBench-style multi-turn conversations (clarify → simulated user reply → answer → judge) for training, and can also be used to adapt other QA benchmarks into AskMind/AskOverconfidence-style evaluation data (by generating variant questions + checklist/rubrics).

See `data_pipeline/README.md` for the recommended entry points and parameters.

<a id="training"></a>
## 🎯 Rubric-guided reward (RLVR)

The `reward/` directory contains **VERL-compatible** reward functions that implement the paper’s rubric-guided, turn-level shaping:

- AskMind (intent-deficient / missing info): `reward/ask_mind_qa.py` (`data_source = ask_mind_qa`)
- AskOverconfidence (misleading premises): `reward/overconfidence_qa.py` (`data_source = overconfidence_qa`)

These scripts are meant to be copied into VERL (`verl/utils/reward_score/`) and registered in `default_compute_score()`. Configure judge endpoints via `API_URLS` / `JUDGE_MODEL_NAME`. See `reward/readme` for step-by-step integration, and `reward/readme_for_ai.md` for code-level notes.

For a sanitized reference training launcher (VERL + Ray + DAPO/GRPO), see `reward/train.sh`.

## 📚 Citation

If you use this codebase, please cite the paper:

```bibtex
@misc{askbench2026,
  title        = {When and What to Ask: AskBench and Rubric-Guided RLVR for LLM Clarification},
  author       = {Anonymous},
  year         = {2026},
  note         = {Anonymous ACL submission},
}
```
