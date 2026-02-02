# ask_eval: Evaluation Pipeline

`ask_eval` is a unified evaluation scaffold for both:

- **Single-turn benchmarks** (e.g., Math500, MedQA, GPQA, BBH), and
- **Judge-driven multi-turn protocols** (e.g., AskBench-style interactive evaluation), where a judge model both grades final answers and simulates user follow-ups when the tested model asks for clarification.

This README focuses on how to install and run the evaluation pipeline. For implementation details and code pointers, see `ask_eval/readme_for_ai.md`. A Chinese copy of the original documentation is preserved as `ask_eval/README_zh.md`.

## Quickstart

```bash
# 1) Install dependencies (virtualenv recommended)
pip install -e .

# 2) Edit the base config as needed
vim config/base.ini

# 3) Run evaluation (example)
python scripts/main.py --config config/base.ini
# Or use run.sh: edit variables at the top, then run
./run.sh
```

After the run finishes, outputs are written to `results/<task>/<task_name>/`, and a summary line is appended to `results/final_result.txt`.
