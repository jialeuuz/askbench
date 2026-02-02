`ask_eval` is the evaluation pipeline, and `data_pipeline` is the data construction pipeline.

- `ask_eval/README.md`: usage guide for `ask_eval`.
- `ask_eval/readme_for_ai.md`: implementation notes (useful for debugging/modifying the codebase with an LLM).
- `ask_eval/run.sh`: entry script for running evaluations.
- `data_pipeline/README.md`: usage guide for `data_pipeline`.
- `data_pipeline/readme_for_ai.md`: implementation notes for debugging/modifying `data_pipeline`.
- `data_pipeline/main.py`: entry script for data construction.
- `reward/readme`: how to plug the RLVR rewards into VERL.
- `reward/readme_for_ai.md`: code-level notes for the reward modules (prompts, JSON schema, and control flow).
- `reward/ask_mind_qa.py`: AskMind (intent-deficient) reward implementation (VERL-compatible).
- `reward/overconfidence_qa.py`: AskOverconfidence reward implementation (VERL-compatible).
- `tools/vllm.sh`: deploys an OpenAI-compatible API via vLLM (serve a local model as an API).
- `tools/merge.sh`: merges/exports training outputs (e.g., sharded checkpoints) into an inference-ready HuggingFace model directory for vLLM.
- `paper.pdf`: the paper PDF.

Chinese version: `readme_for_ai_zh.md`.
