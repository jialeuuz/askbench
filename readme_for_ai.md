`ask_eval` is the evaluation pipeline, and `data_pipeline` is the data construction pipeline.

- `ask_eval/README.md`: user-facing usage guide for `ask_eval`.
- `ask_eval/run.sh`: entry script for running evaluations.
- `data_pipeline/README.md`: user-facing (open-source friendly) usage guide for `data_pipeline`.
- `data_pipeline/readme_for_ai.md`: implementation notes for debugging/modifying `data_pipeline`.
- `data_pipeline/main.py`: entry script for data construction.
- `tools/vllm.sh`: starts a vLLM OpenAI-compatible API server for serving a local model.
- `tools/merge.sh`: converts/merges sharded training checkpoints into an inference-ready HuggingFace model directory.

Chinese copies of the original docs are preserved with a `_zh` suffix (for example, `readme_for_ai_zh.md`).
