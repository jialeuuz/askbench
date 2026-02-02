ask_eval是评测pipeline，data_pipeline是数据构建pipeline。
ask_eval/README.md记录着ask_eval的使用说明，ask_eval/readme_for_ai.md记录实现细节（方便定位与改码），用于llm调试和修改代码。ask_eval/run.sh是评测框架的入口脚本。
data_pipeline/README.md记录着数据构建pipeline的使用说明，data_pipeline/readme_for_ai.md记录实现细节（方便定位与改代码），用于llm调试和修改代码。data_pipeline/main.py是数据构建的入口脚本。
reward/readme记录着如何把RLVR reward接入到VERL中使用。
reward/readme_for_ai.md记录reward模块的代码结构、prompt/JSON schema与控制流（给LLM调试/改码用）。
reward/ask_mind_qa.py是AskMind（意图缺失维度）的VERL兼容reward实现。
reward/overconfidence_qa.py是AskOverconfidence（过度自信维度）的VERL兼容reward实现。
reward/train.sh是已脱敏的训练启动脚本参考（VERL + Ray + DAPO/GRPO）。
tools/vllm.sh用于用vLLM部署OpenAI-compatible API（把本地模型以API形式提供给评测/调用）。
tools/merge.sh用于将训练产物（如分片checkpoint）合并/导出成可被vLLM加载推理的HuggingFace模型目录。
paper.pdf是这个工作的论文。
