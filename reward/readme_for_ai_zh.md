## Reward 模块说明（给 LLM 调试/改码用）

本目录包含两个可直接接入 **VERL** 的 reward 模块，对应论文中的 rubric-guided、turn-level shaping：

- `reward/ask_mind_qa.py`：AskMind（意图缺失 / 信息不足）reward
- `reward/overconfidence_qa.py`：AskOverconfidence（过度自信 / 错误前提）reward

两份代码都提供与 VERL reward 接口兼容的顶层入口函数：

- AskMind：`compute_score_ask_mind_qa(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`
- Overconfidence：`compute_score_overconfidence_qa(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float`

### 总体流程（两份代码一致）

1）**`compute_score_*` 解析 `extra_info`**
   - 判断是否为最终轮（`is_final_turn`）
   - 读取原始问题、当前问题、对话上下文、rubric/checklist 条目等

2）**非最终轮 shaping**
   - AskMind 使用 `required_points: List[str]` 作为 checklist
   - Overconfidence 使用 `misleading_points: List[str]` 作为 checklist

   两者都会要求 judge 输出严格 JSON：
   - `answered_final: bool`（在非最终轮是否提前给最终答案）
   - `hits: List[bool]`（逐条 checklist 是否被显式覆盖；长度必须等于 checklist 长度）
   - `irrelevant_or_redundant: bool`
   - `notes: List[str]`（可选短备注；不参与打分）

   然后把 judge 输出映射到论文里的离散 reward 集合：

   - `answered_final == True` → `-2.0`
   - `answered_final == False` 且覆盖数为 0 → `-0.8`
   - 部分覆盖 → `0.8`
   - 全覆盖 → `1.0`

3）**最终轮评分**
   - judge 返回 `{"decision": "still_asking" | "wrong" | "correct"}`
   - 映射到 `-2.0 / -1.0 / 1.0`

4）**网络调用与鲁棒性**
   - `call_llm_api_json()` 调用 OpenAI-compatible 的 `/v1/chat/completions`
   - system prompt 强制 JSON-only + `_extract_json` 容错解析
   - `API_URLS` 多端点随机选取并重试
   - 异常/解析失败时返回保守默认分（`DEFAULT_*_FAIL`）

### 常见改动点

- `API_URLS`：judge 端点列表（OpenAI-compatible）
- `JUDGE_MODEL_NAME`：需要与 vLLM 的 `--served-model-name` 一致
- 默认分：`DEFAULT_NON_FINAL_FAIL`, `DEFAULT_FINAL_FAIL`
- prompts 与 JSON schema：如果希望稳定解析，尽量保持 schema 不变

### `extra_info` 字段（最小集合）

AskMind（`ask_mind_qa.py`）：

- `is_final_turn: bool`
- `ori_question: str`
- `degraded_info: str`
- `required_points: List[str]`（推荐；开启 checklist 模式）
- `question: str`（当前用户轮）
- `context: str`（对话历史）
- `expected_answer: str`（最终轮目标；为空则回退 `ground_truth`）

AskOverconfidence（`overconfidence_qa.py`）：

- `is_final_turn: bool`
- `ori_question: str`
- `overconfidence_info: str`
- `misleading_points: List[str]`（推荐；开启 checklist 模式）
- `question: str`
- `context: str`
- `expected_answer: str`（最终轮目标；为空则回退 `ground_truth`）
