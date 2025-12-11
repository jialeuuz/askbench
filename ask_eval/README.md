# ask_eval 评测框架说明

本仓库提供了一套针对多模任务的统一评测脚手架，既支持传统单轮基准（Math500、MedQA 等），也支持 AskBench 等由裁判模型驱动的多轮对话评测。本文档旨在快速梳理整体架构、关键模块与使用方式，方便后续查阅。

## 快速开始

```bash
# 1. 安装依赖（建议使用虚拟环境）
pip install -e .

# 2. 按需修改基础配置
vim config/base.ini

# 3. 启动评测（示例）
python scripts/main.py --config config/base.ini
# 或者使用 run.sh，它会先覆盖 base.ini 再运行 main.py
./run.sh -u http://<model_host>/v1/chat/completions -t math500,medqa
```

运行结束后，结果会写入 `results/<task>/<task_name>/`，并追加汇总条目到 `results/final_result.txt`。

## 目录速览

| 目录/文件 | 作用 |
| --- | --- |
| `scripts/main.py` | 评测入口：逐任务读取配置并调度相应的运行脚本 |
| `scripts/run.py` | 单轮评测主循环（Math/MedQA 等） |
| `scripts/run_ask.py` | AskBench 多轮对话评测（被测模型 + 裁判模型） |
| `scripts/run_ask_lone.py` | AskLone 评测入口，执行通过率估计 + 最终答复判定 |
| `ask_eval/models/` | 模型封装，统一 API 调用、批量推理与保活逻辑 |
| `ask_eval/data/` | 数据加载器（当前默认读取 `test.jsonl`） |
| `ask_eval/evaluators/` | 评估器实现与评测策略 |
| `ask_eval/utils/config.py` | 配置读取、合并与结果汇总工具 |
| `config/base.ini` | 全局默认配置；`config/common/*.ini` 为任务差分配置 |
| `run.sh` | 快速覆盖配置并启动评测的脚本 |
| `data/fata/*` | FATA 任务数据目录（按任务名放置 `test.jsonl`） |

## 评测执行流程

1. **加载基础配置**：`scripts/main.py` 读取 `--config` 指定的 INI 文件，解析 `[tasks] enabled` 列表。
2. **逐任务调度**：
   - 默认情况下拼接 `config/common/<task>.ini` 后调用 `scripts/run.py`。
   - 任务名包含 `ask_lone` 时，调用 `scripts/run_ask_lone.py` 执行通过率评估 + 最终答复评分。
   - 任务名包含 `fata`、`ask` 或 `quest_bench` 时，统一调用 `scripts/run_ask.py` 触发 Judge 驱动的多轮评测（FATA 逻辑见下文）。
   - 配置中若将 `tasks_config_path` 指向其它模板（如 EvalScope/OpenCompass），会走相应分支并在结束后写入最终指标。
3. **结果写出**：
   - 单轮评测：生成 `api_responses.json`、`summary_results.json`、`results.txt`。
   - AskBench：生成 `askbench_detailed_results.json`、`results.txt`。
   - 所有任务完成后，调用 `write_final_result_file` 或 `write_final_evalscope_result_file` 追加汇总信息。

整体流程可以概括为：

```
INI 配置 -> Merge 任务配置 -> 加载数据 -> 模型批量推理
       -> 评估器比对答案/调用裁判 -> 写入 attempt 与汇总日志
```

## 配置体系

- **基础配置 (`config/base.ini`)**
  - `[model]`：被测模型的 API、鉴权、系统提示等。
  - `[generateconfig]`：推理参数（`max_tokens`、`temperature`、`max_concurrent`、`n_attempts` 等）。
  - `[tasks]`：任务开关与任务配置目录。
  - `[path]`：默认数据及结果根目录。
  - `[evaluatorconfig]`：裁判模型配置，除 AskBench/AskLone 任务外，也会被 math500 / medqa / gpqa 等单轮任务复用。
- **任务配置 (`config/common/<task>.ini`)**
  - 覆写数据路径、任务别名、默认 API 等。
  - `load_merged_config` 会以基础配置为主导，逐段覆盖任务配置。
- **运行脚本 (`run.sh`)**
  - 支持以命令行参数覆盖配置（模型 URL、任务列表、温度、并发等）。
  - 在运行前备份 `base.ini`，结束后恢复，避免污染默认配置。
  - 新增 `--guidance-mode` / 环境变量 `GUIDANCE_MODE` 控制首轮引导策略，可选 `none`（默认）、`weak`、`strong`、`fata`；其中 `fata` 会在首轮用户消息中追加官方 FATA 引导文案，便于按需对比 baseline。

## 数据加载层

`ask_eval/data/data_map.py` 将任务名映射到数据加载器（当前均为 `JsonlLoader`），默认读取 `data/<group>/<task>/test.jsonl`。样本结构因任务类型略有区别：

- **数学与常规 QA**：包含 `problem` / `ori_question` 与 `expected_answer`。
- **降质任务（*_de）**：使用 `degraded_question` 和 `expected_answer`。
- **AskBench 系列**：样本包含 `degraded_question`、`ori_question`、`expected_answer`、`degraded_info`，以及 `required_points`（列出必须补齐的关键信息，现已覆盖 ask_mind* 与 quest_bench），用于多轮对话模拟。
- **in3_interaction**：原始数据只提供 `task`、`vague`、`missing_details` 以及示例交互。评测器会将 `task` 重命名为 `ori_question`/`degraded_question`，把 `missing_details` 中每个元素的 `description` 汇总成 `required_points`，并把整段 `missing_details` 转写成 `degraded_info`。由于没有 `expected_answer`，该基准只衡量澄清问答行为（ask-rate/覆盖率/冗余提问等），不计算 Accuracy。
- **AskLone 系列**：只使用 `ori_question` 与 `expected_answer`，用于单轮作答与“不会做就承认”评估。
- **HealthBench**：`prompt` 内直接提供多轮对话消息列表（`role`/`content`），`rubrics` 为带 points 的评分项。被测模型在现有对话上生成回复，再将完整对话与单条 rubric 传给裁判模型（模板见 `data/common/healthbench/grader_prompt.py`）；命中 rubric 得到对应分值（含负分），最后用全部正分和做归一得到 0-1 区间得分（最低截断为 0）。
- **ask_mind 汇总集**：`data/ask_bench/ask_mind/test.jsonl` 由 `ask_mind_math500de/medqade/gpqade/bbhde` 各采样 100 题（总计 400 题）拼接而成，可通过 `python data/ask_bench/ask_mind/build_combined_eval.py` 复现。任务名 `ask_mind` 沿用 AskBench 逻辑与综合得分计算。

如需引入新任务，可在 `LOADER_MAP` 注册自定义加载器，或沿用 JSONL 格式。

## 模型抽象

所有模型实现均继承自 `ask_eval/models/base_api_model.BaseAPIModel`，提供统一的同步/异步推理接口。

- **`CustomAPI`**：面向自建推理服务，支持 `enable_thinking`、自定义 header、自动拆分 `<think></think>` 内容。
- **`GPTAPI`**：适配 GPT-4o / GPT-5 等接口，内置 QPS 控制与图片、Developer prompt 等请求格式处理。
- **健康检查与保活**：`create_model` 在实例化后会调用 `check_urls_health`，确保至少找到一个可用 URL。

批量推理通过 `infer_batch_async` 搭配 `max_concurrent` 控制并发，返回响应正文、链路推理（thinking）以及截断标记。

## 评估器体系

评估器均从 `ask_eval/evaluators/base_evaluator.BaseEvaluator` 派生，实现：

- `format_example`：准备发送给模型的 prompt（支持 few-shot）。
- `extract_answer`：从模型输出中摘取候选答案。
- `validate_answer`：将预测与标准答案比对（可同步/异步）。
- `evaluate_responses`：写出 `api_responses.json` 并统计精准率、截断比例。

任务到评估器的绑定由 `ask_eval/evaluators/evaluator_map.EVALUATOR_MAP` 管理。典型实现包括：

- **数学系列**：`MathEvaluator` 提供 LaTeX 归一化、SymPy 化简、数字匹配等能力；`Math500Evaluator` 等实现按需覆写提示与抽取逻辑。
- **降质数学**：`MathDeEvaluator` 继承通用数学逻辑，但 prompt/字段针对 `degraded_question`。
- **MedQA / GPQA 等**：各自实现领域裁剪的提取与校验。
- **AskEvaluator**：多轮对话核心。裁判模型兼任三种角色：判断是否给出最终答案、评估答案正确性、在模型提问后模拟用户回复。评测循环包含：
  1. 被测模型生成下一轮回复（最后一轮会强制输出最终答案）。
  2. 裁判模型判定回复是否是终结回答及其正确性。
  3. 若未终结且仍有轮次，请裁判模型根据隐藏的 `ori_question` 与场景上下文（如 `degraded_info` / `overconfidence_info`）生成符合人类行为的追加信息。
  4. 记录回合日志，直至模型回答或轮次耗尽。

### 单轮 Judge 判分

`math500`、`medqa` 与 `gpqa` 现统一通过裁判模型判分，以替代脆弱的正则比对：

- 每个样本都会把题干、标准答案以及正则提取到的候选答案一起交给 Judge。
- 裁判需先输出 `Reasoning: ...`，再给出一个 JSON 代码块，字段固定为 `{"reason": "...", "result": "correct" | "incorrect"}`，从而满足“先解释、后给结论”的需求。
- JSON 解析失败会自动重试，最多 10 次。若仍然失败，则跳过该样本（`skipped=true`），不会纳入准确率/Pass@1，且在 `api_responses.json` 中记录失败原因。
- `[evaluatorconfig]` 中的裁判模型配置会被这些单轮任务自动复用，无需逐任务重复填写。

最后会生成 `askbench_detailed_results.json`，记录每个样本的对话轨迹、裁判判定与失败原因统计。

## 结果产出

标准单轮任务的输出结构：

- `api_responses.json`：逐样本详情（原始回复、抽取答案、正确与否、思维链、截断状态）。
- `summary_results.json`：按题目聚合的多尝试结果（包含 `pass@1` 统计）。
- `results.txt`：人类可读的摘要（准确率 / Pass@1、时间开销、截断汇总等）。
- `results/final_result.txt`：所有任务完成后由 `write_final_result_file` 追加的行式汇总。

AskBench 额外生成 `askbench_detailed_results.json`（包含回合日志和失败原因分布）。对于 EvalScope / OpenCompass 等特殊路径，会在任务目录内寻找最新时间戳文件夹并解析对应的 `results.txt`，保持统一的最终汇总格式。

## AskMind 指标扩展

- **新增数据字段**：`data/ask_bench/ask_mind/*/test.jsonl` 以及 `data/ask_bench/quest_bench/test.jsonl` 现包含 `required_points`，用于列出所有被劣化/缺失的关键信息，方便裁判判断模型是否已经问完必要的问题。例如：
  ```json
  {
    "degraded_question": "...",
    "required_points": [
      "Exact value of the first lifetime (10^-9 sec)",
      "Exact value of the second lifetime (10^-8 sec)"
    ]
  }
  ```
- **Ask Overconfidence 字段**：`data/ask_bench/ask_overconfidence/*/test.jsonl` 使用 `overconfidence_question`、`overconfidence_info`、`misleading_points`，分别对应暴露给模型的带有错误暗示的题面、错误论断与正确事实说明，以及必须被模型质疑/修正的误导点清单。AskEvaluator 会把这些字段自动映射成场景上下文与“必查点”，逻辑与 `required_points` 一致。
- **裁判输出规范**：AskEvaluator 会要求裁判模型先给出一行 `Reasoning:`，再输出一个严格的 ```json 代码块，字段包含 `is_final_answer`、`is_correct`、`all_required_points_resolved`、`missing_required_points` 与可选的 `notes`。若未能解析出 JSON，将自动重试，最多 10 次；若仍失败，则跳过该样本（不计入最终分数，并在结果中标记 `JudgeJSONParseFailed`）。
- **指标拆解**：`askbench_detailed_results.json` 会记录每轮覆盖了哪些 `required_points`、是否出现“信息已经齐全却继续提问”的事件，以及最终答案是否在信息缺失的情况下给出。
- **结果统计**：`results.txt` 与 CLI 输出会同时给出：
  - 只统计有效样本的准确率；
  - “必要提问率” (ask_rate)——在所有有效样本中，被测模型是否至少发起过一次澄清问题的样本占比（例如 500 条中有 300 条曾经发问，则 ask_rate = 300/500）；
  - “合规率” (cov_rate)——在给出最终答案前是否补齐全部 `required_points`；
  - “冗余追问信率” (unq_rate)——信息已经齐全仍继续提问的样本数与事件数；
  - “综合得分” (score)——适用于 `ask_mind_math500de/medqade/gpqade/bbhde`、`ask_overconfidence(+_math500/+_medqa)` 以及 `quest_bench`，按照 `0.5 * acc + 0.3 * cov_rate + 0.2 * (1 - unq_rate)` 汇总，`unq_rate` 越低越好；
  - 全量原因分布（含被跳过样本），方便定位问题。
- **in3_interaction 特例**：沿用同一套 ask 指标，但由于缺少 `expected_answer`，最终日志只会给出 “Vague Ask Rate / Clear-task Direct Rate / cov_rate / unq_rate”等行为指标，不再输出 Accuracy 或综合得分，并在 `results.txt` 首行额外记录 `Vague Ask Rate` 以便 `final_result.txt` 汇总。

## FATA 双阶段评测

- **数据来源**：`fata_math500` 与 `fata_medqa` 直接复用 AskMind 数据（分别从 `ask_mind_math500de` 与 `ask_mind_medqade` 中复制），统一存放在 `data/fata/<task>/test.jsonl`。
- **交互流程**：
  1. 被测模型收到官方 FATA prompt，并在其中看到降质题面：
     ```
     User request: <degraded_question>.
     To better assist me, before offering advice, please adopt the perspective of an expert in the relevant field
     and ask questions to help you identify any missing key information.
     Please ensure the problem is structured clearly and expressed concisely, with example guidance,
     just like how experts ask users questions during consultations to gather key information before providing solutions.

     After I provide additional information, please then offer a more personalized and practical solution as an expert in that field.
     If all key information has already been provided, please directly give the solution.
     Note: Maintain a positive attitude, and do not request phone numbers, ID numbers, or other sensitive data.
     ```
     模型可以在第一轮提出一次澄清问题，或直接作答。
  2. 每轮回复都会交给裁判（Judge）模型。裁判拿到 `ori_question`、`degraded_info`、`required_points` 与 `expected_answer`，判断当前回复是否在补充信息：
     - 若确实在提问，裁判会按照原题事实写出用户补充信息，并把这些内容作为第二轮输入传给被测模型；
     - 若已经开始作答，则直接判定正误。
  3. 最多只允许两轮。第二轮若仍然追问，会被视为违反“只问一次就给答案”的规则而判错。
- **判分机制**：
  - 裁判输出 JSON，包含 `needs_more_info`、`user_reply`（可选）、`is_correct` 与 `reason`。当 `needs_more_info=false` 时，会基于 `expected_answer` 判定最终是否正确。
  - 输出文件沿用 AskBench 规格：`askbench_detailed_results.json` 记录完整对话轨迹与裁判结论，`summary_results.json`/`results.txt` 则统计准确率、是否触发澄清、第二轮仍提问的失败案例以及裁判解析失败数。

## 扩展指南

1. **新数据集**：准备 `data/<group>/<task>/test.jsonl`，编写或复用数据加载器，并在 `LOADER_MAP` 登记。
2. **新评估逻辑**：继承 `BaseEvaluator`，实现格式化/抽取/验证逻辑，将类注册到 `EVALUATOR_MAP`。
3. **新任务配置**：创建 `config/common/<task>.ini`，指定 `[evalset] evalsetname`、模型、生成与路径参数。
4. **多模型评测**：可在 `run.sh` 中批量覆盖 `--url`、`--tasks`，或编写外层调度脚本循环调用。

## 常用参数提示

- `generateconfig.n_attempts`：同一题目多次采样，评估平均准确率与 `pass@1`。
- `generateconfig.max_concurrent`：并发请求上限，避免压垮模型服务。AskBench / QuestBench 多轮评测现已严格按照该值控制被测模型的异步调用。
- `[evaluatorconfig].max_concurrent`：裁判模型并发上限。Judge 负责仲裁与模拟用户两个角色，同样会遵循该限制以免向外部 GPT 服务一次性发出太多请求。
- `evaluatorconfig.max_turns`：AskBench 中最多对话轮数，默认 5。
- `model.extra_prompt`、`model.system_prompt`：通过 `BaseAPIModel.format_messages` 自动拼接进用户或系统对话。

## 注意事项

- `tasks_config_path` 中若包含 `EvalScope`，`main.py` 将尝试调用 `scripts.run_evalscope_origin`。该脚本未随仓库一起提交，使用前需补齐。
- 仓库默认依赖列表见 `setup.py`，其中 `sympy`、`latex2sympy2` 等被用于数学表达式归一化。
- AskBench 评测依赖裁判模型的稳定性与一致性，建议在 `evaluatorconfig` 中将温度设为 0 并控制并发。

至此，评测框架的核心逻辑与扩展点已经覆盖。后续查看 README 时，可据此快速定位需要修改的模块或配置。

## AskLone 评测逻辑

- **通过率估计**：`AskLoneEvaluator` 会使用原题 (`ori_question`) 触发被测模型独立作答 16 次，通过裁判模型逐次判定正误，得到题目的个人通过率。
- **最终作答**：随后以提示 “Do your best to solve this problem... If you feel you are unable...” 再次向模型提问，引导其要么给出答案，要么明确表示不会。
- **计分方式**：最终响应由裁判模型判定为 `correct` / `incorrect` / `refusal`。判定正确记 1 分，错误记 0 分，若模型清晰承认不会，则记 `1 - 通过率` 分，鼓励对难题坦诚放弃。
- **文件布局**：相关脚本为 `scripts/run_ask_lone.py`，详细日志写入 `ask_lone_detailed_results.json`；生成的 JSON 仅包含 `ori_question` 与 `expected_answer`。

## 内置任务与评测逻辑速览

| 任务标识 | 数据目录 | 评估器 | 交互模式 | 核心逻辑 |
| --- | --- | --- | --- | --- |
| `math500` | `data/common/math500` | `Math500Evaluator` | 单轮 + 裁判 | 单轮作答，由 Judge 根据题干与标准答案判断是否正确，并记录跳过/失败原因。 |
| `math500_de` | `data/degrade/math500_de` | `Math500DeEvaluator` | 单轮 | 降质版数学题，Prompt 使用 `degraded_question`，答案提取逻辑同 Math500。 |
| `medqa` | `data/common/medqa` | `MedQAEvaluator` | 单轮 + 裁判 | 单轮多选题，由 Judge 读取题干与参考答案 JSON 判定正误。 |
| `medqa_de` | `data/degrade/medqa_de` | `MedQADeEvaluator` | 单轮 | 降质版 MedQA，题干为 `degraded_question`，答案仍是选项匹配。 |
| `gpqa` | `data/common/gpqa` | `GpqaEvaluator` | 单轮 + 裁判 | 通识问答集，同样将模型答案交由 Judge 判定。 |
| `ask_overconfidence` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | 多轮裁判 | AskBench 子任务，存在裁判模型；被测模型需通过提问补全信息，裁判负责判定终止与正误。 |
| `ask_overconfidence_math500` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | 多轮裁判 | Math500 子集的 overconfidence 版本，模型需识别并修正误导点后再作答。 |
| `ask_overconfidence_medqa` | `data/ask_bench/ask_overconfidence` | `AskEvaluator` | 多轮裁判 | MedQA 子集的 overconfidence 版本，字段与 ask_overconfidence_math500 相同。 |
| `ask_mind` | `data/ask_bench/ask_mind` | `AskEvaluator` | 多轮裁判 | AskBench 主任务，逻辑同上，题干为 `degraded_question`，真题存放于 `ori_question`，默认数据为四个 ask_mind 子集各 100 题的 400 条混合集。 |
| `ask_lone` | `data/ask_bench/ask_lone` | `AskLoneEvaluator` | 单轮 + 裁判 | 先估 16 次通过率，再根据最终作答/认输计算得分。 |
| `ask_lone_bbhde` | `data/ask_bench/ask_lone` | `AskLoneEvaluator` | 单轮 + 裁判 | AskLone 逻辑 + BBH 原题（`ori_question`），题目来源 `ask_mind_bbhde`。 |
| `ask_lone_gpqade` | `data/ask_bench/ask_lone` | `AskLoneEvaluator` | 单轮 + 裁判 | AskLone 逻辑 + GPQA 原题（`ori_question`），题目来源 `ask_mind_gpqade`。 |
| `ask_lone_math500de` | `data/ask_bench/ask_lone` | `AskLoneEvaluator` | 单轮 + 裁判 | AskLone 逻辑 + Math500 原题（`ori_question`），题目来源 `ask_mind_math500de`。 |
| `ask_lone_medqade` | `data/ask_bench/ask_lone` | `AskLoneEvaluator` | 单轮 + 裁判 | AskLone 逻辑 + MedQA 原题（`ori_question`），题目来源 `ask_mind_medqade`。 |
| `ask_mind_math500de` | `data/ask_bench/ask_mind` | `AskEvaluator` | 多轮裁判 | AskBench 与 Math500 降质结合，测试模型是否能主动提问补全信息。 |
| `ask_mind_medqade` | `data/ask_bench/ask_mind` | `AskEvaluator` | 多轮裁判 | AskBench + MedQA 降质组合，裁判流程同上。 |
| `ask_mind_gpqade` | `data/ask_bench/ask_mind` | `AskEvaluator` | 多轮裁判 | AskBench + GPQA 降质组合。 |
| `ask_mind_bbhde` | `data/ask_bench/ask_mind` | `AskEvaluator` | 多轮裁判 | AskBench + BBH 降质组合。 |
| `fata_math500` | `data/fata/fata_math500` | `AskEvaluator` | 双轮（澄清+最终回答） | 官方 prompt 先引导模型提问一次，Judge 判断是否需要补充信息并模拟用户回复，再由同一 Judge 判定最终答案是否正确。 |
| `fata_medqa` | `data/fata/fata_medqa` | `AskEvaluator` | 双轮（澄清+最终回答） | 流程与 `fata_math500` 相同，只是题源换为 MedQA。 |
| `quest_bench` | `data/ask_bench/quest_bench` | `AskEvaluator` | 多轮裁判 | QuestBench 任务，沿用 AskEvaluator + `required_points` 清单，Judge 会按 ask_mind 体系判定合规性。 |
| `in3_interaction` | `data/ask_bench/in3_interaction` | `In3InteractionEvaluator` | 多轮裁判 | IN3 Interaction 新基准：`task` 视为原始问题，`missing_details` 会被拆成 `required_points`，裁判只衡量澄清问答合规性，不再计算 Accuracy。 |

> 注：所有 `ask_*`、`quest_bench` 以及 `math500` / `medqa` / `gpqa` 均依赖 `[evaluatorconfig]` 定义的裁判模型；其余传统任务则仍使用正则或数值对比。新增任务时可对照该表快速定位所需的数据结构与评估器。
