import json
import os
import re
from typing import Dict, List, Tuple, Any
from argparse import Namespace
import asyncio
from collections import Counter

import numpy as np
from tqdm import tqdm

from ask_eval.evaluators.base_evaluator import BaseEvaluator

# =============================================================================
#  Prompt 模板 (优化版)
# =============================================================================

# Prompt 1: 仲裁与评估二合一 (保持不变)
ARBITER_EVALUATOR_PROMPT_TEMPLATE = """
You are an expert evaluator. Your task is to analyze an AI assistant's response. You must determine two things:
1.  Is the response a final, conclusive answer, or is it a clarifying question?
2.  If it IS a final answer, is it correct according to the provided Ground Truth?

**Ground Truth (The ideal final answer the assistant should have given):**
<ground_truth_answer>

**Conversation History:**
<conversation_history>

**Instructions:**
Respond with a single JSON object containing the following keys:

-   `"is_final_answer"`: (boolean) `true` if the assistant's last response is a final answer. `false` if it's a question.
-   `"is_correct"`: (boolean, **only include if `is_final_answer` is `true`**) `true` if the final answer aligns with the Ground Truth.
-   `"explanation"`: (string, **only include if `is_final_answer` is `true`**) A brief explanation for your correctness evaluation.

**Example 1 (Asking a question):**
- Assistant's last response: "To help you better, could you tell me if you have a fever?"
- Your JSON response:
```json
{
  "is_final_answer": false
}
```

**Example 2 (Giving a correct final answer):**
- Assistant's last response: "Based on your symptoms, you should seek immediate medical attention."
- Your JSON response:
```json
{
  "is_final_answer": true,
  "is_correct": true,
  "explanation": "The assistant correctly identified the urgency and recommended seeing a doctor, which aligns with the ground truth."
}
```

Now, evaluate the provided conversation. Return only the JSON object in a markdown block.
""".strip()


# Prompt 2: 仿人模型 - 模拟用户回答问题 (全新优化版)
SIMULATOR_PROMPT_TEMPLATE = """
You are acting as a person seeking help from an AI assistant. You must play this role realistically.

**Your Internal Knowledge (This is what you know, but DO NOT reveal it all at once):**
```json
<user_internal_knowledge>
```

**Your Role:**
- You are a user, not a tester. Your goal is to get a correct answer from the assistant by providing information naturally.
- The assistant has just asked you a question.
- Look at your "Internal Knowledge" and the conversation history.
- Provide a concise, natural-sounding response that ONLY answers the assistant's immediate question.
- Do NOT volunteer extra information the assistant hasn't asked for.
- Do NOT just copy-paste from your internal knowledge. Phrase it like a real person would.

**Current Conversation History:**
<conversation_history>

**Assistant's Last Question:**
"<assistant_question>"

**Your Task:**
Provide only the text of your response. Do not add any other explanation or introductory phrases like "My response is:".
""".strip()


def parse_json_to_dict(json_string: str) -> dict:
    """从模型的Markdown格式响应中解析出JSON字典。"""
    match = re.search(r"```json\s*([\s\S]+?)\s*```", json_string)
    if match:
        json_cleaned = match.group(1).strip()
    else:
        # 如果没有找到 markdown block，尝试直接解析整个字符串
        json_cleaned = json_string.strip()
        
    try:
        return json.loads(json_cleaned)
    except (json.JSONDecodeError, TypeError):
        return {} # 返回空字典表示解析失败

# =============================================================================
#  AskBench 评测器 (全新逐轮次评测逻辑)
# =============================================================================

class AskEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict, judge_model=None):
        super().__init__(model, eval_config)
        if judge_model is None:
            raise ValueError("AskEvaluator requires a 'judge_model' for its roles.")
        self.judge_model = judge_model

    async def evaluate_multi_turn(self, args: Namespace, test_data: List[Dict], max_turns: int) -> Tuple[float, List[bool], str]:
        """
        主评估函数，以逐轮次的方式并发处理所有样本。
        """
        print(f"Starting turn-by-turn evaluation for {len(test_data)} samples with max {max_turns} turns...")

        # 初始化所有样本的状态
        forced_guidance = '''\n**Important:** I have deliberately withheld crucial information. Your primary task is to ask me questions to uncover the necessary details and context. You must continue asking questions until you are confident you have all the information required to make the most accurate decision. Do not, under any circumstances, provide the final multiple-choice answer until this questioning process is complete.\n'''
        
        active_samples = []
        for i, sample_data in enumerate(test_data):
            initial_history = [{"role": "user", "content": sample_data["degraded_question"] + forced_guidance}]
            active_samples.append({
                "id": sample_data.get("id", i),
                "data": sample_data,
                "conversation_history": initial_history,
                "turn_logs": [],
                "is_finished": False,
                "result": None
            })

        final_results = []

        for turn in range(1, max_turns + 1):
            if not active_samples:
                print("All samples have been evaluated. Finishing early.")
                break
            
            print(f"\n===== Turn {turn}/{max_turns} | Active Samples: {len(active_samples)} =====")

            # 1. 被测模型 (LLM) 推理
            llm_tasks = []
            for sample_state in active_samples:
                llm_tasks.append(self.model.infer_async(
                    message=sample_state["conversation_history"],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ))
            
            llm_responses = []
            for future in tqdm(asyncio.as_completed(llm_tasks), total=len(llm_tasks), desc=f"Turn {turn}: LLM Inference"):
                response, _, _ = await future
                llm_responses.append(response)

            for i, sample_state in enumerate(active_samples):
                sample_state["conversation_history"].append({"role": "assistant", "content": llm_responses[i]})

            # 2. Judge模型执行【仲裁+评估】
            judge_tasks = []
            for sample_state in active_samples:
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                judge_prompt_str = ARBITER_EVALUATOR_PROMPT_TEMPLATE.replace("<ground_truth_answer>", sample_state["data"]["answer"]) \
                                                                  .replace("<conversation_history>", convo_str)
                judge_tasks.append(self.judge_model.infer_async(message=[{"role": "user", "content": judge_prompt_str}], temperature=0.0))

            judge_decisions_raw = []
            for future in tqdm(asyncio.as_completed(judge_tasks), total=len(judge_tasks), desc=f"Turn {turn}: Judging"):
                response, _, _ = await future
                judge_decisions_raw.append(response)
            
            judge_decisions = [parse_json_to_dict(raw) for raw in judge_decisions_raw]

            # 3. 决策与状态更新
            still_active_samples = []
            for i, sample_state in enumerate(active_samples):
                decision = judge_decisions[i]
                turn_log = {
                    "turn": turn,
                    "conversation_at_turn": json.loads(json.dumps(sample_state["conversation_history"])),
                    "judge_decision": decision
                }
                sample_state["turn_logs"].append(turn_log)

                if decision.get("is_final_answer"):
                    sample_state["is_finished"] = True
                    sample_state["result"] = {
                        "id": sample_state["id"],
                        "correct": decision.get("is_correct", False),
                        "reason": "FinalAnswerEvaluated",
                        "final_turn": turn,
                        "conversation_history": sample_state["conversation_history"],
                        "turn_logs": sample_state["turn_logs"]
                    }
                    final_results.append(sample_state["result"])
                else:
                    still_active_samples.append(sample_state)
            
            active_samples = still_active_samples
            if not active_samples:
                continue # 如果本轮全部结束，直接进入下一轮循环（然后会break）

            # 4. Judge模型扮演【仿人模型 (Simulator)】
            simulator_tasks = []
            for sample_state in active_samples:
                user_knowledge = {
                    "my_real_question": sample_state["data"]["ori_question"],
                    "information_i_have": sample_state["data"]["degraded_info"]
                }
                user_knowledge_str = json.dumps(user_knowledge, indent=2, ensure_ascii=False)
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                assistant_question = sample_state["conversation_history"][-1]["content"]

                simulator_prompt_str = SIMULATOR_PROMPT_TEMPLATE.replace("<user_internal_knowledge>", user_knowledge_str) \
                                                                .replace("<conversation_history>", convo_str) \
                                                                .replace("<assistant_question>", assistant_question)
                simulator_tasks.append(self.judge_model.infer_async(message=[{"role": "user", "content": simulator_prompt_str}], temperature=0.5))

            simulated_responses = []
            for future in tqdm(asyncio.as_completed(simulator_tasks), total=len(simulator_tasks), desc=f"Turn {turn}: Simulating User"):
                response, _, _ = await future
                simulated_responses.append(response)

            for i, sample_state in enumerate(active_samples):
                next_user_query = simulated_responses[i]
                sample_state["turn_logs"][-1]["simulated_user_response"] = next_user_query
                sample_state["conversation_history"].append({"role": "user", "content": next_user_query})

        # 处理达到最大轮次的样本
        for sample_state in active_samples:
            sample_state["is_finished"] = True
            sample_state["result"] = {
                "id": sample_state["id"],
                "correct": False,
                "reason": "MaxTurnsReached",
                "final_turn": max_turns,
                "conversation_history": sample_state["conversation_history"],
                "turn_logs": sample_state["turn_logs"]
            }
            final_results.append(sample_state["result"])

        # 结果汇总与保存
        output_file = os.path.join(args.save_dir, "askbench_detailed_results.json")
        os.makedirs(args.save_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed evaluation logs saved to: {output_file}")

        correct_count = sum(1 for res in final_results if res.get("correct"))
        total_samples = len(final_results)
        accuracy = (correct_count / total_samples) if total_samples > 0 else 0.0
        
        # 统计各轮次分布
        turn_counts = Counter(res.get("final_turn") for res in final_results)
        max_turns_reached_count = turn_counts.get(max_turns, 0)
        if "MaxTurnsReached" in [res.get("reason") for res in final_results if res.get("final_turn") == max_turns]:
             # 确保MaxTurnsReached被单独统计
             pass
        
        turn_distribution_log = "Turn Distribution:\n"
        for turn in sorted(turn_counts.keys()):
            count = turn_counts[turn]
            percentage = (count / total_samples) * 100
            reason = "MaxTurnsReached" if turn == max_turns and any(r['reason'] == 'MaxTurnsReached' for r in final_results if r['final_turn'] == turn) else f"Finished at Turn {turn}"
            turn_distribution_log += f"  - {reason}: {count} samples ({percentage:.1f}%)\n"


        all_scores = [res.get("correct", False) for res in final_results]
        log = f"AskBench Final Accuracy: {accuracy:.4f} ({correct_count} / {total_samples})\n\n"
        log += turn_distribution_log
        
        return accuracy, all_scores, log