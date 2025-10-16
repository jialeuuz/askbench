import json
import os
import re
from typing import Dict, List, Tuple, Any
from argparse import Namespace
import asyncio
from collections import Counter

from tqdm import tqdm

from ask_eval.evaluators.base_evaluator import BaseEvaluator

# --- 在这里配置首轮对话的引导模式 ---
# 可选值: 'none', 'weak', 'strong'
# 'none': 不添加任何引导
# 'weak': 添加弱引导，暗示模型可以提问
# 'strong': 添加强引导，明确要求模型必须先提问
GUIDANCE_MODE = 'none'

# Weak guidance: Politely suggests that the model can ask for more information.
WEAK_GUIDANCE_PROMPT = "If you need more information to provide a better and more complete answer, please feel free to ask me any questions."
# Strong guidance: A direct instruction that the model MUST ask questions before answering.
STRONG_GUIDANCE_PROMPT = "Important: My question is likely incomplete. You must ask clarifying questions to gather all the necessary information before you provide a final answer."


ARBITER_EVALUATOR_PROMPT_TEMPLATE = """
You are an expert evaluator. Your task is to analyze an AI assistant's response. Your primary goal is to distinguish between a **final answer** that attempts to resolve the user's request, and a **clarifying question** that seeks more information.

Based on this, you must determine two things:
1.  Is the response a final, conclusive answer, or is it a clarifying question?
2.  If it IS a final answer, is it correct according to the provided Ground Truth?

**Key Definitions:**
-   **Clarifying Question (`"is_final_answer": false`)**: The assistant's response is a question aimed at gathering more information, details, or clarification from the user. It is a step in the conversation, not the conclusion.
-   **Final Answer (`"is_final_answer": true`)**: The assistant provides a conclusive statement, answer, or action that attempts to resolve the user's request. It does **not** ask for more information to proceed. This answer can be correct or incorrect.

**Ground Truth (The ideal final answer the assistant should have given):**
<ground_truth_answer>

**Conversation History:**
<conversation_history>

**Instructions:**
Respond with a single JSON object containing the following keys:

-   `"is_final_answer"`: (boolean) `true` if the assistant's last response is a final answer. `false` if it's a clarifying question.
-   `"is_correct"`: (boolean, **only include if `is_final_answer` is `true`**) `true` if the final answer aligns with the Ground Truth.
-   `"explanation"`: (string, **only include if `is_final_answer` is `true`**) A brief explanation for your correctness evaluation.

**Example 1 (Asking a question):**
- Assistant's last response: "To help you better, could you tell me which city you're in?"
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

**Example 3 (Giving an incorrect final answer):**
- Ground Truth: "The capital of France is Paris."
- Assistant's last response: "The capital of France is Berlin."
- Your JSON response:
```json
{
  "is_final_answer": true,
  "is_correct": false,
  "explanation": "The assistant provided a definitive but incorrect answer. The capital of France is Paris, not Berlin."
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

# Prompt 3: 强制最终回答 
FORCE_FINAL_ANSWER_PROMPT = """
\n**This is the final turn.** Based on the information you have gathered so far, you MUST provide a conclusive, final answer. Do not ask any more questions.
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

class AskEvaluator(BaseEvaluator):
    def __init__(self, model, eval_config: Dict, judge_model=None):
        super().__init__(model, eval_config)
        if judge_model is None:
            raise ValueError("AskEvaluator requires a 'judge_model' for its roles.")
        self.judge_model = judge_model

    async def evaluate_multi_turn(self, args: Namespace, test_data: List[Dict], max_turns: int) -> Tuple[float, List[bool], str]:
        print(f"Starting turn-by-turn evaluation for {len(test_data)} samples with max {max_turns} turns...")

        forced_guidance = ''
        
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

        # for turn in range(1, max_turns + 1):
        #     if not active_samples:
        #         print("All samples have been evaluated. Finishing early.")
        #         break
            
        #     print(f"\n===== Turn {turn}/{max_turns} | Active Samples: {len(active_samples)} =====")

        #     # --- 修正部分：带进度条的并发执行 ---
        #     async def run_tasks_with_progress(tasks_coroutines, description):
        #         tasks = []
        #         with tqdm(total=len(tasks_coroutines), desc=description) as pbar:
        #             def update_pbar(future):
        #                 pbar.update(1)
                    
        #             for coro in tasks_coroutines:
        #                 task = asyncio.create_task(coro)
        #                 task.add_done_callback(update_pbar)
        #                 tasks.append(task)
                    
        #             results = await asyncio.gather(*tasks)
        #         return results

        #     # 1. 被测模型 (LLM) 推理
        #     llm_coroutines = []
        #     for sample_state in active_samples:
        #         messages_for_llm = list(sample_state["conversation_history"])
        #         if turn == max_turns and messages_for_llm:
        #             last_message = messages_for_llm[-1].copy()
        #             last_message["content"] += "\n" + FORCE_FINAL_ANSWER_PROMPT
        #             messages_for_llm[-1] = last_message

        #         llm_coroutines.append(self.model.infer_async(
        #             message=messages_for_llm,
        #             max_tokens=self.max_tokens,
        #             temperature=self.temperature
        #         ))
        for turn in range(1, max_turns + 1):
            if not active_samples:
                print("All samples have been evaluated. Finishing early.")
                break
            
            print(f"\n===== Turn {turn}/{max_turns} | Active Samples: {len(active_samples)} =====")

            # --- 修正部分：带进度条的并发执行 ---
            async def run_tasks_with_progress(tasks_coroutines, description):
                tasks = []
                with tqdm(total=len(tasks_coroutines), desc=description) as pbar:
                    def update_pbar(future):
                        pbar.update(1)
                    
                    for coro in tasks_coroutines:
                        task = asyncio.create_task(coro)
                        task.add_done_callback(update_pbar)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                return results

            # 1. 被测模型 (LLM) 推理
            llm_coroutines = []
            for sample_state in active_samples:
                messages_for_llm = list(sample_state["conversation_history"])

                if turn == 1 and GUIDANCE_MODE != 'none' and messages_for_llm:
                    prompt_to_add = ""
                    if GUIDANCE_MODE == 'weak':
                        prompt_to_add = WEAK_GUIDANCE_PROMPT
                    elif GUIDANCE_MODE == 'strong':
                        prompt_to_add = STRONG_GUIDANCE_PROMPT

                    if prompt_to_add:
                        # 同样使用 .copy() 来避免修改原始历史记录
                        last_message = messages_for_llm[-1].copy()
                        # 使用两个换行符让提示更清晰
                        last_message["content"] += f"\n\n{prompt_to_add}"
                        messages_for_llm[-1] = last_message

                if turn == max_turns and messages_for_llm:
                    last_message = messages_for_llm[-1].copy()
                    last_message["content"] += "\n" + FORCE_FINAL_ANSWER_PROMPT
                    messages_for_llm[-1] = last_message

                llm_coroutines.append(self.model.infer_async(
                    message=messages_for_llm,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                ))
            
            llm_responses_raw = await run_tasks_with_progress(llm_coroutines, f"Turn {turn}: LLM Inference")
            llm_responses = [res[0] for res in llm_responses_raw]

            for i, sample_state in enumerate(active_samples):
                sample_state["conversation_history"].append({"role": "assistant", "content": llm_responses[i]})

            # 2. Judge模型执行【仲裁+评估】
            judge_coroutines = []
            for sample_state in active_samples:
                convo_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in sample_state["conversation_history"]])
                judge_prompt_str = ARBITER_EVALUATOR_PROMPT_TEMPLATE.replace("<ground_truth_answer>", sample_state["data"]["expected_answer"]) \
                                                                  .replace("<conversation_history>", convo_str)
                judge_coroutines.append(self.judge_model.infer_async(message=[{"role": "user", "content": judge_prompt_str}], temperature=0.0))

            judge_decisions_raw_tuples = await run_tasks_with_progress(judge_coroutines, f"Turn {turn}: Judging")
            judge_decisions_raw = [res[0] for res in judge_decisions_raw_tuples]
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
                        "ground_truth_answer": sample_state["data"]["expected_answer"],
                        "turn_logs": sample_state["turn_logs"]
                    }
                    final_results.append(sample_state["result"])
                else:
                    if turn < max_turns:
                        still_active_samples.append(sample_state)
                    else:
                        sample_state["is_finished"] = True
                        sample_state["result"] = {
                            "id": sample_state["id"],
                            "correct": False,
                            "reason": "FailedToAnswerOnLastTurn",
                            "final_turn": max_turns,
                            "conversation_history": sample_state["conversation_history"],
                            "ground_truth_answer": sample_state["data"]["expected_answer"],
                            "turn_logs": sample_state["turn_logs"]
                        }
                        final_results.append(sample_state["result"])

            active_samples = still_active_samples
            if not active_samples:
                continue

            # 4. Judge模型扮演【仿人模型 (Simulator)】
            simulator_coroutines = []
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
                simulator_coroutines.append(self.judge_model.infer_async(message=[{"role": "user", "content": simulator_prompt_str}], temperature=0.5))

            simulated_responses_raw = await run_tasks_with_progress(simulator_coroutines, f"Turn {turn}: Simulating User")
            simulated_responses = [res[0] for res in simulated_responses_raw]

            for i, sample_state in enumerate(active_samples):
                next_user_query = simulated_responses[i]
                sample_state["turn_logs"][-1]["simulated_user_response"] = next_user_query
                sample_state["conversation_history"].append({"role": "user", "content": next_user_query})

        for sample_state in active_samples:
            if not sample_state["is_finished"]:
                sample_state["is_finished"] = True
                sample_state["result"] = {
                    "id": sample_state["id"],
                    "correct": False,
                    "reason": "MaxTurnsReached",
                    "final_turn": max_turns,
                    "conversation_history": sample_state["conversation_history"],
                    "ground_truth_answer": sample_state["data"]["expected_answer"],
                    "turn_logs": sample_state["turn_logs"]
                }
                final_results.append(sample_state["result"])

        output_file = os.path.join(args.save_dir, "askbench_detailed_results.json")
        os.makedirs(args.save_dir, exist_ok=True)
        final_results.sort(key=lambda x: x['id'])
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed evaluation logs saved to: {output_file}")

        correct_count = sum(1 for res in final_results if res.get("correct"))
        total_samples = len(final_results)
        accuracy = (correct_count / total_samples) if total_samples > 0 else 0.0
        
        reason_counts = Counter(res.get("reason") for res in final_results)
        
        turn_distribution_log = "Evaluation Outcome Distribution:\n"
        for reason, count in reason_counts.most_common():
            percentage = (count / total_samples) * 100
            turn_distribution_log += f"  - {reason}: {count} samples ({percentage:.1f}%)\n"

        all_scores = [res.get("correct", False) for res in final_results]
        log = f"AskBench Final Accuracy: {accuracy:.4f} ({correct_count} / {total_samples})\n\n"
        log += turn_distribution_log
        
        return accuracy, all_scores, log