import json
import re
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable
from post_api import CustomAPI

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _parse_llm_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match: raise ValueError("在回复中未找到JSON对象")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if 'degraded_question' in parsed_json and 'degraded_info' in parsed_json:
            return parsed_json, None
        raise KeyError("JSON对象中缺少 'degraded_question' 或 'degraded_info' 键")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e

def _parse_judge_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match: raise ValueError("在 Judge 回复中未找到JSON对象")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if 'is_correct' in parsed_json and 'reason' in parsed_json:
            valid_reasons = [None, 'insufficient_asking', 'reasoning_error']
            if parsed_json.get('reason') not in valid_reasons:
                raise ValueError(f"无效的 reason 值: '{parsed_json.get('reason')}'. 只允许 {valid_reasons}")
            return parsed_json, None
        raise KeyError("JSON对象中缺少 'is_correct' 或 'reason' 键")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e

async def _run_batch_step_with_retry(
    api_client: CustomAPI,
    items: List[Dict[str, Any]],
    step_name: str,
    prompt_template: str,
    prompt_format_keys: List[str],
    parser_func: Callable[[str], Tuple[Optional[Any], Optional[Exception]]],
    api_params: Dict[str, Any],
    max_retries: int = 2,
    retry_delay: float = 3.0
) -> Tuple[List[Tuple[Dict, Any]], List[Dict]]:
    if not items:
        return [], []

    print(f"  - {step_name}: 开始处理 {len(items)} 项...")
    item_indices_to_process = list(range(len(items)))
    successful_results = {}

    for attempt in range(max_retries + 1):
        if not item_indices_to_process: break
        if attempt > 0:
            print(f"    - {step_name}: 第 {attempt}/{max_retries} 轮重试，剩余 {len(item_indices_to_process)} 项...")
            await asyncio.sleep(retry_delay)
        
        prompts_for_this_batch = []
        valid_indices_for_batch = []
        for index in item_indices_to_process:
            try:
                format_args = {key: items[index][key] for key in prompt_format_keys}
                if 'conversation_history' in format_args:
                    format_args['conversation_history'] = json.dumps(format_args['conversation_history'], ensure_ascii=False, indent=2)
                prompts_for_this_batch.append(prompt_template.format(**format_args))
                valid_indices_for_batch.append(index)
            except KeyError as e:
                successful_results[index] = (items[index], None)
        
        item_indices_to_process = [idx for idx in item_indices_to_process if idx in valid_indices_for_batch]
        if not prompts_for_this_batch: continue

        try:
            responses, _, _ = await api_client.infer_batch_async(messages=prompts_for_this_batch, **api_params)
        except Exception as e:
            continue

        failed_indices_for_next_round = []
        for i, raw_response in enumerate(responses):
            original_item_index = item_indices_to_process[i]
            parsed_result, error = parser_func(raw_response)
            if error is None:
                successful_results[original_item_index] = (items[original_item_index], parsed_result)
            else:
                failed_indices_for_next_round.append(original_item_index)
        item_indices_to_process = failed_indices_for_next_round

    final_successful = []
    final_failed = []
    for i, item in enumerate(items):
        if i in successful_results and successful_results[i][1] is not None:
            final_successful.append(successful_results[i])
        else:
            # ### 优化点 1 改动 ###: 确保失败的item是原始的，不包含中间字段
            # 我们需要从原始的 `items` 列表中获取，并移除可能被添加的临时字段
            clean_item = item.copy()
            clean_item.pop('degraded_question', None)
            clean_item.pop('degraded_info', None)
            clean_item.pop('conversation_history', None)
            clean_item.pop('temp_answer', None)
            clean_item.pop('generated_answer', None)
            clean_item.pop('solution_section', None) # 确保这个临时字段也被移除
            final_failed.append(clean_item)
            
    if final_failed:
        print(f"  - {step_name}: 完成。成功: {len(final_successful)} 项, 失败并丢弃: {len(final_failed)} 项。")
    return final_successful, final_failed

# ---------------------------------------------------------------------------
# 策略 1 
# ---------------------------------------------------------------------------
async def generate_degraded_question_and_info(
    api_client: CustomAPI, data: List[Dict[str, Any]], templates: Dict[str, str], max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (成功项列表, 失败项列表)"""
    step_name = "步骤 1: 生成劣化问题和信息"
    prompt_template = templates['template_generate_degraded_question_and_info']
    api_params = {'max_tokens': 16000, 'temperature': 0.7}
    successful_results, failed_items = await _run_batch_step_with_retry(
        api_client, data, step_name, prompt_template,
        ['ori_question', 'expected_answer'],
        _parse_llm_json_response, api_params, max_retries
    )
    processed_successful_items = []
    for item, parsed_json in successful_results:
        item['degraded_question'] = parsed_json['degraded_question']
        item['degraded_info'] = parsed_json['degraded_info']
        processed_successful_items.append(item)
    return processed_successful_items, failed_items

# ---------------------------------------------------------------------------
# 策略 2: 生成完整的多轮对话训练数据 (V4 - 带失败数据保存和条件solution)
# ---------------------------------------------------------------------------
async def generate_multi_turn_training_data(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
    max_ask_loops: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: # ### 优化点 1 改动 ###: 修改返回类型
    """
    策略：构建一个完整的多轮对话，并根据优化要求进行调整。
    V4: 
    - 返回 (成功列表, 失败列表) 以便保存失败数据。
    - 在强制修正答案时，如果原始数据包含 'solution' 字段，则将其加入 prompt。
    """
    print("\n============================================================")
    print("开始执行策略: generate_multi_turn_training_data (V4)")
    print(f"初始数据量: {len(data)} 项")
    print("============================================================\n")

    items_to_process, failed_step1 = await generate_degraded_question_and_info(api_client, data, templates)
    total_discarded_items = failed_step1
    
    if not items_to_process:
        print("\n所有数据在步骤1中失败，流程终止。")
        return [], total_discarded_items # ### 优化点 1 改动 ###

    for item in items_to_process:
        item['conversation_history'] = [{"role": "user", "content": item['degraded_question']}]
    
    completed_items = []
    current_loop = 0
    
    while items_to_process and current_loop < max_ask_loops:
        # ... (循环内部的前面步骤无变化) ...
        round_start_count = len(items_to_process)
        print(f"\n{'='*20} 对话轮次 {current_loop + 1} (处理 {round_start_count} 项) {'='*20}")

        # --- 步骤 2: 批量生成澄清问题 ---
        tpl_name = 'template_assistant_ask_first_question' if current_loop == 0 else 'template_assistant_ask_follow_up_question'
        ask_results, failed_ask = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 2: 生成澄清问题", templates[tpl_name],
            ['degraded_question', 'degraded_info', 'conversation_history'],
            lambda r: (r, None), {'max_tokens': 2048, 'temperature': 0.5}
        )
        total_discarded_items.extend(failed_ask)
        if not ask_results: break
        items_to_process = [item for item, _ in ask_results]
        for item, response in ask_results:
            item['conversation_history'].append({"role": "assistant", "content": response})

        # --- 步骤 3: 批量模拟用户回复 ---
        reply_results, failed_reply = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 3: 模拟用户回复", templates['template_simulate_user_reply'],
            ['conversation_history', 'degraded_info'],
            lambda r: (r, None), {'max_tokens': 2048, 'temperature': 0.5}
        )
        total_discarded_items.extend(failed_reply)
        if not reply_results: break
        items_to_process = [item for item, _ in reply_results]
        for item, response in reply_results:
            item['conversation_history'].append({"role": "user", "content": response})

        # --- 步骤 4: 批量生成最终答案 ---
        answer_results, failed_answer = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 4: 生成最终答案", templates['template_generate_final_answer'],
            ['conversation_history'],
            lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.7}
        )
        total_discarded_items.extend(failed_answer)
        if not answer_results: break
        items_to_process = [item for item, _ in answer_results]
        for item, response in answer_results:
            item['temp_answer'] = response
            item['conversation_history'].append({"role": "assistant", "content": response})

        # --- 步骤 5: 批量判断答案 ---
        for item in items_to_process:
            item['generated_answer'] = item['temp_answer']
        
        judge_results, failed_judge = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 5: 判断答案", templates['template_judge_answer'],
            ['conversation_history', 'generated_answer', 'expected_answer', 'degraded_info'],
            _parse_judge_json_response, {'max_tokens': 1024, 'temperature': 0.0}
        )
        total_discarded_items.extend(failed_judge)
        if not judge_results: break
        
        # --- 分流逻辑 (无变化) ---
        next_round_items = []
        items_to_force_correct = []
        current_round_completed_count = 0

        for item, judgement in judge_results:
            if judgement['is_correct']:
                completed_items.append(item)
                current_round_completed_count += 1
            elif judgement['reason'] == 'insufficient_asking':
                item['conversation_history'].pop()
                next_round_items.append(item)
            else:
                item['conversation_history'].pop()
                items_to_force_correct.append(item)
        
        print(f"--- 本轮分流小结 ---")
        print(f"  - {current_round_completed_count} 项在本轮处理完成。")
        print(f"  - {len(next_round_items)} 项因追问不充分，将进入下一轮。")
        print(f"  - {len(items_to_force_correct)} 项因推理错误，将进行强制修正。")

        # --- 步骤 6: 强制修正 ---
        if items_to_force_correct:
            # ### 优化点 2 改动 ###: 动态准备 prompt 内容
            for item in items_to_force_correct:
                if item.get('solution'):
                    item['solution_section'] = f"\n# Detailed Solution for Reference:\n{item['solution']}\n"
                else:
                    item['solution_section'] = "" # 确保该键存在，即使为空

            force_correct_results, failed_force_correct = await _run_batch_step_with_retry(
                api_client, items_to_force_correct, "步骤 6: 强制修正答案", 
                templates['template_force_correct_answer'],
                # ### 优化点 2 改动 ###: 添加新键
                ['conversation_history', 'expected_answer', 'solution_section'],
                lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.7}
            )
            for item, corrected_answer in force_correct_results:
                item['conversation_history'].append({"role": "assistant", "content": corrected_answer})
                completed_items.append(item)
            
            if failed_force_correct:
                total_discarded_items.extend(failed_force_correct)
            
            print(f"--- 强制修正小结 ---")
            print(f"  - {len(force_correct_results)} 项修正成功并完成。")
            print(f"  - {len(failed_force_correct)} 项修正失败被丢弃。")

        items_to_process = next_round_items
        current_loop += 1

    if items_to_process:
        print(f"\n达到最大追问次数({max_ask_loops})，剩余 {len(items_to_process)} 项将被丢弃。")
        total_discarded_items.extend(items_to_process)

    print("\n============================================================")
    print("所有数据条目处理完毕。")
    
    # 清理临时字段
    for item in completed_items:
        item.pop('temp_answer', None)
        item.pop('generated_answer', None)
        # ### 优化点 2 改动 ###: 清理临时 solution 字段
        item.pop('solution_section', None)
    
    print(f"最终结果统计:")
    print(f"  - 初始数据量: {len(data)}")
    print(f"  - 成功生成完整对话: {len(completed_items)}")
    print(f"  - 失败并丢弃总数: {len(total_discarded_items)}")
    print("============================================================\n")

    # ### 优化点 1 改动 ###: 返回成功和失败的列表
    return completed_items, total_discarded_items

# ---------------------------------------------------------------------------
# 策略 3: 直接回答并修正 (Direct Answer & Correct)
# ---------------------------------------------------------------------------
async def strategy_direct_answer_and_correct(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    策略：直接回答问题，然后判断答案。如果错误，则根据标准答案和解题思路重构。
    1. LLM 根据 ori_question 生成初步答案。
    2. LLM Judge 判断初步答案是否正确。
    3. 如果正确，则构建对话并完成。
    4. 如果错误，则使用 expected_answer 和 solution (如果存在) 重构一个完美的答案，然后构建对话。
    """
    print("\n============================================================")
    print("开始执行策略: strategy_direct_answer_and_correct")
    print(f"初始数据量: {len(data)} 项")
    print("============================================================\n")

    completed_items = []
    total_discarded_items = []

    # --- 步骤 1: 批量生成初步答案 ---
    # 要求LLM在回答时包含关键点、分析和最终结果
    answer_results, failed_answer = await _run_batch_step_with_retry(
        api_client, data, "步骤 1: 生成初步答案",
        templates['template_direct_answer'],
        ['ori_question'],
        lambda r: (r, None),
        {'max_tokens': 8192, 'temperature': 0.7},
        max_retries=2
    )
    total_discarded_items.extend(failed_answer)
    if not answer_results:
        print("所有数据在步骤1中失败，流程终止。")
        return [], total_discarded_items

    items_to_judge = []
    for item, generated_answer in answer_results:
        item['generated_answer'] = generated_answer
        items_to_judge.append(item)

    # --- 步骤 2: 批量判断答案 ---
    # 使用一个专门为本策略简化的Judge Prompt
    judge_results, failed_judge = await _run_batch_step_with_retry(
        api_client, items_to_judge, "步骤 2: 判断答案",
        templates['template_judge_direct_answer'],
        ['ori_question', 'generated_answer', 'expected_answer'],
        _parse_judge_json_response,
        {'max_tokens': 1024, 'temperature': 0.0},
        max_retries=2
    )
    total_discarded_items.extend(failed_judge)
    if not judge_results:
        print("所有数据在步骤2中失败，流程终止。")
        return completed_items, total_discarded_items

    # --- 步骤 3: 分流处理 ---
    items_to_reconstruct = []
    correct_count = 0
    for item, judgement in judge_results:
        if judgement['is_correct']:
            # 答案正确，直接构建最终对话
            item['conversation_history'] = [
                {"role": "user", "content": item['ori_question']},
                {"role": "assistant", "content": item['generated_answer']}
            ]
            completed_items.append(item)
            correct_count += 1
        else:
            # 答案错误，加入待重构列表
            items_to_reconstruct.append(item)

    print(f"--- 判断小结 ---")
    print(f"  - {correct_count} 项初步答案正确，已完成。")
    print(f"  - {len(items_to_reconstruct)} 项答案错误，将进行重构。")

    # --- 步骤 4: 批量重构错误答案 ---
    if items_to_reconstruct:
        # 动态准备 prompt 内容，加入 solution (如果存在)
        for item in items_to_reconstruct:
            if item.get('solution'):
                item['solution_section'] = f"\n# Detailed Solution for Reference:\n{item['solution']}\n"
            else:
                item['solution_section'] = ""  # 确保该键存在，即使为空

        reconstruct_results, failed_reconstruct = await _run_batch_step_with_retry(
            api_client, items_to_reconstruct, "步骤 4: 重构错误答案",
            templates['template_reconstruct_answer'],
            ['ori_question', 'generated_answer', 'expected_answer', 'solution_section'],
            lambda r: (r, None),
            {'max_tokens': 8192, 'temperature': 0.5}, # 使用稍低的温度以确保答案质量
            max_retries=2
        )
        total_discarded_items.extend(failed_reconstruct)

        for item, reconstructed_answer in reconstruct_results:
            # 使用重构后的完美答案构建最终对话
            item['conversation_history'] = [
                {"role": "user", "content": item['ori_question']},
                {"role": "assistant", "content": reconstructed_answer}
            ]
            completed_items.append(item)
        
        print(f"--- 重构小结 ---")
        print(f"  - {len(reconstruct_results)} 项重构成功并完成。")
        print(f"  - {len(failed_reconstruct)} 项重构失败被丢弃。")

    # --- 清理并总结 ---
    for item in completed_items:
        item.pop('generated_answer', None)
        item.pop('solution_section', None)

    print("\n============================================================")
    print("所有数据条目处理完毕。")
    print(f"最终结果统计:")
    print(f"  - 初始数据量: {len(data)}")
    print(f"  - 成功生成对话: {len(completed_items)}")
    print(f"  - 失败并丢弃总数: {len(total_discarded_items)}")
    print("============================================================\n")

    return completed_items, total_discarded_items