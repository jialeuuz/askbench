import json
import re
import asyncio
from functools import partial
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable
from post_api import CustomAPI

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _coerce_points_list(points: Any) -> List[str]:
    if points is None:
        return []
    if isinstance(points, list):
        cleaned: List[str] = []
        for entry in points:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                cleaned.append(text)
        return cleaned
    if isinstance(points, str):
        stripped = points.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                loaded = json.loads(stripped)
                if isinstance(loaded, list):
                    return _coerce_points_list(loaded)
            except Exception:
                pass
        return [stripped]
    return [str(points).strip()]


def _format_required_points_text(points: Any) -> str:
    points_list = _coerce_points_list(points)
    if not points_list:
        return "- None provided (the assistant may answer once confident)."
    return "\n".join(f"{idx}. {point}" for idx, point in enumerate(points_list, start=1))


def _format_conversation_history_text(conversation_history: Any) -> str:
    if conversation_history is None:
        return ""
    if isinstance(conversation_history, str):
        return conversation_history.strip()
    if not isinstance(conversation_history, list):
        return str(conversation_history).strip()
    lines: List[str] = []
    for msg in conversation_history:
        if not isinstance(msg, dict):
            lines.append(str(msg))
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _extract_last_assistant_message(conversation_history: Any) -> str:
    if not isinstance(conversation_history, list):
        return ""
    for msg in reversed(conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def _build_user_internal_knowledge(
    item: Dict[str, Any],
    *,
    scenario_type: str,
    scenario_context: str,
    checklist_header: str,
    checklist_points: Any,
) -> str:
    user_knowledge = {
        "my_real_question": item.get("ori_question", ""),
        "scenario_context": scenario_context,
        "scenario_type": scenario_type,
        "checklist_header": checklist_header,
        "checklist_points": _coerce_points_list(checklist_points),
    }
    return json.dumps(user_knowledge, indent=2, ensure_ascii=False)


def _safe_format(template: str, variables: Dict[str, Any]) -> str:
    """仅替换已知占位符 {key}，保留其他花括号为字面量。
    这样可避免模板中的 JSON 示例触发 str.format 的 KeyError。
    """
    if not variables:
        return template
    # 构造正则，仅匹配 {key} 形式，key 限定在 variables 中
    pattern = re.compile(r"\{(" + "|".join(re.escape(k) for k in variables.keys()) + r")\}")
    return pattern.sub(lambda m: str(variables.get(m.group(1), "")), template)
def _clean_for_failure(item: Dict[str, Any]) -> Dict[str, Any]:
    clean_item = item.copy()
    clean_item.pop('degraded_question', None)
    clean_item.pop('degraded_info', None)
    clean_item.pop('overconfidence_question', None)
    clean_item.pop('overconfidence_info', None)
    clean_item.pop('misleading_points', None)
    clean_item.pop('conversation_history', None)
    clean_item.pop('temp_answer', None)
    clean_item.pop('generated_answer', None)
    clean_item.pop('solution_section', None)
    return clean_item

def _attach_failure_meta(item: Dict[str, Any], step: str, reason: str, attempts: int = 0, response_preview: Optional[str] = None) -> Dict[str, Any]:
    clean_item = _clean_for_failure(item)
    meta = {"step": step, "reason": reason, "attempts": attempts}
    if response_preview:
        meta["response_preview"] = response_preview
    clean_item["_failure"] = meta
    return clean_item
def _parse_question_variant_json_response(raw_response: str, *, question_key: str, info_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match: raise ValueError("在回复中未找到JSON对象")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if question_key in parsed_json and info_key in parsed_json:
            return parsed_json, None
        raise KeyError(f"JSON对象中缺少 '{question_key}' 或 '{info_key}' 键")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e

def _parse_coverage_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("在 覆盖自检 回复中未找到JSON对象")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if 'all_covered' in parsed_json and 'missing' in parsed_json:
            if not isinstance(parsed_json['all_covered'], bool):
                raise ValueError("'all_covered' 必须为布尔值")
            if not isinstance(parsed_json['missing'], list):
                raise ValueError("'missing' 必须为数组")
            return parsed_json, None
        raise KeyError("JSON对象中缺少 'all_covered' 或 'missing' 键")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e

def _parse_judge_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match: raise ValueError("在 Judge 回复中未找到JSON对象")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if 'is_correct' not in parsed_json:
            raise KeyError("JSON对象中缺少 'is_correct' 键")

        # 兼容新版 overconfidence 裁判输出（无 reason 字段，但包含覆盖信息）
        valid_reasons = [None, 'insufficient_asking', 'reasoning_error']
        reason = parsed_json.get('reason')
        if reason is None and any(k in parsed_json for k in ('missing_required_points', 'all_required_points_resolved', 'is_final_answer')):
            if parsed_json.get('is_correct') is True:
                reason = None
            elif parsed_json.get('all_required_points_resolved') is False:
                reason = 'insufficient_asking'
            elif parsed_json.get('is_correct') is False:
                reason = 'reasoning_error'
            else:
                # is_correct 为空或其他情况，视为未满足覆盖
                reason = 'insufficient_asking'
            parsed_json['reason'] = reason

        if parsed_json.get('reason') not in valid_reasons:
            raise ValueError(f"无效的 reason 值: '{parsed_json.get('reason')}'. 只允许 {valid_reasons}")
        return parsed_json, None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e


def _coerce_to_text_block(value: Any) -> str:
    """
    模型有时会把 info 字段输出成列表；这里统一转换为多行字符串，方便后续模板使用。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        normalized_lines = []
        for entry in value:
            if entry is None:
                continue
            if isinstance(entry, str):
                line = entry.strip()
            else:
                line = json.dumps(entry, ensure_ascii=False)
            if not line:
                continue
            if line.startswith(("-", "*")):
                normalized_lines.append(line)
            else:
                normalized_lines.append(f"- {line}")
        return "\n".join(normalized_lines).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()

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
    failure_reasons = {}
    last_response_preview = {}
    attempt_counts = {i: 0 for i in range(len(items))}

    debug_logged = 0
    for attempt in range(max_retries + 1):
        if not item_indices_to_process: break
        if attempt > 0:
            print(f"    - {step_name}: 第 {attempt}/{max_retries} 轮重试，剩余 {len(item_indices_to_process)} 项...")
            await asyncio.sleep(retry_delay)
        
        prompts_for_this_batch = []
        valid_indices_for_batch = []
        for index in item_indices_to_process:
            try:
                item = items[index]
                format_args = {key: item[key] for key in prompt_format_keys}

                raw_conversation_history = item.get("conversation_history")
                if isinstance(raw_conversation_history, list):
                    format_args.setdefault(
                        "conversation_history_text",
                        _format_conversation_history_text(raw_conversation_history),
                    )
                    assistant_message = _extract_last_assistant_message(raw_conversation_history)
                    format_args.setdefault("assistant_message", assistant_message)
                    format_args.setdefault("assistant_question", assistant_message)
                    if "conversation_history" in format_args:
                        format_args["conversation_history"] = json.dumps(raw_conversation_history, ensure_ascii=False, indent=2)
                if 'conversation_history' in format_args and not isinstance(format_args['conversation_history'], str):
                    format_args['conversation_history'] = json.dumps(format_args['conversation_history'], ensure_ascii=False, indent=2)

                scenario_question = (
                    item.get("overconfidence_question")
                    or item.get("degraded_question")
                    or item.get("ori_question", "")
                )
                scenario_context = item.get("overconfidence_info") or item.get("degraded_info") or ""
                is_overconfidence = bool(item.get("overconfidence_question"))
                checklist_header = (
                    "Misleading claims that must be addressed before answering"
                    if is_overconfidence
                    else "Required clarification points (must be obtained before answering)"
                )
                if is_overconfidence:
                    checklist_points = item.get("misleading_points") or item.get("required_points")
                else:
                    checklist_points = item.get("required_points")

                format_args.setdefault("ori_question", item.get("ori_question", ""))
                format_args.setdefault("scenario_question", scenario_question or "")
                format_args.setdefault("scenario_context", scenario_context or "")
                format_args.setdefault("checklist_header", checklist_header)
                format_args.setdefault("required_points_text", _format_required_points_text(checklist_points))
                format_args.setdefault(
                    "user_internal_knowledge",
                    _build_user_internal_knowledge(
                        item,
                        scenario_type=("overconfidence" if is_overconfidence else "missing_info"),
                        scenario_context=scenario_context or "",
                        checklist_header=checklist_header,
                        checklist_points=checklist_points,
                    ),
                )

                if 'required_points' in format_args and not isinstance(format_args['required_points'], str):
                    try:
                        format_args['required_points'] = json.dumps(format_args['required_points'], ensure_ascii=False, indent=2)
                    except Exception:
                        format_args['required_points'] = str(format_args['required_points'])
                if 'misleading_points' in format_args and not isinstance(format_args['misleading_points'], str):
                    try:
                        format_args['misleading_points'] = json.dumps(format_args['misleading_points'], ensure_ascii=False, indent=2)
                    except Exception:
                        format_args['misleading_points'] = str(format_args['misleading_points'])
                # 使用安全替换，避免模板中的 JSON 花括号触发 KeyError
                prompts_for_this_batch.append(_safe_format(prompt_template, format_args))
                valid_indices_for_batch.append(index)
            except KeyError as e:
                # 记录格式化缺失键导致的失败原因
                successful_results[index] = (items[index], None)
                failure_reasons[index] = f"prompt_format_key_error: {str(e)}"
        
        item_indices_to_process = [idx for idx in item_indices_to_process if idx in valid_indices_for_batch]
        if not prompts_for_this_batch: continue

        try:
            responses, _, _ = await api_client.infer_batch_async(messages=prompts_for_this_batch, **api_params)
        except Exception as e:
            # 记录本轮批量 API 错误，对所有仍待处理的索引记一次失败
            for idx in item_indices_to_process:
                attempt_counts[idx] += 1
                failure_reasons[idx] = f"api_error: {repr(e)}"
            continue

        failed_indices_for_next_round = []
        for i, raw_response in enumerate(responses):
            original_item_index = item_indices_to_process[i]
            parsed_result, error = parser_func(raw_response)
            if error is None:
                successful_results[original_item_index] = (items[original_item_index], parsed_result)
            else:
                if debug_logged < 3:
                    preview = str(raw_response)[:400].replace("\n", " ")
                    print(f"    解析失败示例[{debug_logged+1}]: {preview}")
                    debug_logged += 1
                attempt_counts[original_item_index] += 1
                failure_reasons[original_item_index] = f"parse_error: {repr(error)}"
                last_response_preview[original_item_index] = str(raw_response)[:400]
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
            # 追加失败元信息，便于后续定位问题
            meta = {
                "step": step_name,
                "reason": failure_reasons.get(i, "unknown"),
                "attempts": attempt_counts.get(i, 0)
            }
            if i in last_response_preview:
                meta["response_preview"] = last_response_preview[i]
            clean_item["_failure"] = meta
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
    api_params = {'max_tokens': 16000, 'temperature': 0.2}
    question_parser = partial(
        _parse_question_variant_json_response,
        question_key='degraded_question',
        info_key='degraded_info'
    )
    successful_results, failed_items = await _run_batch_step_with_retry(
        api_client, data, step_name, prompt_template,
        ['ori_question', 'expected_answer'],
        question_parser, api_params, max_retries
    )
    processed_successful_items = []
    for item, parsed_json in successful_results:
        item['degraded_question'] = parsed_json['degraded_question']
        item['degraded_info'] = _coerce_to_text_block(parsed_json['degraded_info'])
        # 保存结构化缺失点清单（若存在）；否则给一个空清单，确保后续模板格式化安全
        if isinstance(parsed_json, dict) and 'required_points' in parsed_json:
            item['required_points'] = parsed_json['required_points']
        else:
            item['required_points'] = []
        processed_successful_items.append(item)
    return processed_successful_items, failed_items


async def generate_overconfidence_question_and_info(
    api_client: CustomAPI, data: List[Dict[str, Any]], templates: Dict[str, str], max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 (成功项列表, 失败项列表)，用于构造过度自信样本。"""
    step_name = "步骤 1: 生成过度自信问题和纠错信息"
    prompt_template = templates['template_generate_overconfidence_question_and_info']
    api_params = {'max_tokens': 16000, 'temperature': 0.2}
    question_parser = partial(
        _parse_question_variant_json_response,
        question_key='overconfidence_question',
        info_key='overconfidence_info'
    )
    successful_results, failed_items = await _run_batch_step_with_retry(
        api_client, data, step_name, prompt_template,
        ['ori_question', 'expected_answer'],
        question_parser, api_params, max_retries
    )
    processed_successful_items = []
    for item, parsed_json in successful_results:
        # 直接采用模型产出的独立问题文本（要求其保留原题 givens，但以自然口吻融入“自信但错误”的断言）
        item['overconfidence_question'] = parsed_json.get('overconfidence_question', '')
        item['overconfidence_info'] = _coerce_to_text_block(parsed_json['overconfidence_info'])
        if isinstance(parsed_json, dict) and 'misleading_points' in parsed_json:
            item['misleading_points'] = parsed_json['misleading_points']
        else:
            item['misleading_points'] = []
        processed_successful_items.append(item)
    return processed_successful_items, failed_items

# ---------------------------------------------------------------------------
# 多轮对话策略通用实现
# ---------------------------------------------------------------------------
async def _run_multi_turn_strategy(
    *,
    strategy_label: str,
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
    generator_func: Callable[[CustomAPI, List[Dict[str, Any]], Dict[str, str]], Awaitable[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]],
    question_key: str,
    info_key: str,
    checklist_key: str,
    template_config: Dict[str, str],
    max_ask_loops: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    通用多轮追问-作答-判定的流水线实现。
    """
    print("\n============================================================")
    print(f"开始执行策略: {strategy_label}")
    print(f"初始数据量: {len(data)} 项")
    print("============================================================\n")

    items_to_process, failed_step1 = await generator_func(api_client, data, templates)
    total_discarded_items = failed_step1

    if not items_to_process:
        print("\n所有数据在步骤1中失败，流程终止。")
        return [], total_discarded_items

    for item in items_to_process:
        item.setdefault(checklist_key, [])
        item['conversation_history'] = [{"role": "user", "content": item.get(question_key, "")}]

    completed_items = []
    current_loop = 0

    while items_to_process and current_loop < max_ask_loops:
        round_start_count = len(items_to_process)
        print(f"\n{'='*20} 对话轮次 {current_loop + 1} (处理 {round_start_count} 项) {'='*20}")

        # --- 步骤 2: 批量生成澄清问题 ---
        if current_loop == 0:
            ask_template_name = template_config['ask_first']
        elif current_loop == max_ask_loops - 1:
            ask_template_name = template_config['ask_all']
        else:
            ask_template_name = template_config['ask_follow_up']
        ask_results, failed_ask = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 2: 生成澄清问题", templates[ask_template_name],
            [question_key, info_key, 'conversation_history', checklist_key],
            lambda r: (r, None), {'max_tokens': 2048, 'temperature': 0.5}
        )
        total_discarded_items.extend(failed_ask)
        if not ask_results: break
        items_to_process = [item for item, _ in ask_results]
        for item, response in ask_results:
            item['conversation_history'].append({"role": "assistant", "content": response})

        # --- 步骤 3: 批量模拟用户回复 ---
        reply_results, failed_reply = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 3: 模拟用户回复", templates[template_config['simulate_user']],
            ['conversation_history', info_key, checklist_key],
            lambda r: (r, None), {'max_tokens': 2048, 'temperature': 0.5}
        )
        total_discarded_items.extend(failed_reply)
        if not reply_results: break
        items_to_process = [item for item, _ in reply_results]
        for item, response in reply_results:
            item['conversation_history'].append({"role": "user", "content": response})

        # --- 步骤 3.5: 覆盖自检 ---
        pre_next_round_items = []
        pre_force_correct_items = []
        coverage_results, failed_coverage = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 3.5: 覆盖自检", templates[template_config['coverage_check']],
            ['conversation_history', checklist_key],
            _parse_coverage_json_response, {'max_tokens': 512, 'temperature': 0.0}
        )
        for failed_item in failed_coverage:
            if current_loop == max_ask_loops - 1:
                pre_force_correct_items.append(failed_item)
            else:
                pre_next_round_items.append(failed_item)

        items_ready_to_answer = []
        for item, cov in coverage_results:
            if cov.get('all_covered') is True:
                items_ready_to_answer.append(item)
            else:
                if current_loop == max_ask_loops - 1:
                    pre_force_correct_items.append(item)
                else:
                    pre_next_round_items.append(item)

        if not items_ready_to_answer:
            if pre_force_correct_items:
                print("覆盖自检后进入强制修正（最后一轮）...")
                for it in pre_force_correct_items:
                    if it.get('solution'):
                        it['solution_section'] = f"\n# Detailed Solution for Reference:\n{it['solution']}\n"
                    else:
                        it['solution_section'] = ""
                force_correct_results, failed_force = await _run_batch_step_with_retry(
                    api_client, pre_force_correct_items, "步骤 6: 强制修正答案",
                    templates[template_config['force_correct']],
                    ['conversation_history', 'expected_answer', 'solution_section'],
                    lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.2}
                )
                for it, corrected_answer in force_correct_results:
                    it['conversation_history'].append({"role": "assistant", "content": corrected_answer})
                    completed_items.append(it)
                if failed_force:
                    total_discarded_items.extend(failed_force)
            items_to_process = pre_next_round_items
            current_loop += 1
            continue

        items_to_process = items_ready_to_answer

        # --- 步骤 4: 批量生成最终答案 ---
        answer_results, failed_answer = await _run_batch_step_with_retry(
            api_client, items_to_process, "步骤 4: 生成最终答案", templates[template_config['generate_answer']],
            ['conversation_history'],
            lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.2}
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
            api_client, items_to_process, "步骤 5: 判断答案", templates[template_config['judge_answer']],
            ['conversation_history', 'generated_answer', 'expected_answer', info_key, checklist_key],
            _parse_judge_json_response, {'max_tokens': 1024, 'temperature': 0.0}
        )
        total_discarded_items.extend(failed_judge)
        if not judge_results: break

        next_round_items = pre_next_round_items.copy()
        items_to_force_correct = pre_force_correct_items.copy()
        current_round_completed_count = 0

        for item, judgement in judge_results:
            if judgement['is_correct']:
                completed_items.append(item)
                current_round_completed_count += 1
            elif judgement['reason'] == 'insufficient_asking':
                item['conversation_history'].pop()
                if current_loop == max_ask_loops - 1:
                    items_to_force_correct.append(item)
                else:
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
            for item in items_to_force_correct:
                if item.get('solution'):
                    item['solution_section'] = f"\n# Detailed Solution for Reference:\n{item['solution']}\n"
                else:
                    item['solution_section'] = ""

            force_correct_results, failed_force_correct = await _run_batch_step_with_retry(
                api_client, items_to_force_correct, "步骤 6: 强制修正答案",
                templates[template_config['force_correct']],
                ['conversation_history', 'expected_answer', 'solution_section'],
                lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.2}
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
        enriched = [
            _attach_failure_meta(it, step="循环结束", reason="max_loops_reached", attempts=0)
            for it in items_to_process
        ]
        total_discarded_items.extend(enriched)

    print("\n============================================================")
    print("所有数据条目处理完毕。")

    for item in completed_items:
        item.pop('temp_answer', None)
        item.pop('generated_answer', None)
        item.pop('solution_section', None)

    print(f"最终结果统计:")
    print(f"  - 初始数据量: {len(data)}")
    print(f"  - 成功生成完整对话: {len(completed_items)}")
    print(f"  - 失败并丢弃总数: {len(total_discarded_items)}")
    print("============================================================\n")

    return completed_items, total_discarded_items

# ---------------------------------------------------------------------------
# 策略 2: 劣化问法的多轮对话训练数据
# ---------------------------------------------------------------------------
async def generate_multi_turn_degraded_training_data(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
    max_ask_loops: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    劣化场景：从模糊/缺失意图的问题出发，逐轮澄清并作答。
    """
    template_config = {
        'ask_first': 'template_assistant_ask_first_question',
        'ask_follow_up': 'template_assistant_ask_follow_up_question',
        'ask_all': 'template_assistant_ask_all_remaining',
        'simulate_user': 'template_simulate_user_reply',
        'coverage_check': 'template_coverage_check',
        'generate_answer': 'template_generate_final_answer',
        'judge_answer': 'template_judge_answer',
        'force_correct': 'template_force_correct_answer',
    }
    return await _run_multi_turn_strategy(
        strategy_label="generate_multi_turn_degraded_training_data (V4)",
        api_client=api_client,
        data=data,
        templates=templates,
        generator_func=generate_degraded_question_and_info,
        question_key='degraded_question',
        info_key='degraded_info',
        checklist_key='required_points',
        template_config=template_config,
        max_ask_loops=max_ask_loops,
    )

# ---------------------------------------------------------------------------
# 策略 2b: 过度自信问法的多轮对话训练数据
# ---------------------------------------------------------------------------
async def generate_multi_turn_overconfidence_training_data(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
    max_ask_loops: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    过度自信场景：原问题语气强势但条件错误，策略需要识别并纠正这些误导点。
    """
    template_config = {
        'ask_first': 'template_overconfidence_assistant_ask_first_question',
        'ask_follow_up': 'template_overconfidence_assistant_ask_follow_up_question',
        'ask_all': 'template_overconfidence_assistant_ask_all_remaining',
        'simulate_user': 'template_overconfidence_simulate_user_reply',
        'coverage_check': 'template_overconfidence_coverage_check',
        'generate_answer': 'template_generate_overconfidence_final_answer',
        'judge_answer': 'template_overconfidence_judge_answer',
        'force_correct': 'template_force_correct_answer',
    }
    return await _run_multi_turn_strategy(
        strategy_label="generate_multi_turn_overconfidence_training_data (V1)",
        api_client=api_client,
        data=data,
        templates=templates,
        generator_func=generate_overconfidence_question_and_info,
        question_key='overconfidence_question',
        info_key='overconfidence_info',
        checklist_key='misleading_points',
        template_config=template_config,
        max_ask_loops=max_ask_loops,
    )

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
