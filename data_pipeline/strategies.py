import json
import re
import asyncio
from functools import partial
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable
from post_api import CustomAPI

# ---------------------------------------------------------------------------
# Helper functions
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
    """Only replace known placeholders {key}; treat other braces as literals.

    This avoids str.format KeyError when templates include JSON examples with braces.
    """
    if not variables:
        return template
    # Build a regex that matches only {key} placeholders where key exists in variables.
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
        if not json_match:
            raise ValueError("No JSON object found in the response.")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if question_key in parsed_json and info_key in parsed_json:
            return parsed_json, None
        raise KeyError(f"JSON object is missing '{question_key}' or '{info_key}'.")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e

def _parse_coverage_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the coverage-check response.")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if 'all_covered' in parsed_json and 'missing' in parsed_json:
            if not isinstance(parsed_json['all_covered'], bool):
                raise ValueError("'all_covered' must be a boolean")
            if not isinstance(parsed_json['missing'], list):
                raise ValueError("'missing' must be a list")
            return parsed_json, None
        raise KeyError("JSON object is missing 'all_covered' or 'missing'.")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e

def _parse_judge_json_response(raw_response: str) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
    try:
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in the judge response.")
        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)
        if 'is_correct' not in parsed_json:
            raise KeyError("JSON object is missing 'is_correct'.")

        # Backward-compatible with newer overconfidence judge outputs (no reason field, but includes coverage info).
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
                # is_correct missing/other: treat as insufficient coverage.
                reason = 'insufficient_asking'
            parsed_json['reason'] = reason

        if parsed_json.get('reason') not in valid_reasons:
            raise ValueError(f"Invalid reason value: '{parsed_json.get('reason')}'. Allowed: {valid_reasons}")
        return parsed_json, None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, e


def _coerce_to_text_block(value: Any) -> str:
    """
    Models sometimes output the `info` field as a list; normalize it into a multi-line string.
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

    print(f"  - {step_name}: processing {len(items)} item(s)...")
    item_indices_to_process = list(range(len(items)))
    successful_results = {}
    failure_reasons = {}
    last_response_preview = {}
    attempt_counts = {i: 0 for i in range(len(items))}

    debug_logged = 0
    for attempt in range(max_retries + 1):
        if not item_indices_to_process: break
        if attempt > 0:
            print(f"    - {step_name}: retry {attempt}/{max_retries}, remaining {len(item_indices_to_process)} item(s)...")
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
                # Use safe formatting to avoid KeyError from JSON braces in templates.
                prompts_for_this_batch.append(_safe_format(prompt_template, format_args))
                valid_indices_for_batch.append(index)
            except KeyError as e:
                # Record missing-key formatting failures.
                successful_results[index] = (items[index], None)
                failure_reasons[index] = f"prompt_format_key_error: {str(e)}"
        
        item_indices_to_process = [idx for idx in item_indices_to_process if idx in valid_indices_for_batch]
        if not prompts_for_this_batch: continue

        try:
            responses, _, _ = await api_client.infer_batch_async(messages=prompts_for_this_batch, **api_params)
        except Exception as e:
            # Record batch API error; count a failure for each remaining item.
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
                    print(f"    Parse failure example [{debug_logged+1}]: {preview}")
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
            # Ensure failed items are the original inputs, without intermediate fields.
            clean_item = item.copy()
            clean_item.pop('degraded_question', None)
            clean_item.pop('degraded_info', None)
            clean_item.pop('conversation_history', None)
            clean_item.pop('temp_answer', None)
            clean_item.pop('generated_answer', None)
            clean_item.pop('solution_section', None)  # ensure this temporary field is removed
            # Attach failure metadata to help debugging.
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
        print(f"  - {step_name}: done. success={len(final_successful)}, dropped={len(final_failed)}.")
    return final_successful, final_failed

# ---------------------------------------------------------------------------
# Strategy 1
# ---------------------------------------------------------------------------
async def generate_degraded_question_and_info(
    api_client: CustomAPI, data: List[Dict[str, Any]], templates: Dict[str, str], max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (successful_items, failed_items)."""
    step_name = "Step 1: generate degraded question and context"
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
        # Persist structured required_points if present; otherwise use an empty list for safe formatting downstream.
        if isinstance(parsed_json, dict) and 'required_points' in parsed_json:
            item['required_points'] = parsed_json['required_points']
        else:
            item['required_points'] = []
        processed_successful_items.append(item)
    return processed_successful_items, failed_items


async def generate_overconfidence_question_and_info(
    api_client: CustomAPI, data: List[Dict[str, Any]], templates: Dict[str, str], max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (successful_items, failed_items) for building overconfidence samples."""
    step_name = "Step 1: generate overconfidence question and correction info"
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
        # Use the model-generated standalone question text (should preserve original givens, but includes a confident-yet-wrong claim).
        item['overconfidence_question'] = parsed_json.get('overconfidence_question', '')
        item['overconfidence_info'] = _coerce_to_text_block(parsed_json['overconfidence_info'])
        if isinstance(parsed_json, dict) and 'misleading_points' in parsed_json:
            item['misleading_points'] = parsed_json['misleading_points']
        else:
            item['misleading_points'] = []
        processed_successful_items.append(item)
    return processed_successful_items, failed_items

# ---------------------------------------------------------------------------
# Shared implementation for multi-turn dialogue strategies
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
    Shared multi-turn ask→answer→judge pipeline implementation.
    """
    print("\n============================================================")
    print(f"Starting strategy: {strategy_label}")
    print(f"Initial items: {len(data)}")
    print("============================================================\n")

    items_to_process, failed_step1 = await generator_func(api_client, data, templates)
    total_discarded_items = failed_step1

    if not items_to_process:
        print("\nAll items failed at step 1; aborting.")
        return [], total_discarded_items

    for item in items_to_process:
        item.setdefault(checklist_key, [])
        item['conversation_history'] = [{"role": "user", "content": item.get(question_key, "")}]

    completed_items = []
    current_loop = 0

    while items_to_process and current_loop < max_ask_loops:
        round_start_count = len(items_to_process)
        print(f"\n{'='*20} Round {current_loop + 1} (processing {round_start_count}) {'='*20}")

        # --- Step 2: generate clarification question(s) ---
        if current_loop == 0:
            ask_template_name = template_config['ask_first']
        elif current_loop == max_ask_loops - 1:
            ask_template_name = template_config['ask_all']
        else:
            ask_template_name = template_config['ask_follow_up']
        ask_results, failed_ask = await _run_batch_step_with_retry(
            api_client, items_to_process, "Step 2: generate clarification questions", templates[ask_template_name],
            [question_key, info_key, 'conversation_history', checklist_key],
            lambda r: (r, None), {'max_tokens': 2048, 'temperature': 0.5}
        )
        total_discarded_items.extend(failed_ask)
        if not ask_results: break
        items_to_process = [item for item, _ in ask_results]
        for item, response in ask_results:
            item['conversation_history'].append({"role": "assistant", "content": response})

        # --- Step 3: simulate user reply ---
        reply_results, failed_reply = await _run_batch_step_with_retry(
            api_client, items_to_process, "Step 3: simulate user reply", templates[template_config['simulate_user']],
            ['conversation_history', info_key, checklist_key],
            lambda r: (r, None), {'max_tokens': 2048, 'temperature': 0.5}
        )
        total_discarded_items.extend(failed_reply)
        if not reply_results: break
        items_to_process = [item for item, _ in reply_results]
        for item, response in reply_results:
            item['conversation_history'].append({"role": "user", "content": response})

        # --- Step 3.5: coverage self-check ---
        pre_next_round_items = []
        pre_force_correct_items = []
        coverage_results, failed_coverage = await _run_batch_step_with_retry(
            api_client, items_to_process, "Step 3.5: coverage self-check", templates[template_config['coverage_check']],
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
                print("Coverage check triggers forced correction (last round)...")
                for it in pre_force_correct_items:
                    if it.get('solution'):
                        it['solution_section'] = f"\n# Detailed Solution for Reference:\n{it['solution']}\n"
                    else:
                        it['solution_section'] = ""
                force_correct_results, failed_force = await _run_batch_step_with_retry(
                    api_client, pre_force_correct_items, "Step 6: force-correct answers",
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

        # --- Step 4: generate final answer ---
        answer_results, failed_answer = await _run_batch_step_with_retry(
            api_client, items_to_process, "Step 4: generate final answers", templates[template_config['generate_answer']],
            ['conversation_history'],
            lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.2}
        )
        total_discarded_items.extend(failed_answer)
        if not answer_results: break
        items_to_process = [item for item, _ in answer_results]
        for item, response in answer_results:
            item['temp_answer'] = response
            item['conversation_history'].append({"role": "assistant", "content": response})

        # --- Step 5: judge answer ---
        for item in items_to_process:
            item['generated_answer'] = item['temp_answer']

        judge_results, failed_judge = await _run_batch_step_with_retry(
            api_client, items_to_process, "Step 5: judge answers", templates[template_config['judge_answer']],
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

        print("--- Round summary ---")
        print(f"  - completed this round: {current_round_completed_count}")
        print(f"  - to next round (insufficient asking): {len(next_round_items)}")
        print(f"  - to forced correction (reasoning error): {len(items_to_force_correct)}")

        # --- Step 6: forced correction ---
        if items_to_force_correct:
            for item in items_to_force_correct:
                if item.get('solution'):
                    item['solution_section'] = f"\n# Detailed Solution for Reference:\n{item['solution']}\n"
                else:
                    item['solution_section'] = ""

            force_correct_results, failed_force_correct = await _run_batch_step_with_retry(
                api_client, items_to_force_correct, "Step 6: force-correct answers",
                templates[template_config['force_correct']],
                ['conversation_history', 'expected_answer', 'solution_section'],
                lambda r: (r, None), {'max_tokens': 8192, 'temperature': 0.2}
            )
            for item, corrected_answer in force_correct_results:
                item['conversation_history'].append({"role": "assistant", "content": corrected_answer})
                completed_items.append(item)

            if failed_force_correct:
                total_discarded_items.extend(failed_force_correct)

            print("--- Forced-correction summary ---")
            print(f"  - corrected and completed: {len(force_correct_results)}")
            print(f"  - correction failed/dropped: {len(failed_force_correct)}")

        items_to_process = next_round_items
        current_loop += 1

    if items_to_process:
        print(f"\nReached max_ask_loops ({max_ask_loops}); dropping remaining {len(items_to_process)} item(s).")
        enriched = [
            _attach_failure_meta(it, step="loop_end", reason="max_loops_reached", attempts=0)
            for it in items_to_process
        ]
        total_discarded_items.extend(enriched)

    print("\n============================================================")
    print("All items processed.")

    for item in completed_items:
        item.pop('temp_answer', None)
        item.pop('generated_answer', None)
        item.pop('solution_section', None)

    print("Final stats:")
    print(f"  - initial items: {len(data)}")
    print(f"  - completed dialogues: {len(completed_items)}")
    print(f"  - dropped items: {len(total_discarded_items)}")
    print("============================================================\n")

    return completed_items, total_discarded_items

# ---------------------------------------------------------------------------
# Strategy 2: multi-turn dialogues for missing-info questions
# ---------------------------------------------------------------------------
async def generate_multi_turn_degraded_training_data(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
    max_ask_loops: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Missing-info scenario: start from an ambiguous/underspecified question, ask clarifications, then answer.
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
# Strategy 2b: multi-turn dialogues for overconfidence questions
# ---------------------------------------------------------------------------
async def generate_multi_turn_overconfidence_training_data(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
    max_ask_loops: int = 4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Overconfidence scenario: the original question is assertive but contains incorrect conditions; the strategy must detect and correct them.
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
# Strategy 3: direct answer & correct
# ---------------------------------------------------------------------------
async def strategy_direct_answer_and_correct(
    api_client: CustomAPI,
    data: List[Dict[str, Any]],
    templates: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Strategy: answer first, then judge; if incorrect, reconstruct using the reference answer (and solution if provided).

    1) LLM generates an initial answer for ori_question.
    2) A judge LLM decides whether the answer is correct.
    3) If correct, build the final dialogue and finish.
    4) If incorrect, reconstruct a "perfect" answer using expected_answer (+ solution if present), then build the final dialogue.
    """
    print("\n============================================================")
    print("Starting strategy: strategy_direct_answer_and_correct")
    print(f"Initial items: {len(data)}")
    print("============================================================\n")

    completed_items = []
    total_discarded_items = []

    # --- Step 1: generate initial answers ---
    answer_results, failed_answer = await _run_batch_step_with_retry(
        api_client, data, "Step 1: generate initial answers",
        templates['template_direct_answer'],
        ['ori_question'],
        lambda r: (r, None),
        {'max_tokens': 8192, 'temperature': 0.7},
        max_retries=2
    )
    total_discarded_items.extend(failed_answer)
    if not answer_results:
        print("All items failed at step 1; aborting.")
        return [], total_discarded_items

    items_to_judge = []
    for item, generated_answer in answer_results:
        item['generated_answer'] = generated_answer
        items_to_judge.append(item)

    # --- Step 2: judge answers ---
    judge_results, failed_judge = await _run_batch_step_with_retry(
        api_client, items_to_judge, "Step 2: judge answers",
        templates['template_judge_direct_answer'],
        ['ori_question', 'generated_answer', 'expected_answer'],
        _parse_judge_json_response,
        {'max_tokens': 1024, 'temperature': 0.0},
        max_retries=2
    )
    total_discarded_items.extend(failed_judge)
    if not judge_results:
        print("All items failed at step 2; aborting.")
        return completed_items, total_discarded_items

    # --- Step 3: route by judgement ---
    items_to_reconstruct = []
    correct_count = 0
    for item, judgement in judge_results:
        if judgement['is_correct']:
            # Correct: build final dialogue directly.
            item['conversation_history'] = [
                {"role": "user", "content": item['ori_question']},
                {"role": "assistant", "content": item['generated_answer']}
            ]
            completed_items.append(item)
            correct_count += 1
        else:
            # Incorrect: queue for reconstruction.
            items_to_reconstruct.append(item)

    print("--- Judgement summary ---")
    print(f"  - correct and completed: {correct_count}")
    print(f"  - incorrect; to reconstruct: {len(items_to_reconstruct)}")

    # --- Step 4: reconstruct incorrect answers ---
    if items_to_reconstruct:
        # Dynamically build prompt fields, adding solution (if present).
        for item in items_to_reconstruct:
            if item.get('solution'):
                item['solution_section'] = f"\n# Detailed Solution for Reference:\n{item['solution']}\n"
            else:
                item['solution_section'] = ""  # ensure the key exists

        reconstruct_results, failed_reconstruct = await _run_batch_step_with_retry(
            api_client, items_to_reconstruct, "Step 4: reconstruct incorrect answers",
            templates['template_reconstruct_answer'],
            ['ori_question', 'generated_answer', 'expected_answer', 'solution_section'],
            lambda r: (r, None),
            {'max_tokens': 8192, 'temperature': 0.5},  # slightly lower temperature for answer quality
            max_retries=2
        )
        total_discarded_items.extend(failed_reconstruct)

        for item, reconstructed_answer in reconstruct_results:
            # Build final dialogue using the reconstructed answer.
            item['conversation_history'] = [
                {"role": "user", "content": item['ori_question']},
                {"role": "assistant", "content": reconstructed_answer}
            ]
            completed_items.append(item)
        
        print("--- Reconstruction summary ---")
        print(f"  - reconstructed and completed: {len(reconstruct_results)}")
        print(f"  - reconstruction failed/dropped: {len(failed_reconstruct)}")

    # --- Cleanup & summary ---
    for item in completed_items:
        item.pop('generated_answer', None)
        item.pop('solution_section', None)

    print("\n============================================================")
    print("All items processed.")
    print("Final stats:")
    print(f"  - initial items: {len(data)}")
    print(f"  - completed dialogues: {len(completed_items)}")
    print(f"  - dropped items: {len(total_discarded_items)}")
    print("============================================================\n")

    return completed_items, total_discarded_items
