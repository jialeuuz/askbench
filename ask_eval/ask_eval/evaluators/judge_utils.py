import json
import re
from typing import Dict

# Shared retry limit for parsing judge outputs.
MAX_JUDGE_JSON_RETRIES = 10


def parse_json_to_dict(response_text: str) -> Dict:
    """
    Extract the first JSON object from a judge model response.
    Responses are expected to optionally start with a `Reasoning:` line
    followed by a fenced ```json block. We fall back to parsing the raw
    string when no fenced block is present.
    """
    if not response_text:
        return {}

    match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text)
    if match:
        json_payload = match.group(1).strip()
    else:
        json_payload = response_text.strip()

    try:
        return json.loads(json_payload)
    except (json.JSONDecodeError, TypeError):
        return {}
