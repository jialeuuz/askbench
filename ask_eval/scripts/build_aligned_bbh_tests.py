import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{idx}: {e}") from e
    return rows


Key = Tuple[str, str]


def row_key(row: Dict) -> Key:
    q = row.get("ori_question")
    a = row.get("expected_answer")
    if not q or a is None:
        raise ValueError("Each row must contain `ori_question` and `expected_answer`")
    return (str(q), str(a))


def build_unique_map(rows: Sequence[Dict], *, name: str) -> Dict[Key, Dict]:
    mapping: Dict[Key, Dict] = {}
    for row in rows:
        key = row_key(row)
        existing = mapping.get(key)
        if existing is not None:
            continue
        mapping[key] = row
    return mapping


def ordered_unique_keys(rows: Sequence[Dict]) -> List[Key]:
    keys: List[Key] = []
    seen = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample aligned BBH test sets by the same (ori_question, expected_answer) keys, "
            "and write them to the three corresponding test.jsonl files."
        )
    )
    parser.add_argument("--k", type=int, default=1000, help="Number of samples to pick")
    parser.add_argument("--seed", type=int, default=20250301, help="Random seed for sampling")
    parser.add_argument(
        "--common-ori",
        type=Path,
        default=Path("ask_eval/data/common/bbh/test_ori.jsonl"),
        help="Input: common BBH test_ori.jsonl",
    )
    parser.add_argument(
        "--mind-ori",
        type=Path,
        default=Path("ask_eval/data/ask_bench/ask_mind/ask_mind_bbhde/test_ori.jsonl"),
        help="Input: ask_mind_bbhde test_ori.jsonl",
    )
    parser.add_argument(
        "--over-ori",
        type=Path,
        default=Path("ask_eval/data/ask_bench/ask_overconfidence/ask_overconfidence_bbh/test_ori.jsonl"),
        help="Input: ask_overconfidence_bbh test_ori.jsonl",
    )
    parser.add_argument(
        "--common-out",
        type=Path,
        default=Path("ask_eval/data/common/bbh/test.jsonl"),
        help="Output: common BBH test.jsonl",
    )
    parser.add_argument(
        "--mind-out",
        type=Path,
        default=Path("ask_eval/data/ask_bench/ask_mind/ask_mind_bbhde/test.jsonl"),
        help="Output: ask_mind_bbhde test.jsonl",
    )
    parser.add_argument(
        "--over-out",
        type=Path,
        default=Path("ask_eval/data/ask_bench/ask_overconfidence/ask_overconfidence_bbh/test.jsonl"),
        help="Output: ask_overconfidence_bbh test.jsonl",
    )
    parser.add_argument(
        "--keys-out",
        type=Path,
        default=None,
        help="Optional: write picked keys as JSON (list of [ori_question, expected_answer])",
    )
    args = parser.parse_args()

    common_rows = load_jsonl(args.common_ori)
    mind_rows = load_jsonl(args.mind_ori)
    over_rows = load_jsonl(args.over_ori)

    common_map = build_unique_map(common_rows, name="common")
    mind_map = build_unique_map(mind_rows, name="ask_mind_bbhde")
    over_map = build_unique_map(over_rows, name="ask_overconfidence_bbh")

    common_keys = set(common_map.keys())
    mind_keys = set(mind_map.keys())
    over_keys = set(over_map.keys())

    intersect_keys = common_keys & mind_keys & over_keys
    if len(intersect_keys) < args.k:
        raise ValueError(
            f"Not enough intersected keys to sample: intersection={len(intersect_keys)} < k={args.k}"
        )

    base_keys = [k for k in ordered_unique_keys(over_rows) if k in intersect_keys]
    if len(base_keys) < args.k:
        raise ValueError(
            f"Overconfidence base keys after intersection is insufficient: {len(base_keys)} < k={args.k}"
        )

    rng = random.Random(args.seed)
    picked_set = set(rng.sample(base_keys, args.k))
    picked_keys = [k for k in base_keys if k in picked_set]

    common_out_rows = [common_map[k] for k in picked_keys]
    mind_out_rows = [mind_map[k] for k in picked_keys]
    over_out_rows = [over_map[k] for k in picked_keys]

    write_jsonl(args.common_out, common_out_rows)
    write_jsonl(args.mind_out, mind_out_rows)
    write_jsonl(args.over_out, over_out_rows)

    if args.keys_out is not None:
        args.keys_out.parent.mkdir(parents=True, exist_ok=True)
        with args.keys_out.open("w", encoding="utf-8") as f:
            json.dump([list(k) for k in picked_keys], f, ensure_ascii=False, indent=2)

    print(
        "Done. "
        f"common={args.common_out} mind={args.mind_out} over={args.over_out} "
        f"(k={len(picked_keys)}, seed={args.seed}, intersection={len(intersect_keys)})"
    )


if __name__ == "__main__":
    main()
