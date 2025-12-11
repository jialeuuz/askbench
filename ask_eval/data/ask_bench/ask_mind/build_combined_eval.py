import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "test.jsonl"
SAMPLE_PER_SOURCE = 100
SHUFFLE_SEED = 20250301

SOURCES = [
    ("ask_mind_math500de", ROOT / "ask_mind_math500de" / "test.jsonl"),
    ("ask_mind_medqade", ROOT / "ask_mind_medqade" / "test.jsonl"),
    ("ask_mind_gpqade", ROOT / "ask_mind_gpqade" / "test.jsonl"),
    ("ask_mind_bbhde", ROOT / "ask_mind_bbhde" / "test.jsonl"),
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    combined = []

    for idx, (name, path) in enumerate(SOURCES):
        if not path.exists():
            raise FileNotFoundError(f"数据源不存在: {path}")
        data = load_jsonl(path)
        if len(data) < SAMPLE_PER_SOURCE:
            raise ValueError(f"{name} 数据不足 {SAMPLE_PER_SOURCE} 条（当前 {len(data)} 条）")

        rng = random.Random(SHUFFLE_SEED + idx)
        picked = rng.sample(data, SAMPLE_PER_SOURCE)
        for sample in picked:
            item = dict(sample)
            item["source_task"] = name
            combined.append(item)

    random.Random(SHUFFLE_SEED).shuffle(combined)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"生成完成: {OUTPUT_PATH} （总计 {len(combined)} 条）")


if __name__ == "__main__":
    main()
