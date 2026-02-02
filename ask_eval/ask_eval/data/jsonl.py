# ask_eval/data/jsonl.py
import json
from typing import List, Dict
import os
from .base import BaseDataLoader

class JsonlLoader(BaseDataLoader):
    """JSONL file loader."""
    def __init__(self, data_dir: str, filename: str = "test.jsonl"):
        super().__init__(data_dir)
        self.filename = filename

    def load_data(self) -> List[Dict]:
        file_path = os.path.join(self.data_dir, self.filename)
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
