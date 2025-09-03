# ask_eval/data/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import os

class BaseDataLoader(ABC):
    """数据加载器基类"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @abstractmethod
    def load_data(self) -> List[Dict]:
        """加载数据"""
        pass