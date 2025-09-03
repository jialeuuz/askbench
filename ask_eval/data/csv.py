# ask_eval/data/csv.py
import pandas as pd
from typing import List, Dict, Optional
import os
from .base import BaseDataLoader

class CsvLoader(BaseDataLoader):
    """CSV文件基础加载器"""
    def __init__(self, data_dir: str, filename: str = "test.csv"):
        super().__init__(data_dir)
        self.filename = filename

    def load_data(self) -> List[Dict]:
        file_path = os.path.join(self.data_dir, self.filename)
        df = pd.read_csv(file_path)
        return df.to_dict('records')

class MultiCsvLoader(BaseDataLoader):
    """多CSV文件加载器"""
    def __init__(self, data_dir: str, filenames: Optional[List[str]] = None):
        super().__init__(data_dir)
        self.filenames = filenames

    def _get_all_csv_files(self) -> List[str]:
        """获取目录下所有CSV文件"""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

    def load_data(self) -> Dict[str, List[Dict]]:
        """加载所有CSV文件"""
        filenames = self.filenames or self._get_all_csv_files()
        all_data = {}
        
        for filename in filenames:
            name = os.path.splitext(filename)[0]  # 移除.csv后缀
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                all_data[name] = df.to_dict('records')
                
        return all_data