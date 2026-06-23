import torch
from torch.utils.data import IterableDataset

import os
from typing import List
import subprocess
import sqlite3

# Ensure the extension is built via setup.py
try:
    from _nnue_dataset import ChunkedDataset
except ImportError:
    raise ImportError("C++ extension _nnue_dataset not found. Please run 'python setup.py build_ext --inplace' in the nnue directory first.")

def count_lines_in_file(file_path: str) -> int:
    output = subprocess.check_output(['wc', '-l', file_path])
    return int(output.split()[0])

class NnueDataset(IterableDataset):
    def __init__(self, file_paths: List[str], chunk_size: int = 128):
        super().__init__()
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.total_lines = 0
        with sqlite3.connect('wl.db') as conn:
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS file_line_counts (file_path TEXT PRIMARY KEY, line_count INTEGER, modified_time INTEGER)')
            for file_path in self.file_paths:
                modified_time = os.path.getmtime(file_path)
                cursor.execute('SELECT line_count, modified_time FROM file_line_counts WHERE file_path = ?', (file_path,))
                row = cursor.fetchone()
                if row is None or row[1] != modified_time:
                    print(f"Counting lines in {file_path}...")
                    line_count = count_lines_in_file(file_path)
                    cursor.execute('REPLACE INTO file_line_counts (file_path, line_count, modified_time) VALUES (?, ?, ?)', (file_path, line_count, modified_time))
                    conn.commit()
                    self.total_lines += line_count
                else:
                    self.total_lines += row[0]
                
    def __len__(self):
        return (self.total_lines + self.chunk_size - 1) // self.chunk_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        assigned_files = self.file_paths
        
        # Split files across workers if using DataLoader
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            assigned_files = [f for i, f in enumerate(self.file_paths) if i % num_workers == worker_id]
            
        if not assigned_files:
            return iter([])
            
        return iter(ChunkedDataset(assigned_files, self.chunk_size))
