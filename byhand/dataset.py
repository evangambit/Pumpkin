import torch
from torch.utils.data import IterableDataset
import os
import glob
from typing import List

# Ensure the extension is built via setup.py
try:
    from _byhand_dataset import ChunkedDataset, feature_name
except ImportError:
    raise ImportError("C++ extension _byhand_dataset not found. Please run 'python setup.py build_ext --inplace --force' in the byhand directory first.")

class ByHandDataset(IterableDataset):
    def __init__(self, file_paths: List[str], chunk_size: int = 128, total_lines: int = None):
        super().__init__()
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.total_lines = total_lines
    
    def feature_name(self, index: int) -> str:
        return feature_name(index)

    def __len__(self):
        if self.total_lines is None:
            self.total_lines = 0
            import subprocess
            for path in self.file_paths:
                output = subprocess.check_output(['wc', '-l', path])
                self.total_lines += int(output.split()[0])
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
