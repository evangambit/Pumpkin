import numpy as np
import torch.utils.data as tdata
from torch import nn
from sharded_matrix import ShardedLoader

class DynamicShardedMatrixIterator:
  def __init__(self, path, chunk_size=1):
    self.path = path
    self.chunk_size = chunk_size
    with open(path + "-lengths", "rb") as f:
      f.seek(0, 2)
      self._length = f.tell() // 2

  def __iter__(self):
    self.values_file = open(self.path + "-values", "rb")
    self.lengths_file = open(self.path + "-lengths", "rb")
    while True:
      length_bytes = self.lengths_file.read(self.chunk_size * 2)
      if not length_bytes:
        break
      lengths = np.frombuffer(length_bytes, dtype=np.int16)
      values = np.frombuffer(self.values_file.read(lengths.sum() * 2), dtype=np.int16)
      yield values, lengths
  
  def __len__(self):
    return self._length // self.chunk_size

class SingleShardedMatrixIterator:
  def __init__(self, xpath, chunk_size=1):
    self.X = ShardedLoader(xpath)
    self.chunk_size = chunk_size

  def __iter__(self):
    shard_index = 0
    offset = 0
    x = self.X.load_shard(shard_index)
    while True:
      if offset + self.chunk_size >= x.shape[0]:
        first_half = x[offset:]
        shard_index += 1
        if shard_index >= self.X.num_shards:
          break
        x = self.X.load_shard(shard_index)
        second_half = x[0:self.chunk_size - first_half.shape[0]]
        yield np.concatenate([first_half, second_half], axis=0).copy()
        offset = self.chunk_size - first_half.shape[0]
      else:
        yield x[offset:offset + self.chunk_size].copy()
        offset += self.chunk_size
  
  def __len__(self):
    return self.X.num_rows // self.chunk_size

class ShardedMatricesIterableDataset(tdata.IterableDataset):
  def __init__(self, *iterators):
    super().__init__()
    self.iterators = iterators
  
  def __iter__(self):
    its = [iter(it) for it in self.iterators]
    yield from zip(*its)
  
  def __len__(self):
    return len(self.iterators[0])