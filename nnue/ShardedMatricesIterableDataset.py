import numpy as np
import torch.utils.data as tdata
from sharded_matrix import ShardedLoader

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
  def __init__(self, *paths, chunk_size=1):
    super().__init__()
    self.iterators = [SingleShardedMatrixIterator(p, chunk_size=chunk_size) for p in paths]
  
  def __iter__(self):
    its = [iter(it) for it in self.iterators]
    yield from zip(*its)
  
  def __len__(self):
    return len(self.iterators[0])