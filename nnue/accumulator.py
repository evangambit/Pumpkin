"""
NNUE embedding layer.

This is the accumulator in c++, but in PyTorch we
implement it as a (non-incremental) embedding layer.
"""

import torch
from torch import nn
from features import kMaxNumOnesInInput


class SumBag(nn.Module):
  """
  Similar to EmbeddingBag but without the actual embedding lookup. Instead,
  we assume you've already done the embedding lookup and are passing in the values directly.
  """
  def forward(self, values: torch.Tensor, lengths: torch.Tensor):
    assert values.dim() == 2
    assert lengths.dim() == 1
    assert lengths.sum() == values.shape[0]
    indicies = torch.repeat_interleave(
        torch.arange(len(lengths), device=values.device), 
        lengths
    )
    out = torch.zeros(len(lengths), values.shape[1], device=values.device, dtype=values.dtype)
    out.index_add_(0, indicies, values)    
    return out

class Emb(nn.Module):
  def __init__(self, dout):
    super().__init__()
    din = Emb.k * 8 * 8

    s = (1.0 / kMaxNumOnesInInput) ** 0.5    
    self.pieces = nn.Parameter(torch.randn(Emb.k, 1, 1, dout) * s)
    self.ranks = nn.Parameter(torch.randn(Emb.k, 8, 1, dout) * s)
    self.files = nn.Parameter(torch.randn(Emb.k, 1, 8, dout) * s)
    self.tiles = nn.Parameter(torch.randn(Emb.k, 8, 8, dout) * s)
    self.zeros = nn.Parameter(torch.zeros(1, dout), requires_grad=False)
    self.bagger = SumBag()
  
  def zero_(self):
    with torch.no_grad():
      self.pieces.zero_()
      self.ranks.zero_()
      self.files.zero_()
      self.tiles.zero_()
  
  @staticmethod
  def flip_vertical(x):
    # Flip the pieces (White <-> Black) and the ranks.
    return x.flip(1).roll(Emb.k // 2, dims=0)

  def weight(self):
    tiles = self.tiles + self.pieces + self.ranks + self.files
    return torch.cat([tiles.reshape(Emb.k * 8 * 8, -1), self.zeros], dim=0)
  
  def forward(self, values, lengths):
    w = self.weight()
    flipped = self.flip_vertical(w)
    a = self.bagger(w[values.to(torch.int64)], lengths.to(torch.int64))
    b = self.bagger(flipped[values.to(torch.int64)], lengths.to(torch.int64))
    return a, b
Emb.k = 14
