"""
NNUE embedding layer.

This is the accumulator in c++, but in PyTorch we
implement it as a (non-incremental) embedding layer.
"""

import torch
from torch import nn
from features import kMaxNumOnesInInput

class Emb(nn.Module):
  def __init__(self, dout):
    super().__init__()
    din = 768

    s = (1.0 / kMaxNumOnesInInput) ** 0.5    
    self.pieces = nn.Parameter(torch.randn(12, 1, 1, dout) * s)
    self.ranks = nn.Parameter(torch.randn(1, 8, 1, dout) * s)
    self.files = nn.Parameter(torch.randn(1, 1, 8, dout) * s)
    self.tiles = nn.Parameter(torch.randn(12, 8, 8, dout) * s)
    self.zeros = nn.Parameter(torch.zeros(1, dout), requires_grad=False)
  
  def zero_(self):
    with torch.no_grad():
      self.pieces.zero_()
      self.ranks.zero_()
      self.files.zero_()
      self.tiles.zero_()
  
  @staticmethod
  def flip_vertical(x):
    # Flip the pieces (White <-> Black) and the ranks.
    return x.flip(1).roll(6, dims=0)

  def weight(self):
    tiles = self.tiles + self.pieces + self.ranks + self.files
    return torch.cat([tiles.reshape(768, -1), self.zeros], dim=0)
  
  def forward(self, x):
    assert len(x.shape) == 2
    assert x.shape[1] == kMaxNumOnesInInput, f"x.shape={x.shape}"
    w = self.weight()
    flipped = self.flip_vertical(w)
    return w[x.to(torch.int32)].sum(1), flipped[x.to(torch.int32)].sum(1)
