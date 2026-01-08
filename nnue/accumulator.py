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
    self.tiles = nn.Parameter(torch.randn(12, 8, 8, dout) * s)
    self.zeros = nn.Parameter(torch.zeros(1, dout), requires_grad=False)
  
  def zero_(self):
    with torch.no_grad():
      self.tiles.zero_()

  def weight(self, vertically_flipped=False):
    """
    Return the embedding weight matrix.

    We could implement this whole class as as an nn.Embedding,
    but this way we can
    # 1) support vertical flipping
    # 2) apply higher regularization (theoretically at least) 

    Shape: (768 + 1, dout)
    """
    tiles = self.tiles
    if vertically_flipped:
      # Flip the pieces (White <-> Black) and the ranks.
      tiles = tiles.flip(1).roll(6, dims=0)

    batch_size = tiles.shape[0]
    return torch.cat([tiles.reshape(768, -1), self.zeros], dim=0)
  
  def forward(self, x, vertically_flipped=False):
    assert len(x.shape) == 2
    assert x.shape[1] == kMaxNumOnesInInput, f"x.shape={x.shape}"
    return self.weight(vertically_flipped=vertically_flipped)[x.to(torch.int32)].sum(1)
