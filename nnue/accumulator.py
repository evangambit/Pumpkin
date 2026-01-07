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

    # Roughly speaking
    # variance(z[i]) = (var(t1) + var(t2) + var(t3) + var(t4) + var(t5) + var(t6)) * kMaxNumOnesInInput
    # So if we want variance(z[i]) = 1.0, we want each var(ti) = 1.0 / (kMaxNumOnesInInput * 6)
    num_components = 6
    targeted_variance_per_component = 1.0
    s = (targeted_variance_per_component / (kMaxNumOnesInInput * num_components)) ** 0.5
    
    self.tiles = nn.Parameter(torch.randn(12, 8, 8, dout) * s)
    self.coord = nn.Parameter(torch.randn(1, 8, 8, dout) * s)
    self.piece = nn.Parameter(torch.randn(12, 1, 1, dout) * s)
    self.row = nn.Parameter(torch.randn(1, 8, 1, dout) * s)
    self.col = nn.Parameter(torch.randn(1, 1, 8, dout) * s)
    self.tilecolor = nn.Parameter(torch.randn(1, 1, 1, dout) * s)

    # We juse pawns on the first/last rank to indicate things like castling rights,
    # so we don't want coord, piece, row, col, tilecolor to apply to them. We do,
    # however, still want to flip them in the vertically_flipped case.
    self.special_mask = nn.Parameter(torch.zeros(12, 8, 8, dout), requires_grad=False)
    with torch.no_grad():
      for piece in range(12):
        for y in range(8):
          for x in range(8):
            if (piece % 6 == 0) and (y == 0 or y == 7):
              self.special_mask[piece, y, x, :] = 1.0

    self.white_tile_mask = nn.Parameter(torch.zeros(1, 8, 8, 1), requires_grad=False)
    with torch.no_grad():
      for y in range(8):
        for x in range(8):
          self.white_tile_mask[0, y, x, 0] = (y + x) % 2 == 0
    
    self.zeros = nn.Parameter(torch.zeros(1, dout), requires_grad=False)
  
  def zero_(self):
    with torch.no_grad():
      self.tiles.zero_()
      self.coord.zero_()
      self.piece.zero_()
      self.row.zero_()
      self.col.zero_()
      self.tilecolor.zero_()

  def weight(self, vertically_flipped=False):
    """
    Return the embedding weight matrix.

    We could implement this whole class as as an nn.Embedding,
    but this way we can
    # 1) support vertical flipping
    # 2) apply higher regularization (theoretically at least) 

    Shape: (768 + 1, dout)
    """
    tilecolor = self.tilecolor * self.white_tile_mask
    
    factorized_terms = (
      self.coord
      +
      self.piece
      +
      self.row
      +
      self.col
      +
      tilecolor
    ) * (1.0 - self.special_mask)

    T = factorized_terms + self.tiles

    if vertically_flipped:
      # Flip the pieces (White <-> Black) and the ranks.
      T = T.flip(1).roll(6, dims=0)

    batch_size = T.shape[0]
    return torch.cat([T.reshape(768, -1), self.zeros], dim=0)
  
  def forward(self, x, vertically_flipped=False):
    assert len(x.shape) == 2
    assert x.shape[1] == kMaxNumOnesInInput, f"x.shape={x.shape}"
    return self.weight(vertically_flipped=vertically_flipped)[x.to(torch.int32)].sum(1)
