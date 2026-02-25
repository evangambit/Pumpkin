"""
NNUE embedding layer.

This is the accumulator in c++, but in PyTorch we
implement it as a (non-incremental) embedding layer.
"""

import torch
from torch import nn
from features import kRoughEstimateOfNumberOfOnesInInput


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

    s = (1.0 / kRoughEstimateOfNumberOfOnesInInput) ** 0.5    

    # Factorized version of self.tiles for increased regularization and faster convergence.
    self.pieces = nn.Parameter(torch.randn(Emb.k, 1, 1, dout) * s)
    self.ranks = nn.Parameter(torch.randn(Emb.k, 8, 1, dout) * s)
    self.files = nn.Parameter(torch.randn(Emb.k, 1, 8, dout) * s)

    # Full rank representation so we don't lose expressivity.
    self.tiles = nn.Parameter(torch.randn(Emb.k, 8, 8, dout) * s)

    # TODO: we don't use padding anymore, so we can remove this.
    self.zeros = nn.Parameter(torch.zeros(1, dout), requires_grad=False)

    # We don't want to apply pawn-level factorization to 1st and 8th rank embeddings, since
    # these embeddings are used for non-pawn things (castling, empty files, etc.).
    self.factorization_mask = nn.Parameter(torch.ones(Emb.k, 8, 8, 1), requires_grad=False)
    with torch.no_grad():
      i, j = 0, Emb.k // 2
      self.factorization_mask[i,0,:,:] = 0.0
      self.factorization_mask[i,7,:,:] = 0.0
      self.factorization_mask[j,0,:,:] = 0.0
      self.factorization_mask[j,7,:,:] = 0.0
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

  def weight(self, merged_tiles):
    return torch.cat([merged_tiles.reshape(Emb.k * 8 * 8, -1), self.zeros], dim=0)

  def flipped_weight(self, merged_tiles):
    flipped_tiles = self.flip_vertical(merged_tiles)
    return torch.cat([flipped_tiles.reshape(Emb.k * 8 * 8, -1), self.zeros], dim=0)
  
  def merged_tiles(self):
    tiles = self.tiles + (self.pieces + self.ranks + self.files) * self.factorization_mask
    return tiles
  
  def forward(self, values, lengths):
    merged_tiles = self.merged_tiles()
    w = self.weight(merged_tiles)
    flipped = self.flipped_weight(merged_tiles)
    a = self.bagger(w[values.to(torch.int64)], lengths.to(torch.int64))
    b = self.bagger(flipped[values.to(torch.int64)], lengths.to(torch.int64))
    return a.clip(0, 1), b.clip(0, 1)
Emb.k = 24
