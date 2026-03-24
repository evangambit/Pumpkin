"""
NNUE embedding layer.

This is the accumulator in c++, but in PyTorch we
implement it as a (non-incremental) embedding layer.
"""

import torch
from torch import nn
from features import kRoughEstimateOfNumberOfOnesInInput


import torch.nn.functional as F

class Emb(nn.Module):
  def __init__(self, dout):
    super().__init__()

    s = (1.0 / kRoughEstimateOfNumberOfOnesInInput) ** 0.5    

    # Factorized version of self.tiles for increased regularization and faster convergence.
    # Shape: (64 king squares, k piece types, rank, file, dout)
    self.pieces = nn.Parameter(torch.randn(64, Emb.k, 1, 1, dout) * s)
    self.ranks = nn.Parameter(torch.randn(64, Emb.k, 8, 1, dout) * s)
    self.files = nn.Parameter(torch.randn(64, Emb.k, 1, 8, dout) * s)

    # Full rank representation so we don't lose expressivity.
    self.tiles = nn.Parameter(torch.randn(64, Emb.k, 8, 8, dout) * s)

    # We don't want to apply pawn-level factorization to 1st and 8th rank embeddings, since
    # these embeddings are used for non-pawn things (castling, empty files, etc.).
    self.factorization_mask = nn.Parameter(torch.ones(64, Emb.k, 8, 8, 1), requires_grad=False)
    with torch.no_grad():
      i, j = 0, Emb.k // 2
      self.factorization_mask[:,i,0,:,:] = 0.0
      self.factorization_mask[:,i,7,:,:] = 0.0
      self.factorization_mask[:,j,0,:,:] = 0.0
      self.factorization_mask[:,j,7,:,:] = 0.0

  
  def zero_(self):
    with torch.no_grad():
      self.pieces.zero_()
      self.ranks.zero_()
      self.files.zero_()
      self.tiles.zero_()
  
  @staticmethod
  def flip_vertical(x):
    # Flip the pieces (White <-> Black), the ranks, and the king square (vertically).
    # x shape: (64, k, 8, 8, dout)
    # Flip pieces and ranks
    x = x.flip(2).roll(Emb.k // 2, dims=1)
    # Flip king square vertically: sq -> (7 - sq//8)*8 + sq%8
    perm = torch.arange(64)
    perm = (7 - perm // 8) * 8 + perm % 8
    return x[perm]

  def weight(self, merged_tiles):
    return merged_tiles.reshape(64 * Emb.k * 64, -1)

  def flipped_weight(self, merged_tiles):
    flipped_tiles = self.flip_vertical(merged_tiles)
    return flipped_tiles.reshape(64 * Emb.k * 64, -1)
  
  def merged_tiles(self):
    tiles = self.tiles + (self.pieces + self.ranks + self.files) * self.factorization_mask
    return tiles
  
  def forward(self, values, lengths, kings):
    merged_tiles = self.merged_tiles()
    w = self.weight(merged_tiles)
    flipped = self.flipped_weight(merged_tiles)

    # Convert lengths to offsets for embedding_bag
    # lengths: [3, 2, 4] -> offsets: [0, 3, 5]
    offsets = lengths.cumsum(0) - lengths

    # HalfKP: offset piece indices by king square
    # kings shape: [batch, 2] where [:,0]=mover king, [:,1]=waiter king
    mover_king = kings[:, 0].to(torch.int64)   # [batch]
    waiter_king = kings[:, 1].to(torch.int64)  # [batch]

    # Expand king squares to per-feature level
    mover_king_exp = torch.repeat_interleave(mover_king, lengths.to(torch.int64))  # [total_features]
    waiter_king_exp = torch.repeat_interleave(waiter_king, lengths.to(torch.int64))  # [total_features]

    piece_dim = Emb.k * 64  # 640
    mover_values = mover_king_exp * piece_dim + values.to(torch.int64)
    waiter_values = waiter_king_exp * piece_dim + values.to(torch.int64)

    a = F.embedding_bag(mover_values, w, offsets=offsets.to(torch.int64), mode='sum')
    b = F.embedding_bag(waiter_values, flipped, offsets=offsets.to(torch.int64), mode='sum')
    return a.clip(0, 1), b.clip(0, 1)
Emb.k = 10
