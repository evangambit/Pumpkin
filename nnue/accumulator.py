"""
NNUE embedding layer.

This is the accumulator in c++, but in PyTorch we
implement it as a (non-incremental) embedding layer.
"""

import torch
from torch import nn
from features import kRoughEstimateOfNumberOfOnesInInput


import torch.nn.functional as F

# If you change this, make sure to also change
# src/eval/nnue/NnueFeatureBitmapType.h and
# re-train the model.
kKingBuckets = torch.tensor([
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  3, 3, 0, 0, 1, 0, 2, 2,
], dtype=torch.int64)
kNumKingBuckets = kKingBuckets.max().item() + 1
# </end of things that need to be changed if kKingBuckets is changed>

class Emb(nn.Module):
  def __init__(self, dout):
    super().__init__()

    s = (1.0 / kRoughEstimateOfNumberOfOnesInInput) ** 0.5    

    # Factorized version of self.tiles for increased regularization and faster convergence.
    # Shape: (64 king squares, k piece types, rank, file, dout)
    self.pieces = nn.Parameter(torch.randn(kNumKingBuckets, Emb.k, 1, 1, dout) * s)
    self.ranks = nn.Parameter(torch.randn(kNumKingBuckets, Emb.k, 8, 1, dout) * s)
    self.files = nn.Parameter(torch.randn(kNumKingBuckets, Emb.k, 1, 8, dout) * s)
    self.noking = nn.Parameter(torch.randn(1, Emb.k, 8, 8, dout) * s)

    # Full rank representation so we don't lose expressivity.
    self.tiles = nn.Parameter(torch.randn(kNumKingBuckets, Emb.k, 8, 8, dout) * s)

    # We don't want to apply pawn-level factorization to 1st and 8th rank embeddings, since
    # these embeddings are used for non-pawn things (castling, empty files, etc.).
    self.factorization_mask = nn.Parameter(torch.ones(kNumKingBuckets, Emb.k, 8, 8, 1), requires_grad=False)
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
      self.noking.zero_()
      self.tiles.zero_()
  
  @staticmethod
  def flip_values(values):
    # Flip feature indices: swap piece colors and flip squares vertically.
    piece_type = values // 64
    square = values % 64
    flipped_piece = (piece_type + Emb.k // 2) % Emb.k
    flipped_square = (7 - square // 8) * 8 + square % 8
    return flipped_piece * 64 + flipped_square

  def weight(self, merged_tiles):
    return merged_tiles.reshape(kNumKingBuckets * Emb.k * 64, -1)
  
  def merged_tiles(self):
    tiles = self.tiles + (self.pieces + self.ranks + self.files) * self.factorization_mask + self.noking
    return tiles
  
  def forward(self, values, lengths, kings):
    merged_tiles = self.merged_tiles()
    w = self.weight(merged_tiles)

    # Convert lengths to offsets for embedding_bag
    # lengths: [3, 2, 4] -> offsets: [0, 3, 5]
    offsets = lengths.cumsum(0) - lengths

    # kings shape: [batch, 2] where [:,0]=mover king, [:,1]=waiter king
    mover_king = kings[:, 0].to(torch.int64)   # [batch]
    waiter_king = kings[:, 1].to(torch.int64)  # [batch]
    buckets = kKingBuckets.to(mover_king.device)
    mover_bucket = buckets[mover_king]  # [batch]
    # Flip waiter's king to their perspective before bucket lookup
    waiter_king_flipped = (7 - waiter_king // 8) * 8 + waiter_king % 8
    waiter_bucket = buckets[waiter_king_flipped]  # [batch]

    # Expand king buckets to per-feature level
    lengths_i64 = lengths.to(torch.int64)
    mover_bucket_exp = torch.repeat_interleave(mover_bucket, lengths_i64)  # [total_features]
    waiter_bucket_exp = torch.repeat_interleave(waiter_bucket, lengths_i64)  # [total_features]

    piece_dim = Emb.k * 64  # 640
    values_i64 = values.to(torch.int64)
    mover_values = mover_bucket_exp * piece_dim + values_i64
    # Flip feature values for waiter perspective (swap colors + flip squares)
    waiter_values = waiter_bucket_exp * piece_dim + Emb.flip_values(values_i64)

    a = F.embedding_bag(mover_values, w, offsets=offsets.to(torch.int64), mode='sum')
    b = F.embedding_bag(waiter_values, w, offsets=offsets.to(torch.int64), mode='sum')
    return a.clip(0, 1), b.clip(0, 1)
Emb.k = 10
