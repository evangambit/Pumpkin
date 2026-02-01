"""
NNUE feature representation conversion functions.
"""

import chess
import numpy as np

kMaxNumOnesInInput = 32 + 4

def x2board(vec, turn):
  assert len(vec.shape) == 1
  assert vec.shape[0] == kMaxNumOnesInInput
  board = chess.Board.empty()
  board.turn = turn
  castling = ''
  for val in vec:
    if val >= 768:
      continue
    if (val >= 7 and val < 56) or val > 63:
      piece = val // 64
      val = val % 64
      y = 7 - val // 8
      x = val % 8
      if not ((piece == 0 or piece == 6) and (y == 0 or y == 7)):
        # Ignore pawns on first or last rank, since these are used
        # for special features (see castling rights below).
        sq = chess.square(x, y)
        board.set_piece_at(sq, chess.Piece(piece % 6 + 1, chess.WHITE if piece < 6 else chess.BLACK))
        continue
    if val == 0:
      castling += 'K'
    if val == 1:
      castling += 'Q'
    if val == 440:
      castling += 'k'
    if val == 441:
      castling += 'q'
  # board.set_castling_fen(castling) doesn't work?
  parts = board.fen().split(' ')
  parts[2] = castling if castling else '-'
  return chess.Board(' '.join(parts))

def board2x(board):
  vec = np.ones(kMaxNumOnesInInput, dtype=np.int16) * 768
  i = 0
  for sq, piece in board.piece_map().items():
    val = 0
    val += 'PNBRQKpnbrqk'.index(piece.symbol()) * 64
    val += 8 * (7 - sq // 8)
    val += sq % 8
    assert val not in [0, 1, 440, 441], f"Piece on square {chess.square_name(sq)} conflicts with castling rights encoding."
    assert val < 768
    vec[i] = val
    i += 1
  if board.has_kingside_castling_rights(chess.WHITE):
    vec[i] = 0  # White pawn on a1
    i += 1
  if board.has_queenside_castling_rights(chess.WHITE):
    vec[i] = 1  # White pawn on b1
    i += 1
  if board.has_kingside_castling_rights(chess.BLACK):
    vec[i] = 440  # Black pawn on a8
    i += 1
  if board.has_queenside_castling_rights(chess.BLACK):
    vec[i] = 441  # Black pawn on b8
    i += 1
  assert i <= kMaxNumOnesInInput
  vec.sort()
  return vec
