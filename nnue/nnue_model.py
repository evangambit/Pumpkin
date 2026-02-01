import torch
from torch import nn
import numpy as np
import chess

from accumulator import Emb
from features import kMaxNumOnesInInput, board2x, x2board

class CReLU(nn.Module):
  def forward(self, x):
    return x.clip(0, 1)

class NNUE(nn.Module):
  def __init__(self, input_size, hidden_sizes: list[int], output_size: int):
    super(NNUE, self).__init__()
    self.emb = Emb(dout=hidden_sizes[0])
    layers = []
    # hidden_sizes[0] *= 2
    for i in range(len(hidden_sizes) - 1):
      layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
      layers.append(CReLU())
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    for layer in layers:
      if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)
    self.mlp = nn.Sequential(*layers)

  def embed(self, x, turn):
    assert len(x.shape) == 2
    assert len(turn.shape) == 2
    assert x.shape[0] == turn.shape[0]
    assert turn.shape[1] == 1
    z = self.emb(x, vertically_flipped=False)
    z_flipped = self.emb(x, vertically_flipped=True)
    z = torch.where(turn == 1, z, z_flipped)
    return z

  def forward(self, x, turn):
    # Turn is 1 for white to move, -1 for black to move
    z = self.embed(x, turn)
    layers = [z]
    for layer in self.mlp:
      z = layer(z)
      if isinstance(layer, nn.Linear):
        layers.append(z)
    return z, layers


def test():
  model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[768], output_size=2)
  model.emb.zero_()
  for i in range(768):
    with torch.no_grad():
      model.emb.tiles[i // 64, (i % 64) // 8, (i % 64) % 8, i] = 1.0

  def remove_all(lst, val):
    return [x for x in lst if x != val]

  def foo(fen: str):
    x = torch.tensor(board2x(chess.Board(fen)), dtype=torch.int16).unsqueeze(0)
    z_white = model.emb(x, vertically_flipped=False)
    z_black = model.emb(x, vertically_flipped=True)
    if ' w ' in fen:
      return x, torch.cat([z_white, z_black], dim=1)
    else:
      return x, torch.cat([z_black, z_white], dim=1)
  start_pos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
  e4_c5 = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR'
  c4_e5 = 'rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR'
  white_missing_pieces = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1K3'
  black_missing_pieces = 'rnb1k3/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

  x0, z0 = foo(f'{start_pos} w KQkq - 0 1')  # starting position
  x1, z1 = foo(f'{e4_c5} w KQkq - 0 2')  # 1. e4 c5 (sicilian)
  x2, z2 = foo(f'{c4_e5} w KQkq - 0 2')  # 1. c4 e5 (reverse sicilian)

  x3, z3 = foo(f'{c4_e5} b KQkq - 0 2')  # 1. c4 e5 but black to move (should match z1)
  x4, z4 = foo(f'{e4_c5} b KQkq - 0 2')  # 1. e4 c5 but black to move (should match z2)
  x5, z5 = foo(f'{white_missing_pieces} w Qkq - 0 1')  # white missing pieces
  x6, z6 = foo(f'{black_missing_pieces} b KQq - 0 1')  # black missing pieces

  x0 = remove_all(x0.squeeze().tolist(), 768)
  x1 = remove_all(x1.squeeze().tolist(), 768)
  x2 = remove_all(x2.squeeze().tolist(), 768)
  x3 = remove_all(x3.squeeze().tolist(), 768)
  x4 = remove_all(x4.squeeze().tolist(), 768)
  x5 = remove_all(x5.squeeze().tolist(), 768)
  x6 = remove_all(x6.squeeze().tolist(), 768)
  def f(v):
    if v // 64 % 6 == 0 and (v % 64) // 8 in [0,7]:
      return ('w castle' if v < 384 else 'b castle')
    return ('w' if v < 384 else 'b', t[(v // 64) % 6], v % 64)

  t = 'pnbrqk'
  # print('xxx', [f(v) for v in list(set(x1) - set(x0))], [f(v) for v in list(set(x0) - set(x1))])
  # print('xxx', [f(v) for v in list(set(x2) - set(x0))], [f(v) for v in list(set(x0) - set(x2))])
  # print('xxx', [f(v) for v in list(set(x3) - set(x0))], [f(v) for v in list(set(x0) - set(x3))])
  # print('xxx', [f(v) for v in list(set(x4) - set(x0))], [f(v) for v in list(set(x0) - set(x4))])
  # print('xxx', [f(v) for v in list(set(x5) - set(x0))], [f(v) for v in list(set(x0) - set(x5))])
  # print('xxx', [f(v) for v in list(set(x6) - set(x0))], [f(v) for v in list(set(x0) - set(x6))])

  # On white's turn, the first half of z is white's pieces.
  assert np.nonzero(z0.squeeze()).squeeze().tolist()[:len(x0)] == x0
  assert np.nonzero(z1.squeeze()).squeeze().tolist()[:len(x1)] == x1
  assert np.nonzero(z2.squeeze()).squeeze().tolist()[:len(x2)] == x2
  assert np.nonzero(z5.squeeze()).squeeze().tolist()[:len(x5)] == x5
  
  # On black's turn, the second half of z is black
  assert (np.nonzero(z3.squeeze()).squeeze() - 768).tolist()[len(x3):] == x3
  assert (np.nonzero(z4.squeeze()).squeeze() - 768).tolist()[len(x4):] == x4
  assert (np.nonzero(z6.squeeze()).squeeze() - 768).tolist()[len(x6):] == x6

  i5 = np.nonzero(z5[0].detach().numpy())[0]
  i6 = np.nonzero(z6[0].detach().numpy())[0]

  assert torch.allclose(z1, z3)
  assert torch.allclose(z2, z4)
  assert torch.allclose(z5, z6)
  assert not torch.allclose(z1, z2)
  assert not torch.allclose(z1, z4)
  assert not torch.allclose(z2, z3)
  assert not torch.allclose(z3, z4)

test()

class Emb2(nn.Module):
  def __init__(self, dout):
    super().__init__()
    din = 768
    s = (1.0 / (kMaxNumOnesInInput * kMaxNumOnesInInput)) ** 0.5
    self.tiles = nn.Parameter(torch.randn(12, 8, 8, 12, 8, 8, dout) * s)
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
      tiles = tiles.flip(1).flip(4).roll(6, dims=0).roll(6, dims=3)
    
    with torch.no_grad():
      tiles[0,0,3,:] = 0.0
      tiles[:,:,:,0,0,3,:] = 0.0

    batch_size = tiles.shape[0]
    r = torch.cat([tiles.reshape(768**2, self.tiles.shape[-1]), self.zeros], dim=0)
    return r
  
  def forward(self, x, vertically_flipped=False):
    assert len(x.shape) == 2
    return self.weight(vertically_flipped=vertically_flipped)[x.to(torch.int32)].sum(1)


class NNUE(nn.Module):
  def __init__(self, input_size, hidden_sizes=None, output_size: int = 1):
    super(NNUE, self).__init__()
    self.emb = Emb2(dout=output_size)
  
  def compute_x_black(self, x_white):
    mask = x_white < 768
    piece = x_white // 64
    square = x_white % 64
    rank = square // 8
    file = square % 8
    x_black = ((piece + 6) % 12) * 64 + (7 - rank) * 8 + file
    return torch.where(mask, x_black, torch.tensor(768, device=x_white.device, dtype=x_white.dtype))

  def compute_x2(self, x_white):
    assert len(x_white.shape) == 2, f"x.shape = {x_white.shape}"

    mask = x_white < 768

    x_black = self.compute_x_black(x_white)

    x_white, x_black = x_white.to(torch.int32), x_black.to(torch.int32)

    x2_white = (x_white.reshape(-1, 1, 36) + x_white.reshape(-1, 36, 1) * 768).reshape(-1, 36 * 36)
    x2_black = (x_black.reshape(-1, 1, 36) + x_black.reshape(-1, 36, 1) * 768).reshape(-1, 36 * 36)

    # An interaction is only valid if neither piece is padding
    mask_2d_white = (x_white.reshape(-1, 1, 36) < 768) & (x_white.reshape(-1, 36, 1) < 768)
    mask_2d_black = (x_black.reshape(-1, 1, 36) < 768) & (x_black.reshape(-1, 36, 1) < 768)

    mask_2d_white = mask_2d_white.reshape(-1, 36 * 36)
    mask_2d_black = mask_2d_black.reshape(-1, 36 * 36)

    # Invalid interactions get mapped to the padding index
    x2_white = torch.where(mask_2d_white, x2_white, 3)
    x2_black = torch.where(mask_2d_black, x2_black, 443)

    return x2_white, x2_black

  def embed(self, x_white, turn):
    assert len(x_white.shape) == 2, f"x.shape = {x_white.shape}"
    assert len(turn.shape) == 2, f"turn.shape = {turn.shape}"
    assert x_white.shape[0] == turn.shape[0], f"x.shape[0] = {x_white.shape[0]} vs turn.shape[0] = {turn.shape[0]}"
    assert turn.shape[1] == 1, f"turn.shape = {turn.shape}"
    x2_white, x2_black = self.compute_x2(x_white)
    z = self.emb(x2_white, vertically_flipped=False)
    z_flipped = self.emb(x2_black, vertically_flipped=True)
    z = torch.where(turn == 1, z, z_flipped)
    return z

  def forward(self, x, turn):
    # Turn is 1 for white to move, -1 for black to move
    z = self.embed(x, turn).sum(1, keepdim=True)
    assert len(z.shape) == 2, f'z.shape = {z.shape}'
    return z, torch.zeros(1, dtype=z.dtype, device=z.device)


def test():
  model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[768], output_size=1)
  model.emb.zero_()

  view = model.emb.tiles.flatten()
  with torch.no_grad():
    view[:] = 0.0
    view[:] += torch.arange(view.shape[0], dtype=torch.float32)

  def remove_all(lst, val):
    return [x for x in lst if x != val]

  def foo(fen: str):
    x = torch.tensor(board2x(chess.Board(fen)), dtype=torch.int16).unsqueeze(0)
    return x, model.embed(x, turn=torch.tensor([[1.0 if ' w ' in fen else -1.0]]))
  start_pos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
  e4_c5 = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR'
  c4_e5 = 'rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR'
  white_missing_pieces = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1K3'
  black_missing_pieces = 'rnb1k3/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'

  # x0, z0 = foo(f'{start_pos} w KQkq - 0 1')  # starting position
  # x1, z1 = foo(f'{e4_c5} w KQkq - 0 2')  # 1. e4 c5 (sicilian)
  # x2, z2 = foo(f'{c4_e5} w KQkq - 0 2')  # 1. c4 e5 (reverse sicilian)

  # x3, z3 = foo(f'{c4_e5} b KQkq - 0 2')  # 1. c4 e5 but black to move (should match z1)
  # x4, z4 = foo(f'{e4_c5} b KQkq - 0 2')  # 1. e4 c5 but black to move (should match z2)
  # x5, z5 = foo(f'{white_missing_pieces} w Qkq - 0 1')  # white missing pieces
  # x6, z6 = foo(f'{black_missing_pieces} b KQq - 0 1')  # black missing pieces

  # with torch.no_grad():
  #   w1, b1 = model.compute_x2(torch.tensor(x1))
  #   w1 = torch.sort(w1[0]).values
  #   b1 = torch.sort(b1[0]).values
  #   w2, b2 = model.compute_x2(torch.tensor(x2))
  #   w2 = torch.sort(w2[0]).values
  #   b2 = torch.sort(b2[0]).values
  #   w3, b3 = model.compute_x2(torch.tensor(x3))
  #   w3 = torch.sort(w3[0]).values
  #   b3 = torch.sort(b3[0]).values
  #   w4, b4 = model.compute_x2(torch.tensor(x4))
  #   w4 = torch.sort(w4[0]).values
  #   b4 = torch.sort(b4[0]).values
  #   w5, b5 = model.compute_x2(torch.tensor(x5))
  #   w5 = torch.sort(w5[0]).values
  #   b5 = torch.sort(b5[0]).values
  #   w6, b6 = model.compute_x2(torch.tensor(x6))
  #   w6 = torch.sort(w6[0]).values
  #   b6 = torch.sort(b6[0]).values

  # x0 = remove_all(x0.squeeze().tolist(), 768)
  # x1 = remove_all(x1.squeeze().tolist(), 768)
  # x2 = remove_all(x2.squeeze().tolist(), 768)
  # x3 = remove_all(x3.squeeze().tolist(), 768)
  # x4 = remove_all(x4.squeeze().tolist(), 768)
  # x5 = remove_all(x5.squeeze().tolist(), 768)
  # x6 = remove_all(x6.squeeze().tolist(), 768)
  # def f(v):
  #   if v // 64 % 6 == 0 and (v % 64) // 8 in [0,7]:
  #     return ('w castle' if v < 384 else 'b castle')
  #   return ('w' if v < 384 else 'b', t[(v // 64) % 6], v % 64)

  # i1 = np.nonzero(w1[0].detach().numpy())[0]
  # i2 = np.nonzero(w2[0].detach().numpy())[0]
  # i3 = np.nonzero(w3[0].detach().numpy())[0]
  # i4 = np.nonzero(w4[0].detach().numpy())[0]
  # i5 = np.nonzero(w5[0].detach().numpy())[0]
  # i6 = np.nonzero(w6[0].detach().numpy())[0]

  # print('torch.allclose(z1, z3)', torch.abs(z1 - z3).mean().item())
  # print('torch.allclose(z2, z4)', torch.abs(z2 - z4).mean().item())
  # print('torch.allclose(z5, z6)', torch.abs(z5 - z6).mean().item())
  # print('not torch.allclose(z1, z2)', torch.abs(z1 - z2).mean().item())
  # print('not torch.allclose(z1, z4)', torch.abs(z1 - z4).mean().item())
  # print('not torch.allclose(z2, z3)', torch.abs(z2 - z3).mean().item())
  # print('not torch.allclose(z3, z4)', torch.abs(z3 - z4).mean().item())

  # breakpoint()

  # assert torch.allclose(z1, z3)
  # assert torch.allclose(z2, z4)
  # assert torch.allclose(z5, z6)
  # assert not torch.allclose(z1, z2)
  # assert not torch.allclose(z1, z4)
  # assert not torch.allclose(z2, z3)
  # assert not torch.allclose(z3, z4)

  only_kings = 'k7/8/8/8/8/8/8/K7 w - - 0 1'  # kings on a1/a8
  x1, z1 = foo(only_kings)
  x2, z2 = foo('k7/8/8/8/8/8/8/K7 b - - 0 1')  # kings on a1/a8, black to move

  w1, b1 = model.compute_x2(torch.tensor(x1))
  w1 = torch.sort(w1[0]).values
  b1 = torch.sort(b1[0]).values
  w2, b2 = model.compute_x2(torch.tensor(x2))
  w2 = torch.sort(w2[0]).values
  b2 = torch.sort(b2[0]).values

  breakpoint()

test()
