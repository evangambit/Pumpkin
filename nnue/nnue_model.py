import torch
from torch import nn
import numpy as np

from accumulator import Emb

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
    return self.emb(x, vertically_flipped=False)

  def forward(self, x, turn):
    # Turn is 1 for white to move, -1 for black to move
    z = self.embed(x, turn)
    out = self.mlp(z)
    return out * turn

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

  x0, z0 = foo(f'{start_pos} w KQkq - 0 1')  # starting position
  x1, z1 = foo(f'{e4_c5} w KQkq - 0 2')  # 1. e4 c5 (sicilian)
  x2, z2 = foo(f'{c4_e5} w KQkq - 0 2')  # 1. c4 e5 (reverse sicilian)

  x3, z3 = foo(f'{c4_e5} b KQkq - 0 2')  # 1. c4 e5 but black to move (should match z1)
  x4, z4 = foo(f'{e4_c5} b KQkq - 0 2')  # 1. e4 c5 but black to move (should match z2)
  x5, z5 = foo(f'{white_missing_pieces} w Qkq - 0 1')  # white missing pieces

  x0 = remove_all(x0.squeeze().tolist(), 768)
  x1 = remove_all(x1.squeeze().tolist(), 768)
  x2 = remove_all(x2.squeeze().tolist(), 768)
  x3 = remove_all(x3.squeeze().tolist(), 768)
  x4 = remove_all(x4.squeeze().tolist(), 768)
  x5 = remove_all(x5.squeeze().tolist(), 768)

  # On white's turn, the first half of z is white's pieces.
  assert np.nonzero(z0.squeeze()).squeeze().tolist()[:len(x0)] == x0
  assert np.nonzero(z1.squeeze()).squeeze().tolist()[:len(x1)] == x1
  assert np.nonzero(z2.squeeze()).squeeze().tolist()[:len(x2)] == x2
  assert np.nonzero(z5.squeeze()).squeeze().tolist()[:len(x5)] == x5
  
  # On black's turn, the second half of z is black
  assert (np.nonzero(z3.squeeze()).squeeze() - 768).tolist()[len(x3):] == x3
  assert (np.nonzero(z4.squeeze()).squeeze() - 768).tolist()[len(x4):] == x4

  i5 = np.nonzero(z5[0].detach().numpy())[0]

  assert torch.allclose(z1, z3)
  assert torch.allclose(z2, z4)
  assert not torch.allclose(z1, z2)
  assert not torch.allclose(z3, z4)

test()
