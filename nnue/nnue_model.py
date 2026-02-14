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
    hidden_sizes[0] *= 2
    for i in range(len(hidden_sizes) - 1):
      layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
      layers.append(CReLU())
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    for layer in layers:
      if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)
    self.mlp = nn.Sequential(*layers)

  def embed(self, x):
    assert len(x.shape) == 2
    z_us, z_them = self.emb(x)
    return torch.cat([z_us, z_them], dim=1)


  def forward(self, x):
    # Turn is 1 for white to move, -1 for black to move
    z = self.embed(x)
    layers = []
    for layer in self.mlp:
      z = layer(z)
      if isinstance(layer, nn.Linear):
        layers.append(z)
    return z, layers

