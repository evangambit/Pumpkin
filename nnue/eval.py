import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import chess
from collections import defaultdict
from tqdm import tqdm
import io
import os
import datetime

import torch.utils.data as tdata
from sharded_matrix import ShardedLoader
from ShardedMatricesIterableDataset import ShardedMatricesIterableDataset, SingleShardedMatrixIterator, DynamicShardedMatrixIterator
from features import board2x, x2board
from accumulator import Emb
from nnue_model import NNUE

model = NNUE(hidden_sizes=[1], output_size=1)
with open('hanging-1d.pt', 'rb') as f:
  model.load_state_dict(torch.load(f, map_location='cpu'))

print(model.emb.weight()[:64,0])

def moves2fen(*moves):
  board = chess.Board()
  for move in moves:
    board.push_san(move)
  return board.fen()

CHUNK_SIZE = 1
initial_fen = chess.Board().fen()
fens = [
  (initial_fen, ''),
  (moves2fen('e4'), 'e4'),
  (moves2fen('d4'), 'd4'),
  (moves2fen('f3'), 'f3'),
  (moves2fen('Na3'), 'Na3'),
  (moves2fen('e4', 'g8f6', 'd4', 'f6g8'), 'e4 d4'),
  (initial_fen, ''),  # There is a bug where the last position is dropped, so we add padding here. (TODO: fix the bug).
]
with open('/tmp/pos.txt', 'w') as f:
  for (fen, _) in fens:
    f.write(fen + '|+0.00\n')

# Run "cat /tmp/pos.txt | ./mt --output /tmp/x --emit_nnue"
import subprocess
subprocess.run('cat /tmp/pos.txt | ./mt --output /tmp/x --emit_nnue', shell=True, check=True)

dataset = ShardedMatricesIterableDataset(
  DynamicShardedMatrixIterator(f'/tmp/x-nnue-sparse', chunk_size=CHUNK_SIZE),
  SingleShardedMatrixIterator(f'/tmp/x-eval', chunk_size=CHUNK_SIZE),
  SingleShardedMatrixIterator(f'/tmp/x-piece-counts', chunk_size=CHUNK_SIZE),
)
it = iter(dataset)
for row, (fen, name) in zip(it, fens):
  (values, lengths), y, piece_counts = row
  values = torch.from_numpy(values)
  lengths = torch.from_numpy(lengths)
  yhat = model(values, lengths)[0].squeeze().detach().numpy()
  print(name, yhat)
