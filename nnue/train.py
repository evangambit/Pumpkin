import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import chess
from collections import defaultdict
from tqdm import tqdm
import io

import torch.utils.data as tdata
from sharded_matrix import ShardedLoader
from ShardedMatricesIterableDataset import ShardedMatricesIterableDataset
from features import board2x, x2board, kMaxNumOnesInInput
from accumulator import Emb

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

  def embed(self, x, turn):
    assert len(x.shape) == 2
    assert len(turn.shape) == 2
    assert x.shape[0] == turn.shape[0]
    assert turn.shape[1] == 1
    white_z = self.emb(x, vertically_flipped=False)
    black_z = self.emb(x, vertically_flipped=True)
    mover = turn * white_z + (1 - turn) * black_z
    non_mover = (1 - turn) * white_z + turn * black_z
    z = torch.cat([mover, non_mover], dim=1)
    return z

  def forward(self, x, turn):
    z = self.embed(x, turn)
    out = self.mlp(z)
    return out

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

# We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
# much for non-trivial networks though.
BATCH_SIZE = 2048
CHUNK_SIZE = 64
assert BATCH_SIZE % CHUNK_SIZE == 0

device = torch.cuda.current_device()

print("Loading dataset...")
dataset = ShardedMatricesIterableDataset(
  f'data/de6-md2/tables-nnue',
  f'data/de6-md2/tables-eval',
  f'data/de6-md2/tables-turn',
  f'data/de6-md2/tables-piece-counts',
  chunk_size=CHUNK_SIZE,
)
dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

print("Creating model...")
 # [512, 128]
model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[1024, 128], output_size=16).to(device)

print("Creating optimizer...")
opt = torch.optim.AdamW([model.emb.piece], lr=0.0, weight_decay=0.1)

earliness_weights = torch.tensor([
  # p    n    b    r    q
  0.0, 1.0, 1.0, 1.0, 3.0,
  0.0, 1.0, 1.0, 1.0, 3.0,
]).to(device) / 18.0

metrics = defaultdict(list)
print("Starting training...")
NUM_EPOCHS = 1
for epoch in range(NUM_EPOCHS):
  for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    opt.zero_grad()

    if batch_idx == 10:
      for pg in opt.param_groups:
        pg['lr'] = np.linspace(3e-3, 3e-5, NUM_EPOCHS)[epoch]
    
    batch = [v.reshape((BATCH_SIZE,) + v.shape[2:]).to(device) for v in batch]

    x, y_white_perspective, turn, piece_counts = batch
    y_white_perspective = y_white_perspective.float() / 1000.0

    win_mover_perspective = y_white_perspective[:,0:1] * (turn == 1).float() + y_white_perspective[:,2:3] * (turn == -1).float()
    draw_mover_perspective = y_white_perspective[:,1:2]
    lose_mover_perspective = y_white_perspective[:,2:3] * (turn == 1).float() + y_white_perspective[:,0:1] * (turn == -1).float()

    earliness = piece_counts.float().matmul(earliness_weights).unsqueeze(1)  # Shape: (batch_size, 1)

    columns = torch.cat([
      earliness,
      (1.0 - earliness),
    ], 1)

    output = model(x, (turn + 1) // 2)
    output = output[:,:6].reshape((output.shape[0], 2, 3))
    output = (nn.functional.softmax(output, dim=2) * columns.unsqueeze(2)).sum(1)  # Shape: (batch_size, 3) (win, draw, loss)
    yhat = output[:,0:1] + output[:,1:2] * 0.5

    label = win_mover_perspective + draw_mover_perspective * 0.5

    # TODO: for some reason my network is learning negative scores.
    # TODO: try predicting wdl instead of score.

    loss = nn.functional.mse_loss(yhat, label, reduction='mean')
    loss.backward()
    opt.step()
    metrics["loss"].append(loss.item())
    if batch_idx % 500 == 0:
      print(f"loss: {np.mean(metrics['loss'][-100:]):.4f}")

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

# Save the model
with open('model.bin', 'wb') as f:
  save_tensor(model.emb.weight(False), 'embedding', f)
  save_tensor(model.mlp[0].weight, 'linear0.weight', f)
  save_tensor(model.mlp[0].bias, 'linear0.bias', f)
  save_tensor(model.mlp[2].weight, 'linear1.weight', f)
  save_tensor(model.mlp[2].bias, 'linear1.bias', f)

plt.figure(figsize=(10,10))
yhat = yhat.squeeze().cpu().detach().numpy()
label = label.squeeze().cpu().detach().numpy()
I = np.argsort(yhat)
yhat, label = yhat[I], label[I]
plt.scatter(yhat, label, alpha=0.1)
plt.scatter(np.convolve(yhat, np.ones(100)/100, mode='valid'), np.convolve(label, np.ones(100)/100, mode='valid'), color='red', label='moving average')
plt.savefig('nnue-scatter.png')

plt.figure(figsize=(10,10))
plt.plot(np.convolve(metrics['loss'][500:], np.ones(100)/100, mode='valid'), label='loss')
plt.legend()
plt.savefig('nnue-loss.png')


board = chess.Board()
X = []
T = []
moves = np.array(list(board.legal_moves))
for move in moves:
  board.push(move)
  X.append(torch.tensor(board2x(board)).unsqueeze(0).to(device))
  T.append(-1)
  board.pop()

output = model(torch.cat(X, dim=0), torch.tensor(T, device=device).unsqueeze(1))

I = output[:,0].cpu().detach().numpy().argsort()[::-1]
for i in I:
  move = moves[i]
  score = output[i,0].item()
  print(f"{board.san(move):6s} {score: .4f}")

