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
from ShardedMatricesIterableDataset import ShardedMatricesIterableDataset

def x2board(x, turn):
  x = x.reshape(-1, 8, 8)
  board = chess.Board()
  board.clear_board()
  for i, p in enumerate('PpNnBbRrQqKk'):
    for rank in range(8):
      for file in range(8):
        if x[i, rank, file] > 0.5:
          piece_type = 'PNBRQKpnbrqk'.index(p) % 6 + 1
          color = chess.WHITE if (p == p.upper()) == (turn == 1) else chess.BLACK
          square = chess.square(file, rank)
          board.set_piece_at(square, chess.Piece(piece_type, color))
  board.turn = chess.WHITE if turn == 1 else chess.BLACK
  return board

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

# Cosine learning rate scheduler with warmup
class CosineAnnealingWithWarmup:
  def __init__(self, optimizer, max_lr=3e-3, min_lr=1e-5, warmup_steps=100, total_steps=None):
    self.optimizer = optimizer
    self.max_lr = max_lr
    self.min_lr = min_lr
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.current_step = 0

  def step(self):
    if self.current_step < self.warmup_steps:
      # Linear warmup phase
      lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_step / self.warmup_steps)
    else:
      # Cosine annealing phase
      progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
      lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    for pg in self.optimizer.param_groups:
      pg['lr'] = lr
    
    self.current_step += 1

class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.linear = nn.Linear(768, 64)
  
  def weight(self):
    w = self.raw.reshape(2, -1, 8, 8) + self.simple.reshape(2, -1, 1, 1)
    w += self.files.reshape(2, -1, 8, 1)
    w += self.ranks.reshape(2, -1, 1, 8)
    return w
  
  def weight2(self):
    w = self.weight()
    flipped = -w.flip(dims=(2,))

    # Interleave mover and waiter weights, where waiter weights are
    # flipped, negative version of mover weights.
    y = torch.zeros((2, w.shape[1] * 2, 8, 8), device=w.device, dtype=w.dtype)
    y[:,0::2] = w
    y[:,1::2] = flipped

    return y

  def forward(self, x):
    weights = self.weight2()
    yhat = nn.functional.linear(x, weights.reshape(2, -1), self.bias)
    return yhat

def warmup_length(beta, c = 2.0):
  # The amount of warmup needs to increase as beta approaches 1,
  # since we need to see more data before the moving averages stabilize
  # to its long-run variability.
  return int(c / (1 - beta))

NUM_EPOCHS = 4
BATCH_SIZE = 2048  # 2048
maxlr = 3e-3
BETAS = (0.9, 0.9)  # (0.9, 0.999)
WEIGHT_DECAY = 0.1

if __name__ == '__main__':
  # We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
  # much for non-trivial networks though.
  CHUNK_SIZE = 128
  assert BATCH_SIZE % CHUNK_SIZE == 0
  print("Loading dataset...")
  # dataset_name = 'de6-md2'  # Accuracy: 78%, MSE: 0.077179
  # dataset_name = 'de7-md4'  # Data quality: Accuracy: 92%, MSE: 0.037526
  dataset_name = 'stock'
  dataset = ShardedMatricesIterableDataset(
    f'data/moveorder-pieces',
    f'data/moveorder-moves',
    chunk_size=CHUNK_SIZE,
  )
  print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

  # Create a directory for this run
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  run_dir = os.path.join("runs", timestamp)
  os.makedirs(run_dir, exist_ok=True)
  with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
    f.write(f'Batch size: {BATCH_SIZE}\n')
    f.write(f'Chunk size: {CHUNK_SIZE}\n')
    f.write(f'Max learning rate: {maxlr}\n')
    f.write(f'Weight decay: {WEIGHT_DECAY}\n')


  print("Creating model...")
  from_lin = nn.Embedding(64 * 6 + 1, 1)
  to_lin = nn.Embedding(64 * 6 + 1, 1)
  piece_lin = nn.Embedding(6 + 1, 1)

  print("Creating optimizer...")
  opt = torch.optim.AdamW(list(
    from_lin.parameters()) + list(to_lin.parameters()) + list(piece_lin.parameters()
  ), lr=0.0, weight_decay=WEIGHT_DECAY, betas=BETAS)

  # Calculate total steps
  steps_per_epoch = len(dataloader)
  total_steps = NUM_EPOCHS * steps_per_epoch
  warmup_steps = warmup_length(max(opt.param_groups[0]['betas']))
  assert warmup_steps < total_steps // 10, f"Warmup steps {warmup_steps} seems too long for total steps {total_steps} (more than 10%)"

  scheduler = CosineAnnealingWithWarmup(
    opt,
    max_lr=maxlr,
    min_lr=1e-5,
    warmup_steps=warmup_steps,
    total_steps=total_steps
  )

  def wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective):
    assert len(win_mover_perspective.shape) == 1
    assert len(draw_mover_perspective.shape) == 1
    assert len(lose_mover_perspective.shape) == 1
    assert win_mover_perspective.shape == draw_mover_perspective.shape
    assert win_mover_perspective.shape == lose_mover_perspective.shape
    return win_mover_perspective + draw_mover_perspective * 0.5

  metrics = defaultdict(list)
  for epoch in range(NUM_EPOCHS):
    print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
      opt.zero_grad()

      # Update learning rate
      scheduler.step()
      
      batch = [v.reshape((BATCH_SIZE,) + v.shape[2:]).to(device) for v in batch]

      x, moves = batch

      from_squares = moves[:,0::2]
      to_squares = moves[:,1::2]

      yhat = from_lin(from_squares.to(torch.int64)) + to_lin(to_squares.to(torch.int64)) + piece_lin((from_squares // 64).to(torch.int64))
      yhat = yhat.squeeze()   # (BATCH_SIZE, 10)
      num_moves = (from_squares != 64).sum(1, keepdim=True)  # (BATCH_SIZE, 1)

      # The first move is always the correct move.
      # loss = -torch.log_softmax(yhat / (num_moves + 1), dim=1)[:,0].mean()

      loss = -torch.log_softmax(yhat, dim=1)[:,0]
      loss = (loss / num_moves.squeeze()).mean()

      baseline = -torch.log(1 / num_moves.float()).mean().item()

      loss.backward()
      opt.step()
      metrics["loss"].append(loss.item())
      metrics["mse"].append((loss / baseline).item())
      if (batch_idx + 1) % 100 == 0:
        print(f"loss: {np.mean(metrics['loss'][-1000:]):.4f}, mse: {np.mean(metrics['mse'][-1000:]):.4f}")

  loss = np.array(metrics["loss"])
  mse = np.array(metrics["mse"])
  with open(os.path.join(run_dir, 'metrics.txt'), 'w') as f:
    f.write(f'Final loss: %.5f (%.5f)\n' % (loss[-100:].mean(), loss.std() / 10))
    f.write(f'Final mse: %.5f (%.5f)\n' % (mse[-100:].mean(), mse.std() / 10))

  for lin in [from_lin, to_lin]:
    avg = lin.weight.mean()
    for i in range(0, 6):
      piece = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'][i]
      print(f'// {piece}')
      w = torch.round((lin.weight[i*64:(i+1)*64,0].reshape(8,8) - avg) * 10).detach().numpy()
      for rank in range(8):
        line = []
        for file in range(8):
          line.append(str(int(w[rank, file])).rjust(4))
        print(', '.join(f'{x}' for x in line) + ',')
