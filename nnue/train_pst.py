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
from features import board2x, x2board

kMaxNumOnesInInput = 32

class PST(nn.Module):
  def __init__(self):
    super(PST, self).__init__()
    # 6 pieces + empty square
    self.piece = nn.Parameter(torch.randn(6) * 0.1, requires_grad=True)
    self.pst_early = nn.Parameter(torch.randn(6 * 64) * 0.01, requires_grad=True)
    self.pst_late = nn.Parameter(torch.randn(6 * 64) * 0.01, requires_grad=True)

  def embed(self, indices, turn):
    assert len(indices.shape) == 2
    assert len(turn.shape) == 2
    assert indices.shape[0] == turn.shape[0]
    assert turn.shape[1] == 1
    indices = indices.to(torch.int32)

    is_not_empty = (indices < 12 * 64).to(torch.float32)
    is_white = (indices < 6 * 64).to(torch.float32)
    color_sign = is_white * 2.0 - 1.0
    piece = (indices // 64) % 6

    rank = (indices % 64) // 8
    file = (indices % 64) % 8

    white_indices = piece * 64 + rank * 8 + file
    black_indices = piece * 64 + (7 - rank) * 8 + file

    piece_values = self.piece[piece]
    assert piece_values.shape == (indices.shape[0], kMaxNumOnesInInput), f"piece_values.shape={piece_values.shape}, indices.shape={indices.shape}"

    early = ((self.pst_early[white_indices] * is_white + self.pst_early[black_indices] * (1.0 - is_white)))
    assert early.shape == (indices.shape[0], kMaxNumOnesInInput), f"early.shape={early.shape}, indices.shape={indices.shape}"
    late = ((self.pst_late[white_indices] * is_white + self.pst_late[black_indices] * (1.0 - is_white)))
    assert late.shape == (indices.shape[0], kMaxNumOnesInInput), f"late.shape={late.shape}, indices.shape={indices.shape}"
    return torch.cat([
      ((piece_values + early) * color_sign * is_not_empty).sum(1, keepdim=True),
      ((piece_values + late) * color_sign * is_not_empty).sum(1, keepdim=True),
    ], dim=1)
  
  def table(self):
    early_table = self.pst_early.detach().cpu().numpy().reshape((6, 8, 8))
    late_table = self.pst_late.detach().cpu().numpy().reshape((6, 8, 8))
    for i in range(6):
      early_table[i, :, :] += self.piece[i].item()
      late_table[i, :, :] += self.piece[i].item()
    return early_table, late_table

  def forward(self, x, turn):
    # Turn is 1 for white to move, -1 for black to move
    z = self.embed(x, turn)
    return z * turn

# We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
# much for non-trivial networks though.
BATCH_SIZE = 2048
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

device = torch.cuda.current_device()

print("Loading dataset...")
# dataset_name = 'de6-md2'  # Accuracy: 78%, MSE: 0.077179
dataset_name = 'de7-md4'  # Data quality: Accuracy: 92%, MSE: 0.037526
dataset = ShardedMatricesIterableDataset(
  f'data/{dataset_name}/tables-nnue',
  f'data/{dataset_name}/tables-eval',
  f'data/{dataset_name}/tables-turn',
  f'data/{dataset_name}/tables-piece-counts',
  chunk_size=CHUNK_SIZE,
)

dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
model = PST().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)

earliness_weights = torch.tensor([
  # p    n    b    r    q
  0.0, 1.0, 1.0, 1.0, 3.0,
  0.0, 1.0, 1.0, 1.0, 3.0,
]).to(device) / 18.0

def wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective):
  return win_mover_perspective + draw_mover_perspective * 0.5

def loss1(x, piece_counts, output, win_mover_perspective, draw_mover_perspective, lose_mover_perspective):
  """
  Very simple loss that just interpolates between early and late game based centipawns.
  """
  earliness = piece_counts.float().matmul(earliness_weights).unsqueeze(1)  # Shape: (batch_size, 1)
  output = torch.sigmoid(output[:,0] * earliness + output[:,1] * (1.0 - earliness))
  label = wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective)
  loss = nn.functional.mse_loss(output, label, reduction='mean')
  return loss, output

def loss2(x, piece_counts, output, win_mover_perspective, draw_mover_perspective, lose_mover_perspective):
  """
  Expects model to output 6 values: early win, early draw, early loss, late win, late draw, late loss.
  Applies softmax to each of early and late, then combines them based on earliness.
  Computes negative log likelihood loss.
  """
  earliness = piece_counts.float().matmul(earliness_weights).unsqueeze(1)  # Shape: (batch_size, 1)
  columns = torch.cat([
    earliness,
    (1.0 - earliness),
  ], 1)

  output = output[:,:6].reshape((output.shape[0], 2, 3))
  output = (nn.functional.softmax(output, dim=2) * columns.unsqueeze(2)).sum(1)  # Shape: (batch_size, 3) (win, draw, loss)
  
  log_likelihood = win_mover_perspective * torch.log(output[:,0])
  log_likelihood += draw_mover_perspective * torch.log(output[:,1])
  log_likelihood += lose_mover_perspective * torch.log(output[:,2])

  return -log_likelihood.mean(), output[:,0] + output[:,1] * 0.5

earliness_weights = torch.tensor([
  # p    n    b    r    q
  0.0, 1.0, 1.0, 1.0, 3.0,
  0.0, 1.0, 1.0, 1.0, 3.0,
]).to(device) / 18.0

metrics = defaultdict(list)
print("Starting training...")
NUM_EPOCHS = 3
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

    if epoch == 0:
      if batch_idx == 0:
        for i in range(10):
          board = x2board(x[i].cpu().numpy(), turn[i].item())
          print(f"Sample {i}:\n{board.fen()} | Win: {win_mover_perspective[i].item():.3f}, Draw: {draw_mover_perspective[i].item():.3f}, Loss: {lose_mover_perspective[i].item():.3f}\n")

    output = model(x, turn)
    earliness =  piece_counts.to(torch.float32) @ earliness_weights.unsqueeze(1)  # Shape: (batch_size, 1)

    yhat = torch.sigmoid(output[:,0:1] * earliness + output[:,1:2] * (1.0 - earliness))

    label = wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective)
    assert yhat.shape == label.shape



    loss = nn.functional.mse_loss(
      yhat, label, reduction='mean',
    )
    mse = loss.item()
    baseline = ((label - 0.5) ** 2).mean().item()

    loss.backward()
    opt.step()
    metrics["loss"].append(loss.item())
    metrics["relative_mse"].append(mse / baseline)
    if batch_idx % 500 == 0:
      print(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.6f}, Relative MSE={mse / baseline:.6f}")

early, late = model.table()

for mat in [early, late]:
  for i, name in enumerate(['P', 'N', 'B', 'R', 'Q', 'K']):
    print(f'// {name}')
    for y in range(8):
      line = ', '.join([v.rjust(5) for v in (mat[i, y, :] * 100).astype(np.int32).astype(str).tolist()])
      print(f'   {line},')
    print('')
