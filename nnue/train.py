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
from features import board2x, x2board, kMaxNumOnesInInput
from accumulator import Emb
from nnue_model import NNUE


def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

def evaluate_data_quality(dataset):
  import chess
  from chess import engine as chess_engine
  engine = chess_engine.SimpleEngine.popen_uci('/opt/homebrew/bin/stockfish')
  chunk = next(iter(dataset))
  correct, mse = 0, 0
  for i in range(100):
    x, y, piece_counts = chunk
    x, y, piece_counts = x[i], y[i], piece_counts[i]
    assert x.shape == (kMaxNumOnesInInput,)
    assert y.shape == (3,)
    assert piece_counts.shape == (10,)
    board = x2board(x, True)
    lines = engine.analyse(board, chess_engine.Limit(depth=7), multipv=1)
    wdl = lines[0]['score'].white().wdl()
    wdl = np.array([wdl.wins, wdl.draws, wdl.losses], dtype=np.int32)
    if y.argmax() == wdl.argmax():
      correct += 1
    mse += (((y.astype(np.float32) / 1000.0) - (wdl.astype(np.float32) / wdl.sum())) ** 2).mean()
  incorrect = 100 - correct
  print(f"Data quality: Accuracy: {correct}%, MSE: {mse / 100:.6f}")

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


# We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
# much for non-trivial networks though.
BATCH_SIZE = 2048
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

# Create a directory for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("runs", timestamp)
os.makedirs(run_dir, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

print("Loading dataset...")
dataset = ShardedMatricesIterableDataset(
  f'data/x-nnue',
  f'data/x-eval',
  f'data/x-piece-counts',
  chunk_size=CHUNK_SIZE,
)
evaluate_data_quality(dataset)


print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

print("Creating model...")
 # [512, 128]
model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[512, 8], output_size=1).to(device)

print("Creating optimizer...")
opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)

def warmup_length(beta, c = 2.0):
  # The amount of warmup needs to increase as beta approaches 1,
  # since we need to see more data before the moving averages stabilize
  # to its long-run variability.
  return int(c / (1 - beta))

# Calculate total steps
NUM_EPOCHS = 1
steps_per_epoch = len(dataloader)
total_steps = NUM_EPOCHS * steps_per_epoch
warmup_steps = warmup_length(0.999) # AdamW's beta is 0.999.
assert warmup_steps < total_steps // 10, "You probably made a mistake."

scheduler = CosineAnnealingWithWarmup(
  opt,
  max_lr=3e-3,
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

    x, y_white_perspective, piece_counts = batch
    y_white_perspective = y_white_perspective.float() / 1000.0

    win_mover_perspective = y_white_perspective[:,0]
    draw_mover_perspective = y_white_perspective[:,1]
    lose_mover_perspective = y_white_perspective[:,2]

    output, layers = model(x)

    penalty = 0.0
    for layer_output in layers:
      penalty += (layer_output.mean() ** 2 + (layer_output.std() - 1.0) ** 2)

    output = torch.sigmoid(output)[:,0]

    label = wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective)
    assert output.shape == label.shape, f"{output.shape} vs {label.shape}"
    loss = nn.functional.mse_loss(
      output, label, reduction='mean',
    )
    mse = loss.item()
    baseline = ((label - label.mean()) ** 2).mean().item()

    (loss + penalty * 0.02).backward()
    opt.step()
    metrics["loss"].append(loss.item())
    metrics["mse"].append(mse / baseline)
    metrics["penalty"].append(penalty.item())
    if (batch_idx + 1) % 500 == 0:
      print(f"loss: {np.mean(metrics['loss'][-1000:]):.4f}, mse: {np.mean(metrics['mse'][-1000:]):.4f}, penalty: {np.mean(metrics['penalty'][-1000:]):.4f}")

# Save the model

with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
  torch.save(model.state_dict(), f)

with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
  save_tensor(model.emb.weight()[:-1], 'embedding', f)
  save_tensor(model.mlp[0].weight, 'linear0.weight', f)
  save_tensor(model.mlp[0].bias, 'linear0.bias', f)
  save_tensor(model.mlp[2].weight, 'linear1.weight', f)
  save_tensor(model.mlp[2].bias, 'linear1.bias', f)

plt.figure(figsize=(10,10))
output = output.squeeze().cpu().detach().numpy()
label = label.squeeze().cpu().detach().numpy()
I = np.argsort(output)
output, label = output[I], label[I]
plt.scatter(output, label, alpha=0.1)
plt.scatter(np.convolve(output, np.ones(100)/100, mode='valid'), np.convolve(label, np.ones(100)/100, mode='valid'), color='red', label='moving average')
plt.savefig(os.path.join(run_dir, 'nnue-scatter.png'))

plt.figure(figsize=(10,10))
plt.plot(np.convolve(metrics['loss'][500:], np.ones(100)/100, mode='valid'), label='loss')
plt.legend()
plt.savefig(os.path.join(run_dir, 'nnue-loss.png'))


board = chess.Board()
X = []
moves = np.array(list(board.legal_moves))
for move in moves:
  board.push(move)
  X.append(torch.tensor(board2x(board)).unsqueeze(0).to(device))
  board.pop()

output = model(torch.cat(X, dim=0))[0].squeeze()

I = output.cpu().detach().numpy().argsort()[::-1]
for i in I:
  move = moves[i]
  score = output[i].item()
  print(f"{board.san(move):6s} {score: .4f}")

white_winning = 'rnbqkbnr/pppppppp/8/8/2BPPB2/2N2N2/PPP2PPP/R2Q1RK1 w Qkq - 0 1'
board = chess.Board(white_winning)
output = model(torch.tensor(board2x(board)).unsqueeze(0).to(device))[0][:,0]
print(f"White winning position score: {output[0].item():.4f}")

# loss: 0.0194, mse: 0.5199, penalty: 0.0200
# e4      0.6946
# c4      0.6531
# c3      0.6399
# e3      0.6351
# d4      0.6285
# Nc3     0.6163
# Nf3     0.6140
# a3      0.5890
# g3      0.5652
# h3      0.5644
# b3      0.5609
# b4      0.5340
# a4      0.5203
# Na3     0.5192
# d3      0.5174
# Nh3     0.4823
# h4      0.4611
# f3      0.4248
# f4      0.4180
# g4      0.3641
# White winning position score: 0.8122