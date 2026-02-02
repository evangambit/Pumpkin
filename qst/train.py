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

names = [
  'our_p|base_psq',
  'our_n|base_psq',
  'our_b|base_psq',
  'our_r|base_psq',
  'our_q|base_psq',
  'our_k|base_psq',
  'tir_p|base_psq',
  'tir_n|base_psq',
  'tir_b|base_psq',
  'tir_r|base_psq',
  'tir_q|base_psq',
  'tir_k|base_psq',

  'our_psd_pwns',
  'tir_psd_pwns',
  'our_iso_pwns',
  'tir_iso_pwns',
  'our_dbl_pwns',
  'tir_dbl_pwns',

  'bad_our_p',
  'bad_our_n',
  'bad_our_b',
  'bad_our_r',
  'bad_our_q',
  'bad_our_k',
  'bad_tir_p',
  'bad_tir_n',
  'bad_tir_b',
  'bad_tir_r',
  'bad_tir_q',
  'bad_tir_k',

  # Conditional on them having a queen.
  'our_p|tir_q',
  'our_n|tir_q',
  'our_b|tir_q',
  'our_r|tir_q',
  'our_q|tir_q',
  'our_k|tir_q',
  'tir_p|tir_q',
  'tir_n|tir_q',
  'tir_b|tir_q',
  'tir_r|tir_q',
  'tir_q|tir_q',
  'tir_k|tir_q',

  # Conditional on us having a queen.
  'our_p|our_q',
  'our_n|our_q',
  'our_b|our_q',
  'our_r|our_q',
  'our_q|our_q',
  'our_k|our_q',
  'tir_p|our_q',
  'tir_n|our_q',
  'tir_b|our_q',
  'tir_r|our_q',
  'tir_q|our_q',
  'tir_k|our_q',
]

for name in names:
  assert len(name) <= 14

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


# We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
# much for non-trivial networks though.
BATCH_SIZE = 2048
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

# Create a directory for this run
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join("runs", timestamp)
os.makedirs(run_dir, exist_ok=True)

device = torch.cuda.current_device()

print("Loading dataset...")
# dataset_name = 'de6-md2'  # Accuracy: 78%, MSE: 0.077179
dataset_name = 'de7-md4'  # Data quality: Accuracy: 92%, MSE: 0.037526
dataset = ShardedMatricesIterableDataset(
  f'data/{dataset_name}/qst-qst',
  f'data/{dataset_name}/qst-eval',
  f'data/{dataset_name}/qst-turn',
  f'data/{dataset_name}/qst-piece-counts',
  chunk_size=CHUNK_SIZE,
)

class MyModel(nn.Module):
  def __init__(self, input_dim):
    super(MyModel, self).__init__()
    self.bias = nn.Parameter(torch.zeros((2,)), requires_grad=True)
    self.raw = nn.Parameter(torch.zeros((input_dim, 2)), requires_grad=True)
    self.simple = nn.Parameter(torch.zeros((input_dim // 64, 2)), requires_grad=True)
    self.files = nn.Parameter(torch.zeros((input_dim // 8, 2)), requires_grad=True)
    self.ranks = nn.Parameter(torch.zeros((input_dim // 8, 2)), requires_grad=True)
    nn.init.zeros_(self.raw)
    nn.init.zeros_(self.simple)
    nn.init.zeros_(self.files)
    nn.init.zeros_(self.ranks)
  
  def weight(self):
    w = self.raw.reshape(2, -1, 8, 8)
    w = w + self.simple.reshape(2, -1, 1, 1)
    w = w + self.files.reshape(2, -1, 8, 1)
    w = w + self.ranks.reshape(2, -1, 1, 8)
    return w

  def forward(self, x):
    weights = self.weight()
    return nn.functional.linear(x, weights.reshape(2, -1), self.bias)


print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

print("Creating model...")
 # [512, 128]
dim = next(iter(dataset))[0].shape[1]
assert dim == len(names) * 64, f"{dim} vs {len(names)}"
model = MyModel(dim).to(device)

print("Creating optimizer...")
opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)

def warmup_length(beta, c = 2.0):
  # The amount of warmup needs to increase as beta approaches 1,
  # since we need to see more data before the moving averages stabilize
  # to its long-run variability.
  return int(c / (1 - beta))

# Calculate total steps
NUM_EPOCHS = 4
steps_per_epoch = len(dataloader)
total_steps = NUM_EPOCHS * steps_per_epoch
warmup_steps = warmup_length(0.999) # AdamW's beta is 0.999.
# assert warmup_steps < total_steps // 10, "You probably made a mistake."

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

    x, y_white_perspective, turn, piece_counts = batch
    y_white_perspective = y_white_perspective.float() / 1000.0

    win_mover_perspective = y_white_perspective[:,0] * (turn.squeeze() == 1).float() + y_white_perspective[:,2] * (turn.squeeze() == -1).float()
    draw_mover_perspective = y_white_perspective[:,1]
    lose_mover_perspective = y_white_perspective[:,2] * (turn.squeeze() == 1).float() + y_white_perspective[:,0] * (turn.squeeze() == -1).float()

    output = model(x.to(torch.float32))

    earliness = piece_counts[:,1] + piece_counts[:,2] + piece_counts[:,3] + piece_counts[:,4] * 3
    earliness += piece_counts[:,6] + piece_counts[:,7] + piece_counts[:,8] + piece_counts[:,9] * 3
    earliness = earliness.float() / 18.0

    output = output[:,0] * earliness + output[:,1] * (1.0 - earliness)
    output = torch.sigmoid(output)

    label = wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective)
    assert output.shape == label.shape, f"{output.shape} vs {label.shape}"
    loss = nn.functional.mse_loss(
      output, label, reduction='mean',
    )
    mse = loss.item()
    baseline = ((label - 0.5) ** 2).mean().item()

    loss.backward()
    opt.step()
    metrics["loss"].append(loss.item())
    metrics["mse"].append(mse / baseline)
    if (batch_idx + 1) % 100 == 0:
      print(f"loss: {np.mean(metrics['loss'][-1000:]):.4f}, mse: {np.mean(metrics['mse'][-1000:]):.4f}")

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

w = model.weight().detach().cpu().numpy()
b = model.bias.detach().cpu().numpy()

e = w[0]

scale = 100.0 / e[0,3:-1].mean()

for i, name in enumerate(names):
  plt.figure(figsize=(5,5))
  plt.imshow(e[i] * scale, cmap='seismic', vmin=e[i].min() * scale, vmax=e[i].max() * scale)
  plt.colorbar()
  plt.title(name)
  plt.savefig(os.path.join(run_dir, f'embedding-slice-{i}.png'))

# Save the model

with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
  torch.save(model.state_dict(), f)

with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
  for i, name in enumerate(names):
    save_tensor(torch.tensor(w[0,i]), 'e_' + name, f)
    save_tensor(torch.tensor(w[1,i]), 'l_' + name, f)
  save_tensor(torch.tensor(b), 'biases', f)

# plt.figure(figsize=(10,10))
# output = output.squeeze().cpu().detach().numpy()
# label = label.squeeze().cpu().detach().numpy()
# I = np.argsort(output)
# output, label = output[I], label[I]
# plt.scatter(output, label, alpha=0.1)
# plt.scatter(np.convolve(output, np.ones(100)/100, mode='valid'), np.convolve(label, np.ones(100)/100, mode='valid'), color='red', label='moving average')
# plt.savefig(os.path.join(run_dir, 'nnue-scatter.png'))

# plt.figure(figsize=(10,10))
# plt.plot(np.convolve(metrics['loss'][500:], np.ones(100)/100, mode='valid'), label='loss')
# plt.legend()
# plt.savefig(os.path.join(run_dir, 'nnue-loss.png'))


# board = chess.Board()
# X = []
# T = []
# moves = np.array(list(board.legal_moves))
# for move in moves:
#   board.push(move)
#   X.append(torch.tensor(board2x(board)).unsqueeze(0).to(device))
#   T.append(-1)
#   board.pop()

# output = model(torch.cat(X, dim=0))

# I = output.cpu().detach().numpy().argsort()[::-1]
# for i in I:
#   move = moves[i]
#   score = output[i].item()
#   print(f"{board.san(move):6s} {score: .4f}")

# white_winning = 'rnbqkbnr/pppppppp/8/8/2BPPB2/2N2N2/PPP2PPP/R2Q1RK1 w Qkq - 0 1'
# board = chess.Board(white_winning)
# output = model(
#   torch.tensor(board2x(board)).unsqueeze(0).to(device),
#   torch.tensor([1], device=device).unsqueeze(0),
# )[0][:,0]
# print(f"White winning position score: {output[0].item():.4f}")

# # loss: 0.0339, mse: 0.2576, penalty: 0.0019
# # Nf3     0.1300
# # e4      0.0932
# # d4      0.0859
# # e3      0.0574
# # Nc3     0.0509
# # c4      0.0509
# # g3      0.0504
# # b4      0.0237
# # g4      0.0216
# # d3      0.0211
# # c3      0.0112
# # a3      0.0069
# # h3      0.0011
# # Nh3    -0.0027
# # a4     -0.0108
# # h4     -0.0155
# # b3     -0.0220
# # Na3    -0.0391
# # f4     -0.1045
# # f3     -0.1073
# # White winning position score: 3.7974
