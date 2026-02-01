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

def compute_x_black(x_white):
  mask = x_white < 768
  piece = x_white // 64
  square = x_white % 64
  rank = square // 8
  file = square % 8
  x_black = ((piece + 6) % 12) * 64 + (7 - rank) * 8 + file
  return torch.where(mask, x_black, torch.tensor(768, device=x_white.device, dtype=x_white.dtype))

def compute_x2(x_white):
  assert len(x_white.shape) == 2, f"x.shape = {x_white.shape}"

  mask = x_white < 768

  x_black = compute_x_black(x_white)

  x_white, x_black = x_white.to(torch.int32), x_black.to(torch.int32)

  x2_white = (x_white.reshape(-1, 1, 36) + x_white.reshape(-1, 36, 1) * 768).reshape(-1, 36 * 36)
  x2_black = (x_black.reshape(-1, 1, 36) + x_black.reshape(-1, 36, 1) * 768).reshape(-1, 36 * 36)

  # An interaction is only valid if neither piece is padding
  mask_2d_white = (x_white.reshape(-1, 1, 36) < 768) & (x_white.reshape(-1, 36, 1) < 768)
  mask_2d_black = (x_black.reshape(-1, 1, 36) < 768) & (x_black.reshape(-1, 36, 1) < 768)

  mask_2d_white = mask_2d_white.reshape(-1, 36 * 36)
  mask_2d_black = mask_2d_black.reshape(-1, 36 * 36)

  # Invalid interactions get mapped to the padding index
  x2_white = torch.where(mask_2d_white, x2_white, 768 * 768)
  x2_black = torch.where(mask_2d_black, x2_black, 768 * 768)

  return x2_white, x2_black

# class PST2(nn.Module):
#   def __init__(self):
#     super(PST2, self).__init__()
#     self.emb = nn.EmbeddingBag(768 * 768 + 1, 1, mode='sum', sparse=False)
#     nn.init.normal_(self.emb.weight, mean=0.0, std=np.sqrt(1 / (6 * kMaxNumOnesInInput * kMaxNumOnesInInput)))
#     self.piece = nn.EmbeddingBag(12 * 12 + 1, 1, mode='sum', sparse=False)
#     nn.init.normal_(self.piece.weight, mean=0.0, std=np.sqrt(1 / (6 * kMaxNumOnesInInput * kMaxNumOnesInInput)))
#     self.nofile = nn.EmbeddingBag(12 * 8 * 12 * 8 + 1, 1, mode='sum', sparse=False)
#     nn.init.normal_(self.nofile.weight, mean=0.0, std=np.sqrt(1 / (6 * kMaxNumOnesInInput * kMaxNumOnesInInput)))
#     self.norank = nn.EmbeddingBag(12 * 8 * 12 * 8 + 1, 1, mode='sum', sparse=False)
#     nn.init.normal_(self.norank.weight, mean=0.0, std=np.sqrt(1 / (6 * kMaxNumOnesInInput * kMaxNumOnesInInput)))
#     self.nofilerank = nn.EmbeddingBag(12 * 8 * 12 * 8 + 1, 1, mode='sum', sparse=False)
#     nn.init.normal_(self.nofilerank.weight, mean=0.0, std=np.sqrt(1 / (6 * kMaxNumOnesInInput * kMaxNumOnesInInput)))
#     self.norankfile = nn.EmbeddingBag(12 * 8 * 12 * 8 + 1, 1, mode='sum', sparse=False)
#     nn.init.normal_(self.norankfile.weight, mean=0.0, std=np.sqrt(1 / (6 * kMaxNumOnesInInput * kMaxNumOnesInInput)))

#     # emb[(a, WHITE), (a, WHITE)] = - emb[(a, BLACK), (a, BLACK)]

#     # emb[(a, WHITE), (b, WHITE)] = emb[(b, WHITE), (a, WHITE)]

#     # emb[(a, WHITE), (b, WHITE)] = - emb[(a, BLACK), (b, WHITE)]

#     # emb[(a, WHITE), (b, BLACK)] = - emb[(a, BLACK), (b, WHITE)]

#   def forward(self, x):
#     x = x.to(torch.int32)
#     assert len(x.shape) == 2
#     assert x.shape[1] == kMaxNumOnesInInput
#     batch_size = x.shape[0]
#     x2_white, x2_black = compute_x2(x)
#     offsets = torch.arange(0, batch_size * x2_white.shape[1], x2_white.shape[1], device=x.device, dtype=torch.int32)

#     i = x2_white // 768
#     j = x2_white % 768

#     piece_1 = i // 64
#     piece_2 = j // 64

#     sq_1 = i % 64
#     sq_2 = j % 64

#     rank_1 = sq_1 // 8
#     file_1 = sq_1 % 8
#     rank_2 = sq_2 // 8
#     file_2 = sq_2 % 8

#     mask = x2_white < 768 * 768

#     pieces_index = piece_1 * 12 + piece_2
#     pieces_index = torch.where(
#       mask, pieces_index, torch.tensor(12 * 12, device=x.device, dtype=pieces_index.dtype)
#     )

#     nofile_index = piece_1 * 8 * 12 * 8 + rank_1 * 12 * 8 + piece_2 * 8 + rank_2
#     norank_index = piece_1 * 8 * 12 * 8 + file_1 * 12 * 8 + piece_2 * 8 + file_2
#     nofile_rank_index = piece_1 * 8 * 12 * 8 + rank_1 * 12 * 8 + piece_2 * 8 + file_2
#     norank_file_index = piece_1 * 8 * 12 * 8 + file_1 * 12 * 8 + piece_2 * 8 + rank_2

#     output = self.emb(x2_white.reshape(-1), offsets)
#     # output -= self.emb(x2_black.reshape(-1), offsets)
#     output += self.piece(pieces_index.reshape(-1), offsets)
#     output += self.nofile(nofile_index.reshape(-1), offsets)
#     output += self.norank(norank_index.reshape(-1), offsets)
#     output += self.nofilerank(nofile_rank_index.reshape(-1), offsets)
#     output += self.norankfile(norank_file_index.reshape(-1), offsets)
#     return output.squeeze()

"""
The basic idea is we can have efficient, conditional, quantized piece-square tables by using bitmaps.

Ex:

x: Bitmap

weights:
 4  0  0 -4
 0  0  0  4
 0  0  4  0
 0  0  0  0

Can be computed easily as:

val = (popcount(x & positive_mask) - popcount(x & negative_mask)) * 4

This opens up cool possibilities like having different PSTs based on piece counts, threats, etc.

For example, we can learn weights for each square that indicates how important the safety of
that square is. Then we can do stuff like.

for piece in pieces:
  x = threats.badFor[piece]  # Bitmap for whether it is safe for the piece to be on each square

  score += popcount(x & threats_pst[piece].positive_mask) * threats_pst[piece].positive_weight
  score -= popcount(x & threats_pst[piece].negative_mask) * threats_pst[piece].negative_weight

  x &= kDistFromKing[kingSquare][4]
  score += popcount(x & king_safety_pst[piece].positive_mask) * king_safety_pst[piece].positive_weight
  score -= popcount(x & king_safety_pst[piece].negative_mask) * king_safety_pst[piece].negative_weight

We can also do PSTs conditioned on king-location:

for piece in pieces:
  w = king_location_pst[piece][kingSquare]
  score += popcount(x & w.positive_mask) * w.positive_weight
  score -= popcount(x & w.negative_mask) * w.negative_weight

"""

class PST2(nn.Module):
  def __init__(self, dim_embedding=2048):
    super(PST2, self).__init__()
    self.pst = nn.Embedding(12 * 64 + 1, 1)
    self.emb = nn.Embedding(12 * 64 + 1, dim_embedding)
    nn.init.normal_(self.emb.weight, mean=0.0, std=np.sqrt(1 / (kMaxNumOnesInInput * dim_embedding)))
    nn.init.normal_(self.pst.weight, mean=0.0, std=np.sqrt(1 / (kMaxNumOnesInInput)))
  
  def forward(self, x):
    pst = self.pst(x.to(torch.int64)).sum((1, 2))  # (BS,)
    if np.random.rand() < 1.0:
      return pst
    z = self.emb(x.to(torch.int64))  # (BS, 36, dim_embedding)
    dot = torch.bmm(z, z.transpose(1, 2))  # (BS, 36, 36)
    return dot.sum((1, 2)) + pst

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
  f'data/{dataset_name}/tables-nnue',
  f'data/{dataset_name}/tables-eval',
  f'data/{dataset_name}/tables-turn',
  f'data/{dataset_name}/tables-piece-counts',
  chunk_size=CHUNK_SIZE,
)


print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

print("Creating model...")
 # [512, 128]
model = PST2().to(device)

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

    x, y_white_perspective, turn, piece_counts = batch
    y_white_perspective = y_white_perspective.float() / 1000.0

    win_mover_perspective = y_white_perspective[:,0] * (turn.squeeze() == 1).float() + y_white_perspective[:,2] * (turn.squeeze() == -1).float()
    draw_mover_perspective = y_white_perspective[:,1]
    lose_mover_perspective = y_white_perspective[:,2] * (turn.squeeze() == 1).float() + y_white_perspective[:,0] * (turn.squeeze() == -1).float()

    output = model(x) * turn.squeeze()

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

# Save the model

with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
  torch.save(model.state_dict(), f)

with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
  save_tensor(model.emb.weight(False)[:-1], 'embedding', f)
  save_tensor(model.mlp[0].weight, 'linear0.weight', f)
  save_tensor(model.mlp[0].bias, 'linear0.bias', f)
  save_tensor(model.mlp[2].weight, 'linear1.weight', f)
  save_tensor(model.mlp[2].bias, 'linear1.bias', f)
  save_tensor(model.mlp[4].weight, 'linear2.weight', f)
  save_tensor(model.mlp[4].bias, 'linear2.bias', f)

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
T = []
moves = np.array(list(board.legal_moves))
for move in moves:
  board.push(move)
  X.append(torch.tensor(board2x(board)).unsqueeze(0).to(device))
  T.append(-1)
  board.pop()

output = model(torch.cat(X, dim=0))

I = output.cpu().detach().numpy().argsort()[::-1]
for i in I:
  move = moves[i]
  score = output[i].item()
  print(f"{board.san(move):6s} {score: .4f}")

white_winning = 'rnbqkbnr/pppppppp/8/8/2BPPB2/2N2N2/PPP2PPP/R2Q1RK1 w Qkq - 0 1'
board = chess.Board(white_winning)
output = model(
  torch.tensor(board2x(board)).unsqueeze(0).to(device),
  torch.tensor([1], device=device).unsqueeze(0),
)[0][:,0]
print(f"White winning position score: {output[0].item():.4f}")

# loss: 0.0339, mse: 0.2576, penalty: 0.0019
# Nf3     0.1300
# e4      0.0932
# d4      0.0859
# e3      0.0574
# Nc3     0.0509
# c4      0.0509
# g3      0.0504
# b4      0.0237
# g4      0.0216
# d3      0.0211
# c3      0.0112
# a3      0.0069
# h3      0.0011
# Nh3    -0.0027
# a4     -0.0108
# h4     -0.0155
# b3     -0.0220
# Na3    -0.0391
# f4     -0.1045
# f3     -0.1073
# White winning position score: 3.7974
