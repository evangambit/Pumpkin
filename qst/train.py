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

# shuf data/stock/pos.txt > data/stock/pos.shuf.txt

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

names = [
  'p|base_psq',
  'n|base_psq',
  'b|base_psq',
  'r|base_psq',
  'q|base_psq',
  'k|base_psq',

  'k|no_castle',
  'passed_pawns',
  'isolated_pawns',
  'doubled_pawns',

  'bad_for_p',
  'bad_for_n',
  'bad_for_b',
  'bad_for_r',
  'bad_for_q',
  'bad_for_k',

  'hanging_p',
  'hanging_n',
  'hanging_b',
  'hanging_r',
  'hanging_q',
  'hanging_k',

  'badSqNearK_p',
  'badSqNearK_b',
  'badSqNearK_k',
  'pInFrontOfK',
  'pawnStorm',
  'adjacentPawns',
  'diagonalPawns',
]

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

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

class MyModel(nn.Module):
  def __init__(self, input_dim):
    super(MyModel, self).__init__()
    input_dim //= 2  # mover/waiter symmetry
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

NUM_EPOCHS = 1
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
    f'data/{dataset_name}/qst-qst',
    f'data/{dataset_name}/qst-eval',
    f'data/{dataset_name}/qst-turn',
    f'data/{dataset_name}/qst-piece-counts',
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
    f.write(f'Dataset: de7-md4\n')
    f.write(f'Batch size: {BATCH_SIZE}\n')
    f.write(f'Chunk size: {CHUNK_SIZE}\n')
    f.write(f'Max learning rate: {maxlr}\n')
    f.write(f'Weight decay: {WEIGHT_DECAY}\n')


  print("Creating model...")
  # [512, 128]
  dim = next(iter(dataset))[0].shape[1]
  assert dim // 64 == len(names) * 2, f"{dim // 64} vs {len(names) * 2}"
  model = MyModel(dim).to(device)

  print("Creating optimizer...")
  opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=WEIGHT_DECAY, betas=BETAS)

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

      x, wdl, turn, piece_counts = batch
      wdl = wdl.float() / 1000.0
      label = wdl2score(wdl[:,0], wdl[:,1], wdl[:,2])

      if epoch == 0 and batch_idx == 0:
        regr = np.linalg.lstsq(x[:,:768].reshape(-1, 12, 64).sum(2).cpu().detach().numpy(), label.cpu().detach().numpy())[0]
        print("Initial linear regression weights:", regr)
      
      output = model(x.to(torch.float32))

      earliness = piece_counts[:,1] + piece_counts[:,2] + piece_counts[:,3] + piece_counts[:,4] * 3
      earliness += piece_counts[:,6] + piece_counts[:,7] + piece_counts[:,8] + piece_counts[:,9] * 3
      earliness = earliness.float() / 18.0

      output = output[:,0] * earliness + output[:,1] * (1.0 - earliness)
      output = torch.sigmoid(output)

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

  loss = np.array(metrics["loss"])
  mse = np.array(metrics["mse"])
  with open(os.path.join(run_dir, 'metrics.txt'), 'w') as f:
    f.write(f'Final loss: %.5f (%.5f)\n' % (loss[-100:].mean(), loss.std() / 10))
    f.write(f'Final mse: %.5f (%.5f)\n' % (mse[-100:].mean(), mse.std() / 10))

  w = model.weight().detach().cpu().numpy()
  b = model.bias.detach().cpu().numpy()

  e = w[0]

  scale = 100.0 / e[0,3:-1].mean()

  if not os.path.exists(os.path.join(run_dir, 'imgs')):
    os.makedirs(os.path.join(run_dir, 'imgs'))

  plt.figure(figsize=(5,5))
  plt.plot(np.convolve(loss, np.ones(100)/100, mode='valid'), label='loss')
  plt.savefig(os.path.join(run_dir, 'nnue-loss.png'))

  plt.figure(figsize=(10,10))
  plt.scatter(output.cpu().detach().numpy(), label.cpu().detach().numpy(), alpha=0.1)
  plt.xlabel('output')
  plt.ylabel('label')
  plt.savefig(os.path.join(run_dir, 'nnue-scatter.png'))

  for i, name in enumerate(names):
    plt.figure(figsize=(5,5))
    plt.imshow(e[i] * scale, cmap='seismic', vmin=e[i].min() * scale, vmax=e[i].max() * scale)
    plt.colorbar()
    plt.title(name)
    plt.savefig(os.path.join(run_dir, 'imgs', f'embedding-slice-{i}.png'))

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
