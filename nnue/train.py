import gc, time, math, os, random, copy, json, re, shutil

import torch
from torch import nn, optim
from torch.utils import data as tdata

import matplotlib.pyplot as plt
from scipy.ndimage import filters
import numpy as np

ss = np.swapaxes
cat = np.concatenate
gf = filters.uniform_filter1d
pjoin = os.path.join

# cd Pumpkin/nnue
# python setup.py build_ext --inplace

import dataset as ndata
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

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

def read_tensor(in_file: io.BufferedReader):
  name = ''.join([chr(c) for c in np.frombuffer(in_file.read(16), dtype=np.uint8)])
  dims = np.frombuffer(in_file.read(4), dtype=np.int32)[0]
  shape = np.frombuffer(in_file.read(dims * 4), dtype=np.int32)
  return name, np.frombuffer(in_file.read(), dtype=np.float32).reshape(shape)

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

def collate_fn(rows):
  values, lengths, labels, turns = zip(*rows)
  values = torch.from_numpy(np.concatenate(values))
  lengths = torch.from_numpy(np.concatenate(lengths))
  labels = torch.from_numpy(np.stack(labels))
  turns = torch.from_numpy(np.stack(turns))
  labels = labels.reshape(labels.shape[0] * labels.shape[1], *labels.shape[2:])
  turns = turns.reshape(turns.shape[0] * turns.shape[1], *turns.shape[2:])
  return values, lengths, labels, turns

def warmup_length(beta, c = 2.0):
  # The amount of warmup needs to increase as beta approaches 1,
  # since we need to see more data before the moving averages stabilize
  # to its long-run variability.
  return int(c / (1 - beta))

def wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective):
  assert len(win_mover_perspective.shape) == 1
  assert len(draw_mover_perspective.shape) == 1
  assert len(lose_mover_perspective.shape) == 1
  assert win_mover_perspective.shape == draw_mover_perspective.shape
  assert win_mover_perspective.shape == lose_mover_perspective.shape
  return win_mover_perspective + draw_mover_perspective * 0.5

if __name__ == '__main__':
  # We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
  # much for non-trivial networks though.
  BATCH_SIZE = 2048
  CHUNK_SIZE = 2048
  assert BATCH_SIZE % CHUNK_SIZE == 0

  # Create a directory for this run
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  run_dir = os.path.join("runs", timestamp)
  os.makedirs(run_dir, exist_ok=True)

  print("Run dir: ", run_dir)

  device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
  # device = torch.device('cpu')

  print("Loading dataset...")
  dataset = ndata.NnueDataset(['../data/pos.shuf.txt'])

  # evaluate_data_quality(dataset)

  # parts.aa  parts.ab  parts.ac  parts.ad	

  # Computing the length is the slow part
  print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

  dataloader = tdata.DataLoader(
      dataset, batch_size=BATCH_SIZE//CHUNK_SIZE,
      shuffle=False, num_workers=4,
      pin_memory=True, drop_last=True, collate_fn=collate_fn)


  # print("Creating model...")
  # teacher = NNUE(hidden_sizes=[2048, 512], output_size=1).to(device)
  # with open('/content/drive/My Drive/nnue_runs/20260223-151132/model.pt', 'rb') as f:
  #   teacher.load_state_dict(torch.load(f))
  # teacher.eval()
  # teacher = teacher.to(device)
  teacher = None

  # Emb.k = 24 + 4
  model = NNUE(hidden_sizes=[256, 32], output_size=1).to(device)
  # model = NNUE(hidden_sizes=[512, 32], output_size=1).to(device)
  # model = NNUE(hidden_sizes=[1024, 32], output_size=1).to(device)

  print("Creating optimizer...")
  from itertools import chain
  opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)


  # Calculate total steps
  NUM_EPOCHS = 3
  steps_per_epoch = len(dataloader)
  total_steps = NUM_EPOCHS * steps_per_epoch
  warmup_steps = warmup_length(0.999) # AdamW's beta is 0.999.
  assert warmup_steps < total_steps // 10, "You probably made a mistake."

  print(repr(model))
  print(device)

  with open(os.path.join(run_dir, 'model.txt'), 'w') as f:
    f.write(repr(model))

  scheduler = CosineAnnealingWithWarmup(
    opt,
    # max_lr=3e-3,
    # max_lr=3e-2,  # worse than 3e-3
    max_lr=1e-3,
    min_lr=3e-5,
    warmup_steps=warmup_steps,
    total_steps=total_steps
  )


  metrics = defaultdict(list)
  CLIP_VALUE = 1.0 # Added for gradient clipping

  for epoch in range(NUM_EPOCHS):
    print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
      t0 = time.time()
      opt.zero_grad()

      # Update learning rate
      scheduler.step()

      batch = [x.to(device) for x in batch]
      values, lengths, label, turns = batch

      output, layers = model(values, lengths)

      penalty = 0.0
      for layer_output in layers:
        penalty += (layer_output.mean() ** 2 + (layer_output.std() - 1.0) ** 2)

      output = torch.sigmoid(output)[:,0]
      label = torch.sigmoid(label)

      assert output.shape == label.shape, f"{output.shape} vs {label.shape}"
      loss = nn.functional.mse_loss(
        output, label, reduction='mean',
      )

      mse = loss.item()
      baseline = ((label - label.mean()) ** 2).mean().item()

      (loss + penalty * 0.02).backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE) # Added gradient clipping
      opt.step()
      metrics["loss"].append(loss.item())
      metrics["mse"].append(mse / baseline)
      metrics["penalty"].append(penalty.item())
      if (batch_idx + 1) % 500 == 0:
        print(f"loss: {np.mean(metrics['loss'][-1000:]):.4f}, mse: {np.mean(metrics['mse'][-1000:]):.4f}, penalty: {np.mean(metrics['penalty'][-1000:]):.4f}")
    
      values, lengths, label = None, None, None
      output, layers, loss = None, None, None
      t1 = time.time()

      num_iters = len(metrics['loss'])
      if num_iters % 400_000 == 0:
        name = num_iters // 400_000
        print('Saving model to', os.path.join(run_dir, f'model-{name}.pt'))
        with open(os.path.join(run_dir, f'model-{name}.pt'), 'wb') as f:
          torch.save(model.state_dict(), f)

        with open(os.path.join(run_dir, f'model-{name}.bin'), 'wb') as f:
          merged_tiles = model.emb.merged_tiles()
          save_tensor(model.emb.weight(merged_tiles)[:-1], 'embedding', f)
          save_tensor(model.mlp[0].weight, 'linear0.weight', f)
          save_tensor(model.mlp[0].bias, 'linear0.bias', f)
          save_tensor(model.mlp[2].weight, 'linear1.weight', f)
          save_tensor(model.mlp[2].bias, 'linear1.bias', f)

  # Save the model

  print('Saving model to', os.path.join(run_dir, 'model.pt'))
  with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
    torch.save(model.state_dict(), f)

  with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
    merged_tiles = model.emb.merged_tiles()
    save_tensor(model.emb.weight(merged_tiles)[:-1], 'embedding', f)
    save_tensor(model.mlp[0].weight, 'linear0.weight', f)
    save_tensor(model.mlp[0].bias, 'linear0.bias', f)
    save_tensor(model.mlp[2].weight, 'linear1.weight', f)
    save_tensor(model.mlp[2].bias, 'linear1.bias', f)

  plt.figure(figsize=(10,10))
  plt.plot(np.convolve(metrics['loss'][500:], np.ones(100)/100, mode='valid'), label='loss')
  plt.ylim(0.005, 0.013)
  plt.grid()
  plt.legend()
  plt.savefig(os.path.join(run_dir, 'nnue-loss.png'))

