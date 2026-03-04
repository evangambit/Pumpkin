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
import time

import torch.utils.data as tdata
from sharded_matrix import ShardedLoader
from ShardedMatricesIterableDataset import ShardedMatricesIterableDataset, SingleShardedMatrixIterator, DynamicShardedMatrixIterator
from features import board2x, x2board
from accumulator import Emb
from nnue_model import NNUE

import dataset as ndata

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

def collate_fn(rows):
  values, lengths, labels, turns = zip(*rows)
  values = torch.from_numpy(np.concatenate(values))
  lengths = torch.from_numpy(np.concatenate(lengths))
  labels = torch.from_numpy(np.stack(labels))
  turns = torch.from_numpy(np.stack(turns))
  labels = labels.reshape(labels.shape[0] * labels.shape[1], *labels.shape[2:])
  turns = turns.reshape(turns.shape[0] * turns.shape[1], *turns.shape[2:])
  return values, lengths, labels, turns


if __name__ == "__main__":

  # Create a directory for this run
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  run_dir = os.path.join("runs", timestamp)
  os.makedirs(run_dir, exist_ok=True)

  device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

  print("Loading dataset...")
  dataset = ndata.NnueDataset(['../data/pos.shuf.txt'])

  print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

  dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

  print("Creating model...")
  model = NNUE(hidden_sizes=[512, 8], output_size=1).to(device)

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
  warmup_steps = warmup_length(0.9) # AdamW's beta is 0.999.
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
    t0 = time.time()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
      t_data = time.time()
      
      opt.zero_grad()
      
      # Update learning rate
      scheduler.step()
      
      batch = [x.to(device) for x in batch]
      values, lengths, label, turn = batch
      t_transfer = time.time()

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
      t_forward = time.time()
      
      mse = loss.item()
      baseline = ((label - label.mean()) ** 2).mean().item()

      (loss + penalty * 0.02).backward()
      opt.step()
      t_backward = time.time()
      metrics["loss"].append(loss.item())
      metrics["mse"].append(mse / baseline)
      metrics["penalty"].append(penalty.item())
      if batch_idx < 10:
        print(f"\nBatch {batch_idx}: data={t_data-t0:.4f}s transfer={t_transfer-t_data:.4f}s forward={t_forward-t_transfer:.4f}s backward={t_backward-t_forward:.4f}s")
      if (batch_idx + 1) % 500 == 0:
        print(f"loss: {np.mean(metrics['loss'][-1000:]):.4f}, mse: {np.mean(metrics['mse'][-1000:]):.4f}, penalty: {np.mean(metrics['penalty'][-1000:]):.4f}")
      
      t0 = time.time()

  # Save the model

  with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
    torch.save(model.state_dict(), f)

  with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
    save_tensor(model.emb.weight(model.emb.merged_tiles())[:-1], 'embedding', f)
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

