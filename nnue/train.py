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
import json

import torch.utils.data as tdata
from sharded_matrix import ShardedLoader
from ShardedMatricesIterableDataset import ShardedMatricesIterableDataset, SingleShardedMatrixIterator, DynamicShardedMatrixIterator
from features import board2x, x2board
from accumulator import Emb, kKingBuckets
from nnue_model import NNUE, OLDNNUE

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
BATCH_SIZE = 4096
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

def collate_fn(rows):
  values, lengths, labels, kings, lateness = zip(*rows)
  values = torch.from_numpy(np.concatenate(values))
  lengths = torch.from_numpy(np.concatenate(lengths))
  labels = torch.from_numpy(np.stack(labels))
  kings = torch.from_numpy(np.concatenate(kings))
  lateness = torch.from_numpy(np.concatenate(lateness))
  labels = labels.reshape(labels.shape[0] * labels.shape[1], *labels.shape[2:])
  return values, lengths, labels, kings, lateness


if __name__ == "__main__":

  # Create a directory for this run
  timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  run_dir = os.path.join("runs", timestamp)
  os.makedirs(run_dir, exist_ok=True)

  # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cpu')

  print("Loading dataset...")
  dataset = ndata.NnueDataset([r'data/de7-md4/pos.shuf.txt'])
  # dataset = ndata.NnueDataset([r'data/de7-md4/tiny.txt'])

  print(f'Dataset loaded with {len(dataset) * CHUNK_SIZE} rows.')

  dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

  print("Creating model...")
  # Simple hiiden_sizes=[1] yields (loss: 0.0262, mse: 0.6076, penalty: 0.2377)
  # loss: 0.0142, mse: 0.3296, penalty: 0.0075

  # teacher = None
  teacher = OLDNNUE(hidden_sizes=[1024, 256, 128], output_size=1).to(device)
  with open('runs/20260618-021209/model.pt', 'rb') as f:
    teacher.load_state_dict(torch.load(f))
  teacher.eval()

  model = NNUE(hidden_sizes=[384, 16], output_size=1).to(device)
  # model = NNUE(hidden_sizes=[1024, 256, 128], output_size=1).to(device)

  print("Creating optimizer...")
  opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=1.0)

  def warmup_length(beta, c = 2.0):
    # The amount of warmup needs to increase as beta approaches 1,
    # since we need to see more data before the moving averages stabilize
    # to its long-run variability.
    return int(c / (1 - beta))

  # Calculate total steps
  NUM_EPOCHS = 2
  steps_per_epoch = len(dataloader)
  total_steps = NUM_EPOCHS * steps_per_epoch
  warmup_steps = warmup_length(0.999) # AdamW's beta is 0.999.
  assert warmup_steps < total_steps // 10, "You probably made a mistake."
  print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

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

  config = {
    'batch_size': BATCH_SIZE,
    'model': str(model).splitlines(),
    'dataset': dataset.file_paths,
    'scheduler': {
      'min_lr': scheduler.min_lr,
      'max_lr': scheduler.max_lr,
      'warmup_steps': scheduler.warmup_steps,
      'total_steps': scheduler.total_steps,
    },
    'opt': str(opt).splitlines(),
  }

  metrics = defaultdict(list)
  num_models_saved = 0
  last_save_time = 0
  for epoch in range(NUM_EPOCHS):
    print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
    t0 = time.time()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=steps_per_epoch):
      t_data = time.time()
      
      opt.zero_grad()
      
      # Update learning rate
      scheduler.step()
      
      batch = [x.to(device) for x in batch]
      values, lengths, label, kings, lateness = batch
      lateness = lateness.float().clip(0, 18) / 18.0
      t_transfer = time.time()

      if batch_idx == 0:
        print(values.max(), lengths.max())

      output, layers, _ = model(values, lengths, kings, lateness.unsqueeze(1))
      assert len(output.shape) == 2 and output.shape[1] == 1

      penalty = 0.0
      for layer_output in layers:
        penalty += (layer_output.mean() ** 2 + (layer_output.std() - 1.0) ** 2)

      output = torch.sigmoid(output)[:,0]

      if teacher is not None:
        with torch.no_grad():
          teacher_output, _, _ = teacher(values, lengths, kings)
          teacher_output = torch.sigmoid(teacher_output)[:,0]
          label = label * 0.5 + teacher_output * 0.5

      assert output.shape == label.shape, f"{output.shape} vs {label.shape}"
      loss = (torch.abs(output - label)**2.5).mean()
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
      
      # Save a model every 10 minutes.
      if time.time() - last_save_time > 10 * 60:
        with open(os.path.join(run_dir, f'model-{num_models_saved}.bin'), 'wb') as f:
          save_tensor(model.emb.weight(model.emb.merged_tiles()), 'embedding', f)
          for i, layer in enumerate(model.mlp[::2]):
            assert isinstance(layer, nn.Linear)
            save_tensor(layer.weight, f'linear{i}.weight', f)
            save_tensor(layer.bias, f'linear{i}.bias', f)
        num_models_saved += 1
        last_save_time = time.time()
      
      t0 = time.time()

  # Save the model

  with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
    torch.save(model.state_dict(), f)

  with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
    save_tensor(model.emb.weight(model.emb.merged_tiles()), 'embedding', f)
    for i, layer in enumerate(model.mlp[::2]):
      assert isinstance(layer, nn.Linear)
      save_tensor(layer.weight, f'linear{i}.weight', f)
      save_tensor(layer.bias, f'linear{i}.bias', f)

  plt.figure(figsize=(10,10))
  output = output.squeeze().cpu().detach().numpy()
  label = label.squeeze().cpu().detach().numpy()
  I = np.argsort(output)
  output, label = output[I], label[I]
  plt.xlabel('Predicted Score')
  plt.ylabel('Actual Score')
  plt.scatter(output, label, alpha=0.1)
  plt.scatter(np.convolve(output, np.ones(50)/50, mode='valid'), np.convolve(label, np.ones(50)/50, mode='valid'), color='red', label='moving average (50)')
  plt.scatter(np.convolve(output, np.ones(200)/200, mode='valid'), np.convolve(label, np.ones(200)/200, mode='valid'), color='red', label='moving average (200)')
  plt.savefig(os.path.join(run_dir, 'nnue-scatter.png'))

  plt.figure(figsize=(10,10))
  plt.plot(np.convolve(metrics['loss'][1000:], np.ones(50)/50, mode='valid'), label='loss (smooth=50)')
  plt.plot(np.convolve(metrics['loss'][1000:], np.ones(500)/500, mode='valid'), label='loss (smooth=200)')
  plt.plot(metrics['loss'][1000:], label='loss', alpha=0.3)
  plt.grid()
  plt.legend()
  plt.savefig(os.path.join(run_dir, 'nnue-loss.png'))

  with open(os.path.join(run_dir, 'config.json'), 'w') as f:
    config['batches/epoch'] = batch_idx + 1
    config['epochs'] = epoch + 1
    config['sched_final'] = str(opt)
    json.dump(config, f, indent=2)
