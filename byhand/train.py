import torch
from torch import nn
import numpy as np
import os
import datetime
import time
from tqdm import tqdm
import torch.utils.data as tdata
import math
import io

import dataset as ndata

BATCH_SIZE = 2048
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

def collate_fn(rows):
    values, labels, turns = zip(*rows)
    values = torch.cat(values, dim=0)
    labels = torch.cat(labels, dim=0)
    turns = torch.cat(turns, dim=0)
    return values, labels, turns

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
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        
        self.current_step += 1

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    dataset = ndata.ByHandDataset(['../pos.shuf.txt'])
    
    print(f'Dataset chunk size: {CHUNK_SIZE}. Total length calculated dynamically.')

    dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    # Fetch one batch to get the number of features
    first_batch = next(iter(dataloader))
    values, labels, turns = first_batch
    num_features = values.shape[1]
    print(f"Number of features: {num_features}")

    model = nn.Linear(num_features, 2).to(device)
    
    # Optional: Initialize weights with small values
    nn.init.normal_(model.weight, 0, 0.01)
    nn.init.constant_(model.bias, 0.0)

    opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.01)
    
    NUM_EPOCHS = 1
    # We may not know the exact length if total_lines is not provided, 
    # but the dataloader len is estimated via wc -l by ByHandDataset.
    steps_per_epoch = len(dataloader)
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = total_steps // 20

    scheduler = CosineAnnealingWithWarmup(
        opt,
        max_lr=3e-3,
        min_lr=1e-5,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_losses = []
        
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            opt.zero_grad()
            scheduler.step()
            
            values, labels, turns = [x.to(device) for x in batch]
            values = values.to(torch.float32)

            earliness = values[:,1] + values[:,2] + values[:,3] + values[:,4] * 3
            earliness += values[:,6] + values[:,7] + values[:,8] + values[:,9] * 3
            earliness = earliness.clip(0, 18) / 18
            
            # Predict
            # Convert values to float for Linear layer
            output = model(values.float())
            output = output[:,0] * (1.0 - earliness) + output[:,1] * earliness
            
            # Use Sigmoid to match eval output
            output_sig = torch.sigmoid(output)
            label_sig = torch.sigmoid(labels)
            
            loss = nn.functional.mse_loss(output_sig, label_sig, reduction='mean')
            
            loss.backward()
            opt.step()
            
            epoch_losses.append(loss.item())
            
            if (batch_idx + 1) % 500 == 0:
                print(f"Step {batch_idx + 1}/{total_steps} | Loss: {np.mean(epoch_losses[-1000:]):.4f}")

    # Save the model
    print("Saving model to " + os.path.join(run_dir, 'model.pt'))
    with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)

    print("Saving model to " + os.path.join(run_dir, 'model.bin'))
    with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
        save_tensor(model.weight.data, 'weights', f)
        save_tensor(model.bias.data, 'bias', f)

    # Print out learned feature weights
    print("\nLearned Weights:")
    weights = model.weight.data.detach().cpu().numpy()
    bias = model.bias.detach().cpu().numpy()
    for i, w in enumerate(weights.T):
        print(f"Feature {i}: {np.round(w * 100)}")
    print(f"Bias: {np.round(bias * 100)}")
