import torch
from torch import nn
import numpy as np
import os
import datetime
import time
from tqdm import tqdm
import torch.utils.data as tdata
import math

import dataset as ndata

BATCH_SIZE = 2048
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

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

    model = nn.Linear(num_features, 1).to(device)
    
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
            
            # Predict
            # Convert values to float for Linear layer
            output = model(values.float()).squeeze(-1)
            
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
    with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)

    # Print out learned feature weights
    print("\nLearned Weights:")
    weights = model.weight.data.squeeze().cpu().numpy()
    bias = model.bias.item()
    for i, w in enumerate(weights):
        print(f"Feature {i}: {w:.5f}")
    print(f"Bias: {bias:.5f}")
