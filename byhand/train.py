import torch
from torch import nn
from torch.nn import functional as F
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

WEIGHT_STAGES_EQUALLY = True
LABEL_SCALE = 1 / 2.0

def smooth(x):
    assert len(x.shape) == 1
    return nn.functional.conv1d(
        torch.cat([
            x[:1],
            x,
            x[-1:],
        ], dim=0).reshape(1, 1, -1),
        torch.ones((1,1,3), device=x.device) * (1.0 / 3.0),
    ).reshape(-1)

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

def collate_fn(rows):
    values, labels, turns, pst_values, pst_lengths = zip(*rows)
    values = torch.cat(values, dim=0)
    labels = torch.cat(labels, dim=0)
    turns = torch.cat(turns, dim=0)
    pst_values = torch.cat(pst_values, dim=0)
    pst_lengths = torch.cat(pst_lengths, dim=0)
    return values, labels, turns, pst_values, pst_lengths

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

class MyModel(nn.Module):
    def __init__(self, num_features):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(num_features, 2)
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0.0)
        self._pst = nn.Parameter(torch.zeros(6 * 64, 2), requires_grad=True)
    
    @property
    def pst(self):
        # We subtract the mean so that the PST values are centered around 0.
        # This prevents colinearities with the linear layer.
        r = self._pst.view(6, 64, 2)
        r = r - r.mean(dim=1, keepdim=True)
        return r.reshape(-1, 2)

    def forward(self, x, earliness, pst_values, pst_lengths):
        feature_hat = self.linear(x)
        pst_values = pst_values.to(torch.int64)
        pst_lengths = pst_lengths.to(torch.int64)
        offsets = pst_lengths.cumsum(0) - pst_lengths
        pst = self.pst
        pst_hat = F.embedding_bag(
            pst_values, 
            torch.cat([pst, -pst], dim=0),
            offsets=offsets,
            mode='sum'
        )
        z = feature_hat + pst_hat
        return z[:,0] * (1.0 - earliness) + z[:,1] * earliness

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    dataset = ndata.ByHandDataset(['../data/pos.100m.txt'])
    
    print(f'Dataset chunk size: {CHUNK_SIZE}. Total length calculated dynamically.')

    dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    # Fetch one batch to get the number of features
    first_batch = next(iter(dataloader))
    values, labels, turns, pst_values, pst_lengths = first_batch
    num_features = values.shape[1]
    print(f"Number of features: {num_features}")

    model = MyModel(num_features).to(device)

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

    if WEIGHT_STAGES_EQUALLY:
        earliness_values = torch.arange(19, dtype=torch.float32).to(device)
        earliness_weights = torch.ones(earliness_values.shape, dtype=torch.float32).to(device)

    step_count = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Starting Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_losses = []
        
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            step_count += 1
            opt.zero_grad()
            scheduler.step()
            
            values, labels, turns, pst_values, pst_lengths = [x.to(device) for x in batch]
            values = values.to(torch.float32)

            earliness = values[:,dataset.earliness_index]

            if WEIGHT_STAGES_EQUALLY:
                # 1-hot encoding of which bucket a datapoint falls into.
                which_bucket = (earliness_values.reshape(-1, 1) == earliness.reshape(1, -1)).to(torch.int32)
                with torch.no_grad():
                    earliness_weights *= 0.9
                    earliness_weights += which_bucket.sum(1) + 1.0
            
            earliness = earliness.clip(0, dataset.max_earliness) / dataset.max_earliness
            
            output = model(values.float(), earliness, pst_values, pst_lengths)
            
            # Use Sigmoid to match eval output
            output_sig = torch.sigmoid(output)
            label_sig = torch.sigmoid(labels * LABEL_SCALE)
            
            if WEIGHT_STAGES_EQUALLY:
                loss = nn.functional.mse_loss(output_sig, label_sig, reduction='none')
                weights = 1.0 / smooth(earliness_weights)[which_bucket.argmax(0)]
                weights = weights / weights.sum()
                loss = loss * weights
                loss = loss.sum()
            else:
                loss = nn.functional.mse_loss(output_sig, label_sig, reduction='mean')
            
            loss.backward()
            opt.step()
            
            epoch_losses.append(loss.item())
            
            if (batch_idx + 1) % 500 == 0:
                print(f"Step {batch_idx + 1}/{total_steps} | Loss: {np.mean(epoch_losses[-1000:]):.4f}")
            
            if step_count >= total_steps:
                break
        if step_count >= total_steps:
            break

    # Save the model
    print("Saving model to " + os.path.join(run_dir, 'model.pt'))
    with open(os.path.join(run_dir, 'model.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)

    print("Saving model to " + os.path.join(run_dir, 'model.bin'))
    with open(os.path.join(run_dir, 'model.bin'), 'wb') as f:
        save_tensor(model.linear.weight.data, 'weights', f)
        save_tensor(model.linear.bias.data, 'bias', f)
        save_tensor(model.pst[:,0].reshape(6, 64), 'pst_late', f)
        save_tensor(model.pst[:,1].reshape(6, 64), 'pst_early', f)

    # Print out learned feature weights
    print("\nLearned Weights:")
    weights = model.linear.weight.data.detach().cpu().numpy()
    bias = model.linear.bias.data.detach().cpu().numpy()
    scale = 100 / weights[0,1].mean()  # Make endgame pawns == 100
    for i, w in enumerate(weights.T):
        print(f"{dataset.feature_name(i)}: {np.round(w * scale)}")
    print(f"Bias: {np.round(bias * scale)}")
    for name, pst in [("early", model.pst[:,1]), ("late", model.pst[:,0])]:
        print(f"\n{name} PST:")
        pst = pst.view(6, 8, 8)
        for i in range(6):
            print(f"{dataset.feature_name(i + 1)}\n{np.round(pst[i].detach().cpu().numpy() * scale)}")

    dataset = ndata.ByHandDataset(['../data/pos.10m-test.txt'])
    dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    for batch in dataloader:
        values, labels, turns, pst_values, pst_lengths = [x.to(device) for x in batch]
        values = values.to(torch.float32)
        earliness = values[:,dataset.earliness_index].clip(0, dataset.max_earliness) / dataset.max_earliness
        output = model(values.float(), earliness, pst_values, pst_lengths)
        output_sig = torch.sigmoid(output)
        label_sig = torch.sigmoid(labels * LABEL_SCALE)
        print("Output:", output_sig[:10].cpu().detach().numpy())
        print("Labels:", label_sig[:10].cpu().detach().numpy())
        error = torch.abs(output_sig - label_sig).detach().cpu().numpy()
        I = np.argsort(-error)
        for i in I[:10]:
            print(f"Output: {output_sig[i].item():.4f} | Label: {label_sig[i].item():.4f} | Line: {i + 1}")
        break
