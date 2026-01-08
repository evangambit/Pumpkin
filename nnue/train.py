import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import chess
from collections import defaultdict
from tqdm import tqdm
import io

import torch.utils.data as tdata
from sharded_matrix import ShardedLoader
from ShardedMatricesIterableDataset import ShardedMatricesIterableDataset
from features import board2x, x2board, kMaxNumOnesInInput
from accumulator import Emb
from nnue_model import NNUE

# We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
# much for non-trivial networks though.
BATCH_SIZE = 2048
CHUNK_SIZE = 128
assert BATCH_SIZE % CHUNK_SIZE == 0

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

def evaluate_data_quality(dataset):
  import chess
  from chess import engine as chess_engine
  engine = chess_engine.SimpleEngine.popen_uci('/usr/games/stockfish')
  chunk = next(iter(dataset))
  correct, mse = 0, 0
  for i in range(100):
    x, y, turn, piece_counts = chunk
    x, y, turn, piece_counts = x[i], y[i], turn[i], piece_counts[i]
    assert x.shape == (kMaxNumOnesInInput,)
    assert y.shape == (3,)
    assert turn.shape == (1,)
    assert piece_counts.shape == (10,)
    board = x2board(x, turn)
    lines = engine.analyse(board, chess_engine.Limit(depth=7), multipv=1)
    wdl = lines[0]['score'].white().wdl()
    wdl = np.array([wdl.wins, wdl.draws, wdl.losses], dtype=np.int32)
    if y.argmax() == wdl.argmax():
      correct += 1
    mse += (((y.astype(np.float32) / 1000.0) - (wdl.astype(np.float32) / wdl.sum())) ** 2).mean()
  incorrect = 100 - correct
  print(f"Data quality: Accuracy: {correct}%, MSE: {mse / 100:.6f}")

evaluate_data_quality(dataset)

print("Creating model...")
 # [512, 128]
model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[512, 64], output_size=16).to(device)

print("Creating optimizer...")
opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.1)

earliness_weights = torch.tensor([
  # p    n    b    r    q
  0.0, 1.0, 1.0, 1.0, 3.0,
  0.0, 1.0, 1.0, 1.0, 3.0,
]).to(device) / 18.0

def wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective):
  return win_mover_perspective + draw_mover_perspective * 0.5

metrics = defaultdict(list)
print("Starting training...")
NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS):
  for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    opt.zero_grad()

    if batch_idx == 0:
      for pg in opt.param_groups:
        pg['lr'] = np.linspace(3e-3, 3e-5, NUM_EPOCHS)[epoch]
    
    batch = [v.reshape((BATCH_SIZE,) + v.shape[2:]).to(device) for v in batch]

    x, y_white_perspective, turn, piece_counts = batch
    y_white_perspective = y_white_perspective.float() / 1000.0

    win_mover_perspective = y_white_perspective[:,0:1] * (turn == 1).float() + y_white_perspective[:,2:3] * (turn == -1).float()
    draw_mover_perspective = y_white_perspective[:,1:2]
    lose_mover_perspective = y_white_perspective[:,2:3] * (turn == 1).float() + y_white_perspective[:,0:1] * (turn == -1).float()

    output = torch.sigmoid(model(x, turn))[:,0:1]
    label = wdl2score(win_mover_perspective, draw_mover_perspective, lose_mover_perspective)
    assert output.shape == label.shape
    loss = nn.functional.mse_loss(
      output, label, reduction='mean',
    )
    mse = loss.item()
    baseline = ((label - 0.5) ** 2).mean().item()

    loss.backward()
    opt.step()
    metrics["loss"].append(loss.item())
    metrics["mse"].append(mse / baseline)
    if batch_idx % 500 == 0:
      print(f"loss: {np.mean(metrics['loss'][-100:]):.4f}, mse: {np.mean(metrics['mse'][-100:]):.4f}")

def save_tensor(tensor: torch.Tensor, name: str, out: io.BufferedWriter):
  tensor = tensor.cpu().detach().numpy()
  name = name.ljust(16)
  assert len(name) == 16
  out.write(np.array([ord(c) for c in name], dtype=np.uint8).tobytes())
  out.write(np.array(len(tensor.shape), dtype=np.int32).tobytes())
  out.write(np.array(tensor.shape, dtype=np.int32).tobytes())
  out.write(tensor.tobytes())

# Save the model
with open('model.bin', 'wb') as f:
  save_tensor(model.emb.weight(False)[:-1], 'embedding', f)
  save_tensor(model.mlp[0].weight, 'linear0.weight', f)
  save_tensor(model.mlp[0].bias, 'linear0.bias', f)
  save_tensor(model.mlp[2].weight, 'linear1.weight', f)
  save_tensor(model.mlp[2].bias, 'linear1.bias', f)

plt.figure(figsize=(10,10))
yhat = yhat.squeeze().cpu().detach().numpy()
label = label.squeeze().cpu().detach().numpy()
I = np.argsort(yhat)
yhat, label = yhat[I], label[I]
plt.scatter(yhat, label, alpha=0.1)
plt.scatter(np.convolve(yhat, np.ones(100)/100, mode='valid'), np.convolve(label, np.ones(100)/100, mode='valid'), color='red', label='moving average')
plt.savefig('nnue-scatter.png')

plt.figure(figsize=(10,10))
plt.plot(np.convolve(metrics['loss'][500:], np.ones(100)/100, mode='valid'), label='loss')
plt.legend()
plt.savefig('nnue-loss.png')


board = chess.Board()
X = []
T = []
moves = np.array(list(board.legal_moves))
for move in moves:
  board.push(move)
  X.append(torch.tensor(board2x(board)).unsqueeze(0).to(device))
  T.append(-1)
  board.pop()

# Flip bc these are all from black's perspective
output = -model(torch.cat(X, dim=0), torch.tensor(T, device=device).unsqueeze(1))[:,0]

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
)[:,0]
print(f"White winning position score: {output[0].item():.4f}")

