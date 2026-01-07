import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import chess
from collections import defaultdict
from tqdm import tqdm

import torch.utils.data as tdata
from sharded_matrix import ShardedLoader

kMaxNumOnesInInput = 32 + 4

def x2board(vec, turn):
  assert len(vec.shape) == 1
  assert vec.shape[0] == kMaxNumOnesInInput
  board = chess.Board.empty()
  board.turn = turn
  castling = ''
  for val in vec:
    if val >= 768:
      continue
    if (val >= 7 and val < 56) or val > 63:
      piece = val // 64
      val = val % 64
      y = 7 - val // 8
      x = val % 8
      if not ((piece == 0 or piece == 6) and (y == 0 or y == 7)):
        # Ignore pawns on first or last rank, since these are used
        # for special features (see castling rights below).
        sq = chess.square(x, y)
        board.set_piece_at(sq, chess.Piece(piece % 6 + 1, chess.WHITE if piece < 6 else chess.BLACK))
        continue
    if val == 0:
      castling += 'K'
    if val == 1:
      castling += 'Q'
    if val == 440:
      castling += 'k'
    if val == 441:
      castling += 'q'
  # board.set_castling_fen(castling) doesn't work?
  parts = board.fen().split(' ')
  parts[2] = castling if castling else '-'
  return chess.Board(' '.join(parts))

def board2x(board):
  vec = np.ones(kMaxNumOnesInInput, dtype=np.int16) * 768
  i = 0
  for sq, piece in board.piece_map().items():
    val = 0
    val += 'PNBRQKpnbrqk'.index(piece.symbol()) * 64
    val += 8 * (7 - sq // 8)
    val += sq % 8
    assert val not in [0, 1, 440, 441], f"Piece on square {chess.square_name(sq)} conflicts with castling rights encoding."
    assert val < 768
    vec[i] = val
    i += 1
  if board.has_kingside_castling_rights(chess.WHITE):
    vec[i] = 0
    i += 1
  if board.has_queenside_castling_rights(chess.WHITE):
    vec[i] = 1
    i += 1
  if board.has_kingside_castling_rights(chess.BLACK):
    vec[i] = 440
    i += 1
  if board.has_queenside_castling_rights(chess.BLACK):
    vec[i] = 441
    i += 1
  assert i <= kMaxNumOnesInInput
  vec.sort()
  return vec

class SingleShardedMatrixIterator:
  def __init__(self, xpath, chunk_size=1):
    self.X = ShardedLoader(xpath)
    self.chunk_size = chunk_size

  def __iter__(self):
    shard_index = 0
    offset = 0
    x = self.X.load_shard(shard_index)
    while True:
      if offset + self.chunk_size >= x.shape[0]:
        first_half = x[offset:]
        shard_index += 1
        if shard_index >= self.X.num_shards:
          break
        x = self.X.load_shard(shard_index)
        second_half = x[0:self.chunk_size - first_half.shape[0]]
        yield np.concatenate([first_half, second_half], axis=0).copy()
        offset = self.chunk_size - first_half.shape[0]
      else:
        yield x[offset:offset + self.chunk_size].copy()
        offset += self.chunk_size
  
  def __len__(self):
    return self.X.num_rows // self.chunk_size

class SimpleIterablesDataset(tdata.IterableDataset):
  def __init__(self, *paths, chunk_size=1):
    super().__init__()
    self.iterators = [SingleShardedMatrixIterator(p, chunk_size=chunk_size) for p in paths]
  
  def __iter__(self):
    its = [iter(it) for it in self.iterators]
    yield from zip(*its)
  
  def __len__(self):
    return len(self.iterators[0])

class CReLU(nn.Module):
  def forward(self, x):
    return x.clip(0, 1)

class Emb(nn.Module):
  def __init__(self, dout):
    super().__init__()
    din = 768

    # Roughly speaking
    # variance(z[i]) = (var(t1) + var(t2) + var(t3) + var(t4) + var(t5) + var(t6)) * kMaxNumOnesInInput
    # So if we want variance(z[i]) = 1.0, we want each var(ti) = 1.0 / (kMaxNumOnesInInput * 6)
    num_components = 6
    targeted_variance_per_component = 1.0
    s = (targeted_variance_per_component / (kMaxNumOnesInInput * num_components)) ** 0.5
    
    self.tiles = nn.Parameter(torch.randn(12, 8, 8, dout) * s)
    self.coord = nn.Parameter(torch.randn(1, 8, 8, dout) * s)
    self.piece = nn.Parameter(torch.randn(12, 1, 1, dout) * s)
    self.row = nn.Parameter(torch.randn(1, 8, 1, dout) * s)
    self.col = nn.Parameter(torch.randn(1, 1, 8, dout) * s)
    self.tilecolor = nn.Parameter(torch.randn(1, 1, 1, dout) * s)

    # We juse pawns on the first/last rank to indicate things like castling rights,
    # so we don't want coord, piece, row, col, tilecolor to apply to them. We do,
    # however, still want to flip them in the vertically_flipped case.
    self.special_mask = nn.Parameter(torch.zeros(12, 8, 8, dout), requires_grad=False)
    with torch.no_grad():
      for piece in range(12):
        for y in range(8):
          for x in range(8):
            if (piece % 6 == 0) and (y == 0 or y == 7):
              self.special_mask[piece, y, x, :] = 1.0

    self.white_tile_mask = nn.Parameter(torch.zeros(1, 8, 8, 1), requires_grad=False)
    with torch.no_grad():
      for y in range(8):
        for x in range(8):
          self.white_tile_mask[0, y, x, 0] = (y + x) % 2 == 0
    
    self.zeros = nn.Parameter(torch.zeros(1, dout), requires_grad=False)
  
  def zero_(self):
    with torch.no_grad():
      self.tiles.zero_()
      self.coord.zero_()
      self.piece.zero_()
      self.row.zero_()
      self.col.zero_()
      self.tilecolor.zero_()

  def weight(self, vertically_flipped=False):
    """
    Return the embedding weight matrix.

    We could implement this whole class as as an nn.Embedding,
    but this way we can
    # 1) support vertical flipping
    # 2) apply higher regularization (theoretically at least) 

    Shape: (768 + 1, dout)
    """
    tilecolor = self.tilecolor * self.white_tile_mask
    
    factorized_terms = (
      self.coord
      +
      self.piece
      +
      self.row
      +
      self.col
      +
      tilecolor
    ) * (1.0 - self.special_mask)

    T = factorized_terms + self.tiles

    if vertically_flipped:
      # Flip the pieces (White <-> Black) and the ranks.
      T = T.flip(1).roll(6, dims=0)

    batch_size = T.shape[0]
    return torch.cat([T.reshape(768, -1), self.zeros], dim=0)
  
  def forward(self, x, vertically_flipped=False):
    assert len(x.shape) == 2
    assert x.shape[1] == kMaxNumOnesInInput, f"x.shape={x.shape}"
    return self.weight(vertically_flipped=vertically_flipped)[x.to(torch.int32)].sum(1)


class NNUE(nn.Module):
  def __init__(self, input_size, hidden_sizes: list[int], output_size: int):
    super(NNUE, self).__init__()
    self.emb = Emb(dout=hidden_sizes[0])
    layers = []
    hidden_sizes[0] *= 2
    for i in range(len(hidden_sizes) - 1):
      layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
      layers.append(CReLU())
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    for layer in layers:
      if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)
    self.mlp = nn.Sequential(*layers)

  def embed(self, x, turn):
    assert len(x.shape) == 2
    assert len(turn.shape) == 2
    assert x.shape[0] == turn.shape[0]
    assert turn.shape[1] == 1
    white_z = self.emb(x, vertically_flipped=False)
    black_z = self.emb(x, vertically_flipped=True)
    mover = turn * white_z + (1 - turn) * black_z
    non_mover = (1 - turn) * white_z + turn * black_z
    z = torch.cat([mover, non_mover], dim=1)
    """
    white pawn on f2 when white moves
    ==
    black pawn on f7 when black moves
    """
    return z

  def forward(self, x, turn):
    z = self.embed(x, turn)
    out = self.mlp(z)
    return out

def test():
  model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[768], output_size=2)
  model.emb.zero_()
  for i in range(768):
    with torch.no_grad():
      model.emb.tiles[i // 64, (i % 64) // 8, (i % 64) % 8, i] = 1.0

  def remove_all(lst, val):
    return [x for x in lst if x != val]

  def foo(fen: str):
    x = torch.tensor(board2x(chess.Board(fen)), dtype=torch.int16).unsqueeze(0)
    z_white = model.emb(x, vertically_flipped=False)
    z_black = model.emb(x, vertically_flipped=True)
    if ' w ' in fen:
      return x, torch.cat([z_white, z_black], dim=1)
    else:
      return x, torch.cat([z_black, z_white], dim=1)
  start_pos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
  e4_c5 = 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR'
  c4_e5 = 'rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR'
  white_missing_pieces = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1K3'

  x0, z0 = foo(f'{start_pos} w KQkq - 0 1')  # starting position
  x1, z1 = foo(f'{e4_c5} w KQkq - 0 2')  # 1. e4 c5 (sicilian)
  x2, z2 = foo(f'{c4_e5} w KQkq - 0 2')  # 1. c4 e5 (reverse sicilian)

  x3, z3 = foo(f'{c4_e5} b KQkq - 0 2')  # 1. c4 e5 but black to move (should match z1)
  x4, z4 = foo(f'{e4_c5} b KQkq - 0 2')  # 1. e4 c5 but black to move (should match z2)
  x5, z5 = foo(f'{white_missing_pieces} w Qkq - 0 1')  # white missing pieces

  x0 = remove_all(x0.squeeze().tolist(), 768)
  x1 = remove_all(x1.squeeze().tolist(), 768)
  x2 = remove_all(x2.squeeze().tolist(), 768)
  x3 = remove_all(x3.squeeze().tolist(), 768)
  x4 = remove_all(x4.squeeze().tolist(), 768)
  x5 = remove_all(x5.squeeze().tolist(), 768)

  # On white's turn, the first half of z is white's pieces.
  assert np.nonzero(z0.squeeze()).squeeze().tolist()[:len(x0)] == x0
  assert np.nonzero(z1.squeeze()).squeeze().tolist()[:len(x1)] == x1
  assert np.nonzero(z2.squeeze()).squeeze().tolist()[:len(x2)] == x2
  assert np.nonzero(z5.squeeze()).squeeze().tolist()[:len(x5)] == x5
  
  # On black's turn, the second half of z is black
  assert (np.nonzero(z3.squeeze()).squeeze() - 768).tolist()[len(x3):] == x3
  assert (np.nonzero(z4.squeeze()).squeeze() - 768).tolist()[len(x4):] == x4

  i5 = np.nonzero(z5[0].detach().numpy())[0]

  assert torch.allclose(z1, z3)
  assert torch.allclose(z2, z4)
  assert not torch.allclose(z1, z2)
  assert not torch.allclose(z3, z4)

test()

# We load data in chunks, rather than 1 row at a time, as it is much faster. It doesn't matter
# much for non-trivial networks though.
BATCH_SIZE = 2048
CHUNK_SIZE = 64
assert BATCH_SIZE % CHUNK_SIZE == 0

device = torch.cuda.current_device()

print("Loading dataset...")
dataset = SimpleIterablesDataset(
  f'data/de6-md2/tables-nnue',
  f'data/de6-md2/tables-eval',
  f'data/de6-md2/tables-turn',
  f'data/de6-md2/tables-piece-counts',
  chunk_size=CHUNK_SIZE,
)
dataloader = tdata.DataLoader(dataset, batch_size=BATCH_SIZE//CHUNK_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

print("Creating model...")
 # [512, 128]
model = NNUE(input_size=kMaxNumOnesInInput, hidden_sizes=[1024, 128], output_size=6).to(device)

print("Creating optimizer...")
opt = torch.optim.AdamW([model.emb.piece], lr=0.0, weight_decay=0.1)

earliness_weights = torch.tensor([
  # p    n    b    r    q
  0.0, 1.0, 1.0, 1.0, 3.0,
  0.0, 1.0, 1.0, 1.0, 3.0,
]).to(device) / 18.0

metrics = defaultdict(list)
print("Starting training...")
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
  for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    opt.zero_grad()

    if batch_idx == 10:
      for pg in opt.param_groups:
        pg['lr'] = np.linspace(3e-3, 3e-5, NUM_EPOCHS)[epoch]
    
    batch = [v.reshape((BATCH_SIZE,) + v.shape[2:]).to(device) for v in batch]

    x, y_white_perspective, turn, piece_counts = batch
    y_white_perspective = y_white_perspective.float() / 1000.0

    win_mover_perspective = y_white_perspective[:,0:1] * (turn == 1).float() + y_white_perspective[:,2:3] * (turn == -1).float()
    draw_mover_perspective = y_white_perspective[:,1:2]
    lose_mover_perspective = y_white_perspective[:,2:3] * (turn == 1).float() + y_white_perspective[:,0:1] * (turn == -1).float()

    earliness = piece_counts.float().matmul(earliness_weights).unsqueeze(1)  # Shape: (batch_size, 1)

    columns = torch.cat([
      earliness,
      (1.0 - earliness),
    ], 1)

    output = model(x, (turn + 1) // 2)
    output = output.reshape((output.shape[0], 2, 3))
    output = (nn.functional.softmax(output, dim=2) * columns.unsqueeze(2)).sum(1)  # Shape: (batch_size, 3) (win, draw, loss)
    # yhat = (output * columns).sum(1, keepdim=True)
    yhat = output[:,0:1] + output[:,1:2] * 0.5

    label = win_mover_perspective + draw_mover_perspective * 0.5

    # TODO: for some reason my network is learning negative scores.

    # loss = nn.functional.mse_loss(torch.sigmoid(yhat), label, reduction='mean')
    loss = nn.functional.mse_loss(yhat, label, reduction='mean')
    loss.backward()
    opt.step()
    metrics["loss"].append(loss.item())
    if batch_idx % 500 == 0:
      print(f"loss: {np.mean(metrics['loss'][-100:]):.4f}")
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
board.push_san('e4')
X = []
T = []
moves = np.array(list(board.legal_moves))
for move in moves:
  board.push(move)
  X.append(torch.tensor(board2x(board)).unsqueeze(0).to(device))
  T.append(-1)
  board.pop()

output = model(torch.cat(X, dim=0), torch.tensor(T, device=device).unsqueeze(1))

I = output[:,0].cpu().detach().numpy().argsort()[::-1]
for i in I:
  move = moves[i]
  score = output[i,0].item()
  print(f"{board.san(move):6s} {score: .4f}")

