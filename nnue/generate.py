import argparse
import asyncio
import random
import re
import sqlite3
import os
import subprocess
import time

from collections import defaultdict
from multiprocessing import Process, Queue

import chess
from chess import engine as chess_engine

"""
a=de7-md4
sqlite3 data/${a}/db.sqlite3 "select * from positions" > data/${a}/pos.txt

shuf data/${a}/pos.txt > data/${a}/pos.shuf.txt
or
terashuf < data/${a}/pos.txt > data/${a}/pos.shuf.txt

./make_tables data/${a}/pos.shuf.txt data/${a}/tables
"""

def wdl2score(wdl):
  return (wdl.wins + wdl.draws * 0.5) / (wdl.wins + wdl.draws + wdl.losses)

def score2float(score):
  wdl = score.white().wdl()
  return (wdl.wins + wdl.draws * 0.5) / (wdl.wins + wdl.draws + wdl.losses)

def repetition_key(board):
  return board.epd()

def repetition_state_after_push(board, move, seen_positions, repeated_since_reset):
  is_capture = board.is_capture(move)
  is_pawn_move = board.piece_type_at(move.from_square) == chess.PAWN
  board.push(move)

  if is_capture or is_pawn_move:
    return {repetition_key(board): 1}, False

  key = repetition_key(board)
  seen_positions = dict(seen_positions)
  seen_positions[key] = seen_positions.get(key, 0) + 1
  return seen_positions, repeated_since_reset or seen_positions[key] >= 2

def analyzer(resultQueue, args):
  engine = chess_engine.SimpleEngine.popen_uci(args.engine)
  while True:
    helper(engine, resultQueue, args)

def helper(engine, resultQueue, args):
  board = chess.Board()
  seen_positions = {repetition_key(board): 1}
  repeated_since_reset = False
  t = ''
  while not board.is_game_over() and not board.is_repetition() and board.ply() < 200:
    lines = engine.analyse(board, chess_engine.Limit(depth=args.depth), multipv=args.multipv)

    if len(lines) < args.multipv:
      seen_positions, repeated_since_reset = repetition_state_after_push(
        board,
        lines[0]['pv'][0],
        seen_positions,
        repeated_since_reset,
      )
      continue

    for line in lines:
      wdl = line['score'].white().wdl()
      moves = line['pv']
      if len(moves) < args.min_depth:
        continue
      b = board.copy()
      b_seen_positions, b_repeated_since_reset = repetition_state_after_push(
        b,
        moves[0],
        seen_positions,
        repeated_since_reset,
      )
      is_quiet = False
      for i in range(1, len(moves) - args.min_depth):
        san = b.san(moves[i])
        if 'x' not in san and '+' not in san and '=' not in san and '#' not in san:
          is_quiet = True
          break
        b_seen_positions, b_repeated_since_reset = repetition_state_after_push(
          b,
          moves[i],
          b_seen_positions,
          b_repeated_since_reset,
        )
      if is_quiet and not b_repeated_since_reset and random.randint(0, 99) >= args.dropout:
        resultQueue.put((b.fen(), wdl.wins, wdl.draws, wdl.losses))
      b = None

    # Drop blunders
    lines = [l for l in lines if abs(score2float(l['score']) - score2float(lines[0]['score'])) < 0.1]

    if score2float(lines[0]['score']) < 0.25 and board.turn:
      # If white is losing, make the best move
      line = lines[0]
    elif score2float(lines[0]['score']) > 0.75 and not board.turn:
      # If black is losing, make the best move
      line = lines[0]
    else:
      # Pick a random move (biased towards better moves)
      L = []
      for i, line in enumerate(lines[::-1]):
        L += [line] * (i + 1)
      line = random.choice(L)
    seen_positions, repeated_since_reset = repetition_state_after_push(
      board,
      line['pv'][0],
      seen_positions,
      repeated_since_reset,
    )

from functools import lru_cache
class MyLru:
  def foo(self, x):
    self.answer = False

  def __init__(self, n):
    self.f = lru_cache(maxsize=n)(self.foo)

  def __call__(self, x):
    self.answer = True
    self.f(x)
    return self.answer

def sql_inserter(resultQueue, args, database):
  conn = sqlite3.connect(database)
  c = conn.cursor()
  c.execute('CREATE TABLE IF NOT EXISTS positions (fen TEXT PRIMARY KEY, wins INTEGER, draws INTEGER, losses INTEGER)')
  conn.commit()

  c.execute('SELECT COUNT(1) FROM positions')
  n = c.fetchone()[0]

  t = time.time()

  cache = MyLru(50_000)

  while True:
    row = resultQueue.get()
    fen = row[0]
    if cache(fen):
      continue
    c.execute('INSERT OR IGNORE INTO positions VALUES (' + ', '.join('?' * len(row)) + ')', row)
    n += 1
    if n % 500 == 499:
      print(f'Inserted {("%.4f" % ((n + 1) / 1_000_000)).rjust(7)}M positions ({str(int(500 / (time.time() - t))).rjust(4)} / sec)')
      t = time.time()
      conn.commit()

"""
Rather than saving (64x13 = 832) columns, we can save 32 integers indicating the location and piece type of each piece.
Then we can use nn.Embedding to convert these integers to a vector.

We can also save the castling rights and en passant square as integers.

[P1, P2, ..., P32, castlingWhite, castlingBlack, enPassantWhite, enPassantBlack]

P1, P2, ..., P32 are integers from 0 to (64 * 12 - 1) and we'll use 16-bit integers to store them.

"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--engine', default='/usr/games/stockfish')
  parser.add_argument('--depth', type=int, default=7)
  parser.add_argument('--multipv', type=int, default=5)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--min_depth', type=int, default=4)
  parser.add_argument('--dropout', type=int, default=75, help='Probability of dropping a position (0-100). Helps promote higher diversity.')
  args = parser.parse_args()

  database = os.path.join('data', f'de{args.depth}-md{args.min_depth}', f'db.sqlite3')
  if not os.path.exists(os.path.dirname(database)):
    os.makedirs(os.path.dirname(database))

  resultQueue = Queue()

  analyzers = [Process(target=analyzer, args=(resultQueue, args)) for _ in range(args.num_workers)]
  for p in analyzers:
    p.start()
  
  sql_inserter(resultQueue, args, database)

