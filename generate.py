import argparse
import asyncio
import random
import re
import sqlite3
import os
import subprocess
import time

from collections import defaultdict, OrderedDict
from multiprocessing import Process, Queue

import chess
from chess import engine as chess_engine

class LRUSet:
  def __init__(self, capacity):
    self.capacity = capacity
    self.cache = OrderedDict()

  def test_and_add(self, key):
    if key in self.cache:
      self.cache.move_to_end(key)
      return True
    self.cache[key] = None
    if len(self.cache) > self.capacity:
      self.cache.popitem(last=False)
    return False

def writer(resultQueue, args):
  cache = LRUSet(100_000)
  total_saved = 0
  with open(args.out_file, 'a') as f:
    while True:
      row = resultQueue.get()
      # Use EPD for cache to ignore move counts when deduplicating positions
      if cache.test_and_add(row[2]):
        continue
      f.write(f'{row[0]}|{row[1]}\n')
      total_saved += 1
      if total_saved % 100 == 0:
        print(f'Saved {total_saved} positions.')
      f.flush()

def analyzer(resultQueue, args):
  with chess_engine.SimpleEngine.popen_uci(args.engine) as engine:
    while True:
      helper(engine, resultQueue, args)

def helper(engine, resultQueue, args):
  board = chess.Board()
  t = ''
  while not board.is_game_over() and not board.is_repetition() and board.ply() < 200:
    limit = chess_engine.Limit(time=args.movetime / 1000)
    lines = engine.analyse(board, limit, multipv=args.multipv)

    # Skip moves that are too forced.
    if len(lines) <= 3:
      board.push(lines[0]['pv'][0])
      continue

    best_move = lines[0]['pv'][0]
    is_best_move_quiet = (not board.is_capture(best_move) and 
                          not board.gives_check(best_move) and 
                          not best_move.promotion)
    score = lines[0]['score'].white().score(mate_score=10000) / 100.0
    if board.turn == chess.BLACK:
      score = -score
    
    if board.ply() >= args.min_ply and is_best_move_quiet and random.random() > args.pdrop:
      resultQueue.put((board.fen(), score, board.epd()))

    # Pick a random move (biased towards better moves)
    L = []
    for i, line in enumerate(lines[::-1]):
      L += [line] * (i + 1)
    line = random.choice(L)
    board.push(line['pv'][0])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--engine', default='/opt/homebrew/bin/stockfish')
  parser.add_argument('--movetime', type=int, default=50, help='Time to search for each move in milliseconds')
  parser.add_argument('--multipv', type=int, default=5, help='Number of moves to search for each position')
  parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
  parser.add_argument('--pdrop', type=float, default=0.9, help='Probability of dropping a move')
  parser.add_argument('--min_depth', type=int, default=4, help='Minimum depth to search to')
  parser.add_argument('--min_ply', type=int, default=10)
  parser.add_argument('--out_file', type=str, default='out.txt')
  args = parser.parse_args()

  resultQueue = Queue()
  analyzers = [Process(target=analyzer, args=(resultQueue, args)) for _ in range(args.num_workers)]
  for p in analyzers:
    p.start()

  writer(resultQueue, args)
