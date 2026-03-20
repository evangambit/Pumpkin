#!/usr/bin/env python3
"""Play two UCI chess engines against each other in a match."""

"""
python3 match.py \
--engine1 "./old \"evaluator byhand\"" \
--engine2 "./uci \"evaluator byhand byhand/runs/20260316-172658/model.bin\"" \
--tc movetime=10 --concurrency=4 --games=10000 --opening 6mvs_+90_+99.epd
"""

import argparse
import math
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import chess
import chess.pgn


_active_processes_lock = threading.Lock()
_active_processes = set()


def _kill_all_engines():
  """Kill all active engine subprocesses to unblock workers."""
  with _active_processes_lock:
    for proc in list(_active_processes):
      try:
        proc.kill()
      except Exception:
        pass


class UCIEngine:
  def __init__(self, path, options=None):
    self.path = path
    self.cmd = shlex.split(path)
    self.name = self.cmd[0]
    self.process = subprocess.Popen(
      self.cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
    )
    with _active_processes_lock:
      _active_processes.add(self.process)
    self._send("uci")
    self._wait_for("uciok")
    if options:
      for opt in options:
        name, _, value = opt.partition("=")
        self.set_option(name.strip(), value.strip())
    self._send("isready")
    self._wait_for("readyok")

  def _send(self, cmd):
    try:
      self.process.stdin.write((cmd + "\n").encode())
      self.process.stdin.flush()
    except (BrokenPipeError, OSError):
      raise RuntimeError(f"Engine {self.name} crashed (broken pipe)")

  def _read_line(self):
    line = self.process.stdout.readline().decode()
    if line == "":
      raise RuntimeError(f"Engine {self.name} crashed (empty read)")
    return line.strip()

  def _wait_for(self, token):
    while True:
      line = self._read_line()
      if token == "uciok" and line.startswith("id name "):
        self.name = line[len("id name "):]
      if line.startswith(token):
        return line
    return ""

  def set_option(self, name, value=""):
    if value:
      self._send(f"setoption name {name} value {value}")
    else:
      self._send(f"setoption name {name}")

  def new_game(self):
    self._send("ucinewgame")
    self._send("isready")
    self._wait_for("readyok")

  def go(self, fen, moves, go_params):
    if moves:
      self._send(f"position fen {fen} moves {' '.join(moves)}")
    else:
      self._send(f"position fen {fen}")
    self._send(f"go {go_params}")
    info_lines = []
    while True:
      line = self._read_line()
      if line.startswith("info "):
        info_lines.append(line)
      if line.startswith("bestmove"):
        best = line.split()[1]
        return best, info_lines

  def quit(self):
    with _active_processes_lock:
      _active_processes.discard(self.process)
    try:
      self._send("quit")
      self.process.wait(timeout=5)
    except Exception:
      self.process.kill()


def parse_time_control(tc_str):
  """Parse a time control string.

  Formats:
    nodes=N    -> {"type": "nodes", "nodes": N}
    depth=N    -> {"type": "depth", "depth": N}
    movetime=N   -> {"type": "movetime", "movetime": N}  (milliseconds)
    S+I      -> {"type": "clock", "time": S*1000, "inc": I*1000}  (seconds)
    S      -> {"type": "clock", "time": S*1000, "inc": 0}
  """
  if tc_str.startswith("nodes="):
    return {"type": "nodes", "nodes": int(tc_str.split("=")[1])}
  if tc_str.startswith("depth="):
    return {"type": "depth", "depth": int(tc_str.split("=")[1])}
  if tc_str.startswith("movetime="):
    return {"type": "movetime", "movetime": int(tc_str.split("=")[1])}
  # Clock-based: "60+0.5" or "60"
  parts = tc_str.split("+")
  base = float(parts[0])
  inc = float(parts[1]) if len(parts) > 1 else 0.0
  return {"type": "clock", "time": int(base * 1000), "inc": int(inc * 1000)}


def make_go_params(tc, wtime_ms, btime_ms, is_white):
  if tc["type"] == "nodes":
    return f"nodes {tc['nodes']}"
  if tc["type"] == "depth":
    return f"depth {tc['depth']}"
  if tc["type"] == "movetime":
    return f"movetime {tc['movetime']}"
  return f"wtime {wtime_ms} btime {btime_ms} winc {tc['inc']} binc {tc['inc']}"


def random_opening(n_plies=4):
  board = chess.Board()
  for _ in range(n_plies):
    moves = list(board.legal_moves)
    if not moves:
      break
    board.push(random.choice(moves))
  return board.fen()


def _play_single_game(engine_w, engine_b, tc, opening_fen=chess.STARTING_FEN):
  """Play a single game. Returns (result_str, pgn_game)."""
  board = chess.Board(opening_fen)
  game = chess.pgn.Game()
  game.setup(board)

  game.headers["Event"] = "Engine Match"
  game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
  game.headers["White"] = engine_w.name
  game.headers["Black"] = engine_b.name
  if tc["type"] == "clock":
    base_s = tc["time"] / 1000
    inc_s = tc["inc"] / 1000
    game.headers["TimeControl"] = f"{base_s:g}+{inc_s:g}"
  else:
    game.headers["TimeControl"] = "-"

  try:
    engine_w.new_game()
    engine_b.new_game()
  except RuntimeError as e:
    print(f"  Engine crash: {e}")
    game.headers["Result"] = "1/2-1/2"
    game.headers["Termination"] = "engine crash"
    return "1/2-1/2", game

  wtime_ms = tc.get("time", 0)
  btime_ms = tc.get("time", 0)

  node = game
  fen0 = board.fen()
  moves_uci = []

  while not board.is_game_over(claim_draw=True):
    if board.fullmove_number > 250:
      break

    is_white = board.turn == chess.WHITE
    engine = engine_w if is_white else engine_b
    go_params = make_go_params(tc, wtime_ms, btime_ms, is_white)

    t0 = time.monotonic()
    try:
      best_move, _ = engine.go(fen0, moves_uci, go_params)
    except RuntimeError as e:
      print(f"  Engine crash: {e}")
      # The crashed engine loses
      result = "0-1" if is_white else "1-0"
      game.headers["Result"] = result
      game.headers["Termination"] = "engine crash"
      return result, game

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if tc["type"] == "clock":
      if is_white:
        wtime_ms -= elapsed_ms
        wtime_ms += tc["inc"]
        if wtime_ms <= 0:
          result = "0-1"
          game.headers["Result"] = result
          game.headers["Termination"] = "time forfeit"
          return result, game
      else:
        btime_ms -= elapsed_ms
        btime_ms += tc["inc"]
        if btime_ms <= 0:
          result = "1-0"
          game.headers["Result"] = result
          game.headers["Termination"] = "time forfeit"
          return result, game

    try:
      move = chess.Move.from_uci(best_move)
    except ValueError:
      print(f"  Invalid UCI move: {best_move}")
      result = "0-1" if is_white else "1-0"
      game.headers["Result"] = result
      return result, game

    if move not in board.legal_moves:
      print(f"  Illegal move: {best_move} in position {board.fen()}")
      result = "0-1" if is_white else "1-0"
      game.headers["Result"] = result
      return result, game

    board.push(move)
    moves_uci.append(best_move)
    node = node.add_variation(move)

  outcome = board.outcome(claim_draw=True)
  if outcome is None:
    result = "1/2-1/2"
  elif outcome.winner is None:
    result = "1/2-1/2"
  elif outcome.winner == chess.WHITE:
    result = "1-0"
  else:
    result = "0-1"

  game.headers["Result"] = result
  return result, game


def play_game(e1, e2, tc, opening_fen=chess.STARTING_FEN):
  """Play a game pair from the same position with colors swapped.

  Returns (games, pair_score) where:
    games: list of (result_str, pgn_game) for each game
    pair_score: score from e1's perspective in [-0.5, 0.5]
  """
  result1, pgn1 = _play_single_game(e1, e2, tc, opening_fen)
  result2, pgn2 = _play_single_game(e2, e1, tc, opening_fen)

  def _score(result, e1_is_white):
    if result == "1-0":
      return 0.5 if e1_is_white else -0.5
    if result == "0-1":
      return -0.5 if e1_is_white else 0.5
    return 0.0

  pair_score = (_score(result1, True) + _score(result2, False)) / 2.0
  # e1_was_white: True for game 1, False for game 2
  return [(result1, pgn1, True), (result2, pgn2, False)], pair_score


def play_pair_worker(engine1_path, engine2_path, options1, options2, tc, fen):
  """Worker: spawn engines, play a pair, quit engines."""
  e1 = UCIEngine(engine1_path, options1)
  e2 = UCIEngine(engine2_path, options2)
  try:
    games, pair_score = play_game(e1, e2, tc, fen)
  finally:
    e1.quit()
    e2.quit()
  return games, pair_score


def elo_diff(wins, draws, losses):
  total = wins + draws + losses
  if total == 0:
    return 0.0, 0.0
  score = (wins + draws * 0.5) / total
  if score <= 0.0 or score >= 1.0:
    return float("inf") if score >= 1.0 else float("-inf"), 0.0
  elo = -400.0 * math.log10(1.0 / score - 1.0)
  # Approximate standard error
  win_rate = score
  variance = (wins * (1 - win_rate) ** 2 + draws * (0.5 - win_rate) ** 2 + losses * (0 - win_rate) ** 2) / total
  se = math.sqrt(variance / total) if total > 0 else 0
  if se > 0:
    elo_se = 400.0 * se / (math.log(10) * score * (1 - score))
  else:
    elo_se = 0.0
  return elo, elo_se


def significance_test(pair_scores):
  """One-sample t-test on pair scores. Returns (p_value, avg, stderr)."""
  n = len(pair_scores)
  if n < 5:
    return 1.0, 0.0, 0.0
  avg = sum(pair_scores) / n
  variance = sum((x - avg) ** 2 for x in pair_scores) / (n - 1)
  stderr = math.sqrt(variance / n)
  if stderr == 0:
    return 0.0 if avg != 0 else 1.0, avg, stderr
  t_stat = avg / stderr
  # Two-tailed p-value using normal approximation (good for n >= 5)
  # P(|Z| > |t|) ≈ 2 * erfc(|t| / sqrt(2)) / 2
  p_value = math.erfc(abs(t_stat) / math.sqrt(2))
  return p_value, avg, stderr


def main():
  parser = argparse.ArgumentParser(description="Play two UCI engines against each other")
  parser.add_argument("--engine1", required=True, help="Path to first engine")
  parser.add_argument("--engine2", required=True, help="Path to second engine")
  parser.add_argument("--games", type=int, default=10, help="Number of game pairs (default: 10, so 20 games total)")
  parser.add_argument("--tc", default="nodes=10000",
            help="Time control: nodes=N, depth=N, movetime=N (ms), or S+I (seconds+increment)")
  parser.add_argument("--pgn", default="match.pgn", help="Output PGN file (default: match.pgn)")
  parser.add_argument("--opening", default="startpos",
            help="Opening type: 'startpos', 'random', or path to an EPD file (default: startpos)")
  parser.add_argument("--concurrency", type=int, default=1,
            help="Number of game pairs to play in parallel (default: 1)")
  parser.add_argument("--alpha", type=float, default=0.01,
            help="Significance level for early stopping (default: 0.01, 0 to disable)")
  parser.add_argument("--min_num_games", type=int, default=100,
            help="Minimum number of game pairs before early stopping can occur (default: 5)")
  parser.add_argument("--option1", action="append", default=[],
            help="UCI option for engine1, e.g. 'Hash=64' (repeatable)")
  parser.add_argument("--option2", action="append", default=[],
            help="UCI option for engine2, e.g. 'Hash=64' (repeatable)")
  args = parser.parse_args()

  tc = parse_time_control(args.tc)

  # Probe engines to get their names
  print(f"Starting engine 1: {args.engine1}")
  probe1 = UCIEngine(args.engine1, args.option1)
  e1_name = probe1.name
  probe1.quit()
  print(f"  -> {e1_name}")

  print(f"Starting engine 2: {args.engine2}")
  probe2 = UCIEngine(args.engine2, args.option2)
  e2_name = probe2.name
  probe2.quit()
  print(f"  -> {e2_name}")

  # Load opening positions
  opening_label = args.opening
  epd_fens = []
  if args.opening not in ("startpos", "random"):
    with open(args.opening) as f:
      for line in f:
        line = line.strip()
        if line:
          epd_fens.append(line)
    if not epd_fens:
      print(f"Error: EPD file '{args.opening}' is empty")
      sys.exit(1)
    random.shuffle(epd_fens)
    opening_label = f"{args.opening} ({len(epd_fens)} positions)"

  print(f"\nMatch: {e1_name} vs {e2_name}")
  print(f"Game pairs: {args.games}, TC: {args.tc}, Opening: {opening_label}, Concurrency: {args.concurrency}")
  if args.alpha > 0:
    print(f"Early stopping: alpha={args.alpha}, min_num_games={args.min_num_games}")
  print(f"PGN output: {args.pgn}")
  print()

  wins, draws, losses = 0, 0, 0
  pair_scores = []
  completed = 0

  # Pre-generate openings
  fens = []
  for i in range(args.games):
    if epd_fens:
      fens.append(epd_fens[i % len(epd_fens)])
    elif args.opening == "random":
      fens.append(random_opening())
    else:
      fens.append(chess.STARTING_FEN)

  with open(args.pgn, "w") as pgn_file, \
       ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    futures = {}
    next_pair = 0
    stopped_early = False

    def submit_batch():
      nonlocal next_pair
      # Keep up to concurrency jobs in flight
      while len(futures) < args.concurrency and next_pair < args.games:
        fen = fens[next_pair]
        future = pool.submit(
          play_pair_worker,
          args.engine1, args.engine2,
          args.option1, args.option2,
          tc, fen,
        )
        futures[future] = next_pair + 1
        next_pair += 1

    submit_batch()

    while futures:
      # Wait for next completion
      done_iter = as_completed(futures)
      future = next(done_iter)
      pair_num = futures.pop(future)
      try:
        games, pair_score = future.result()
      except Exception as e:
        print(f"  Pair {pair_num}: ERROR - {e}")
        submit_batch()
        continue

      completed += 1
      pair_scores.append(pair_score)

      for result, pgn_game, e1_was_white in games:
        if result == "1-0":
          if e1_was_white:
            wins += 1
          else:
            losses += 1
        elif result == "0-1":
          if not e1_was_white:
            wins += 1
          else:
            losses += 1
        else:
          draws += 1
        print(pgn_game, file=pgn_file, end="\n\n")
      pgn_file.flush()

      r1, r2 = games[0][0], games[1][0]
      p_value, avg, stderr = significance_test(pair_scores)
      print(f"  Pair {pair_num}: {e1_name} as W: {r1}, as B: {r2}  pair={pair_score:+.2f}  [{completed}/{args.games}]  [+{wins}-{losses}={draws}]    p={p_value:.3f}  [{avg - stderr * 1.96:.4f}, {avg + stderr * 1.96:.4f}]")

      # Early stopping check
      if args.alpha > 0 and len(pair_scores) >= args.min_num_games and p_value < args.alpha:
        print(f"\n  *** Stopping early: p={p_value:.4f} < alpha={args.alpha} after {completed} pairs ***")
        stopped_early = True
        _kill_all_engines()
        for f in futures:
          f.cancel()
        futures.clear()
        break

      submit_batch()

  total_games = wins + draws + losses
  print(f"\n{'='*50}")
  print(f"Match result ({e1_name} vs {e2_name}):")
  print(f"  +{wins} -{losses} ={draws}  ({total_games} games)")
  if total_games > 0:
    score_pct = (wins + draws * 0.5) / total_games * 100
    print(f"  Score: {score_pct:.1f}%")
  if len(pair_scores) > 1:
    avg = sum(pair_scores) / len(pair_scores)
    variance = sum((x - avg) ** 2 for x in pair_scores) / (len(pair_scores) - 1)
    stderr = math.sqrt(variance / len(pair_scores))
    print(f"  Pair score: {avg:+.4f} +/- {stderr:.4f}")
    try:
      elo_lo = math.log(1 / (0.5 + avg - stderr) - 1.0) / math.log(10) * -400
      elo_hi = math.log(1 / (0.5 + avg + stderr) - 1.0) / math.log(10) * -400
      print(f"  Elo: {elo_lo:.1f} to {elo_hi:.1f}")
    except (ValueError, ZeroDivisionError):
      pass
    p_value, _, _ = significance_test(pair_scores)
    if avg > 0:
      print(f"  {e1_name} is stronger (p={p_value:.4f})")
    elif avg < 0:
      print(f"  {e2_name} is stronger (p={p_value:.4f})")
    else:
      print(f"  No difference detected (p={p_value:.4f})")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    _kill_all_engines()
    print("\n  Interrupted.")
