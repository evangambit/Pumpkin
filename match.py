#!/usr/bin/env python3
"""Round-robin tournament between N UCI chess engines."""

"""
python3 match.py \
  --engine "./old \"evaluator byhand\"" \
  --engine "./uci \"evaluator byhand\"" \
  --engine "./uci \"evaluator nnue nnue/runs/.../model.bin\"" \
  --tc movetime=10 --concurrency=4 --games=10000 --opening 6mvs_+90_+99.epd --alpha 0.01
"""

import argparse
import itertools
import math
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import json
import os
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
  def __init__(self, path, options=None, timeout=30):
    self.path = path
    self.cmd = shlex.split(path)
    self.name = self.cmd[0]
    self.timeout = timeout
    self.process = None
    try:
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
    except Exception:
      self._cleanup_process()
      raise

  def _cleanup_process(self):
    if self.process is None:
      return
    with _active_processes_lock:
      _active_processes.discard(self.process)
    try:
      self.process.kill()
    except Exception:
      pass
    try:
      self.process.wait(timeout=5)
    except Exception:
      pass
    self.process = None

  def _send(self, cmd):
    try:
      self.process.stdin.write((cmd + "\n").encode())
      self.process.stdin.flush()
    except (BrokenPipeError, OSError):
      raise RuntimeError(f"Engine {self.name} crashed (broken pipe)")

  def _read_line(self):
    result = [None]
    def _reader():
      result[0] = self.process.stdout.readline()
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=self.timeout)
    if t.is_alive():
      self.process.kill()
      raise RuntimeError(f"Engine {self.name} timed out after {self.timeout}s")
    line = result[0].decode()
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
    if self.process is None:
      return
    proc = self.process
    with _active_processes_lock:
      _active_processes.discard(proc)
    try:
      self._send("quit")
      proc.wait(timeout=5)
    except Exception:
      try:
        proc.kill()
      except Exception:
        pass
    finally:
      self.process = None


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


def random_opening(n_plies):
  if n_plies <= 0:
    n_plies = 4
  board = chess.Board()
  for _ in range(n_plies):
    moves = list(board.legal_moves)
    if not moves:
      break
    board.push(random.choice(moves))
  return board.fen()


def _engine_label_from_path(path):
  cmd = shlex.split(path)
  return cmd[0] if cmd else path


def _make_forfeit_game(white_name, black_name, tc, opening_fen, result, termination):
  board = chess.Board(opening_fen)
  game = chess.pgn.Game()
  game.setup(board)

  game.headers["Event"] = "Engine Match"
  game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
  game.headers["White"] = white_name
  game.headers["Black"] = black_name
  if tc["type"] == "clock":
    base_s = tc["time"] / 1000
    inc_s = tc["inc"] / 1000
    game.headers["TimeControl"] = f"{base_s:g}+{inc_s:g}"
  else:
    game.headers["TimeControl"] = "-"
  game.headers["Result"] = result
  game.headers["Termination"] = termination
  return game


def _play_game_with_fresh_engines(white_path, black_path, white_options, black_options, tc, fen, timeout):
  white_engine = None
  black_engine = None
  white_name = _engine_label_from_path(white_path)
  black_name = _engine_label_from_path(black_path)

  try:
    try:
      white_engine = UCIEngine(white_path, white_options, timeout=timeout)
      white_name = white_engine.name
    except Exception as e:
      result = "0-1"
      game = _make_forfeit_game(
        white_name,
        black_name,
        tc,
        fen,
        result,
        f"engine startup failure: {e}",
      )
      return result, game

    try:
      black_engine = UCIEngine(black_path, black_options, timeout=timeout)
      black_name = black_engine.name
    except Exception as e:
      result = "1-0"
      game = _make_forfeit_game(
        white_name,
        black_name,
        tc,
        fen,
        result,
        f"engine startup failure: {e}",
      )
      return result, game

    return _play_single_game(white_engine, black_engine, tc, fen)
  finally:
    if white_engine is not None:
      white_engine.quit()
    if black_engine is not None:
      black_engine.quit()


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
  except RuntimeError as e:
    print(f"  Engine crash during new_game: {e}")
    game.headers["Result"] = "0-1"
    game.headers["Termination"] = "engine crash"
    return "0-1", game
  try:
    engine_b.new_game()
  except RuntimeError as e:
    print(f"  Engine crash during new_game: {e}")
    game.headers["Result"] = "1-0"
    game.headers["Termination"] = "engine crash"
    return "1-0", game

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
      # The crashed engine loses; fresh engine instances will be used next game.
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


def play_pair_worker(engine1_path, engine2_path, options1, options2, tc, fen, matchup_key=None, timeout=30):
  """Worker: spawn fresh engines for each game in a pair to avoid hash contamination."""

  def _score(result, e1_is_white):
    if result == "1-0":
      return 0.5 if e1_is_white else -0.5
    if result == "0-1":
      return -0.5 if e1_is_white else 0.5
    return 0.0

  # Game 1: engine1 as white, engine2 as black
  result1, pgn1 = _play_game_with_fresh_engines(
    engine1_path,
    engine2_path,
    options1,
    options2,
    tc,
    fen,
    timeout,
  )

  # Game 2: engine2 as white, engine1 as black (fresh instances)
  result2, pgn2 = _play_game_with_fresh_engines(
    engine2_path,
    engine1_path,
    options2,
    options1,
    tc,
    fen,
    timeout,
  )

  pair_score = (_score(result1, True) + _score(result2, False)) / 2.0
  games = [(result1, pgn1, True), (result2, pgn2, False)]
  return games, pair_score, matchup_key


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


def main(concurrency, args):

  if concurrency <= 0:
    cpus = os.cpu_count() or 2
    tc = parse_time_control(args.tc)
    if tc["type"] in ("movetime", "clock"):
      # Use "cpus - 1" for time controls where contention can cause competition.
      concurrency = max(1, (cpus - 1) // 2)
    else:
      concurrency = max(1, cpus // 2)

  if len(args.engine) < 2:
    parser.error("at least 2 --engine flags are required")

  n_engines = len(args.engine)

  # Parse per-engine options: --option 1:Hash=64 -> engine_options[0] = ["Hash=64"]
  engine_options = [[] for _ in range(n_engines)]
  for opt_str in args.option:
    colon_pos = opt_str.find(":")
    if colon_pos < 1:
      parser.error(f"bad --option format '{opt_str}', expected N:key=value")
    try:
      idx = int(opt_str[:colon_pos]) - 1
    except ValueError:
      parser.error(f"bad engine index in --option '{opt_str}'")
    if idx < 0 or idx >= n_engines:
      parser.error(f"engine index {idx + 1} out of range [1, {n_engines}] in --option '{opt_str}'")
    engine_options[idx].append(opt_str[colon_pos + 1:])

  tc = parse_time_control(args.tc)

  # Probe engines to get their names
  engine_names = []
  for i, path in enumerate(args.engine):
    print(f"Engine {i + 1}: {path}")
    probe = UCIEngine(path, engine_options[i], timeout=args.timeout)
    engine_names.append(probe.name)
    probe.quit()
    print(f"  -> {engine_names[-1]}")

  # Disambiguate duplicate engine names
  name_counts = {}
  for name in engine_names:
    name_counts[name] = name_counts.get(name, 0) + 1
  if any(c > 1 for c in name_counts.values()):
    seen = {}
    for i, name in enumerate(engine_names):
      if name_counts[name] > 1:
        seen[name] = seen.get(name, 0) + 1
        engine_names[i] = f"{name} #{seen[name]}"

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

  # Build matchup list
  if args.vs_engine0_only:
    # Only play matches where one engine is engine 0 (index 0), i < j and i==0 or j==0
    matchups = [(0, j) for j in range(1, n_engines)]
    n_matchups = len(matchups)
    print(f"\nTournament: {n_engines} engines, {n_matchups} matchups (all vs engine 0)")
  else:
    # Full round-robin: all pairs (i, j) with i < j
    matchups = list(itertools.combinations(range(n_engines), 2))
    n_matchups = len(matchups)
    print(f"\nTournament: {n_engines} engines, {n_matchups} matchups (round-robin)")
  print(f"Engines: {', '.join(engine_names)}")
  print(f"Game pairs per matchup: {args.games}, TC: {args.tc}, Opening: {opening_label}")
  print(f"Concurrency: {args.concurrency}")
  if args.alpha > 0:
    print(f"Early stopping: alpha={args.alpha}, min_num_games={args.min_num_games}")
  print(f"PGN output: {args.pgn}")
  print()

  # Pre-generate openings per matchup (same FENs for each matchup for fairness)
  matchup_fens = {}
  for key in matchups:
    fens = []
    for g in range(args.games):
      if epd_fens:
        fens.append(epd_fens[g % len(epd_fens)])
      elif args.opening == "random":
        fens.append(random_opening(ply=args.opening_random_ply))
      else:
        fens.append(chess.STARTING_FEN)
    matchup_fens[key] = fens

  # Per-matchup state: wins/draws/losses from engine i's perspective,
  # pair_scores, completed count, stopped_early flag
  matchup_state = {}
  for key in matchups:
    matchup_state[key] = {
      "wins": 0, "draws": 0, "losses": 0,
      "pair_scores": [], "completed": 0, "stopped_early": False,
      "next_pair": 0,
    }

  # Cross-table: score_matrix[i][j] = points scored by engine i vs engine j
  score_matrix = [[0.0] * n_engines for _ in range(n_engines)]
  games_matrix = [[0] * n_engines for _ in range(n_engines)]
  
  state_file = args.pgn + ".state.json"
  if args.resume and os.path.exists(state_file):
    print(f"\nResuming from state file: {state_file}")
    with open(state_file, "r") as f:
      state_data = json.load(f)
      
      # Restore matrices
      score_matrix = state_data["score_matrix"]
      games_matrix = state_data["games_matrix"]
      
      # Restore matchup states. JSON keys are strings, convert back to tuple (int, int)
      for k_str, st in state_data["matchup_state"].items():
        # Parse "i,j" back to (i, j)
        parts = k_str.split(",")
        i, j = int(parts[0]), int(parts[1])
        matchup_state[(i, j)] = st
        
    print(f"Resumed successfully. Continuing games...")

  pgn_mode = "a" if args.resume else "w"
  with open(args.pgn, pgn_mode) as pgn_file, \
       ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    futures = {}

    def submit_work():
      """Submit pairs round-robin across matchups, up to concurrency limit."""
      while len(futures) < args.concurrency:
        submitted = False
        for key in matchups:
          if len(futures) >= args.concurrency:
            break
          st = matchup_state[key]
          if st["stopped_early"] or st["next_pair"] >= args.games:
            continue
          i, j = key
          fen = matchup_fens[key][st["next_pair"]]
          future = pool.submit(
            play_pair_worker,
            args.engine[i], args.engine[j],
            engine_options[i], engine_options[j],
            tc, fen, key, args.timeout,
          )
          futures[future] = (key, st["next_pair"] + 1)
          st["next_pair"] += 1
          submitted = True
        if not submitted:
          break

    submit_work()

    while futures:
      done_iter = as_completed(futures)
      future = next(done_iter)
      (key, pair_num) = futures.pop(future)
      i, j = key

      try:
        games, pair_score, _ = future.result()
      except Exception as e:
        print(f"  [{engine_names[i]} vs {engine_names[j]}] Pair {pair_num}: ERROR - {e}")
        submit_work()
        continue

      st = matchup_state[key]
      st["completed"] += 1
      st["pair_scores"].append(pair_score)

      for result, pgn_game, ei_was_white in games:
        if result == "1-0":
          if ei_was_white:
            st["wins"] += 1
            score_matrix[i][j] += 1.0
          else:
            st["losses"] += 1
            score_matrix[j][i] += 1.0
        elif result == "0-1":
          if not ei_was_white:
            st["wins"] += 1
            score_matrix[i][j] += 1.0
          else:
            st["losses"] += 1
            score_matrix[j][i] += 1.0
        else:
          st["draws"] += 1
          score_matrix[i][j] += 0.5
          score_matrix[j][i] += 0.5
        games_matrix[i][j] += 1
        games_matrix[j][i] += 1
        print(pgn_game, file=pgn_file, end="\n\n")
      pgn_file.flush()
      
      # Write state atomically
      temp_state = state_file + ".tmp"
      with open(temp_state, "w") as f:
        json.dump({
          "score_matrix": score_matrix,
          "games_matrix": games_matrix,
          # Convert tuple keys to strings for JSON
          "matchup_state": {f"{k[0]},{k[1]}": v for k, v in matchup_state.items()}
        }, f)
      os.rename(temp_state, state_file)

      r1, r2 = games[0][0], games[1][0]
      w, d, l = st["wins"], st["draws"], st["losses"]
      p_value, avg, stderr = significance_test(st["pair_scores"])
      total_completed = sum(s["completed"] for s in matchup_state.values())
      total_target = n_matchups * args.games
      print(
        f"  [engine {i} vs engine {j}] "
        f"result={pair_score:+.2f}  "
        f"[{st['completed']}/{args.games}]  +{w}-{l}={d}  {avg:.3f}±{stderr:.3f}  p={p_value:.3f}  "
        f"({total_completed}/{total_target} total)"
      )

      # Per-matchup early stopping
      if (args.alpha > 0
          and len(st["pair_scores"]) >= args.min_num_games
          and p_value < args.alpha):
        print(
          f"  *** [{engine_names[i]} vs {engine_names[j]}] "
          f"Stopping early: p={p_value:.4f} < alpha={args.alpha} "
          f"after {st['completed']} pairs ***"
        )
        st["stopped_early"] = True
        # Cancel queued futures for this matchup
        to_cancel = [f for f, (k, _) in futures.items() if k == key]
        for f in to_cancel:
          f.cancel()
          del futures[f]

      submit_work()

  # ========================== RESULTS ==========================
  print(f"\n{'=' * 60}")
  print("TOURNAMENT RESULTS")
  print(f"{'=' * 60}")

  # Per-matchup summaries
  print(f"\n--- Matchup Results ---")
  for key in matchups:
    i, j = key
    st = matchup_state[key]
    w, d, l = st["wins"], st["draws"], st["losses"]
    total = w + d + l
    tag = " (stopped early)" if st["stopped_early"] else ""
    print(f"\n  {engine_names[i]} vs {engine_names[j]}  [{total} games{tag}]")
    print(f"    +{w} -{l} ={d}")
    if total > 0:
      score_pct = (w + d * 0.5) / total * 100
      print(f"    Score: {score_pct:.1f}% ({engine_names[i]})")
    if len(st["pair_scores"]) > 1:
      avg = sum(st["pair_scores"]) / len(st["pair_scores"])
      variance = sum((x - avg) ** 2 for x in st["pair_scores"]) / (len(st["pair_scores"]) - 1)
      stderr = math.sqrt(variance / len(st["pair_scores"]))
      print(f"    Pair score: {avg:+.4f} +/- {stderr:.4f}")
      try:
        elo_lo = math.log(1 / (0.5 + avg - stderr) - 1.0) / math.log(10) * -400
        elo_hi = math.log(1 / (0.5 + avg + stderr) - 1.0) / math.log(10) * -400
        print(f"    Elo: {elo_lo:.1f} to {elo_hi:.1f}")
      except (ValueError, ZeroDivisionError):
        pass
      p_value, _, _ = significance_test(st["pair_scores"])
      if avg > 0:
        print(f"    {engine_names[i]} is stronger (p={p_value:.4f})")
      elif avg < 0:
        print(f"    {engine_names[j]} is stronger (p={p_value:.4f})")
      else:
        print(f"    No difference detected (p={p_value:.4f})")

  # Cross-table
  print(f"\n--- Cross Table ---")
  max_name = max(len(n) for n in engine_names)
  col_w = 10
  header = " " * (max_name + 2)
  for n in engine_names:
    header += f"{n:>{col_w}}"
  header += f"{'Total':>{col_w}}{'Score%':>{col_w}}"
  print(header)

  standings = []  # (total_score, total_games, engine_index)
  for i in range(n_engines):
    row = f"{engine_names[i]:<{max_name}}  "
    total_score = 0.0
    total_games = 0
    for j in range(n_engines):
      if i == j:
        row += f"{'---':>{col_w}}"
      elif games_matrix[i][j] > 0:
        pct = score_matrix[i][j] / games_matrix[i][j] * 100
        cell = f"{score_matrix[i][j]:g}/{games_matrix[i][j]}"
        row += f"{cell:>{col_w}}"
        total_score += score_matrix[i][j]
        total_games += games_matrix[i][j]
      else:
        row += f"{'':>{col_w}}"
    if total_games > 0:
      total_pct = total_score / total_games * 100
      row += f"{total_score:>{col_w}g}"
      row += f"{total_pct:>{col_w - 1}.1f}%"
    else:
      row += f"{'':>{col_w}}{'':>{col_w}}"
    standings.append((total_score, total_games, i))
    print(row)

  # Standings
  standings.sort(key=lambda x: (-x[0], x[2]))
  print(f"\n--- Standings ---")
  print(f"  {'#':<4}{'Engine':<{max_name + 2}}{'Score':>8}{'Games':>8}{'Score%':>9}")
  for rank, (score, games_played, idx) in enumerate(standings, 1):
    pct = score / games_played * 100 if games_played > 0 else 0
    print(f"  {rank:<4}{engine_names[idx]:<{max_name + 2}}{score:>8g}{games_played:>8}{pct:>8.1f}%")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Round-robin tournament between N UCI chess engines")
  parser.add_argument("--engine", action="append", default=[],
            help="Engine command (repeatable, at least 2 required)")
  parser.add_argument("--games", type=int, default=10,
            help="Number of game pairs per matchup (default: 10, so 20 games per pair)")
  parser.add_argument("--tc", default="nodes=10000",
            help="Time control: nodes=N, depth=N, movetime=N (ms), or S+I (seconds+increment)")
  parser.add_argument("--pgn", default="match.pgn", help="Output PGN file (default: match.pgn)")
  parser.add_argument("--opening", default="startpos",
            help="Opening type: 'startpos', 'random', or path to an EPD file (default: startpos)")
  parser.add_argument("--opening_random_ply", type=int, default=0,
            help="Number of random moves to play before starting the tournament (default: 4)")
  parser.add_argument("--concurrency", type=int, default=0,
            help="Number of game pairs to play in parallel (0 = auto-detect from CPU count)")
  parser.add_argument("--alpha", type=float, default=0.01,
            help="Significance level for per-matchup early stopping (default: 0.01, 0 to disable)")
  parser.add_argument("--min_num_games", type=int, default=20,
            help="Minimum game pairs per matchup before early stopping can occur (default: 100)")
  parser.add_argument("--option", action="append", default=[],
            help="UCI option for engine N: 'N:key=value' (1-indexed, repeatable)")
  parser.add_argument("--timeout", type=int, default=30,
            help="Per-move timeout in seconds; engines killed if unresponsive (default: 30)")
  parser.add_argument("--vs-engine0-only", action="store_true", default=False,
            help="If set, only play matches where one engine is engine 0 (all vs engine 0 mode)")
  parser.add_argument("--resume", action="store_true", default=False,
            help="If set, resume the tournament from the existing state file and append to the PGN")
  args = parser.parse_args()
  try:
    main(concurrency=args.concurrency, args=args)
  except KeyboardInterrupt:
    _kill_all_engines()
    print("\n  Interrupted.")
