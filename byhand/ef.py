#!/usr/bin/env python3
"""
UCI Engine FEN Evaluator - Two Engine Comparison (Threaded with Timeout Recovery)
Processes FENs and compares node efficiency using Sign and Log-Ratio tests.
"""

import subprocess
import argparse
import os
import sys
import time
import re
import threading
import queue
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
import matplotlib.pyplot as plt

import dataset as ndata

cat = np.concatenate

s = 5
x = np.array([-s * 8, -s * 4, -s * 2, -s, 0, s, s * 2, s * 4, s * 8])
kBaselineIndex = 3


class UCIEngine:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.process = None
        self.output_queue = queue.Queue()
        self.reader_thread = None

    def _read_stream(self):
        """Background thread to capture stdout without blocking the main process."""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.output_queue.put(line.strip())
                else:
                    break
        except Exception:
            pass
        finally:
            if self.process and self.process.stdout:
                self.process.stdout.close()

    def start(self, multipv) -> bool:
        """Starts engine and initializes UCI state."""
        try:
            # We use 'evaluator nnue' as per your original subprocess call
            self.process = subprocess.Popen(
                self.engine_path.split('|'),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            self.reader_thread = threading.Thread(target=self._read_stream, daemon=True)
            self.reader_thread.start()

            self.send_command("uci")
            self.wait_for_response("uciok", timeout=5)
            self.send_command("isready")
            self.wait_for_response("readyok", timeout=5)
            self.send_command(f"setoption name MultiPV value {multipv}")
            return True
        except Exception as e:
            print(f"Failed to start {self.engine_path}: {e}")
            self.quit()
            return False

    def send_command(self, command: str):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + '\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                self.process = None

    def wait_for_response(self, expected: str, timeout: float) -> List[str]:
        """Wait for a specific string in the engine output within a timeout."""
        lines = []
        start = time.time()
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Engine timed out waiting for '{expected}'")
            try:
                line = self.output_queue.get(timeout=0.1)
                lines.append(line)
                if expected in line:
                    break
            except queue.Empty:
                continue
        return lines

    def evaluate_position(self, fen: str, nodes: int, timeout: float) -> List[str]:
        """Returns list of best moves (one per PV line at the final depth)."""
        self.send_command(f"position fen {fen}")
        self.send_command("isready")
        self.wait_for_response("readyok", timeout=5)
        
        self.send_command(f"go nodes {nodes}")
        
        output_lines = []
        start = time.time()
        
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Search timed out after {timeout}s")
            try:
                line = self.output_queue.get(timeout=0.1)
                output_lines.append(line)
                
                if line.startswith("bestmove"):
                    break
            except queue.Empty:
                continue

        # Extract best moves from the final depth's multipv lines.
        # Walk backwards to find the last multipv 1 and multipv 2 info lines.
        pv_moves = {}
        for line in reversed(output_lines):
            if line.startswith("bestmove"):
                continue
            if 'multipv' not in line:
                continue
            parts = line.split()
            mpv_idx = int(parts[parts.index('multipv') + 1])
            if mpv_idx not in pv_moves:
                pv_moves[mpv_idx] = parts[parts.index('pv') + 1]
            if len(pv_moves) >= 2:
                break

        # Return moves ordered by PV index
        return [pv_moves[k] for k in sorted(pv_moves.keys())]

    def quit(self):
        """Cleanly stop or forcefully kill the engine process."""
        if self.process:
            try:
                self.send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=1)
            except:
                if self.process:
                    self.process.kill()
            self.process = None
            self.output_queue = queue.Queue() # Clear leftover output

def load_fens_from_file(filename: str, limit: int) -> List[str]:
    lines = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line.split('|'))
                    if len(lines) >= limit:
                        break
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    return lines

def worker_func(worker_id: int, feature_index: int, fen_queue: queue.Queue, results_list: List[Dict[str, Any]], 
                args: argparse.Namespace, print_lock: threading.Lock, total_fens: int):
    """Worker thread that handles engine instances and FEN processing."""
    engines = [
        UCIEngine(args.engine + f'|increment {feature_index} + 1 {val}') for val in x
    ]
    
    while True:
        try:
            work_item = fen_queue.get_nowait()
        except queue.Empty:
            break
            
        index, line = work_item
        fen = line[0]
        moves = line[1::2]
        scores = [float(s) for s in line[2::2]]
        if len(moves) <= 1:
            continue
        if len(moves) != len(scores):
            continue
        if scores[0] != max(scores):
            continue

        try:
            # Recovery/Lazy Start
            for e in engines:
                if not e.process:
                    e.start(args.multipv)

            # Each engine returns [pv1_move, pv2_move]
            engine_scores = []
            for e in engines:
                engine_scores.append([])
                pv_moves = e.evaluate_position(fen, args.nodes, args.timeout)
                # PV1 move
                for m in pv_moves:
                    engine_scores[-1].append(scores[-1] if m not in moves else scores[moves.index(m)])
            
            # Thread-safe printing
            if index < 10 or index % 100 == 0:
                with print_lock:
                    print(f"Position {index}/{total_fens} (Thread {worker_id})")
            
            if sum(len(arr) == len(engine_scores[0]) for arr in engine_scores) == len(engine_scores):
                engine_scores = [sum(arr) / len(arr) for arr in engine_scores]
                results_list.append({
                    'fen': fen,
                    'scores': engine_scores,
                })
            else:
                with print_lock:
                    print("WTF")
            fen_queue.task_done()

        except Exception as e:
            with print_lock:
                print(f"!! [Thread {worker_id}] Error on Position {index}: {e}")
                print(f"   Restarting engines for next task...")
            for e in engines:
                e.quit()
        
        
    for e in engines:
        e.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare UCI Engines with Timeout Recovery")
    parser.add_argument("--engine", "-e", required=True)
    parser.add_argument("--fens", "-f", required=True)  # Create with generate.py
    parser.add_argument("--nodes", "-n", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=30, help="Seconds before killing hung engine")
    parser.add_argument("--concurrency", "-j", type=int, default=4)
    parser.add_argument("--limit", "-l", type=int, default=40_000)
    parser.add_argument("--multipv", "-m", type=int, default=2)
    parser.add_argument("--out", "-o", help="Output directory", default='results')
    args = parser.parse_args()

    lines = load_fens_from_file(args.fens, args.limit)
    print(f"Starting analysis of {len(lines)} positions...")

    os.makedirs(args.out, exist_ok=True)

    for feature_index in range(0, 10):
        feature_name = ndata.feature_name(feature_index)
        fen_queue = queue.Queue()
        for i, fen in enumerate(lines, 1):
            fen_queue.put((i, fen))
            
        results = []
        print_lock = threading.Lock()
        threads = []
        
        start_time = time.time()
        for i in range(args.concurrency):
            t = threading.Thread(target=worker_func, args=(i+1, feature_index, fen_queue, results, args, print_lock, len(lines)))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
            
        # --- Analysis Section ---
        valid = [r for r in results if r['scores'] is not None]
        if not valid:
            print("\nNo valid results collected (all positions failed or timed out).")
            exit(1)

        losses = np.array([r['scores'] for r in valid], dtype=float)
        baseline = losses[:,kBaselineIndex:kBaselineIndex+1]
        losses = losses - baseline

        avg = losses.mean(0)
        stderr = losses.std(0) / np.sqrt(losses.shape[0] - 1) + 1e-6
        print('  Average loss', avg)
        print('Standard error', stderr)
        print('      Z-scores', avg / stderr)
        print(' Disagreements', (losses != 0).sum(0))

        X = np.stack([
            np.ones(len(x)),
            x,
            x**2
        ], 1)

        cov = np.linalg.inv(X.T @ np.diag(1/stderr**2) @ X)
        w = cov @ X.T @ np.diag(1 / stderr**2) @ avg
        print(w)

        plt.figure()
        plt.title(f'{feature_name} {feature_index}')
        plt.plot(x, avg, c='b')
        # plt.plot(x, avg + stderr, c='b')
        # plt.plot(x, avg - stderr, c='b')
        plt.fill(
            cat([x, x[::-1]]),
            cat([avg + stderr, (avg - stderr)[::-1]]),
            c='b',
            alpha=0.2
        )
        plt.fill(
            cat([x, x[::-1]]),
            cat([avg + stderr * 2.0, (avg - stderr * 2.0)[::-1]]),
            c='b',
            alpha=0.2
        )
        plt.scatter(x, avg)

        # px = np.linspace(x.min(), x.max(), 30)
        # plt.plot(px, w[0] + w[1] * px + w[2] * px * px, c='r')
        # for _ in range(10):
        #     u = np.random.multivariate_normal(w, cov)
        #     plt.plot(px, u[0] + u[1] * px + u[2] * px * px, c='r', alpha=0.3, ls='--')

        plt.grid()
        plt.savefig(f'{args.out}/{feature_index}-{feature_name}.png')

