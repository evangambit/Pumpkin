#!/usr/bin/env python3
"""
UCI Engine FEN Evaluator - Two Engine Comparison (Threaded with Timeout Recovery)
Processes FENs and compares node efficiency using Sign and Log-Ratio tests.
"""

import subprocess
import argparse
import sys
import time
import re
import threading
import queue
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats

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

    def start(self) -> bool:
        """Starts engine and initializes UCI state."""
        try:
            # We use 'evaluator nnue' as per your original subprocess call
            self.process = subprocess.Popen(
                [self.engine_path],
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

    def evaluate_position(self, fen: str, depth: int, timeout: float) -> Tuple[Optional[int], List[str]]:
        """Sends position and 'go depth', returning nodes and full output."""
        self.send_command(f"position fen {fen}")
        self.send_command("isready")
        self.wait_for_response("readyok", timeout=5)
        
        self.send_command(f"go depth {depth}")
        
        output_lines = []
        nodes = None
        start = time.time()
        
        while True:
            if time.time() - start > timeout:
                raise TimeoutError(f"Search timed out after {timeout}s")
            try:
                line = self.output_queue.get(timeout=0.1)
                output_lines.append(line)
                
                # Extract nodes using regex
                if "nodes" in line:
                    match = re.search(r'nodes (\d+)', line)
                    if match:
                        nodes = int(match.group(1))
                
                if line.startswith("bestmove"):
                    break
            except queue.Empty:
                continue
        return nodes, output_lines

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
    fens = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Handle format with | or just plain FEN
                    fen = line.split('|')[0].strip() if '|' in line else line
                    fens.append(fen)
                    if len(fens) >= limit:
                        break
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    return fens

def worker_func(worker_id: int, fen_queue: queue.Queue, results_list: List[Dict[str, Any]], 
                args: argparse.Namespace, print_lock: threading.Lock, total_fens: int):
    """Worker thread that handles engine instances and FEN processing."""
    e1 = UCIEngine(args.engine1)
    e2 = UCIEngine(args.engine2)
    
    while True:
        try:
            work_item = fen_queue.get_nowait()
        except queue.Empty:
            break
            
        index, fen = work_item
        res = {
            'position': index, 
            'fen': fen, 
            'nodes1': None, 'nodes2': None,
            'time1': 0.0, 'time2': 0.0
        }

        try:
            # Recovery/Lazy Start
            if not e1.process: e1.start()
            if not e2.process: e2.start()

            # Engine 1
            t0 = time.time()
            n1, out1 = e1.evaluate_position(fen, args.depth, args.timeout)
            res['nodes1'], res['time1'] = n1, time.time() - t0

            # Engine 2
            t0 = time.time()
            n2, out2 = e2.evaluate_position(fen, args.depth, args.timeout)
            res['nodes2'], res['time2'] = n2, time.time() - t0

            # Thread-safe printing
            with print_lock:
                print(f"Position {index}/{total_fens} (Thread {worker_id})")
                if n1 and n2:
                    ratio = n1 / n2 if n2 > 0 else 1.0
                    print(f"  E1: {n1:,} nodes | E2: {n2:,} nodes | Ratio: {ratio:.3f}")
                if args.verbose:
                    print(f"  FEN: {fen}")
                    if n1: print(f"  E1 Out: {out1[-1]}")
                    if n2: print(f"  E2 Out: {out2[-1]}")

        except Exception as e:
            with print_lock:
                print(f"!! [Thread {worker_id}] Error on Position {index}: {e}")
                print(f"   Restarting engines for next task...")
            e1.quit()
            e2.quit()
        
        results_list.append(res)
        fen_queue.task_done()
        
    e1.quit()
    e2.quit()

def main():
    parser = argparse.ArgumentParser(description="Compare UCI Engines with Timeout Recovery")
    parser.add_argument("--engine1", "-e1", required=True)
    parser.add_argument("--engine2", "-e2", required=True)
    parser.add_argument("--fens", "-f", required=True)
    parser.add_argument("--depth", "-d", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=30, help="Seconds before killing hung engine")
    parser.add_argument("--concurrency", "-j", type=int, default=4)
    parser.add_argument("--limit", "-l", type=int, default=1000)
    parser.add_argument("--output", "-o", help="CSV Output file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    fens = load_fens_from_file(args.fens, args.limit)
    print(f"Starting analysis of {len(fens)} positions...")
    
    fen_queue = queue.Queue()
    for i, fen in enumerate(fens, 1):
        fen_queue.put((i, fen))
        
    results = []
    print_lock = threading.Lock()
    threads = []
    
    start_time = time.time()
    for i in range(args.concurrency):
        t = threading.Thread(target=worker_func, args=(i+1, fen_queue, results, args, print_lock, len(fens)))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    # --- Analysis Section ---
    valid = [r for r in results if r['nodes1'] is not None and r['nodes2'] is not None]
    if not valid:
        print("\nNo valid results collected (all positions failed or timed out).")
        return

    n1_vals = np.array([r['nodes1'] for r in valid], dtype=float)
    n2_vals = np.array([r['nodes2'] for r in valid], dtype=float)
    t1_vals = np.array([r['time1'] for r in valid])
    t2_vals = np.array([r['time2'] for r in valid])

    print("\n" + "="*60)
    print(f"FINAL STATS ({len(valid)} positions success, {len(results)-len(valid)} failed)")
    print("-" * 60)
    print(f"Engine 1 searched fewer nodes: {(n1_vals < n2_vals).sum()}")
    print(f"Engine 2 searched fewer nodes: {(n2_vals < n1_vals).sum()}")
    print(f"Ties:                          {(n1_vals == n2_vals).sum()}")
    
    avg_nps1 = n1_vals.sum() / t1_vals.sum() if t1_vals.sum() > 0 else 0
    avg_nps2 = n2_vals.sum() / t2_vals.sum() if t2_vals.sum() > 0 else 0
    print(f"Avg NPS E1: {int(avg_nps1):,} | E2: {int(avg_nps2):,}")
    print("-" * 60)

    # 1. Sign Test
    diffs = n1_vals - n2_vals
    x = np.sign(diffs)
    if np.any(x != 0) and x.std() > 0:
        z_sign = x.mean() / (x.std() / np.sqrt(len(x)))
        p_sign = 2 * stats.norm.cdf(-abs(z_sign))
        print(f"Sign Test:      z = {z_sign:+.3f}, p = {p_sign:.4f}")
    else:
        print("Sign Test:      Insufficient variation for test")

    # 2. Log-Ratio Test
    n1_s, n2_s = np.where(n1_vals <= 0, 1, n1_vals), np.where(n2_vals <= 0, 1, n2_vals)
    log_ratios = np.log(n1_s / n2_s)
    if log_ratios.std() > 0:
        z_log = log_ratios.mean() / (log_ratios.std() / np.sqrt(len(log_ratios)))
        p_log = 2 * stats.norm.cdf(-abs(z_log))
        print(f"Log-Ratio Test: z = {z_log:+.3f}, p = {p_log:.4f}")
        
        if p_log < 0.05:
            winner = "Engine 1" if z_log < 0 else "Engine 2"
            print(f"\nCONCLUSION: {winner} is significantly more efficient (p < 0.05)")
    else:
        print("Log-Ratio Test: No variation in results.")

    if args.output:
        results.sort(key=lambda x: x['position'])
        with open(args.output, 'w') as f:
            f.write("Index,FEN,Nodes1,Nodes2,Time1,Time2\n")
            for r in results:
                f.write(f"{r['position']},\"{r['fen']}\",{r['nodes1']},{r['nodes2']},{r['time1']:.3f},{r['time2']:.3f}\n")
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
