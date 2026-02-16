#!/usr/bin/env python3
"""
UCI Engine FEN Evaluator - Two Engine Comparison (Threaded)

This script evaluates a list of FEN positions using two UCI chess engines
and compares the number of nodes searched by each engine.

It supports multithreading to process multiple positions in parallel.

Usage:
    python evaluate_fens.py --engine1 ./engine1 --engine2 ./engine2 --fens fens.txt --depth 8 --concurrency 4
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
        
    def start(self):
        """Start the UCI engine process"""
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            # Initialize UCI
            self.send_command("uci")
            self.wait_for_response("uciok")
            self.send_command("isready")
            self.wait_for_response("readyok")
        except Exception as e:
            print(f"Error starting engine {self.engine_path}: {e}")
            sys.exit(1)
    
    def send_command(self, command: str):
        """Send a command to the engine"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + '\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
    
    def wait_for_response(self, expected: str) -> List[str]:
        """Wait for a specific response from the engine"""
        lines = []
        while True:
            if not self.process:
                break
            line = self.process.stdout.readline().strip()
            lines.append(line)
            if expected in line:
                break
        return lines
    
    def evaluate_position(self, fen: str, depth: int) -> Tuple[Optional[int], List[str]]:
        """
        Evaluate a position and return the total nodes searched
        Returns (nodes, all_output_lines)
        """
        # Set position
        self.send_command(f"position fen {fen}")
        self.send_command("isready")
        self.wait_for_response("readyok")
        
        # Start search
        self.send_command(f"go depth {depth}")
        
        # Collect output until bestmove
        output_lines = []
        nodes = None
        
        while True:
            line = self.process.stdout.readline().strip()
            output_lines.append(line)
            
            # Look for nodes in info lines
            if line.startswith("info") and "nodes" in line:
                # Extract nodes using regex
                match = re.search(r'nodes (\d+)', line)
                if match:
                    current_nodes = int(match.group(1))
                    # Keep track of the highest node count (final count)
                    if nodes is None or current_nodes > nodes:
                        nodes = current_nodes
            
            # Stop when we get bestmove
            if line.startswith("bestmove"):
                break
        
        return nodes, output_lines
    
    def quit(self):
        """Quit the engine"""
        if self.process:
            self.send_command("quit")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def load_fens_from_file(filename: str, limit: int) -> List[str]:
    """Load FEN positions from a file"""
    fens = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Handle format with additional data (like in pos.txt)
                    if '|' in line:
                        fen = line.split('|')[0].strip()
                    else:
                        fen = line
                    fens.append(fen)
                    if len(fens) >= limit:
                        break
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
        
    return fens


def worker_func(worker_id: int, 
                fen_queue: queue.Queue, 
                results_list: List[Dict[str, Any]], 
                args: argparse.Namespace, 
                print_lock: threading.Lock,
                total_fens: int):
    """
    Worker thread function to process FENs
    """
    # Initialize separate engine instances for this thread
    engine1 = UCIEngine(args.engine1)
    engine2 = UCIEngine(args.engine2)
    
    try:
        engine1.start()
        engine2.start()
        
        while True:
            try:
                # Get work from queue (non-blocking)
                work_item = fen_queue.get_nowait()
            except queue.Empty:
                break
                
            index, fen = work_item
            
            # --- Evaluation Logic ---
            
            # Evaluate with engine 1
            start_time = time.time()
            nodes1, output1 = engine1.evaluate_position(fen, args.depth)
            time1 = time.time() - start_time
            
            # Evaluate with engine 2
            start_time = time.time()
            nodes2, output2 = engine2.evaluate_position(fen, args.depth)
            time2 = time.time() - start_time
            
            # Process results
            res_entry = {
                'position': index,
                'fen': fen,
                'nodes1': nodes1,
                'nodes2': nodes2,
                'time1': time1,
                'time2': time2,
                'nps1': 0,
                'nps2': 0,
                'ratio': 1.0,
                'winner': 'tie'
            }
            
            if nodes1 is not None and nodes2 is not None:
                # Calculate ratio (engine1 / engine2)
                if nodes2 > 0:
                    res_entry['ratio'] = nodes1 / nodes2
                else:
                    res_entry['ratio'] = float('inf') if nodes1 > 0 else 1.0
                
                # Count wins
                if nodes1 < nodes2:
                    res_entry['winner'] = 'e1'
                elif nodes1 > nodes2:
                    res_entry['winner'] = 'e2'
                
                res_entry['nps1'] = int(nodes1 / time1) if time1 > 0 else 0
                res_entry['nps2'] = int(nodes2 / time2) if time2 > 0 else 0
                
                # --- Printing (Thread Safe) ---
                with print_lock:
                    print(f"Position {index}/{total_fens} (Thread {worker_id})")
                    if args.verbose:
                        print(f"FEN: {fen}")
                    print(f"  Engine 1: {nodes1:,} nodes ({res_entry['nps1']:,} nps)")
                    print(f"  Engine 2: {nodes2:,} nodes ({res_entry['nps2']:,} nps)")
                    print(f"  Ratio (E1/E2): {res_entry['ratio']:.3f}")
                    
                    if args.verbose:
                        print("  Engine 1 output (last 3):")
                        for line in output1[-3:]:
                            print(f"    {line}")
                        print("  Engine 2 output (last 3):")
                        for line in output2[-3:]:
                            print(f"    {line}")
                    print()
            else:
                with print_lock:
                    print(f"Position {index}/{total_fens} - Error extracting nodes")
                    
            results_list.append(res_entry)
            fen_queue.task_done()
            
    except Exception as e:
        with print_lock:
            print(f"Worker {worker_id} failed: {e}")
    finally:
        # Cleanup engines
        engine1.quit()
        engine2.quit()


def main():
    parser = argparse.ArgumentParser(description="Compare FEN evaluations between two UCI engines")
    parser.add_argument("--engine1", "-e1", required=True, help="Path to first UCI chess engine")
    parser.add_argument("--engine2", "-e2", required=True, help="Path to second UCI chess engine")
    parser.add_argument("--fens", "-f", help="File containing FEN positions (one per line)")
    parser.add_argument("--depth", "-d", type=int, required=True, help="Search depth")
    parser.add_argument("--limit", "-l", type=int, default=10000, help="Limit analysis to first N positions")
    parser.add_argument("--concurrency", "-j", type=int, default=4, help="Number of concurrent threads (default: 4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed engine output")
    parser.add_argument("--output", "-o", help="Output file to save results")
    
    args = parser.parse_args()
    
    if not args.fens:
        print("Error: Must specify --fens")
        sys.exit(1)
    
    fens = load_fens_from_file(args.fens, args.limit)
    
    print(f"Loaded {len(fens)} FEN position(s)")
    print(f"Engine 1: {args.engine1}")
    print(f"Engine 2: {args.engine2}")
    print(f"Search depth: {args.depth}")
    print(f"Concurrency: {args.concurrency} threads")
    print("-" * 50)
    
    # Setup Queue and Results List
    fen_queue = queue.Queue()
    for i, fen in enumerate(fens, 1):
        fen_queue.put((i, fen))
        
    results = [] # Shared list, append is thread-safe in CPython but we generally rely on GIL. 
                 # For complex ops, a lock is safer, but list.append is atomic.
                 
    print_lock = threading.Lock()
    threads = []
    
    start_total_time = time.time()
    
    # Start Threads
    for i in range(args.concurrency):
        t = threading.Thread(
            target=worker_func, 
            args=(i+1, fen_queue, results, args, print_lock, len(fens))
        )
        t.start()
        threads.append(t)
        
    # Wait for completion
    for t in threads:
        t.join()
        
    elapsed = time.time() - start_total_time
    print("-" * 50)
    print(f"Evaluation complete in {elapsed:.2f} seconds")
    
    # Sort results by position index to maintain order for analysis/output
    results.sort(key=lambda x: x['position'])
    
    # --- Analysis Section (Same as original) ---
    
    nodes1 = np.array([r['nodes1'] for r in results if r['nodes1'] is not None])
    nodes2 = np.array([r['nodes2'] for r in results if r['nodes2'] is not None])

    if len(nodes1) == 0:
        print("No valid results collected.")
        sys.exit(0)

    print((nodes1 < nodes2).sum(), "positions where Engine 1 searched fewer nodes")
    print((nodes1 > nodes2).sum(), "positions where Engine 2 searched fewer nodes")
    
    # Handle division by zero for log ratios
    safe_nodes2 = np.where(nodes2 == 0, 1, nodes2)
    safe_nodes1 = np.where(nodes1 == 0, 1, nodes1)
    
    log_ratios = np.log(safe_nodes1 / safe_nodes2)
    x = np.sign(nodes1 - nodes2) 

    # Sign test
    if x.std() > 0: 
        z = x.mean() / (x.std() / np.sqrt(x.shape[0]))
        if z < 0.0:
          print('Engine 1 is better (sign test: z = {:.3f}, p = {:.3f})'.format(z, 2 * stats.norm.cdf(z)))
        elif z > 0.0:
          print('Engine 2 is better (sign test: z = {:.3f}, p = {:.3f})'.format(z, 2 * (1 - stats.norm.cdf(z))))
        else:
          print('Engines are tied (sign test)')
    else:
        print('All positions have the same winner - no variation for sign test')
    
    # Log-ratio test
    if log_ratios.std() > 0: 
        z = log_ratios.mean() / (log_ratios.std() / np.sqrt(log_ratios.shape[0]))
        if z < 0.0:
          print('Engine 1 is better (log-ratio test: z = {:.3f}, p = {:.3f})'.format(z, 2 * stats.norm.cdf(z)))
        elif z > 0.0:
          print('Engine 2 is better (log-ratio test: z = {:.3f}, p = {:.3f})'.format(z, 2 * (1 - stats.norm.cdf(z))))
        else:
          print('Engines are tied (log-ratio test: z = {:.3f}, p = {:.3f})'.format(z, 0.5))
    else:
        print('All log-ratios are identical - no variation for log-ratio test')

    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("Position,FEN,Nodes1,Nodes2,Ratio,Time1(s),Time2(s),NPS1,NPS2\n")
            for result in results:
                f.write(f"{result['position']},\"{result['fen']}\",{result['nodes1']},{result['nodes2']},{result['ratio']:.6f},{result['time1']:.3f},{result['time2']:.3f},{result['nps1']},{result['nps2']}\n")
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()