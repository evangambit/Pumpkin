#!/usr/bin/env python3
"""
UCI Engine FEN Evaluator - Two Engine Comparison

This script evaluates a list of FEN positions using two UCI chess engines
and compares the number of nodes searched by each engine.

Usage:
    python evaluate_fens.py --engine1 ./engine1 --engine2 ./engine2 --fens fens.txt --depth 8
    python evaluate_fens.py --engine1 ./engine1 --engine2 ./engine2 --fens fens.txt --depth 8 --limit 100

It is primarily used to evaluate move ordering changes (as a (much) faster alternative to running full games).
"""

import subprocess
import argparse
import sys
import time
import re
import numpy as np
from typing import List, Tuple, Optional
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
            print(f"Engine {self.engine_path} started successfully")
        except Exception as e:
            print(f"Error starting engine: {e}")
            sys.exit(1)
    
    def send_command(self, command: str):
        """Send a command to the engine"""
        if self.process and self.process.stdin:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
    
    def wait_for_response(self, expected: str) -> List[str]:
        """Wait for a specific response from the engine"""
        lines = []
        while True:
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
            self.process.wait()


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


def main():
    parser = argparse.ArgumentParser(description="Compare FEN evaluations between two UCI engines")
    parser.add_argument("--engine1", "-e1", required=True, help="Path to first UCI chess engine")
    parser.add_argument("--engine2", "-e2", required=True, help="Path to second UCI chess engine")
    parser.add_argument("--fens", "-f", help="File containing FEN positions (one per line)")
    parser.add_argument("--depth", "-d", type=int, help="Search depth (default: 8)", required=True)
    parser.add_argument("--limit", "-l", type=int, help="Limit analysis to first N positions", required=True)
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
    print("-" * 50)
    
    # Start both engines
    engine1 = UCIEngine(args.engine1)
    engine2 = UCIEngine(args.engine2)
    engine1.start()
    engine2.start()
    
    results = []
    engine1_total_nodes = 0
    engine2_total_nodes = 0
    engine1_wins = 0
    engine2_wins = 0
    ties = 0
    ratios = []
    
    try:
        for i, fen in enumerate(fens, 1):
            print(f"Evaluating position {i}/{len(fens)}")
            if args.verbose:
                print(f"FEN: {fen}")
            
            # Evaluate with engine 1
            start_time = time.time()
            nodes1, output1 = engine1.evaluate_position(fen, args.depth)
            time1 = time.time() - start_time
            
            # Evaluate with engine 2
            start_time = time.time()
            nodes2, output2 = engine2.evaluate_position(fen, args.depth)
            time2 = time.time() - start_time
            
            if nodes1 is not None and nodes2 is not None:
                engine1_total_nodes += nodes1
                engine2_total_nodes += nodes2
                
                # Calculate ratio (engine1 / engine2)
                if nodes2 > 0:
                    ratio = nodes1 / nodes2
                    ratios.append(ratio)
                else:
                    ratio = float('inf') if nodes1 > 0 else 1.0
                
                # Count wins
                if nodes1 < nodes2:
                    engine1_wins += 1
                elif nodes1 > nodes2:
                    engine2_wins += 1
                else:
                    ties += 1
                
                nps1 = int(nodes1 / time1) if time1 > 0 else 0
                nps2 = int(nodes2 / time2) if time2 > 0 else 0
                
                print(f"  Engine 1: {nodes1:,} nodes ({nps1:,} nps)")
                print(f"  Engine 2: {nodes2:,} nodes ({nps2:,} nps)")
                print(f"  Ratio (E1/E2): {ratio:.3f}")
                
                results.append({
                    'position': i,
                    'fen': fen,
                    'nodes1': nodes1,
                    'nodes2': nodes2,
                    'time1': time1,
                    'time2': time2,
                    'nps1': nps1,
                    'nps2': nps2,
                    'ratio': ratio
                })
            else:
                print(f"  Error: Could not extract node counts")
                results.append({
                    'position': i,
                    'fen': fen,
                    'nodes1': nodes1 or 0,
                    'nodes2': nodes2 or 0,
                    'time1': time1,
                    'time2': time2,
                    'nps1': 0,
                    'nps2': 0,
                    'ratio': 1.0
                })
            
            if args.verbose:
                print("  Engine 1 output:")
                for line in output1[-3:]:  # Show last 3 lines
                    print(f"    {line}")
                print("  Engine 2 output:")
                for line in output2[-3:]:  # Show last 3 lines
                    print(f"    {line}")
            
            print()
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    finally:
        engine1.quit()
        engine2.quit()

    # Extract node counts
    nodes1 = np.array([r['nodes1'] for r in results])
    nodes2 = np.array([r['nodes2'] for r in results])

    print((nodes1 < nodes2).sum(), "positions where Engine 1 searched fewer nodes")
    print((nodes1 > nodes2).sum(), "positions where Engine 2 searched fewer nodes")
    
    log_ratios = np.log(nodes1 / nodes2)
    x = np.sign(nodes1 - nodes2)  # 1 if engine1 searched more nodes (i.e. is worse). -1 if engine1 searched fewer nodes (i.e. is better).

    # Sign test - check if one engine consistently searches fewer nodes
    if x.std() > 0:  # Avoid division by zero
        z = x.mean() / (x.std() / np.sqrt(x.shape[0]))
        if z < 0.0:
          print('Engine 1 is better (sign test: z = {:.3f}, p = {:.3f})'.format(z, 2 * stats.norm.cdf(z)))
        elif z > 0.0:
          print('Engine 2 is better (sign test: z = {:.3f}, p = {:.3f})'.format(z, 2 * (1 - stats.norm.cdf(z))))
        else:
          print('Engines are tied (sign test)')
    else:
        print('All positions have the same winner - no variation for sign test')
    
    # Log-ratio test - more sensitive to the magnitude of differences
    if log_ratios.std() > 0:  # Avoid division by zero
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