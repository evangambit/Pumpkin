#!/usr/bin/env python3
"""
SPSA (Simultaneous Perturbation Stochastic Approximation) tuner for UCI chess
engine parameters, with automatic perturbation widening.

Uses match.py's game infrastructure (UCIEngine, play_pair_worker) to play
games directly. Parameters are specified as engine argument templates with
a #PARAM placeholder that gets substituted with the tuned value.

Widening logic:
  - On a draw the perturbation size grows (we're in a flat zone -> escape it).
  - On a decisive result it shrinks (we found a gradient -> refine).
  The draw-rate target controls the balance between exploration and refinement.

Example:
  python spsa.py \
    --engine './uci "evaluator byhand"' \
    --feature "increment 0 + 1 #PARAM" \
    --feature "increment 1 + 1 #PARAM" \
    --feature "increment 2 + 1 #PARAM" \
    --tc nodes=10000 \
    --opening 6mvs_+90_+99.epd \
    --iterations 500
"""

import argparse
import json
import math
import threading
import os
import random
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from match import (
    UCIEngine,
    parse_time_control,
    play_pair_worker,
    elo_diff,
    _kill_all_engines,
)

import chess


# ---------------------------------------------------------------------------
# Engine command builder
# ---------------------------------------------------------------------------

def build_engine_cmd(base_engine, feature_templates, theta):
    """Build an engine command string by substituting #PARAM in each template.

    Each template with a non-zero delta becomes a quoted argument appended
    to the base engine command.  e.g. template "increment 0 + 1 #PARAM"
    with delta=5 becomes: "increment 0 + 1 5"
    """
    cmd = base_engine
    for template, val in zip(feature_templates, theta):
        delta = int(round(val))
        if delta != 0:
            arg = template.replace("#PARAM", str(delta))
            cmd += f' "{arg}"'
    return cmd


def integer_step_vector(c_vec):
    """Map floating perturbation sizes onto integer feature steps.

    The tuned features are rendered as integers in the engine command, so the
    effective perturbation must stay on the integer lattice as well.
    """
    return np.maximum(1, np.rint(c_vec).astype(int))


# ---------------------------------------------------------------------------
# Single SPSA perturbation (play one game pair, return raw results)
# ---------------------------------------------------------------------------

def play_perturbation(
    base_engine,
    feature_templates,
    theta,
    c_vec,
    tc,
    fen,
    timeout,
):
    """Play one game pair between theta+c*delta and theta-c*delta.

    Returns (delta, pair_score) where delta is the perturbation direction vector.
    """
    p = len(theta)

    # Bernoulli +/-1 perturbation
    delta = np.random.choice([-1, 1], size=p)

    theta_center = np.rint(theta).astype(int)
    step_vec = integer_step_vector(c_vec)

    theta_plus = theta_center + step_vec * delta
    theta_minus = theta_center - step_vec * delta

    engine_plus = build_engine_cmd(base_engine, feature_templates, theta_plus)
    engine_minus = build_engine_cmd(base_engine, feature_templates, theta_minus)

    # Play a pair (engine_plus as white then black)
    games, pair_score, _ = play_pair_worker(
        engine_plus, engine_minus,
        [], [],  # no extra UCI options
        tc, fen,
        matchup_key=None,
        timeout=timeout,
    )

    return delta, pair_score, step_vec


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SPSA tuner for UCI engine parameters (with auto-widening)")

    # Engine / features
    parser.add_argument("--engine", required=True,
                        help='Base engine command, e.g. \'./uci "evaluator byhand"\'')
    parser.add_argument("--feature", action="append", default=[],
                        help='Tuning parameter template with #PARAM placeholder (repeatable). '
                             'e.g. "increment 0 + 1 #PARAM"')
    parser.add_argument("--initial", type=float, nargs="+", default=None,
                        help="Initial parameter deltas (default: all zeros)")
    parser.add_argument("--bounds", type=float, nargs=2, action="append", default=None,
                        help="Per-feature (min, max) bounds for deltas (repeatable)")

    # SPSA hyper-parameters
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of SPSA iterations (default: 500)")
    parser.add_argument("--a", type=float, default=1.0,
                        help="Base step size for parameter updates (default: 1.0)")
    parser.add_argument("--c", type=float, default=5.0,
                        help="Initial perturbation size per feature (default: 5.0)")
    parser.add_argument("--A", type=float, default=None,
                        help="Stability constant (default: 10%% of iterations)")
    parser.add_argument("--alpha", type=float, default=0.602,
                        help="Step-size decay exponent (default: 0.602)")

    # Widening parameters
    parser.add_argument("--draw_rate", type=float, default=0.75,
                        help="Targeted draw rate for widening balance (default: 0.75)")
    parser.add_argument("--draws_to_double", type=int, default=50,
                        help="Consecutive draws needed to double perturbation size (default: 50)")

    # Match settings
    parser.add_argument("--tc", default="nodes=10000",
                        help="Time control (default: nodes=10000)")
    parser.add_argument("--opening", default="6mvs_+90_+99.epd",
                        help="Opening book: EPD file path, 'startpos', or 'random'")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-move timeout in seconds (default: 30)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of game pairs to play in parallel per iteration (default: 1)")

    # Output
    parser.add_argument("--out", default="spsa_results.json",
                        help="Output file for results (default: spsa_results.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")

    args = parser.parse_args()

    if not args.feature:
        parser.error("At least one --feature is required")

    feature_templates = args.feature
    p = len(feature_templates)

    # Validate templates
    for i, tmpl in enumerate(feature_templates):
        if "#PARAM" not in tmpl:
            parser.error(f"--feature template {i} is missing #PARAM placeholder: '{tmpl}'")

    # Initial theta
    if args.initial is not None:
        if len(args.initial) != p:
            parser.error(f"--initial requires {p} values, got {len(args.initial)}")
        theta = np.array(args.initial, dtype=float)
    else:
        theta = np.zeros(p, dtype=float)

    # Bounds
    bounds = None
    if args.bounds is not None:
        if len(args.bounds) != p:
            parser.error(f"Need {p} --bounds pairs, got {len(args.bounds)}")
        bounds = args.bounds

    # Stability constant
    A = args.A if args.A is not None else int(0.1 * args.iterations)

    # Perturbation vector (one c per feature, all start the same)
    c_vec = np.full(p, max(1.0, args.c), dtype=float)

    # --- Widening / narrowing factors ---
    target_draw_rate = args.draw_rate
    # widen_factor^n = 2  =>  widen_factor = 2^(1/n)
    widen_factor = 2.0 ** (1.0 / args.draws_to_double)
    # EMA draw rate for adaptive narrowing (~20 iteration half-life)
    ema_decay = 0.97
    ema_draw_rate = target_draw_rate  # initialize to target

    def compute_narrow_factor(ema_dr):
        """Recompute narrow_factor from current EMA draw rate."""
        # Clamp to avoid division by zero or negative factor
        ema_dr = max(0.01, min(0.99, ema_dr))
        omega = widen_factor - 1.0
        rho = (ema_dr * omega) / (1.0 - ema_dr)
        return max(0.9, 1.0 - rho)  # floor at 0.9 to avoid over-shrinking

    narrow_factor = compute_narrow_factor(ema_draw_rate)

    # Time control
    tc = parse_time_control(args.tc)

    # Openings
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

    # --- Resume support ---
    start_iter = 0
    history = []
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            state = json.load(f)
        theta = np.array(state["theta"])
        c_vec = np.array(state["c_vec"])
        start_iter = state["iteration"] + 1
        history = state.get("history", [])
        ema_draw_rate = state.get("stats", {}).get("ema_draw_rate", target_draw_rate)
        narrow_factor = compute_narrow_factor(ema_draw_rate)
        print(f"Resuming from iteration {start_iter}, theta={np.round(theta, 2)}, ema_draw_rate={ema_draw_rate:.2f}")

    # --- Print configuration ---
    print(f"\n{'='*60}")
    print("SPSA Tuner with Auto-Widening")
    print(f"{'='*60}")
    print(f"Engine:       {args.engine}")
    print(f"Features:")
    for i, tmpl in enumerate(feature_templates):
        print(f"  [{i}] {tmpl}")
    print(f"Initial theta:    {np.round(theta, 2)}")
    print(f"Iterations:   {args.iterations}")
    print(f"TC:           {args.tc}")
    print(f"Opening:      {args.opening}")
    print(f"a={args.a}  c={args.c}  A={A}  alpha={args.alpha}")
    print(f"Draw target:  {target_draw_rate*100:.0f}%")
    print(f"Draws to 2x:  {args.draws_to_double}")
    print(f"EMA decay:    {ema_decay}")
    print(f"Widen:        x{widen_factor:.6f}  (on draw)")
    print(f"Narrow:       x{narrow_factor:.6f}  (on decisive, adaptive)")
    if bounds:
        print(f"Bounds:       {bounds}")
    print(f"Concurrency:  {args.concurrency}")
    print()

    # --- Tracking stats ---
    total_draws = sum(1 for h in history if h.get("draw", False))
    total_decisive = len(history) - total_draws

    # --- SPSA loop ---
    concurrency = args.concurrency

    def pick_fen(k):
        if epd_fens:
            return epd_fens[k % len(epd_fens)]
        elif args.opening == "random":
            board = chess.Board()
            for _ in range(random.randint(4, 8)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(random.choice(moves))
            return board.fen()
        else:
            return chess.STARTING_FEN

    try:
        for k in range(start_iter, args.iterations):
            # Decaying step size
            a_k = args.a / ((k + 1 + A) ** args.alpha)

            # Dispatch `concurrency` perturbation game pairs in parallel
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = []
                for c_idx in range(concurrency):
                    fen = pick_fen(k * concurrency + c_idx)
                    futures.append(pool.submit(
                        play_perturbation,
                        args.engine, feature_templates,
                        theta, c_vec,
                        tc, fen, args.timeout,
                    ))

                results = [f.result() for f in futures]

            # Aggregate results from the batch
            batch_draws = 0
            batch_decisive = 0
            gradient_sum = np.zeros(p, dtype=float)

            for delta, pair_score, step_vec in results:
                is_draw = (pair_score == 0.0)
                if is_draw:
                    batch_draws += 1
                else:
                    batch_decisive += 1
                    # pair_score > 0 means theta_plus won -> gradient is positive
                    gradient_sum += pair_score / (step_vec * delta)

            # Update EMA draw rate and recompute narrow_factor
            batch_draw_rate = batch_draws / concurrency
            ema_draw_rate = ema_decay * ema_draw_rate + (1 - ema_decay) * batch_draw_rate
            narrow_factor = compute_narrow_factor(ema_draw_rate)

            # Update c_vec: widen for draws, narrow for decisive
            new_c = c_vec * (widen_factor ** batch_draws) * (narrow_factor ** batch_decisive)
            new_c = np.maximum(1.0, new_c)

            # Update theta: average gradient across decisive results
            new_theta = theta.copy()
            if batch_decisive > 0:
                new_theta = theta + a_k * (gradient_sum / batch_decisive)

            # Apply bounds
            if bounds:
                for i, (lo, hi) in enumerate(bounds):
                    new_theta[i] = np.clip(new_theta[i], lo, hi)

            theta = new_theta
            c_vec = new_c

            # Track
            total_draws += batch_draws
            total_decisive += batch_decisive

            # Log each pair in the batch
            for delta, pair_score, _ in results:
                history.append({
                    "iteration": k,
                    "theta": theta.tolist(),
                    "c_vec": c_vec.tolist(),
                    "pair_score": pair_score,
                    "draw": (pair_score == 0.0),
                    "a_k": a_k,
                })

            # Print progress
            scores_str = ", ".join(f"{ps:+.2f}" for _, ps, _ in results)
            print(
                f"  [{k+1}/{args.iterations}]  "
                f"scores=[{scores_str}]  "
                f"draws={batch_draws}/{concurrency}  "
                f"theta=[{', '.join(f'{v:+.1f}' for v in theta)}]  "
                f"c=[{', '.join(f'{v:.2f}' for v in c_vec)}]  "
                f"a_k={a_k:.4f}  "
                f"ema_dr={ema_draw_rate*100:.1f}%  "
                f"nf={narrow_factor:.4f}"
            )

            # Save checkpoint every iteration
            checkpoint = {
                "iteration": k,
                "theta": theta.tolist(),
                "c_vec": c_vec.tolist(),
                "feature_templates": feature_templates,
                "config": {
                    "engine": args.engine,
                    "a": args.a,
                    "c": args.c,
                    "A": A,
                    "alpha": args.alpha,
                    "draw_rate": target_draw_rate,
                    "draws_to_double": args.draws_to_double,
                    "tc": args.tc,
                    "opening": args.opening,
                },
                "stats": {
                    "draws": total_draws,
                    "decisive": total_decisive,
                    "ema_draw_rate": ema_draw_rate,
                },
                "history": history,
            }
            tmp_out = args.out + ".tmp"
            with open(tmp_out, "w") as f:
                json.dump(checkpoint, f, indent=2)
            os.rename(tmp_out, args.out)

    except KeyboardInterrupt:
        _kill_all_engines()
        print("\n  Interrupted.")

    # --- Final summary ---
    print(f"\n{'='*60}")
    print("SPSA RESULTS")
    print(f"{'='*60}")
    print(f"Iterations completed: {len(history)}")
    total = total_draws + total_decisive
    if total > 0:
        print(f"Draw rate: {total_draws}/{total} ({total_draws/total*100:.1f}%)")
    print(f"\nFinal parameters:")
    for tmpl, val in zip(feature_templates, theta):
        rendered = tmpl.replace("#PARAM", f"{int(round(val)):+d}")
        print(f"  {rendered}")
    print(f"\nFinal perturbation sizes:")
    for i, c in enumerate(c_vec):
        print(f"  [{i}]: c={c:.4f}")
    print(f"\nResults saved to: {args.out}")


if __name__ == "__main__":
    main()
