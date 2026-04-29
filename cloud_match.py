#!/usr/bin/env python3

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict


"""
./cloud_match.py --engine commit=HEAD "arg=increment 1 1 10" \
                 --engine commit=HEAD "arg=increment 1 1 -10" \
                 --min_num_games=50 --opening_random_ply 4 --tc nodes=200000

./cloud_match.py --engine commit=HEAD flag=-DLMR_NULL_A=0.8 \
                 --engine commit=HEAD flag=-DLMR_NULL_A=1.0 \
                 --min_num_games=50 --opening_random_ply 4 --tc nodes=200000
"""

#   # Teardown only:
#   ./cloud_match.sh --teardown

#   # For debugging builds:
#   /opt/homebrew/share/google-cloud-sdk/bin/gcloud compute ssh pumpkin-match \
#   --zone=us-central1-a --command='rm -rf ~/Pumpkin && \
#   git clone https://github.com/evangambit/Pumpkin.git && cd Pumpkin && \
#   bash build.sh uci-test src/uci/main.cpp -DNDEBUG -O3'

# ---------- Configuration (edit these or override via env vars) ----------

PROJECT = os.environ.get("PROJECT", "")
VM_NAME = os.environ.get("VM_NAME", "pumpkin-match")
REPO_URL = "https://github.com/evangambit/Pumpkin.git"
BRANCH = "master"


def run_cmd(cmd: List[str], check: bool = True, capture_output: bool = False, text: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    if capture_output:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    return subprocess.run(cmd, check=check, stdout=stdout, stderr=stderr, text=text)


def gce_cmd(args: argparse.Namespace, cmd: List[str], capture_output: bool = False, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    base = ["gcloud", "compute"] + cmd + [f"--zone={args.zone}"]
    if PROJECT:
        base.append(f"--project={PROJECT}")
    return run_cmd(base, check=check, capture_output=capture_output, quiet=quiet)


def ssh_cmd(args: argparse.Namespace, remote_command: str, capture_output: bool = False, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    return gce_cmd(args, ["ssh", VM_NAME, f"--command={remote_command}"], capture_output=capture_output, check=check, quiet=quiet)


def scp_to_vm(args: argparse.Namespace, local_path: str, remote_path: str, quiet: bool = False) -> None:
    gce_cmd(args, ["scp", local_path, f"{VM_NAME}:{remote_path}"], quiet=quiet)


def scp_from_vm(args: argparse.Namespace, remote_path: str, local_path: str, quiet: bool = False) -> None:
    gce_cmd(args, ["scp", f"{VM_NAME}:{remote_path}", local_path], quiet=quiet)


def resolve_engines(engine_specs: List[List[str]]) -> List[Dict]:
    print("📋 Resolving engine refs locally...")
    resolved_engines = []
    
    for idx, spec in enumerate(engine_specs):
        commit_ref = "HEAD"
        flags = []
        engine_args = []
        for arg in spec:
            if arg.startswith("commit="):
                commit_ref = arg.split("=", 1)[1]
            elif arg.startswith("flag="):
                flags.append(arg.split("=", 1)[1])
            elif arg.startswith("arg"):
                engine_args.append(f'"{arg.split('=', 1)[1]}"')
            else:
                print(f"Error: Unknown engine argument '{arg}'")
                sys.exit(1)
                
        try:
            hash_val = run_cmd(["git", "rev-parse", "--short=10", commit_ref], capture_output=True).stdout.strip()
            short_msg = run_cmd(["git", "log", "--oneline", "-1", hash_val], capture_output=True).stdout.strip()
            full_msg = run_cmd(["git", "log", "--format=%s", "-1", hash_val], capture_output=True).stdout.strip()
            
            flags_str = " ".join(flags)
            engine_args_str = " ".join(engine_args)
            flag_hash = hashlib.md5(flags_str.encode()).hexdigest()[:6]
            arg_hash = hashlib.md5(engine_args_str.encode()).hexdigest()[:6]
            # Binary name only depends on build parameters (commit and flags)
            name = f"uci-{hash_val}-{flag_hash}-{arg_hash}"
            label = f"{short_msg} (flags: {flags_str}) (args: {engine_args_str})"
                
            resolved_engines.append({
                "commit_ref": commit_ref,
                "hash": hash_val,
                "flags": flags,
                "flags_str": flags_str,
                "name": name,
                "label": label,
                "full_msg": full_msg,
                "engine_args": engine_args,
                "engine_args_str": engine_args_str,
            })
            print(f"  engine {idx} ({commit_ref}) → {hash_val} with flags: '{flags_str}' and args: '{engine_args_str}'")
        except subprocess.CalledProcessError:
            print(f"Error: cannot resolve git ref '{commit_ref}'")
            sys.exit(1)
            
    # De-duplicate while preserving order
    unique_engines = []
    seen = set()
    for eng in resolved_engines:
        key = (eng["hash"], eng["flags_str"], eng["engine_args_str"])
        print('KEY', key)
        if key not in seen:
            seen.add(key)
            unique_engines.append(eng)
            

    return unique_engines, resolved_engines


def main():
    parser = argparse.ArgumentParser(
        description="Run Pumpkin engine tournaments on a GCE spot VM.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage examples:
  # Compare HEAD vs HEAD~5:
  ./cloud_match.py --engine commit=HEAD --engine commit=HEAD~5

  # Compare same commit with different build flags:
  ./cloud_match.py --engine commit=HEAD flag=-DLMR_PV_B=0.4 \\
                   --engine commit=HEAD flag=-DLMR_PV_B=0.5

  # Custom time control and game count:
  ./cloud_match.py --engine commit=HEAD --engine commit=HEAD~3 --tc "nodes=50000" --games 2000

  # Skip VM creation (reuse existing VM):
  ./cloud_match.py --engine commit=HEAD --engine commit=HEAD~5 --reuse

  # Teardown only:
  ./cloud_match.py --teardown

  # For debugging builds:
  /opt/homebrew/share/google-cloud-sdk/bin/gcloud compute ssh pumpkin-match \
  --zone=us-central1-a --command='rm -rf ~/Pumpkin && \
  git clone https://github.com/evangambit/Pumpkin.git && cd Pumpkin && \
  bash build.sh uci-test src/uci/main.cpp -DNDEBUG -O3'
"""
    )

    parser.add_argument("--engine", action="append", nargs="+", default=[], help="Specify an engine (e.g., --engine commit=HEAD flag=-DLMR_PV_B=0.4)")
    parser.add_argument("--tc", default=os.environ.get("TC", "nodes=10000"), help="Time control (default: nodes=10000)")
    parser.add_argument("--games", type=int, default=int(os.environ.get("GAMES", 5000)), help="Game pairs per matchup (default: 5000)")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", 0)), help="Parallel game pairs (default: 0 means auto)")
    parser.add_argument("--alpha", type=float, default=float(os.environ.get("ALPHA", 0.01)), help="Alpha value for SPRT (default: 0.01)")
    parser.add_argument("--min_num_games", type=int, default=int(os.environ.get("MIN_NUM_GAMES", 20)), help="Minimum number of games for SPRT (default: 20)")
    parser.add_argument("--opening", default=os.environ.get("OPENING", "6mvs_+90_+99.epd"), help="Opening book file (default: 6mvs_+90_+99.epd)")
    parser.add_argument("--opening_random_ply", type=int, default=0, help="Number of random moves to play before starting the tournament (default: 4)")
    parser.add_argument("--evaluator", choices=["byhand", "nnue"], default=os.environ.get("EVALUATOR", "byhand"), help="'byhand' or 'nnue' (default: byhand)")
    parser.add_argument("--machine", default=os.environ.get("MACHINE_TYPE", "e2-standard-16"), help="GCE machine type (default: e2-standard-16)")
    parser.add_argument("--zone", default=os.environ.get("ZONE", "us-central1-a"), help="GCE zone (default: us-central1-a)")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing VM (skip creation)")
    parser.add_argument("--keep", action="store_true", help="Don't delete VM after tournament")
    parser.add_argument("--teardown", action="store_true", help="Just delete the VM and exit")

    args = parser.parse_args()

    # Handle teardown only
    if args.teardown:
        print(f"🗑️  Deleting VM {VM_NAME}...")
        gce_cmd(args, ["instances", "delete", VM_NAME, "--quiet"], check=False, quiet=True)
        print("Done.")
        sys.exit(0)

    if len(args.engine) < 2:
        print("Error: need at least 2 --engine flags", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    unique_engines, resolved_engines = resolve_engines(args.engine)

    # ---------- Step 1: Create VM ----------
    if args.reuse:
        print(f"\n♻️  Reusing existing VM {VM_NAME}...")
        res = gce_cmd(args, ["instances", "describe", VM_NAME, "--format=json"], capture_output=True, check=False)
        if res.returncode != 0:
            print(f"❌ Error: VM '{VM_NAME}' does not exist! Please run without --reuse to create it.")
            sys.exit(1)
        
        try:
            status = json.loads(res.stdout).get("status")
            if status != "RUNNING":
                print(f"❌ Error: VM '{VM_NAME}' is currently in state: {status}.")
                print("   Start the VM or run without --reuse to recreate it.")
                sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Error: Failed to parse state for VM '{VM_NAME}'.")
            sys.exit(1)
    else:
        print(f"\n🖥️  Creating spot VM {VM_NAME} ({args.machine})...")
        
        # Delete existing VM if present
        gce_cmd(args, ["instances", "delete", VM_NAME, "--quiet"], check=False, quiet=True)

        startup_script = """#!/bin/bash
set -ex
export DEBIAN_FRONTEND=noninteractive
apt-get update -yq
apt-get install -yq git git-lfs g++ libgflags-dev zlib1g-dev python3 python3-pip tmux
pip3 install python-chess --break-system-packages
"""
        gce_cmd(args, [
            "instances", "create", VM_NAME,
            f"--machine-type={args.machine}",
            "--provisioning-model=SPOT",
            "--instance-termination-action=STOP",
            "--image-family=debian-12",
            "--image-project=debian-cloud",
            "--boot-disk-size=50GB",
            f"--metadata=startup-script={startup_script}"
        ])

        print("  Waiting for VM to be ready...")
        time.sleep(30)
        
        for i in range(20):
            if ssh_cmd(args, "echo ok", check=False, quiet=True).returncode == 0:
                break
            print(f"  Waiting for SSH... (attempt {i+1}/20)")
            time.sleep(10)

    # ---------- Step 2: Wait for startup script to finish ----------
    print("\n⏳ Waiting for dependencies to install...")
    for i in range(60):
        if ssh_cmd(args, "which git && which g++ && which tmux && python3 -c 'import chess' 2>/dev/null", check=False, quiet=True).returncode == 0:
            print("  Dependencies ready.")
            break
        
        if i == 59:
            print("Error: dependencies not ready after 10 minutes. Check startup script logs:")
            print(f"  gcloud compute ssh {VM_NAME} --zone={args.zone} --command='sudo journalctl -u google-startup-scripts'")
            sys.exit(1)
        time.sleep(10)

    # ---------- Step 3: Clone repo ----------
    print("\n📦 Cloning repository...")
    git_clone_cmd = f"""
        if [ ! -d ~/Pumpkin/.git ]; then
            git clone {REPO_URL} ~/Pumpkin
        else
            cd ~/Pumpkin && git fetch origin
        fi
        cd ~/Pumpkin && git lfs install && git lfs pull
    """
    ssh_cmd(args, git_clone_cmd)

    # ---------- Step 4: Upload files not in git ----------
    print("\n📤 Uploading opening book and extra files...")
    if os.path.exists("6mvs_+90_+99.epd"):
        scp_to_vm(args, "6mvs_+90_+99.epd", "~/Pumpkin/", quiet=True)
    if os.path.exists("byhand.bin"):
        scp_to_vm(args, "byhand.bin", "~/Pumpkin/", quiet=True)

    if args.evaluator == "byhand":
        models = list(Path(".").glob("byhand/runs/*/model.bin"))
        if models:
            # Sort by modification time (descending)
            models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest = models[0]
            remote_dir = f"~/Pumpkin/{latest.parent}"
            ssh_cmd(args, f"mkdir -p {remote_dir}", quiet=True)
            scp_to_vm(args, str(latest), f"~/Pumpkin/{latest}", quiet=True)
            print(f"  Uploaded {latest}")

    # ---------- Step 5: Build each engine version ----------
    print(f"\n🔨 Building {len(unique_engines)} engine versions...")
    
    for eng in unique_engines:
        print(f"  Building engine {eng['name']} from {eng['hash']}...")
        build_cmd = f"""
            cd ~/Pumpkin
            git checkout -q {eng['hash']} --force
            bash build.sh {eng['name']} src/uci/main.cpp -DNDEBUG -O3 {eng['flags_str']} 2>&1 | tail -3
            git checkout -q {BRANCH} --force
            echo '  → Built {eng['name']}'
        """
        ssh_cmd(args, build_cmd)

    # ---------- Step 6: Build engine flags for match.py ----------
    engine_flags = ""
    for eng in resolved_engines:
        base_cmd = f"./{eng['name']}"
        if args.evaluator == "byhand":
            base_cmd += " \"evaluator byhand\""
        elif args.evaluator == "nnue":
            base_cmd += " \"evaluator nnue\""
            
        if eng.get("engine_args_str"):
            base_cmd += f" {eng['engine_args_str']}"
            
        engine_flags += f" --engine {shlex.quote(base_cmd)}"

    # ---------- Step 7: Run tournament ----------
    reuse_flag = "--resume" if args.reuse else ""
    
    print("\n🏆 Starting tournament...")
    print(f"   Engines:")
    for engine in resolved_engines:
        print(f"     {engine['label']}")
    print(f"   TC: {args.tc}, Games: {args.games}, Concurrency: {args.concurrency}")
    print()

    match_cmd = (
        f"cd ~/Pumpkin && python3 -u match.py "
        f"{engine_flags} "
        f"--tc {shlex.quote(args.tc)} "
        f"--games {args.games} "
        f"--concurrency {args.concurrency} "
        f"--alpha {args.alpha} "
        f"--min_num_games {args.min_num_games} "
        f"--opening {shlex.quote(args.opening)} "
        f"--opening_random_ply {args.opening_random_ply} "
        f"{reuse_flag} "
        f"--pgn match_results.pgn"
    )

    print(match_cmd)

    start_tournament_cmd = f"""
        which tmux &>/dev/null || sudo apt-get install -y tmux
        tmux kill-session -t match 2>/dev/null || true
        cat << 'EOF' > ~/Pumpkin/run_match.sh
#!/bin/bash
{match_cmd} 2>&1 | tee ~/Pumpkin/match_log.txt
echo DONE > ~/Pumpkin/match_done.txt
EOF
        chmod +x ~/Pumpkin/run_match.sh
        tmux new-session -d -s match ~/Pumpkin/run_match.sh
        echo 'Tournament started in tmux session "match"'
    """
    ssh_cmd(args, start_tournament_cmd)

    # ---------- Step 8: Monitor progress ----------
    print("\n📊 Monitoring progress (Ctrl+C to detach — tournament continues on VM)...")
    print("   To re-attach later:")
    print(f"     gcloud compute ssh {VM_NAME} --zone={args.zone} --command='tmux attach -t match'\n")

    try:
        while True:
            # Check if done
            done_check = ssh_cmd(args, "test -f ~/Pumpkin/match_done.txt", check=False, quiet=True)
            if done_check.returncode == 0:
                print("\n✅ Tournament complete!")
                break
                
            # Print latest output
            ssh_cmd(args, "tail -1 ~/Pumpkin/match_log.txt 2>/dev/null", check=False)
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nDetached. Tournament still running on VM.")
        print(f"Re-attach: gcloud compute ssh {VM_NAME} --zone={args.zone} --command='tmux attach -t match'")
        sys.exit(0)

    # ---------- Step 9: Fetch results ----------
    print("\n📥 Downloading results...")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"tournament/cloud_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    scp_from_vm(args, "~/Pumpkin/match_results.pgn", f"{results_dir}/match.pgn", quiet=True)
    scp_from_vm(args, "~/Pumpkin/match_log.txt", f"{results_dir}/log.txt", quiet=True)
    
    print("\n==========================================")
    print(f"RESULTS (also saved to {results_dir}/)")
    print("==========================================")
    
    if os.path.exists(f"{results_dir}/log.txt"):
        with open(f"{results_dir}/log.txt", "r") as f:
            lines = f.readlines()
            # print last 40 lines
            for line in lines[-40:]:
                print(line.strip())

    # Save metadata
    with open(f"{results_dir}/metadata.txt", "w") as f:
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Machine: {args.machine} (spot)\n")
        f.write(f"Zone: {args.zone}\n")
        f.write(f"TC: {args.tc}\n")
        f.write(f"Games: {args.games}\n")
        f.write(f"Concurrency: {args.concurrency}\n")
        f.write(f"Evaluator: {args.evaluator}\n")
        f.write(f"Opening: {args.opening}\n\n")
        f.write("Engine details:\n")
        for eng in resolved_engines:
            f.write(f"  {eng['name']}: {eng['full_msg']}\n")
            f.write(f"    Commit: {eng['commit_ref']} ({eng['hash']})\n")
            f.write(f"    Flags: {eng['flags_str']}\n")
            f.write(f"    Args: {eng['engine_args_str']}\n")
            
    print(f"\n📁 Results saved to {results_dir}/")

    # ---------- Step 10: Teardown ----------
    if not args.keep:
        print()
        reply = input(f"🗑️  Delete VM {VM_NAME}? [Y/n] ").strip().lower()
        if reply in ('', 'y', 'yes'):
            gce_cmd(args, ["instances", "delete", VM_NAME, "--quiet"])
            print("  VM deleted.")
        else:
            print("  VM kept. Delete later with: ./cloud_match.py --teardown")
    else:
        print(f"\n💡 VM kept (--keep). Delete later with: ./cloud_match.py --teardown")

    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
