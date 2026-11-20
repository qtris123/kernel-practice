"""
Local Multi-Process Simulator Runner for Collective Communication (AllReduce)
Allows simulating distributed CPU/GPU collective algorithms on a single machine
using separate OS processes communicating over loopback TCP sockets.
"""

import argparse
import os
import sys
import time
import subprocess

def run_simulation(script_name, num_nodes, tensor_size, latency_ms=0.0, bw_mbps=0.0, base_port=16500, verify=False):
    workspace = os.path.dirname(os.path.abspath(__file__))
    parts = script_name.split()
    script_file = parts[0]
    extra_args = parts[1:]
    script_path = os.path.join(workspace, script_file)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = workspace
    if latency_ms > 0:
        env["SIM_LATENCY_S"] = str(latency_ms / 1000.0)
    if bw_mbps > 0:
        env["SIM_BW_MBPS"] = str(bw_mbps)

    print(f"\n=======================================================")
    print(f"Algorithm:      {script_name}")
    print(f"Nodes (Ranks):  {num_nodes} CPU processes")
    tensor_bytes = tensor_size * 4  # float32 = 4 bytes
    if tensor_bytes >= 1024 * 1024:
        size_str = f"{tensor_bytes / (1024 * 1024):.2f} MB ({tensor_size:,} elements)"
    elif tensor_bytes >= 1024:
        size_str = f"{tensor_bytes / 1024:.2f} KB ({tensor_size:,} elements)"
    else:
        size_str = f"{tensor_bytes} Bytes ({tensor_size} elements)"
    print(f"Tensor Size:    {size_str}")
    if latency_ms > 0 or bw_mbps > 0:
        print(f"Simulated Net:  Latency = {latency_ms} ms, Bandwidth = {bw_mbps or 'unlimited'} Mbps")
    print(f"=======================================================")

    procs = []
    t_start = time.time()
    for rank in range(num_nodes):
        cmd = [
            sys.executable, script_path
        ] + extra_args + [
            "--master-ip", f"127.0.0.1:{base_port}",
            "--num-nodes", str(num_nodes),
            "--rank", str(rank),
            "--tensor-size", str(tensor_size)
        ]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((rank, p))
        time.sleep(0.005)

    outputs = {}
    errors = {}
    for rank, p in procs:
        stdout, stderr = p.communicate()
        outputs[rank] = stdout.strip()
        errors[rank] = stderr.strip()

    t_end = time.time()
    total_wall_time = t_end - t_start

    # Check for failures
    failed = False
    for rank, p in procs:
        if p.returncode != 0:
            failed = True
            print(f"[ERROR] Rank {rank} failed (exit code {p.returncode}):")
            print(errors[rank])

    if not failed:
        for rank in range(min(4, num_nodes)):
            print(f"  [Rank {rank}] {outputs[rank]}")
        if num_nodes > 4:
            print(f"  ... [Ranks 4-{num_nodes-1} also completed successfully]")
        print(f"--> Total Wall Time: {total_wall_time:.4f} seconds\n")

    return not failed, total_wall_time

def main():
    parser = argparse.ArgumentParser(description="Simulate Multi-GPU/Node AllReduce with CPU processes")
    parser.add_argument("--algo", "-a", choices=["ring", "bde", "practice-ring", "practice-bde"], default="practice-ring",
                        help="Collective algorithm to test")
    parser.add_argument("--nodes", "-n", type=int, default=4,
                        help="Number of simulated worker processes (e.g. 2, 4, 8, 16)")
    parser.add_argument("--tensor-size", "-t", type=int, default=1024,
                        help="Number of float32 elements in tensor (1024 = 4KB, 2621440 = 10MB)")
    parser.add_argument("--latency-ms", type=float, default=0.0,
                        help="Simulated network latency (alpha) in ms per message")
    parser.add_argument("--bandwidth-mbps", type=float, default=0.0,
                        help="Simulated network bandwidth limit (beta) in Mbps (e.g. 100 for 100Mbps)")
    parser.add_argument("--port", type=int, default=16500,
                        help="Base TCP port for localhost communication")

    args = parser.parse_args()
    if args.algo == "ring":
        script = "ring_allreduce.py"
    elif args.algo == "bde":
        script = "recursive_hd_reduce.py"
    elif args.algo == "practice-ring":
        script = "practice_allreduce.py --algo ring"
    else:
        script = "practice_allreduce.py --algo bde"
    run_simulation(
        script_name=script,
        num_nodes=args.nodes,
        tensor_size=args.tensor_size,
        latency_ms=args.latency_ms,
        bw_mbps=args.bandwidth_mbps,
        base_port=args.port
    )

if __name__ == "__main__":
    main()
