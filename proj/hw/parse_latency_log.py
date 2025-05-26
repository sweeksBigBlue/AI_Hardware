#!/usr/bin/env python3
import re
import sys

def parse_log(filename):
    latencies = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.match(r'latency_y\d+\s*=\s*([-eE0-9.]+)', line)
            if match:
                latencies.append(float(match.group(1)) * 1e9)  # convert to ns
    return latencies

def print_summary(latencies, filename):
    total_latency_ns = sum(latencies)
    average_latency_ns = total_latency_ns / len(latencies) if latencies else 0

    print("\n=== Memristor Crossbar Latency Summary ===")
    print(f"Log file processed            : {filename}")
    print(f"Number of Columns Measured   : {len(latencies)}")
    if latencies:
        print(f"Latency per Column (first)   : {latencies[0]:.3f} ns")
    else:
        print("Latency per Column (first)   : N/A")
    print(f"Total Inference Latency      : {total_latency_ns:.3f} ns")
    print(f"Average Column Latency       : {average_latency_ns:.3f} ns")
    print("===========================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 parse_latency_log.py <log_file>")
        sys.exit(1)

    latencies = parse_log(sys.argv[1])
    print_summary(latencies, sys.argv[1])

