"""
benchmark.py
------------

Measure the latency and throughput of the end‑to‑end pipeline.  The script
generates a batch of feature vectors, sends them to the scoring service via
HTTP, and records response times.  At the end it reports the average latency,
99th percentile latency, and throughput (requests per second).

Usage::

    python benchmark.py --n 100 --url http://localhost:8000/score --model xgboost

"""

import argparse
import json
import random
import time
import statistics
import requests


def random_feature() -> dict:
    return {
        "user_id": f"user_{random.randint(0, 999)}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "amount": random.uniform(1, 200),
        "sum_5m": random.uniform(1, 500),
        "count_5m": random.randint(1, 10),
        "sum_30m": random.uniform(1, 2000),
        "count_30m": random.randint(1, 50),
        "category": random.choice(["grocery", "electronics", "fashion", "travel"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark scoring service latency")
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--url", default="http://localhost:8000/score", help="Scoring endpoint URL")
    parser.add_argument("--model", default="isolation", help="Model to use for scoring")
    parser.add_argument("--threshold", type=float, default=0.5, help="Suspicion threshold")
    args = parser.parse_args()
    latencies = []
    for _ in range(args.n):
        payload = random_feature()
        start = time.time()
        response = requests.post(args.url, params={"model": args.model, "threshold": args.threshold}, json=payload)
        elapsed = time.time() - start
        latencies.append(elapsed)
        if response.status_code != 200:
            print("Request failed", response.text)
    avg = statistics.mean(latencies)
    p99 = sorted(latencies)[int(len(latencies) * 0.99) - 1]
    throughput = args.n / sum(latencies)
    print(f"Sent {args.n} requests. Average latency: {avg*1000:.2f} ms, 99th percentile: {p99*1000:.2f} ms, throughput: {throughput:.2f} rps")


if __name__ == "__main__":
    main()
