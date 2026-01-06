"""
stream_processor.py
-------------------

Consume raw transactions from Kafka, compute stateful aggregations for each user,
and publish feature vectors to another Kafka topic.  The goal is to enrich
incoming events with features like total spend in the last 5 and 30 minutes,
transaction velocity, and category frequencies.  These features are then used
by the scoring service.

The processor maintains an in‑memory window of recent transactions for each
user.  For a real deployment you should use a state store such as RocksDB via
Apache Flink or Kafka Streams for fault tolerance.

Usage::

    python stream_processor.py --bootstrap-server localhost:9092 --input-topic transactions --output-topic features

"""

import argparse
import json
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone

from kafka import KafkaConsumer, KafkaProducer


class SlidingWindowAggregator:
    """Maintain rolling aggregates for each user over 5 and 30 minute windows."""

    def __init__(self, window_sizes=(5, 30)):
        # For each user we store a deque of (timestamp, amount) events
        self.windows = defaultdict(lambda: deque())
        self.window_sizes = [timedelta(minutes=w) for w in window_sizes]

    def update(self, user_id: str, amount: float, timestamp: datetime) -> dict:
        q = self.windows[user_id]
        # Append new transaction
        q.append((timestamp, amount))
        # Remove events outside the 30‑minute window
        oldest_cutoff = timestamp - max(self.window_sizes)
        while q and q[0][0] < oldest_cutoff:
            q.popleft()
        # Compute features for each window size
        features = {}
        for w in self.window_sizes:
            cutoff = timestamp - w
            amounts = [amt for ts, amt in q if ts >= cutoff]
            features[f"sum_{int(w.total_seconds()/60)}m"] = sum(amounts)
            features[f"count_{int(w.total_seconds()/60)}m"] = len(amounts)
        return features


def iso_to_datetime(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream processor for transaction aggregation")
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--input-topic", default="transactions")
    parser.add_argument("--output-topic", default="features")
    args = parser.parse_args()

    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.bootstrap_server,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    aggregator = SlidingWindowAggregator()

    print(f"Processing topic {args.input_topic} → {args.output_topic} …")
    for message in consumer:
        txn = message.value
        user_id = txn.get("user_id")
        amount = float(txn.get("amount", 0.0))
        ts = iso_to_datetime(txn.get("timestamp")).replace(tzinfo=None)
        features = aggregator.update(user_id, amount, ts)
        # Build feature vector
        feature_event = {
            "user_id": user_id,
            "timestamp": txn["timestamp"],
            "amount": amount,
            **features,
            "category": txn.get("category"),
        }
        # Publish to output topic
        producer.send(args.output_topic, value=feature_event)
        producer.flush()
        # Simulate real‑time processing
        time.sleep(0.001)


if __name__ == "__main__":
    main()
