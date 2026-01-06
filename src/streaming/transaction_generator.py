"""
transaction_generator.py
------------------------

Generate synthetic transaction events and publish them to Kafka.  Each event
contains a user identifier, a timestamp, an amount, a merchant category, and
a label indicating whether the transaction is fraudulent (for training data).

Synthetic data
^^^^^^^^^^^^^^

The generator uses a simple probabilistic model: for each user we draw a normal
distribution of legitimate transactions and inject anomalies with a small
probability.  You can adjust the fraud rate, mean spend and variance via
command‑line arguments.

Usage::

    python transaction_generator.py --bootstrap-server localhost:9092 --topic transactions

Dependencies: kafka-python
"""

import argparse
import json
import random
import string
import time
from datetime import datetime, timezone

from kafka import KafkaProducer


def random_user_id(num_users: int = 1000) -> str:
    """Return a random user identifier from 0 to num_users-1."""
    return f"user_{random.randint(0, num_users - 1)}"


def random_category() -> str:
    categories = [
        "grocery",
        "electronics",
        "fashion",
        "travel",
        "utilities",
        "health",
        "entertainment",
    ]
    return random.choice(categories)


def generate_transaction(user: str, fraud_rate: float, mean: float, std: float) -> dict:
    """Generate a single transaction.

    With probability `fraud_rate` the amount is drawn from a separate distribution
    representing fraudulent activity (e.g., high spend).  Otherwise it is drawn
    from a normal distribution around `mean`.
    """
    is_fraud = random.random() < fraud_rate
    if is_fraud:
        amount = max(1.0, random.gauss(mean * 3, std * 2))
    else:
        amount = max(1.0, random.gauss(mean, std))
    return {
        "user_id": user,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": round(amount, 2),
        "category": random_category(),
        "is_fraud": int(is_fraud),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic transactions and publish to Kafka")
    parser.add_argument("--bootstrap-server", default="localhost:9092", help="Kafka bootstrap server")
    parser.add_argument("--topic", default="transactions", help="Kafka topic name")
    parser.add_argument("--users", type=int, default=1000, help="Number of distinct users")
    parser.add_argument("--fraud-rate", type=float, default=0.001, help="Fraction of transactions that are fraudulent")
    parser.add_argument("--mean", type=float, default=50.0, help="Mean transaction amount")
    parser.add_argument("--std", type=float, default=20.0, help="Standard deviation of transaction amount")
    parser.add_argument("--sleep", type=float, default=0.01, help="Sleep time between transactions (seconds)")
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Generating transactions on topic {args.topic}… press Ctrl+C to stop.")
    try:
        while True:
            user = random_user_id(args.users)
            txn = generate_transaction(user, args.fraud_rate, args.mean, args.std)
            producer.send(args.topic, value=txn)
            # flush occasionally to avoid buffer build‑up
            producer.flush()
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("Generator stopped.")


if __name__ == "__main__":
    main()
