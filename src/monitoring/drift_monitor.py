"""
drift_monitor.py
----------------

Continuously monitor streaming feature data for signs of data or concept drift
using the Evidently AI library.  The monitor consumes feature vectors from a
Kafka topic, maintains a sliding window of recent events, and compares the
distribution of numeric features against a reference dataset (usually the
training data).  If too many columns drift, an alert is printed and written to
a monitoring topic.

This script demonstrates the pattern described by Conduktor for streaming drift
detection【482404642494084†L128-L213】.  In a production system you would
integrate with your alerting infrastructure or trigger a retraining pipeline.
"""

import argparse
import json
import time
from collections import deque

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns


def load_reference(path: str) -> pd.DataFrame:
    """Load a reference dataset (e.g., training data features)."""
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor drift on streaming features")
    parser.add_argument("--bootstrap-server", default="localhost:9092")
    parser.add_argument("--input-topic", default="features")
    parser.add_argument("--alert-topic", default="drift_alerts")
    parser.add_argument("--reference", required=True, help="Path to reference dataset (CSV or Parquet)")
    parser.add_argument("--window-size", type=int, default=1000, help="Number of events per monitoring window")
    args = parser.parse_args()

    # Load reference
    ref = load_reference(args.reference)

    # Define tests: drift on numeric columns and share of drifted columns
    numeric_cols = [c for c in ref.columns if ref[c].dtype != object]
    tests = [TestColumnDrift(column_name=col) for col in numeric_cols]
    tests.append(TestShareOfDriftedColumns(threshold=0.3))
    suite = TestSuite(tests=tests)

    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.bootstrap_server,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    window = deque(maxlen=args.window_size)

    for message in consumer:
        window.append(message.value)
        if len(window) >= args.window_size:
            current_df = pd.DataFrame(list(window))
            suite.run(reference_data=ref, current_data=current_df)
            results = suite.as_dict()
            if not results["summary"]["all_passed"]:
                drifted = [test["column_name"] for test in results["tests"] if test.get("status") == "FAIL"]
                alert = {
                    "timestamp": time.time(),
                    "drifted_columns": drifted,
                    "failed_count": results["summary"]["failed_count"],
                }
                print(f"[DRIFT ALERT] columns={drifted}")
                producer.send(args.alert_topic, value=alert)
                producer.flush()
            # Reset window for next cycle (50% overlap)
            window.clear()


if __name__ == "__main__":
    main()
