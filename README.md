# Real‑Time Fraud/Anomaly Detection Pipeline

This repository contains a reference implementation of a streaming fraud/anomaly detection system.  It demonstrates how to ingest and aggregate transactional data in real‑time, score each transaction with both unsupervised and supervised models, raise alerts, monitor for drift, and retrain models on a schedule.  The goal of this project is to provide an end‑to‑end example that can serve as a starting point for production deployments or educational purposes.

## Key Features

* **Streaming generator and ingestion** – A `transaction_generator.py` script emits synthetic financial transactions and writes them to Apache&nbsp;Kafka.  The `stream_processor.py` consumer reads these events and performs stateful aggregations on a per‑user basis (e.g., total spend in the last 5/30 minutes and transaction velocity).
* **Model scoring** – Two models are included:
  * **Isolation Forest** – an unsupervised anomaly detection algorithm that isolates points that require few splits to separate from the rest【510348481139182†L146-L154】.  Isolation Forest is useful when little or no labelled fraud data is available.
  * **XGBoost** – a gradient boosting classifier suited for highly imbalanced fraud problems【45698869217900†L14-L24】.  The project includes training scripts and example hyper‑parameter settings.
* **Alerts and case management** – After scoring, suspicious transactions are written to a PostgreSQL table (`cases`), which acts as a case‑management system.  A simple FastAPI endpoint can be used to query and update case statuses.
* **Monitoring and drift detection** – The pipeline uses [Evidently AI](https://github.com/evidentlyai/evidently) to continuously compare recent windows of streaming data against a reference dataset.  The included monitoring script runs a sliding window KS test to detect data drift and publishes alerts if too many features drift【482404642494084†L128-L148】.
* **Retraining pipeline** – A retraining workflow uses Airflow‐like tasks to pull the latest labelled data, train new models weekly, log them to MLflow for versioning, and compare performance against the current model.  If the new model outperforms the old one, it is promoted.
* **Benchmarking and post‑mortem** – The `benchmark/` folder contains a script to measure end‑to‑end latency and throughput.  A sample “incident post‑mortem” report in the `docs/` folder illustrates how to investigate and communicate a significant drift event.

## Architecture

Below is an overview of the pipeline.  A synthetic transaction generator writes events to a Kafka topic.  A stream processor consumes these events, performs stateful aggregations, and produces feature vectors.  The scoring service reads the feature vectors, calls the ML models, and writes alerts and scores to a case table.  A monitoring component reads both the feature stream and the predictions to detect drift.  A retraining workflow runs weekly to retrain models and compare their performance.

```text
┌─────────────┐    write          ┌──────────────────┐     aggregate       ┌────────────────┐     score        ┌─────────────────┐
│ Transaction │ ────────────────▶ │ Kafka Topic      │ ──────────────────▶ │ Stream Processor│ ───────────────▶ │ Scoring Service │
│ Generator   │ (transactions)    │ (transactions)    │                    │ (stateful)     │                │ (Isolation/XGB)│
└─────────────┘                    └──────────────────┘                    └────────────────┘                │              │
                                                                                                             │              │
                                                                                                             ▼              │
                                                                                                         PostgreSQL ←─────┘
                                                                                                           (cases)

                                          ▲
                                          │
                      drift reports & alerts│
                                          │
                                      ┌─────────────────┐
                                      │ Monitoring      │
                                      │ (Evidently AI)  │
                                      └─────────────────┘

```

## Getting Started

This repository is organised as a modular Python package so you can run individual components independently.  You will need Python 3.9+ and `poetry` or `pip` for dependency management.  Kafka and PostgreSQL should be available locally or via Docker.

### Installing dependencies

1. Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install and start Kafka and PostgreSQL.  The easiest way is via Docker Compose:

```bash
docker compose up -d
```

This will start a Kafka broker, a Zookeeper instance, and a PostgreSQL database with a default user/password.  See `docker-compose.yml` for configuration.

### Running the pipeline

1. **Start the transaction generator** – run `python src/streaming/transaction_generator.py` to publish synthetic transactions to Kafka.
2. **Run the stream processor** – execute `python src/aggregator/stream_processor.py` to consume transactions, compute rolling features, and publish feature vectors to another Kafka topic.
3. **Start the scoring service** – run `uvicorn src/scoring_service.app:app --reload` to start a FastAPI server that listens for feature vectors, scores them using the chosen model, and writes alerts to Postgres.
4. **Monitor drift** – execute `python src/monitoring/drift_monitor.py` to continuously monitor for data and concept drift using Evidently AI.  If drift is detected, an alert will be raised and a retraining job can be triggered.
5. **Retrain models** – run `python src/retraining/retrain.py` to train new models on the latest labelled data.  This script logs models and metrics to MLflow and compares them against the current production model.

### Benchmarking

The `benchmark/benchmark.py` script measures the end‑to‑end latency from a transaction being produced to the alert being written to Postgres and calculates throughput.  Use it to tune Kafka partitions, batch sizes, and model inference times.

## Contributing

This repository is intentionally opinionated but extensible.  Feel free to open issues or submit pull requests if you find bugs or have suggestions.  For large changes, please open an issue first to discuss what you would like to change.

## References

* **Isolation Forest** – An anomaly detection algorithm that isolates points requiring few splits to separate them from the rest of the data【510348481139182†L146-L154】.
* **XGBoost for fraud detection** – Gradient boosting can handle imbalanced fraud datasets and is widely used for credit‑card fraud detection【45698869217900†L14-L24】.
* **Drift monitoring** – Evidently AI offers open‑source tests for data, concept and prediction drift and integrates with Kafka【482404642494084†L128-L148】.
