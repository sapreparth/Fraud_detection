# Monitoring & Drift Detection Report

This report summarises the current state of the data and concept drift monitoring for the fraud detection pipeline.  We use Evidently AI to continuously compare recent streaming windows against the training reference and alert when more than 30 % of numeric features drift【482404642494084†L128-L213】.

## Drift Detection Example

The plot below illustrates a synthetic drift scenario.  The blue histogram shows the distribution of transaction amounts in the training (reference) dataset, while the red histogram shows the distribution in a recent streaming window.  The streaming distribution has a higher mean and heavier tail, indicating data drift.  Evidently’s column drift tests detect this shift and raise an alert.

![Distribution Drift Example]({{file:file-G4Rir7a6Fw77u2QjH5VJHK}})

In this example, the drift alert included the following information:

* Drifted columns: `amount`
* Failed tests: 1 (column drift) out of 5 total tests
* Severity: medium

## Current Status

At the time of writing, no critical drift alerts are active.  The monitoring component runs continuously and publishes alerts to the `drift_alerts` Kafka topic.  Recent windows show minor fluctuations in spend and velocity but remain within configured thresholds.

## Next Steps

* Continue to monitor drift metrics and calibrate thresholds per feature.
* Integrate business context (e.g., sales events) into alert suppression rules to reduce false positives.
* Extend monitoring to include concept drift by tracking model prediction distributions and performance metrics【482404642494084†L128-L148】.
