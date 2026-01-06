# Incident Post‑Mortem: Data Drift on 2025‑12‑15

## Background

On 15 December 2025 our fraud detection pipeline began raising an unusually high number of drift alerts.  The monitoring component uses Evidently AI’s drift tests to compare the distribution of incoming transaction features against the training data.  According to Conduktor’s guidelines, drift detection in streaming systems should continuously compare sliding windows of events against a reference dataset and trigger alerts when divergence exceeds a threshold【482404642494084†L128-L213】.

## Timeline

| Time (UTC) | Event |
|-----------|------|
| 16:02 | Monitoring service detects data drift in the `amount` feature.  The mean of transaction amounts in the current window shifted from \$50 to \$70.  An alert is published to the `drift_alerts` Kafka topic. |
| 16:03 | Slack notification sent to the Fraud Ops channel.  Engineers begin investigating the cause. |
| 16:10 | Engineers confirm that a holiday sale campaign started at 16:00, resulting in many high‑value purchases.  This legitimate shift in customer behaviour caused the data drift. |
| 16:20 | After confirming that fraud rates remain stable, the team marks the incident as a false positive.  A new feature flag is added to ignore drift alerts for the `amount` feature during the campaign’s timeframe. |
| 17:00 | A patch is rolled out to update the monitoring configuration, reducing sensitivity for the `amount` feature and adding campaign awareness rules. |

## Root Cause

The drift detector correctly identified a significant distribution shift in transaction amounts.  However, the shift was due to a planned marketing campaign rather than fraud or model degradation.  The absence of business context caused the monitoring system to raise critical alerts.

## Lessons Learned and Action Items

* **Contextual Awareness:** Drift detection rules should incorporate business calendars and known campaigns.  Tools like Evidently provide flexible configuration to adjust thresholds per feature【482404642494084†L128-L148】.
* **Alert Calibration:** Not all drift is harmful.  Some features, like `amount`, naturally exhibit high variance during seasonal events.  Thresholds must reflect this variability.
* **Communication:** Prompt communication between the data science team and business stakeholders is essential to differentiate legitimate behaviour changes from anomalies.

## Conclusion

By integrating drift detection with business context, we can avoid unnecessary escalations and focus engineering effort on genuine anomalies.  This incident highlighted the importance of continuous monitoring and the need to tune alerting thresholds based on domain knowledge.
