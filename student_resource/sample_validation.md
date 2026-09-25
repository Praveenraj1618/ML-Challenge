# Sample Validation Report

## Data inspected

- 5,000 labelled Source 1 records
- 13,328 Source 2 records
- 14,055 Source 3 records
- 17,383 labelled cross-source links
- 280 labelled singleton Source 1 entities
- Training countries: US and India
- Test sample countries: US, India and unseen France

## Wide hybrid baseline

| Metric | Result |
|---|---:|
| Candidate pairs | 743,829 |
| Captured labelled links | 17,372 / 17,383 |
| Candidate link recall | 0.999367 |
| Entities retaining every match | 0.997800 |
| Grouped 5-fold macro F0.5 | 0.992088 |
| Selected threshold | 0.8100 |

The validation split was grouped by Source 1 entity to prevent candidate-pair
leakage. Threshold selection used the exact entity-level macro F0.5 behavior,
including singleton scoring.

## Next improvement stage

After confirming the full-data baseline, add multilingual dense retrieval and
reciprocal-rank fusion, then re-mine hard negatives. Retain each addition only
when grouped out-of-fold macro F0.5 improves.

