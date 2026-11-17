# StreamClip AI — Highlight Evaluation Report *(SYNTHETIC GROUND TRUTH)*

- **Evaluated At**: 2026-11-17T15:34:08.816356+00:00
- **Top-K Window**: K = 3 (out of 3 candidates)
- **Ground Truth Pool**: 5 total (3 approved)

## Primary Metrics
| Metric | Value | Meaning |
| :--- | :---: | :--- |
| **Precision@K** | `100.00%` | Fraction of top-K predictions that match approved moments |
| **Recall@K** | `100.00%` | Fraction of all approved ground truth moments captured |
| **Weighted Recall@K** | `100.00%` | Recall weighted by editorial rating (0-5 stars) |
| **Approval Rate@K** | `100.00%` | Editorial acceptance rate among top candidates |
| **Duplicate Rate** | `0.00%` | Rate of redundant duplicate predictions in top-K |
| **Mean IoU** | `0.600` | Average temporal overlap accuracy on matched windows |
| **Median IoU** | `0.533` | Median temporal overlap accuracy |

## Category-Level Breakdown
| Category | GT Count | Matches | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| `chaotic` | 1 | 1 | 100.0% | 100.0% |
| `fail` | 1 | 1 | 100.0% | 100.0% |
| `impressive` | 1 | 1 | 100.0% | 100.0% |
| `ordinary` | 0 | 0 | 0.0% | 0.0% |
