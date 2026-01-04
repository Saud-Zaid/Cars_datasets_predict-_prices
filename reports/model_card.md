# Model Card — Week 3 Baseline

## Problem
- **Predict:** `is_high_value` (Binary classification)
- **Unit of Analysis:** One row per user
- **Decision enabled:** Who gets a retention offer?
- **Constraints:** CPU-only; offline-first; batch inference

## Data (Contract)
- **Feature table:** `data/processed/features.csv`
- **Target column:** `is_high_value` (1 = high value user)
- **Required Features:** `country`, `n_orders`, `avg_amount`, `total_amount`
- **Optional IDs (Passthrough):** `user_id`
- **Forbidden Columns:** None yet (Target is dropped automatically in training)

## Metrics (Draft)
- **Primary:** F1-Score (balance between precision and recall)
-