# Kaggle Knight Prometeo – Log Loss Optimization

This repository contains my solution for the Kaggle Knight Prometeo competition.

## Problem Statement
Binary classification on a highly imbalanced and noisy transaction dataset.
The evaluation metric is **Log Loss**, where lower values indicate better calibrated probability predictions.

## Approach
- Focused on **probability calibration** instead of raw accuracy
- Used **XGBoost** with extreme regularization to reduce overconfidence
- Restricted model complexity using depth-1 trees (decision stumps)
- Applied strong L1/L2 regularization to stabilize predictions
- Avoided overfitting and leaderboard probing

## Key Insight
For Log Loss–driven problems, **simpler and smoother models outperform complex ones**.
Reducing overconfidence is more important than increasing model capacity.

## Result
Achieved a public leaderboard Log Loss of approximately **0.102**, placing in the top tier.

## Tools & Libraries
- Python
- XGBoost
- Pandas
- scikit-learn

## Notes
- Training and test datasets are provided by Kaggle and are not included in this repository.
- This repository contains only modeling code and configuration.
