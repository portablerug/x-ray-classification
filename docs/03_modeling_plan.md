# Phase 3-5 Guide: Modeling Strategy

## Modeling Goals

- Establish a strong, interpretable baseline
- Improve performance with transfer learning
- Evaluate model choices under class imbalance

## Planned Model Tiers

1. Baseline model (simple CNN or lightweight alternative)
2. Transfer learning model (ResNet/EfficientNet or equivalent)
3. Optional advanced optimization (thresholding, focal loss)

## Experiment Tracking Expectations

- Track dataset version and split version
- Record metrics beyond accuracy (precision, recall, F1, ROC-AUC)
- Track confusion matrix and minority-class recall by run
- Record hyperparameters and training assumptions

## Comparison Criteria

- Minority class recall impact
- Precision-recall tradeoff by threshold
- Stability across repeated runs (if performed)
- Practical interpretation for healthcare context
