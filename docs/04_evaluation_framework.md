# Evaluation Framework (Class Imbalance Focus)

## Primary Metrics

- Recall (high priority for disease detection)
- Precision
- F1-score
- ROC-AUC
- Confusion matrix

## Why Accuracy Alone Is Insufficient

Class imbalance can make high accuracy misleading if the model misses positive disease cases.

## Imbalance Techniques to Evaluate

- Class weighting
- Threshold tuning
- Optional focal loss
- Optional feature-space oversampling methods where appropriate

## Interpretation Prompts

- Which label patterns generate most false negatives?
- How does threshold choice change recall vs precision?
- What operational trade-offs are acceptable in healthcare screening contexts?

## Reporting Standard

Every model report should include:

- Metric table (with focus on recall and F1)
- Confusion matrix visualization
- Precision-recall discussion
- Brief risk statement for false negatives
