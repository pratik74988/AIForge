# Classification Evaluation Metrics

Classification metrics should reflect the user's actual objective.

## Accuracy

Accuracy measures the proportion of predictions that are correct.

It can be misleading when classes are strongly imbalanced.

## Precision

Precision measures the proportion of predicted positive examples that are
actually positive.

It is important when false positives are costly.

## Recall

Recall measures the proportion of actual positive examples that are correctly
identified.

It is important when false negatives are costly.

## F1 Score

F1 combines precision and recall into a single metric using their harmonic
mean.

It can be useful when both false positives and false negatives matter.

## ROC AUC

ROC AUC measures ranking performance across classification thresholds.

It can be useful for evaluating binary classifiers, but should not
automatically replace metrics that directly reflect the user's business
objective.

## Planning Guidance

The primary metric should be selected according to the user's stated
objective.

Additional metrics can be reported to provide a broader view of model
performance.