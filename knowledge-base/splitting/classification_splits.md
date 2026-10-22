# Classification Data Splitting

Classification datasets should generally preserve class proportions when
creating train and test partitions.

## Stratified Splitting

Stratified splitting maintains approximately the same class distribution
across the resulting partitions.

It is especially useful when classes are imbalanced.

## Random Splitting

A standard random split may be acceptable when class imbalance is not a
significant concern and the observations are independently distributed.

## Planning Guidance

The split strategy should be selected using the Dataset Analyzer findings and
the characteristics of the prediction task.

Validation and test data must remain separate from the training process.