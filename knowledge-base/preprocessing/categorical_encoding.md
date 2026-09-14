# Categorical Encoding

Categorical features must be transformed into a numerical representation before
being consumed by models that require numerical inputs.

## One-Hot Encoding

One-hot encoding is a strong general-purpose choice for nominal categorical
features.

It creates a binary feature for each category.

Unknown categories encountered during inference should be handled explicitly
rather than causing the pipeline to fail.

## Ordinal Encoding

Ordinal encoding may be appropriate when categories have a meaningful order.

It should not be used merely because categorical values can be converted to
integers.

## Planning Guidance

The choice of encoding should consider the semantic nature of the categorical
feature and the selected model.

Encoding must be fitted using training data only.