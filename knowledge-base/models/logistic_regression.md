# Logistic Regression

Logistic regression is a supervised classification algorithm commonly used for
binary classification and can also support multiclass classification.

## Strengths

- Simple and interpretable baseline.
- Efficient to train.
- Works well with appropriately processed numerical features.
- Often benefits from feature scaling.
- Supports regularization.

## Limitations

- Assumes a relatively simple decision relationship between features and the
  target.
- May underperform when the decision boundary is highly nonlinear.
- High-cardinality categorical features may require careful preprocessing.

## Planning Guidance

Logistic regression is a reasonable candidate when the user requests a
classification baseline, interpretability, or a relatively simple model.

Model selection should also consider the Dataset Analyzer findings and user
requirements.