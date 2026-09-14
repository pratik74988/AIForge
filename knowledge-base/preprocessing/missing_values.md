# Missing Value Handling

## Numerical Features

For numerical features, median imputation is a robust default when the
distribution may contain outliers or be skewed.

Mean imputation can be appropriate when the feature distribution is reasonably
symmetric and extreme values are not a concern.

The imputation value must be learned from the training data only and then
applied to validation and test data.

## Categorical Features

For categorical features, most-frequent imputation is a common default.

A separate category such as "missing" can be useful when the absence of a
value may itself contain information.

## Planning Guidance

Consider missing-value preprocessing when the Dataset Analyzer reports missing
values in usable features.

Do not impute the target variable as part of ordinary feature preprocessing.