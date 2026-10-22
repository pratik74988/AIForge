# Numerical Feature Scaling

Feature scaling can be important for models whose optimization or distance
calculations are sensitive to feature magnitude.

## Standard Scaling

Standard scaling transforms numerical features using their mean and standard
deviation.

It is commonly appropriate for linear models such as logistic regression.

## When Scaling May Not Be Necessary

Tree-based models generally do not require numerical feature scaling because
their splitting decisions are not based on feature magnitude in the same way
as distance-based or gradient-based models.

## Planning Guidance

Scaling should be selected based on the model family rather than applied
automatically to every dataset.