SUPPORTED_TEMPLATES = {
    "sklearn",
}


SUPPORTED_PREPROCESSING_OPERATIONS = {
    "impute",
    "encode",
    "scale",
}


CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "roc_auc",
}


REGRESSION_METRICS = {
    "mae",
    "mse",
    "rmse",
    "r2",
}


SUPPORTED_SPLIT_STRATEGIES = {
    "random",
    "train_test",
}


MAX_CPU = 16
MAX_MEMORY_GB = 64

HUMAN_APPROVAL_CPU_THRESHOLD = 8
HUMAN_APPROVAL_MEMORY_GB_THRESHOLD = 32