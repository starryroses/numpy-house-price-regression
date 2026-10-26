"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    for j in range(X.shape[1]):
        col = X[:,j]
        if np.all(np.isnan(col)):
            X[:,j] = 0.0
            continue
        meanval = np.nanmean(col)
        X[np.isnan(col), j] = meanval
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    lower = []
    upper = []
    for j in range(X.shape[1]):
        col = X[:,j]
        q1 = np.percentile(col,25)
        q3 = np.percentile(col,75)
        iqr = q3-q1
        low = q1-(k*iqr)
        up = q3+(k*iqr)
        lower.append(low)
        upper.append(up)
    return (lower,upper)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    X_copy = X.copy()
    rows,cols = X.shape
    for i in range(rows):
        for j in range(cols):
            if X_copy[i][j] > upper[j]:
                X_copy[i][j] = upper[j]
            elif X_copy[i][j] < lower[j]:
                X_copy[i][j] = lower[j]
    return X_copy

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    ratio = np.zeros(len(numerator))
    for i in range(len(numerator)):
        ratio[i] = (numerator[i]/(denominator[i]+eps))
    return ratio

# Step 5 - append_column (not yet solved)
# TODO: implement

# Step 6 - one_hot_encode (not yet solved)
# TODO: implement

# Step 7 - fit_standardizer (not yet solved)
# TODO: implement

# Step 8 - apply_standardizer (not yet solved)
# TODO: implement

# Step 9 - add_bias_column (not yet solved)
# TODO: implement

# Step 10 - make_shuffled_indices (not yet solved)
# TODO: implement

# Step 11 - partition_indices (not yet solved)
# TODO: implement

# Step 12 - subset_xy (not yet solved)
# TODO: implement

# Step 13 - ols_fit (not yet solved)
# TODO: implement

# Step 14 - ols_predict (not yet solved)
# TODO: implement

# Step 15 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 16 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 17 - r_squared (not yet solved)
# TODO: implement

# Step 18 - residual_summary (not yet solved)
# TODO: implement

# Step 19 - prepare_cleaned_features (not yet solved)
# TODO: implement

# Step 20 - assemble_feature_matrix (not yet solved)
# TODO: implement

# Step 21 - make_train_val_test (not yet solved)
# TODO: implement

# Step 22 - standardize_and_add_bias (not yet solved)
# TODO: implement

# Step 23 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 24 - house_price_pipeline (not yet solved)
# TODO: implement

