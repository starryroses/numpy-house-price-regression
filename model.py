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

# Step 5 - append_column
def append_column(X, col):
    X = np.asarray(X)
    col = np.asarray(col)
    col = col.reshape(-1,1)
    appended_col = np.concatenate([X, col], axis=1)
    return appended_col

# Step 6 - one_hot_encode
import numpy as np

def one_hot_encode(labels):
    unique = np.unique(labels)
    result = np.zeros((len(labels), len(unique)))

    for i, label in enumerate(labels):
        result[i, np.where(unique == label)[0][0]] = 1.0

    return result

# Step 7 - fit_standardizer
def fit_standardizer(X):
    X = np.array(X)
    num_cols = X.shape[1]
    means = []
    stds = []
    for j in range(num_cols):
        means.append(np.mean(X[:,j]))
        stds.append(np.std(X[:,j]) if np.std(X[:,j])>0 else 1.0)

    return np.array(means), np.array(stds)

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    X_copy = X.copy()
    X_copy = (X_copy-mean)/std
    return X_copy

# Step 9 - add_bias_column
def add_bias_column(X):
    bias = np.ones((X.shape[0],1))
    added_bias = np.concatenate([bias, X], axis=1)
    return added_bias

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    idx = np.random.default_rng(seed).permutation(n_samples)
    return idx

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    n = len(indices)
    train_end = int(n*train_ratio)
    val_end = train_end + int(n*val_ratio)
    train = indices[:train_end]
    val = indices[train_end:val_end]
    test = indices[val_end:]
    return train,val,test

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    X_selected = X[indices]
    y_selected = y[indices]
    return X_selected, y_selected

# Step 13 - ols_fit
def ols_fit(X, y):
    A = X.T@X
    B=X.T@y
    ols_weights = np.linalg.solve(A,B)
    return ols_weights

# Step 14 - ols_predict
def ols_predict(X, theta):
    return X@theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    return float(np.mean(abs(y_true-y_pred)))

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

