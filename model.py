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
    idx = np.random.RandomState(seed).permutation(n_samples)
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
    ols_weights, *_ = np.linalg.lstsq(A,B)
    return ols_weights

# Step 14 - ols_predict
def ols_predict(X, theta):
    return X@theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    return float(np.mean(abs(y_true-y_pred)))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(np.mean(pow(y_true-y_pred,2))))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    ss_res = np.sum(pow(y_true-y_pred,2))
    ymean = np.mean(y_true)
    ss_tot = np.sum(pow(y_true-ymean,2))
    return float(1.0-(ss_res/ss_tot)) if ss_tot!=0 else 0.0

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    r = y_true-y_pred
    res = {
        'mean': np.mean(r),
        'std': np.std(r),
        'median_abs':np.median(np.abs(r))
    }
    return res

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    X = impute_nan_with_mean(X)
    lower,upper = compute_iqr_bounds(X,iqr_k)
    X = clip_columns(X,lower,upper)
    return X

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    numerator = X_num[:,ratio_num_idx]
    denominator = X_num[:,ratio_den_idx]
    ratio = make_ratio_feature(numerator,denominator)
    X_extended = append_column(X_num,ratio)

    if cat_labels is not None:
        cat_matrix = one_hot_encode(cat_labels)
        X_extended = np.hstack((X_extended, cat_matrix))

    return X_extended

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    
    indices = make_shuffled_indices(X.shape[0],seed)
    train_idx,val_idx,test_idx = partition_indices(indices,train_ratio,val_ratio)
    
    train_x,  train_y = subset_xy(X,y,train_idx)
    val_x, val_y = subset_xy(X,y,val_idx)
    test_x, test_y = subset_xy(X,y,test_idx)

    return {
        'X_train': train_x,
        'y_train':train_y,

        'X_val':val_x,
        'y_val':val_y,

        'X_test': test_x,      
        'y_test':test_y
        
    }

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    mean, std = fit_standardizer(splits['X_train'])
    res = {}
    for split in ['train','val','test']:
        x_key = f'X_{split}'
        y_key = f'y_{split}'
        x_scaled = apply_standardizer(splits[x_key],mean,std)
        res[x_key] = add_bias_column(x_scaled)
        res[y_key] = splits[y_key]
    
    return res,mean,std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    return {
        'mae':mean_absolute_error(y_true,y_pred),
        'rmse':root_mean_squared_error(y_true,y_pred),
        'r2':r_squared(y_true,y_pred),
        'residual_summary':residual_summary(y_true,y_pred)
    }

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)
    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels)

    splits = make_train_val_test(X_feat,y,train_ratio,val_ratio,seed)
    std_splits, _,_ = standardize_and_add_bias(splits)

    theta = ols_fit(std_splits['X_train'], std_splits['y_train'])
    y_val_pred = ols_predict(std_splits['X_val'],theta)
    y_test_pred = ols_predict(std_splits['X_test'],theta)

    val_metrics = evaluate_predictions(std_splits['y_val'],y_val_pred)
    test_metrics = evaluate_predictions(std_splits['y_test'], y_test_pred)

    return {
        'theta':theta,
        'y_test':std_splits['y_test'],
        'y_test_pred':y_test_pred,
        'test_metrics':test_metrics,
        'val_metrics':val_metrics
    }

