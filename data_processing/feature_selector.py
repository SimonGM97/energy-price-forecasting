import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, 
    f_regression, 
    mutual_info_regression
)
import tsfresh as tsf
from typing import List


# Target-Feature Correlation Based Filtering
def target_feature_correl_filter(
    y: pd.Series,
    X: pd.DataFrame,
    q_thresh: float = 0.3
) -> List[str]:
    # Prepare DataFrames
    intersection = y.index.intersection(X.index)
    num_cols = list(X.select_dtypes(include=['number']).columns)

    y = y.loc[intersection]
    X = X.loc[intersection, num_cols]

    # Calculate Correlations with target
    tf_corr_df: pd.DataFrame = pd.DataFrame(columns=[y.name])
    for c in X.columns:
        tf_corr_df.loc[c] = [abs(y.corr(X[c]))]
    
    tf_corr_df = tf_corr_df.sort_values(by=[y.name], ascending=False)

    # Define threshold
    threshold = np.quantile(tf_corr_df[y.name].dropna(), q_thresh)
    
    return tf_corr_df.loc[tf_corr_df[y.name] > threshold].index.tolist()


def colinear_feature_filter(
    X: pd.DataFrame,
    thresh: float = 0.9
) -> List[str]:
    cm = pd.DataFrame(X.corr().applymap(lambda x: 100 * np.abs(x))).fillna(100)
    filtered_features = cm.columns.tolist()
    
    i = 0
    while i < len(filtered_features):
        keep_feature = filtered_features[i]
        skip_features = cm.loc[
            (cm[keep_feature] < 100) &
            (cm[keep_feature] >= thresh*100)
        ][keep_feature].index.tolist()
        
        if len(skip_features) > 0:
            filtered_features = [c for c in filtered_features if c not in skip_features]
        i += 1
    
    return filtered_features


def select_k_best_features(
    y: pd.Series,
    X: pd.DataFrame,
    perc: float = 0.1
) -> List[str]:
    # Prepare DataFrames
    intersection = y.index.intersection(X.index)
    
    binary_features = [col for col in X.columns if X[col].nunique() == 2]
    non_binary_features = [col for col in X.columns if X[col].nunique() > 2]

    y = y.loc[intersection]
    X_binary = X.loc[intersection, binary_features]
    X_non_binary = X.loc[intersection, non_binary_features]
    
    # Prepare selected_featuers
    selected_featuers = []
    
    # Select Binary Features
    if X_binary.shape[1] > 0:
        k = int(perc * len(binary_features))
        
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X_binary, y)

        selected_featuers.extend(X_binary.columns[selector.get_support()].tolist())
    
    # Select Non-Binary Features
    if X_non_binary.shape[1] > 0:
        k = int(perc * len(non_binary_features))
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_non_binary, y)

        selected_featuers.extend(X_non_binary.columns[selector.get_support()].tolist())
        
    return selected_featuers


def select_tsfresh_features(
    y: pd.Series,
    X: pd.DataFrame,
    p_value: float = 0.01,
    top_k: int = None
) -> List[str]:
    # Prepare DataFrames
    intersection = y.index.intersection(X.index)

    y = y.loc[intersection]
    X = X.loc[intersection]
    
    # Run TSFresh Feature Selection
    relevance_table = tsf.feature_selection.relevance.calculate_relevance_table(X, y)
    relevance_table = relevance_table[relevance_table.relevant].sort_values("p_value")
    relevance_table = relevance_table.loc[relevance_table['p_value'] < p_value]
    
    selected_featuers = list(relevance_table["feature"].values)
    
    # Extract Features
    if top_k is not None:
        return selected_featuers[:top_k]
    
    return selected_featuers


def select_features(
    y: pd.Series,
    X_stand: pd.DataFrame
) -> pd.DataFrame:
    # Run target-feature correlation filter
    correl_filter: List[str] = target_feature_correl_filter(
        y=y.copy(),
        X=X_stand.copy(),
        q_thresh=0.3
    )

    # Filter X_stand
    X_stand = X_stand.loc[:, correl_filter]

    # Run colinear features filter
    colinear_filter: List[str] = colinear_feature_filter(
        X_stand.copy(), 
        thresh=0.9
    )

    # Filter X_stand
    X_stand = X_stand.loc[:, colinear_filter]

    # Run select k best features
    k_best_features: List[str] = select_k_best_features(
        y.copy(),
        X_stand.copy(),
        perc=0.2
    )

    # Run TSFresh features
    tsfresh_features: List[str] = select_tsfresh_features(
        y.copy(),
        X_stand.copy(),
        p_value=0.01,
        top_k=50
    )

    # Merge & order Features
    selected_features: List[str] = list(set(k_best_features + tsfresh_features))
    selected_features = [f for f in X_stand.columns if f in selected_features]

    # Filter X_stand
    X_stand = X_stand.loc[:, selected_features]

    return X_stand

