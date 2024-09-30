import pandas as pd
from sklearn.preprocessing import StandardScaler


"""
- Since all features are numerical, OneHotEncoding will not be needed
- We will use the StandardScaler to transform the numerical featuers
"""

def standardize_X(X: pd.DataFrame) -> pd.DataFrame:
    # Define StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)

    # Standardize numerical features
    X_stand = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns,
        index=X.index
    )

    return X_stand

