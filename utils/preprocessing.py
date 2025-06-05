import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

def robust_preprocess_data(df: pd.DataFrame, feature_cols=None):
    """
    Applies robust scaling to the feature columns of the dataframe.
    Args:
        df: DataFrame with OHLCV and features
        feature_cols: List of columns to scale (if None, all except timestamp/index)
    Returns:
        DataFrame with scaled features
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'date', 'index']]
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
    return df_scaled
