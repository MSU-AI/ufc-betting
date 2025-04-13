import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def fit_scaler(df, scaler_path):
    """Fit a new scaler on the data and save it.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to fit scaler on
    scaler_path : str
        Path to save the fitted scaler

    Returns:
    --------
    sklearn.preprocessing.StandardScaler
        The fitted scaler object
    """
    # Identify columns to scale (exclude non-numeric and target columns)
    cols_to_scale = df.select_dtypes(include=["float64", "int64"]).columns
    cols_to_scale = [col for col in cols_to_scale if col != "target"]

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(df[cols_to_scale])

    # Save the fitted scaler
    joblib.dump(scaler, scaler_path)

    return scaler


def transform_features(df, scaler):
    """Transform features using a pre-fitted scaler.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing features to transform
    scaler : sklearn.preprocessing.StandardScaler
        Pre-fitted scaler to use for transformation

    Returns:
    --------
    pandas.DataFrame
        Transformed DataFrame
    """
    # Identify columns to scale (exclude non-numeric and target columns)
    cols_to_scale = df.select_dtypes(include=["float64", "int64"]).columns
    cols_to_scale = [col for col in cols_to_scale if col != "target"]

    # Transform the data
    df_scaled = df.copy()
    # TODO: transform() can cause shape issues because it returns a np array
    # df_scaled[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df_scaled.loc[:, scaler.feature_names_in_] = scaler.transform(
        df_scaled[scaler.feature_names_in_]
    )

    return df_scaled
