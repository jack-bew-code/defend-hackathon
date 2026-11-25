"""Shared feature engineering. Add functions here rather than copying code between notebooks."""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example: df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
    return df
