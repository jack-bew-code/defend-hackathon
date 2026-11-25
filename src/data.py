"""Shared data loading. Use this from any notebook so file paths always work.

    import sys; sys.path.append('..')   # only needed when running from a subfolder
    from src.data import load_train, load_test
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_train() -> pd.DataFrame:
    return pd.read_csv(ROOT / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(ROOT / "test.csv")
