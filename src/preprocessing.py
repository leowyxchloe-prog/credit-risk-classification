import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

TARGET = "QUALITY_OF_LOAN"
ID_COL = "ACC_NO"


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    # Treat invalid CLIENT_TYPE value as missing
    df["CLIENT_TYPE"] = (df["CLIENT_TYPE"].replace("0", pd.NA).fillna("UNKNOWN"))

    # Fill categorical missing values
    categorical_cols = ["INF_MARITAL_STATUS","INF_GENDER","COMPENSATION_CHARGED"]

    for col in categorical_cols:
        df[col] = df[col].fillna("UNKNOWN")

    # INSTALL_SIZE missingness appears structural
    df["INSTALL_SIZE"] = df["INSTALL_SIZE"].fillna(0)

    return df

def create_target(df):
    return (df[TARGET] == "B").astype(int)

def build_preprocessor(numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num",StandardScaler(),numerical_features),
            ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features)
        ]
    )
    return preprocessor

def create_features(df):
    df = df.copy()

    # Binary indicators
    df["HAS_DUE_PAYMENT"] = (df["DUE_PAYMENT"] > 0).astype(int)

    df["HAS_INSTALLMENT"] = (df["INSTALL_SIZE"] > 0).astype(int)

    # Balance / investment ratio
    df["BALANCE_TO_INVESTMENT"] = (df["ACCCURRENTBALANCE"] /df["INVESTMENT_TOTAL"])

    df["LOG_BALANCE_TO_INVESTMENT"] = np.log1p(df["BALANCE_TO_INVESTMENT"])

    # Installment / investment ratio
    df["INSTALLMENT_TO_INVESTMENT"] = (df["INSTALL_SIZE"] /df["INVESTMENT_TOTAL"])

    df["LOG_INSTALLMENT_TO_INVESTMENT"] = np.log1p(df["INSTALLMENT_TO_INVESTMENT"])

    return df