import joblib
import pandas as pd

from preprocessing import (clean_data, create_features)

def predict_risk(
    borrower_data,
    threshold=0.30
):
    # Load trained objects
    model = joblib.load("../models/best_model.pkl")

    preprocessor = joblib.load("../models/preprocessor.pkl")

    # Convert input to DataFrame
    df = pd.DataFrame([borrower_data])

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = create_features(df)

    # Select features
    numerical_features = [
        "INVESTMENT_TOTAL",
        "ACCCURRENTBALANCE",
        "INSTALL_SIZE",
        "DUE_PAYMENT",
        "HAS_DUE_PAYMENT",
        "HAS_INSTALLMENT",
        "BALANCE_TO_INVESTMENT",
        "LOG_BALANCE_TO_INVESTMENT",
        "INSTALLMENT_TO_INVESTMENT",
        "LOG_INSTALLMENT_TO_INVESTMENT"
    ]

    categorical_features = [
        "INF_MARITAL_STATUS",
        "INF_GENDER",
        "CLIENT_TYPE"
    ]

    feature_cols = (numerical_features + categorical_features)

    X = df[feature_cols]

    # Apply trained preprocessing
    X_processed = preprocessor.transform(X)

    # Probability of Bad
    probability_bad = model.predict_proba(X_processed)[0, 1]

    # Classification
    prediction = int(probability_bad >= threshold)

    return {
        "probability_bad": probability_bad,
        "prediction": prediction
    }