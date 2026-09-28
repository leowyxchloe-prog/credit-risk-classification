import joblib
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_predict
)
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix ) 
from preprocessing import (
    load_data,
    clean_data,
    create_target,
    build_preprocessor,
    create_features
)
from evaluate import evaluate_model
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

import shap
DATA_PATH = "../data/credit_risk_cleaned.csv"


def main():

    # -------------------------
    # 1. Load data
    # -------------------------

    df = load_data(DATA_PATH)

    # -------------------------
    # 2. Clean data
    # -------------------------

    df = clean_data(df)

    # -------------------------
    # 3. Feature engineering
    # -------------------------

    df = create_features(df)

    # -------------------------
    # 4. Define features
    # -------------------------

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

    feature_cols = (numerical_features +categorical_features)

    X = df[feature_cols]
    y = create_target(df)

    # -------------------------
    # 5. Train/test split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


    # -------------------------
    # 7. Define 5-fold CV
    # -------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    # -------------------------
    # 8. Cross-validation
    # -------------------------
    pipeline = Pipeline([
    (
        "preprocessor",
        build_preprocessor(
            numerical_features,
            categorical_features
        )
    ),
    (
        "model",
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs= 1
        )
    )
])
    param_distributions = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5],
    "model__max_features": ["sqrt", "log2"]
}
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest CV PR-AUC:")
    print(search.best_score_)

    best_pipeline = search.best_estimator_
   # -------------------------
   # 10. Threshold tuning
   # -------------------------

    oof_prob = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)

    threshold_results = []

    for threshold in thresholds:

        y_pred = (oof_prob >= threshold).astype(int)

        precision = precision_score(
            y_train,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_train,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_train,
            y_pred,
            zero_division=0
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    for result in threshold_results:
        print(result)

    best_threshold = max(
    threshold_results,
    key=lambda x: x["f1"]
    )

    print("\nBest threshold:")
    print(best_threshold)

    # -------------------------
    # 11. Final test evaluation
    # -------------------------

    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    threshold = best_threshold["threshold"]

    y_test_pred = (
        y_test_prob >= threshold
    ).astype(int)

    evaluate_model(
        y_true=y_test,
        y_prob=y_test_prob,
        threshold=threshold
    )

    # Calculate final test metrics

    accuracy = accuracy_score(
        y_test,
        y_test_pred
    )

    precision = precision_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_test_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_test_prob
    )

    gini = 2 * roc_auc - 1

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=[0, 1]
    )

    print(f"\nROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")
    print(f"Gini:    {gini:.3f}")

    print("\nConfusion Matrix:")
    print(cm)
    # -------------------------
    # 12. Save best model
    # -------------------------

    model_artifact = {
    "model": best_pipeline,

    "threshold": float(
        best_threshold["threshold"]
    ),

    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "gini": float(gini)
    },

    "confusion_matrix": cm.tolist()
    }

    joblib.dump(
        model_artifact,
        "../models/credit_risk_model.pkl"
    )


if __name__ == "__main__":
    main()