
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_prob, threshold=0.5):
    """
    Evaluate a binary credit-risk model.

    Class 0 = Good
    Class 1 = Bad

    Parameters
    ----------
    y_true : array-like
        Actual labels.
    y_prob : array-like
        Predicted probabilities for class 1 (Bad).
    threshold : float
        Probability threshold for classifying a borrower as Bad.
    """

    # Convert probabilities into class predictions
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    # Calculate classification metrics
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision (Bad): {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall (Bad):    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1 (Bad):        {f1_score(y_true, y_pred, zero_division=0):.3f}")

    # Threshold-independent ranking metrics
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_prob):.3f}")

    # Confusion matrix: rows = actual, columns = predicted
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    # Detailed report for both classes
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Good", "Bad"],
            zero_division=0
        )
    )

    return y_pred
