import pandas as pd
import numpy as np


def make_predictions(model, X_test, y_test, risk_thresholds=None, custom_labels=None, classification_threshold=None):

    # Set defaults
    if risk_thresholds is None:
        risk_thresholds = [0.33, 0.66]
    if custom_labels is None:
        custom_labels = ["Low", "Medium", "High"]

    # Generate predictions
    predicted_probs = model.predict_proba(X_test)[:, 1]
    non_default_probs = model.predict_proba(X_test)[:, 0]
    
    # Apply classification threshold if provided, otherwise use model's default
    if classification_threshold is not None:
        predicted_labels = (predicted_probs >= classification_threshold).astype(int)
    else:
        predicted_labels = model.predict(X_test)

    # Start with features
    results = X_test.copy()
    results["actual"] = y_test.values

    # Core predictions
    results["predicted"] = predicted_labels
    results["risk_probability"] = predicted_probs
    results["non_default_probability"] = non_default_probs

    # Risk categorization
    bins = [-0.01] + risk_thresholds + [1.01]
    results["risk_level"] = pd.cut(
        results["risk_probability"],
        bins=bins,
        labels=custom_labels,
    )

    # Prediction confidence and margin
    results["prediction_confidence"] = np.maximum(predicted_probs, non_default_probs)
    results["prediction_margin"] = np.abs(predicted_probs - non_default_probs)

    # Uncertainty metric (how close to decision boundary)
    results["uncertainty"] = 1 - results["prediction_margin"]

    # Correctness tracking
    results["is_correct"] = (results["predicted"] == results["actual"]).astype(int)
    results["is_false_positive"] = (
        (results["predicted"] == 1) & (results["actual"] == 0)
    ).astype(int)
    results["is_false_negative"] = (
        (results["predicted"] == 0) & (results["actual"] == 1)
    ).astype(int)
    results["is_true_positive"] = (
        (results["predicted"] == 1) & (results["actual"] == 1)
    ).astype(int)
    results["is_true_negative"] = (
        (results["predicted"] == 0) & (results["actual"] == 0)
    ).astype(int)

    # High risk wrong predictions
    results["high_confidence_wrong"] = (
        ((results["prediction_confidence"] > 0.8) & (results["is_correct"] == 0)).astype(
            int
        )
    )

    # Low confidence correct predictions
    results["low_confidence_correct"] = (
        ((results["prediction_confidence"] < 0.6) & (results["is_correct"] == 1)).astype(
            int
        )
    )

    # Risk ranking
    results["risk_rank"] = results["risk_probability"].rank(ascending=False)
    results["risk_percentile"] = (
        results["risk_probability"].rank(pct=True) * 100
    ).round(2)

    # Decision boundary flags
    results["near_decision_boundary"] = (results["prediction_margin"] < 0.15).astype(int)

    # Prediction type categorization
    results["prediction_type"] = "Unknown"
    results.loc[results["is_true_positive"] == 1, "prediction_type"] = "True Positive"
    results.loc[results["is_true_negative"] == 1, "prediction_type"] = "True Negative"
    results.loc[results["is_false_positive"] == 1, "prediction_type"] = "False Positive"
    results.loc[results["is_false_negative"] == 1, "prediction_type"] = "False Negative"

    return results


def get_prediction_summary(predictions_df):

    total_predictions = len(predictions_df)
    correct_predictions = predictions_df["is_correct"].sum()
    accuracy = (correct_predictions / total_predictions) * 100

    summary = {
        "Total Predictions": total_predictions,
        "Correct Predictions": correct_predictions,
        "Accuracy": f"{accuracy:.2f}%",
        "True Positives": predictions_df["is_true_positive"].sum(),
        "True Negatives": predictions_df["is_true_negative"].sum(),
        "False Positives": predictions_df["is_false_positive"].sum(),
        "False Negatives": predictions_df["is_false_negative"].sum(),
        "High Confidence Errors": predictions_df["high_confidence_wrong"].sum(),
        "Default Rate (Actual)": f"{(predictions_df['actual'].sum() / total_predictions) * 100:.2f}%",
        "Default Rate (Predicted)": f"{(predictions_df['predicted'].sum() / total_predictions) * 100:.2f}%",
        "Average Risk Probability": f"{predictions_df['risk_probability'].mean():.4f}",
        "Risk Level Distribution": predictions_df["risk_level"].value_counts().to_dict(),
    }

    return summary


def get_high_risk_borrowers(predictions_df, percentile=95, limit=None):
    threshold = predictions_df["risk_probability"].quantile(percentile / 100)
    high_risk = predictions_df[
        predictions_df["risk_probability"] >= threshold
    ].copy()
    high_risk = high_risk.sort_values("risk_probability", ascending=False)

    if limit is not None:
        high_risk = high_risk.head(limit)

    return high_risk[
        [
            "risk_probability",
            "risk_level",
            "prediction_confidence",
            "actual",
            "predicted",
            "is_correct",
            "risk_rank",
        ]
    ]


def get_misclassified_borrowers(predictions_df, sort_by="confidence"):
    misclassified = predictions_df[predictions_df["is_correct"] == 0].copy()

    if sort_by == "confidence":
        misclassified = misclassified.sort_values(
            "prediction_confidence", ascending=False
        )
    elif sort_by == "margin":
        misclassified = misclassified.sort_values("prediction_margin", ascending=False)
    else:
        misclassified = misclassified.sort_values(
            "risk_probability", ascending=False
        )

    return misclassified[
        [
            "actual",
            "predicted",
            "risk_probability",
            "prediction_confidence",
            "prediction_margin",
            "prediction_type",
        ]
    ]


def get_predictions_by_risk_level(predictions_df, risk_level):
    return predictions_df[predictions_df["risk_level"] == risk_level].copy()