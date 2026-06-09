from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc,
)
import pandas as pd


def evaluate_model(model, X_test, y_test, y_pred=None, y_pred_proba=None):
    # Generate predictions if not provided
    if y_pred is None:
        y_pred = model.predict(X_test)
    if y_pred_proba is None:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Specificity and sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall  # Same as recall

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "classification_report": class_report,
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
    }

    return metrics

def print_evaluation_report(metrics):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    print(f"\nAccuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"Sensitivity:     {metrics['sensitivity']:.4f}")
    print(f"Specificity:     {metrics['specificity']:.4f}")
    print(f"F1-Score:        {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.4f}")

    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"True Negatives:  {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Positives:  {metrics['true_positives']}")

    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)

    # Create classification report
    y_true = [0] * (metrics['true_negatives'] + metrics['false_negatives']) + [
        1
    ] * (metrics['false_positives'] + metrics['true_positives'])
    y_pred = [0] * metrics['true_negatives'] + [1] * metrics['false_positives'] + [
        0
    ] * metrics['false_negatives'] + [1] * metrics['true_positives']

    print(classification_report(y_true, y_pred))
    print("=" * 60)


def save_metrics_to_csv(metrics, output_path):
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "Sensitivity",
                "Specificity",
                "F1-Score",
                "ROC-AUC",
                "True Negatives",
                "False Positives",
                "False Negatives",
                "True Positives",
            ],
            "Value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["sensitivity"],
                metrics["specificity"],
                metrics["f1_score"],
                metrics["roc_auc"],
                metrics["true_negatives"],
                metrics["false_positives"],
                metrics["false_negatives"],
                metrics["true_positives"],
            ],
        }
    )

    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def optimize_thresholds(threshold_df, optimization_config=None):
    """Pick the optimal threshold using expected loss and business constraints."""
    if optimization_config is None:
        optimization_config = {}

    min_approval_rate = float(optimization_config.get("constraints", {}).get("min_approval_rate", 0.80)) * 100
    min_precision = float(optimization_config.get("constraints", {}).get("min_precision", 0.90))
    min_recall = float(optimization_config.get("constraints", {}).get("min_recall", 0.70))

    df = threshold_df.copy()
    df["Meets_Constraints"] = (
        (df["Approval_Rate"] >= min_approval_rate)
        & (df["Precision"] >= min_precision)
        & (df["Recall"] >= min_recall)
    )

    if df["Meets_Constraints"].any():
        candidate_df = df[df["Meets_Constraints"]]
        best_row = candidate_df.loc[candidate_df["Expected_Loss"].idxmin()]
        status = "Met constraints"
        warning = None
    else:
        best_row = df.loc[df["Expected_Loss"].idxmin()]
        status = "Fallback used"
        warning = (
            "No threshold met all configured business constraints, so the fallback threshold minimizes expected loss without constraints."
        )

    return best_row.to_dict(), status, warning


def tune_threshold(model, X_test, y_test, thresholds=[0.30, 0.40, 0.50, 0.60], loss_given_default=0.6, profit_margin=0.05, default_loan_amount=10000, optimization_config=None):
    """
    Test multiple classification thresholds and find the best operating threshold based
    on expected loss and business constraints.
    Returns threshold results and the optimal threshold summary.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold_results = []

    # ROC-AUC is threshold-independent but included for every row per user request.
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        roc_auc = 0.5
    
    loan_col = None
    if hasattr(X_test, 'columns'):
        for c in X_test.columns:
            if 'loan' in c.lower() or 'amount' in c.lower():
                loan_col = c
                break

    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Finance metrics
        approved = (y_pred == 0).sum()
        rejected = (y_pred == 1).sum()
        approval_rate = (approved / len(y_test) * 100) if len(y_test) > 0 else 0
        false_approvals = fn
        false_approvals_pct = (fn / approved * 100) if approved > 0 else 0
        false_rejections = fp
        false_rejections_pct = (fp / rejected * 100) if rejected > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Estimate expected financial loss at this threshold.
        lgd = loss_given_default
        pm = profit_margin

        if loan_col is not None:
            loan_values = X_test[loan_col].astype(float).values
            fa_mask = (y_pred == 0) & (y_test.values == 1)
            fr_mask = (y_pred == 1) & (y_test.values == 0)
            sum_false_approvals_amount = loan_values[fa_mask].sum() if fa_mask.any() else 0.0
            sum_false_rejections_amount = loan_values[fr_mask].sum() if fr_mask.any() else 0.0
            expected_loss = (sum_false_approvals_amount * lgd) + (sum_false_rejections_amount * pm)
        else:
            cost_false_approval = default_loan_amount * lgd
            cost_false_rejection = default_loan_amount * pm
            expected_loss = (false_approvals * cost_false_approval) + (false_rejections * cost_false_rejection)
            sum_false_approvals_amount = false_approvals * default_loan_amount
            sum_false_rejections_amount = false_rejections * default_loan_amount

        threshold_results.append({
            "Threshold": round(threshold, 2),
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "Specificity": round(specificity, 4),
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Approved": int(approved),
            "Rejected": int(rejected),
            "Approval_Rate": round(approval_rate, 2),
            "False_Approvals": int(false_approvals),
            "False_Approvals_Pct": round(false_approvals_pct, 2),
            "False_Rejections": int(false_rejections),
            "False_Rejections_Pct": round(false_rejections_pct, 2),
            "False_Approval_Amount": round(float(sum_false_approvals_amount), 2),
            "False_Rejection_Amount": round(float(sum_false_rejections_amount), 2),
            "Expected_Loss": round(float(expected_loss), 2),
        })

    threshold_df = pd.DataFrame(threshold_results)
    optimal_row, status, warning = optimize_thresholds(threshold_df, optimization_config)
    optimal_threshold = optimal_row["Threshold"]
    optimal_expected_loss = optimal_row["Expected_Loss"]

    return threshold_df, {
        "threshold": optimal_threshold,
        "expected_loss": optimal_expected_loss,
        "constraint_status": status,
        "warning": warning,
    }


def print_threshold_tuning_report(model_name, threshold_df, best_threshold):
    """Print threshold tuning results for a model."""
    print(f"\n{'='*80}")
    print(f"THRESHOLD TUNING RESULTS - {model_name.upper().replace('_', ' ')}")
    print('='*80)
    print(threshold_df.to_string(index=False))
    print(f"\n✓ Best Threshold: {best_threshold:.2f} (Highest Accuracy: {threshold_df.loc[threshold_df['Threshold'] == best_threshold, 'Accuracy'].values[0]:.4f})")
    print('='*80)