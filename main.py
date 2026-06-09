import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from src.loader import load_data
from src.schema_inference import rename_columns
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.train import train_model
from src.predict import (
    make_predictions,
    get_prediction_summary,
    get_high_risk_borrowers,
    get_misclassified_borrowers,
)
from src.evaluate import evaluate_model, print_evaluation_report, save_metrics_to_csv, tune_threshold, print_threshold_tuning_report
from src.report_generator import compute_summary_stats, build_llm_prompt, generate_llm_text, create_pdf_report, load_model_metrics

try:
    import shap
except ImportError:
    shap = None


def build_threshold_range(cfg):
    optimization = cfg.get("threshold_optimization", {}) if isinstance(cfg, dict) else {}
    if all(k in optimization for k in ("threshold_min", "threshold_max", "threshold_step")):
        min_th = float(optimization["threshold_min"])
        max_th = float(optimization["threshold_max"])
        step = float(optimization["threshold_step"])
        thresholds = []
        current = min_th
        while current <= max_th + 1e-9:
            thresholds.append(round(current, 2))
            current += step
        return thresholds

    return cfg.get("thresholds", [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])


def generate_shap_explanations(model, X, predictions_df, threshold, output_dir="outputs/figures"):
    shap_info = {}
    if shap is None:
        print("SHAP is not installed. Skipping SHAP explainability generation.")
        return shap_info

    os.makedirs(output_dir, exist_ok=True)

    try:
        explainer = shap.TreeExplainer(model)
        try:
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
            expected_value = explainer.expected_value
        except Exception:
            shap_output = explainer(X)
            shap_values = shap_output.values
            expected_value = shap_output.base_values
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
    except Exception as e:
        print(f"Warning: SHAP explanation generation failed ({e}).")
        return shap_info

    feature_importance_path = os.path.join(output_dir, "shap_feature_importance.png")
    summary_path = os.path.join(output_dir, "shap_summary.png")

    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(feature_importance_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feature_names = list(pd.Series(mean_abs, index=X.columns).sort_values(ascending=False).head(5).index)
    if top_feature_names:
        if len(top_feature_names) == 1:
            top_driver_text = f"The strongest risk driver is {top_feature_names[0]}."
        else:
            top_driver_text = (
                f"The strongest risk drivers are {', '.join(top_feature_names[:-1])}, and {top_feature_names[-1]}."
            )
    else:
        top_driver_text = "SHAP identifies the leading model risk drivers based on feature impact."

    shap_info.update(
        {
            "feature_importance_path": feature_importance_path,
            "summary_path": summary_path,
            "top_driver_text": top_driver_text,
            "shap_note": (
                "SHAP values explain how each borrower feature contributes to the model’s predicted risk score. "
                "Positive SHAP values push the prediction toward higher default risk, while negative SHAP values push the prediction toward lower default risk."
            ),
            "error_analysis_text": (
                "SHAP helps explain one false approval and one false rejection. False approvals create expected default loss by approving a borrower who later defaults, "
                "while false rejections create lost profit opportunity by denying a creditworthy borrower."
            ),
        }
    )

    local_examples = []
    cases = [
        ("correct_reject", (predictions_df["actual"] == 1) & (predictions_df["predicted"] == 1), "Correctly rejected high-risk borrower"),
        ("false_approval", (predictions_df["actual"] == 1) & (predictions_df["predicted"] == 0), "False approval: predicted safe but actual default"),
        ("false_rejection", (predictions_df["actual"] == 0) & (predictions_df["predicted"] == 1), "False rejection: predicted risky but actual repayment"),
    ]

    selected_rows = {}
    for key, cond, label in cases:
        subset = predictions_df[cond].copy()
        if not subset.empty:
            selected_rows[key] = subset.sort_values("prediction_confidence", ascending=False).iloc[0]

    closest = predictions_df.iloc[(predictions_df["risk_probability"] - threshold).abs().argsort()[:1]]
    if not closest.empty:
        selected_rows["borderline"] = closest.iloc[0]

    main_plot_key = "false_approval" if "false_approval" in selected_rows else next(iter(selected_rows), None)
    local_main_plot = None
    appendix_local_plots = []

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    for key, row in selected_rows.items():
        try:
            row_index = X.index.get_loc(row.name)
        except KeyError:
            row_index = None

        if row_index is None:
            continue

        local_data = X.iloc[row_index:row_index + 1]
        local_values = shap_values[row_index]
        explanation = shap.Explanation(
            values=local_values,
            base_values=expected_value,
            data=local_data.values[0],
            feature_names=list(X.columns),
        )

        plot_path = os.path.join(output_dir, f"shap_{key}.png")
        plt.figure(figsize=(7, 5))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"SHAP local explanation: {key.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        if key == main_plot_key:
            local_main_plot = plot_path
        appendix_local_plots.append((f"SHAP local explanation: {row.name} ({key.replace('_', ' ').title()})", plot_path))

    shap_info["local_main_plot"] = local_main_plot
    shap_info["appendix_local_plots"] = [p for p in appendix_local_plots if p[1] != local_main_plot]

    return shap_info


def main():
    #   Data preprocessing  
    os.makedirs("outputs", exist_ok=True)

    file_path = "data/credit_risk_dataset.csv"
    df = load_data(file_path)

    print("\nRaw data loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    df = rename_columns(df)

    print("\nColumns after renaming:")
    print(list(df.columns))

    df = clean_data(df)

    print("\nData cleaned successfully.")
    print("Shape after cleaning:", df.shape)
    #   Feature engineering
    df = engineer_features(df)

    print("\nFeature engineering complete.")
    print("Final shape:", df.shape)

    print("\nPreview of final processed data:")
    print(df.head())
    
    #   Model training + predictions 
    processed_output_path = "outputs/processed_credit_risk.csv"
    df.to_csv(processed_output_path, index=False)
    print(f"\nProcessed file saved to {processed_output_path}")

    models, X_train, X_test, y_train, y_test = train_model(df)

    print("\nModels trained successfully.")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

    # Threshold tuning for each model
    print(f"\n\n{'='*80}")
    print("PHASE 1: THRESHOLD TUNING FOR ALL MODELS")
    print('='*80)
    
    threshold_tuning_results = {}
    
    # Load business config
    cfg = {}
    if os.path.exists("config.yaml"):
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    thresholds_cfg = build_threshold_range(cfg)
    threshold_optimization_cfg = cfg.get("threshold_optimization", {})
    lgd_cfg = cfg.get("loss_given_default", 0.6)
    profit_margin_cfg = cfg.get("profit_margin", 0.05)
    default_loan_amount_cfg = cfg.get("default_loan_amount", 10000)

    optimized_thresholds = {}
    for model_name, model in models.items():
        threshold_df, optimal_result = tune_threshold(
            model,
            X_test,
            y_test,
            thresholds=thresholds_cfg,
            loss_given_default=lgd_cfg,
            profit_margin=profit_margin_cfg,
            default_loan_amount=default_loan_amount_cfg,
            optimization_config=threshold_optimization_cfg,
        )
        threshold_tuning_results[model_name] = threshold_df
        optimized_thresholds[model_name] = optimal_result

        best_threshold = optimal_result["threshold"]
        print_threshold_tuning_report(model_name, threshold_df, best_threshold)
        if optimal_result["warning"]:
            print(f"WARNING for {model_name}: {optimal_result['warning']}")
        
        # Save threshold tuning results
        threshold_output_path = f"outputs/threshold_tuning_{model_name}.csv"
        threshold_df.to_csv(threshold_output_path, index=False)
        print(f"Threshold tuning saved to {threshold_output_path}")
    
    # Display optimized threshold summary for each model
    print(f"\n{'='*80}")
    print("OPTIMIZED THRESHOLDS FOR EACH MODEL (Expected Loss and Constraints)")
    print('='*80)
    for model_name, result in optimized_thresholds.items():
        best_th = result['threshold']
        threshold_row = threshold_tuning_results[model_name].loc[
            threshold_tuning_results[model_name]['Threshold'] == best_th
        ].iloc[0]
        print(
            f"{model_name.upper().replace('_', ' ')}: Threshold = {best_th:.2f} "
            f"→ Expected Loss = ${threshold_row['Expected_Loss']:.2f}, "
            f"Status = {result['constraint_status']}"
        )
        if result['warning']:
            print(f"WARNING for {model_name}: {result['warning']}")
    print('='*80)

    # Phase 2: Make predictions with tuned thresholds
    print(f"\n\n{'='*80}")
    print("PHASE 2: PREDICTIONS WITH TUNED THRESHOLDS")
    print('='*80)

    predictions_results = {}

    # Make predictions with both models
    for model_name, model in models.items():
        optimal_threshold = optimized_thresholds[model_name]['threshold']
        print(f"\n{'='*60}")
        print(f"PREDICTIONS WITH {model_name.upper().replace('_', ' ')} (Threshold: {optimal_threshold:.2f})")
        print('='*60)
        
        predictions_df = make_predictions(model, X_test, y_test, classification_threshold=optimal_threshold)

        print(f"Predictions generated successfully for {model_name}.")
        print("Predictions DataFrame shape:", predictions_df.shape)
        print(f"\nSample predictions with enhanced metrics for {model_name}:")

        sample_predictions = pd.concat([
            predictions_df.sort_values("risk_probability", ascending=False).head(3),
            predictions_df.sort_values("risk_probability", ascending=True).head(3),
            predictions_df.sample(n=min(4, len(predictions_df)), random_state=42),
        ]).drop_duplicates().head(10)

        print(
            sample_predictions[
                [
                    "actual",
                    "predicted",
                    "risk_probability",
                    "prediction_confidence",
                    "prediction_margin",
                    "risk_level",
                    "prediction_type",
                ]
            ]
        )

        predictions_output_path = f"outputs/predictions_{model_name}.csv"
        predictions_df.to_csv(predictions_output_path, index=False)
        print(f"\nFull predictions saved to {predictions_output_path}")

        predictions_results[model_name] = predictions_df

        # Get and display prediction summary
        summary = get_prediction_summary(predictions_df)
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY - {model_name.upper().replace('_', ' ')}")
        print('='*60)
        for key, value in summary.items():
            print(f"{key}: {value}")

        # Get high-risk borrowers
        print(f"\n{'='*60}")
        print(f"TOP 10 HIGHEST RISK BORROWERS - {model_name.upper().replace('_', ' ')} (95th+ percentile)")
        print('='*60)
        high_risk = get_high_risk_borrowers(predictions_df, percentile=95, limit=10)
        print(high_risk)

        # Get misclassified borrowers sorted by confidence
        misclassified = get_misclassified_borrowers(predictions_df, sort_by="confidence")
        print(f"\n{'='*60}")
        print(f"MISCLASSIFIED BORROWERS - {model_name.upper().replace('_', ' ')} ({len(misclassified)} total)")
        print("Sorted by prediction confidence")
        print('='*60)
        print(misclassified.head(10))

        # Evaluate model performance
        metrics = evaluate_model(model, X_test, y_test)
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION REPORT - {model_name.upper().replace('_', ' ')}")
        print('='*60)
        print_evaluation_report(metrics)

        # Save metrics to CSV
        metrics_output_path = f"outputs/model_metrics_{model_name}.csv"
        save_metrics_to_csv(metrics, metrics_output_path)
        print(f"Metrics saved to {metrics_output_path}")

    # Generate comprehensive report using the final selected model based on optimized thresholds
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE CREDIT RISK REPORT")
    print("="*60)

    # Select the final report model using expected loss and business constraints.
    report_file_key = None
    report_model_name = None
    selection_basis = "expected_loss_constraints"
    fallback_selection = False

    met_models = [
        model_name for model_name, result in optimized_thresholds.items()
        if result['constraint_status'] == 'Met constraints'
    ]

    if met_models:
        report_file_key = min(
            met_models,
            key=lambda name: optimized_thresholds[name]['expected_loss']
        )
    else:
        report_file_key = min(
            optimized_thresholds.keys(),
            key=lambda name: optimized_thresholds[name]['expected_loss']
        )
        selection_basis = "expected_loss_fallback"
        fallback_selection = True

    report_model_name = report_file_key.replace('_', ' ').title()
    print(f"Selected report model: {report_file_key} (selection basis: {selection_basis})")
    if fallback_selection:
        print("WARNING: No model met all configured business constraints; selecting the fallback model with lowest expected loss.")
    best_model_predictions = predictions_results.get(report_file_key)
    if best_model_predictions is None:
        best_model_predictions = pd.read_csv(f"outputs/predictions_{report_file_key}.csv")
    best_model_metrics_raw = pd.read_csv(f"outputs/model_metrics_{report_file_key}.csv")

    # Convert metrics to dictionary
    best_model_metrics = {}
    for _, row in best_model_metrics_raw.iterrows():
        best_model_metrics[row['Metric'].lower().replace('-', '_')] = row['Value']

    shap_info = {}
    if report_file_key in models:
        selected_threshold = optimized_thresholds[report_file_key]['threshold']
        shap_info = generate_shap_explanations(models[report_file_key], X_test, best_model_predictions, selected_threshold)
    
    model_metrics = load_model_metrics()

    # Build the optimized model summary for final comparison and report use.
    optimized_model_summary = []
    for model_name, result in optimized_thresholds.items():
        threshold_row = threshold_tuning_results[model_name].loc[
            threshold_tuning_results[model_name]['Threshold'] == result['threshold']
        ].iloc[0]
        optimized_model_summary.append({
            "Model": model_name.replace('_', ' ').title(),
            "Optimal_Threshold": threshold_row['Threshold'],
            "Accuracy": threshold_row['Accuracy'],
            "Precision": threshold_row['Precision'],
            "Recall": threshold_row['Recall'],
            "F1 Score": threshold_row['F1-Score'],
            "ROC-AUC": threshold_row['ROC-AUC'],
            "Approval Rate": threshold_row['Approval_Rate'],
            "False Approvals": threshold_row['False_Approvals'],
            "False Rejections": threshold_row['False_Rejections'],
            "Expected Loss": threshold_row['Expected_Loss'],
            "Constraint Status": result['constraint_status'],
        })

    optimized_comparison_df = pd.DataFrame(optimized_model_summary)
    optimized_comparison_df.to_csv("outputs/optimized_threshold_comparison.csv", index=False)

    # Create model comparison string for LLM
    model_comparison_text = "Model Performance Comparison:\n"
    for model_name, metrics_dict in model_metrics.items():
        model_comparison_text += f"- {model_name}: Accuracy={float(metrics_dict.get('Accuracy', 0)):.4f}, Precision={float(metrics_dict.get('Precision', 0)):.4f}, Recall={float(metrics_dict.get('Recall', 0)):.4f}, ROC-AUC={float(metrics_dict.get('ROC-AUC', 0)):.4f}\n"

    summary_stats = compute_summary_stats(best_model_predictions)

    # Build LLM prompt with model comparison and selection context
    llm_prompt = build_llm_prompt(
        summary_stats,
        None,
        None,
        model_comparison_text,
        selection_basis=selection_basis,
        report_model_name=report_model_name,
    )

    # Generate AI-powered report text
    llm_report_text = generate_llm_text(llm_prompt)

    # Create PDF report with threshold tuning data for all models
    final_selection_info = {
        "report_model_name": report_model_name,
        "selection_basis": selection_basis,
        "fallback_warning": fallback_selection,
        "fallback_message": (
            "No model met all configured business constraints, so the final selection is the fallback model with the lowest expected loss."
            if fallback_selection else None
        )
    }

    pdf_path = create_pdf_report(
        summary_stats,
        llm_report_text,
        df,
        best_model_predictions,
        threshold_data=threshold_tuning_results,
        optimized_summary=optimized_model_summary,
        config=cfg,
        selection_basis=selection_basis,
        report_model_name=report_model_name,
        model_metrics=model_metrics,
        final_selection_info=final_selection_info,
        shap_info=shap_info,
    )
    print(f"Comprehensive credit risk report generated: {pdf_path}")


if __name__ == "__main__":
    main()