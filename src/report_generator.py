import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, LongTable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from openai import OpenAI
from src.summary import summarize_customer_profile
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import math

def load_model_metrics():
    """Load metrics for all trained models"""
    model_files = {
        'Logistic Regression': 'outputs/model_metrics_logistic_regression.csv',
        'Random Forest': 'outputs/model_metrics_random_forest.csv',
        'XGBoost': 'outputs/model_metrics_xgboost.csv',
        'LightGBM': 'outputs/model_metrics_lightgbm.csv'
    }

    metrics_data = {}
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            metrics = {}
            for _, row in df.iterrows():
                metrics[row['Metric']] = row['Value']
            metrics_data[model_name] = metrics

    return metrics_data


def load_threshold_tuning_data():
    """Load threshold tuning CSVs for each model."""
    model_files = {
        'Logistic Regression': 'outputs/threshold_tuning_logistic_regression.csv',
        'Random Forest': 'outputs/threshold_tuning_random_forest.csv',
        'XGBoost': 'outputs/threshold_tuning_xgboost.csv',
        'LightGBM': 'outputs/threshold_tuning_lightgbm.csv'
    }

    threshold_data = {}
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            threshold_data[model_name] = pd.read_csv(file_path)

    return threshold_data


def compute_summary_stats(predictions_df):
    total_records = len(predictions_df)
    avg_risk = predictions_df["risk_probability"].mean()

    risk_counts = predictions_df["risk_level"].value_counts().to_dict()

    summary = {
        "total_records": total_records,
        "avg_risk": round(avg_risk, 4),
        "low_count": risk_counts.get("Low", 0),
        "medium_count": risk_counts.get("Medium", 0),
        "high_count": risk_counts.get("High", 0)
    }

    return summary


def compute_threshold_metrics(predictions_df, thresholds=[0.30, 0.40, 0.50, 0.60]):
    """
    Compute metrics at different classification thresholds.
    Threshold represents the probability cutoff for predicting high risk (default).
    
    - Approval Rate: % of customers predicted as low risk (probability < threshold)
    - False Approvals (FP): % of approved customers who actually defaulted
    - False Rejections (FN): % of rejected customers who didn't default
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    results = []
    
    for threshold in thresholds:
        # Create predictions based on threshold
        predictions_at_threshold = (predictions_df["risk_probability"] >= threshold).astype(int)
        actuals = predictions_df["actual"].astype(int)
        
        # Calculate metrics
        total = len(predictions_df)
        tp = ((predictions_at_threshold == 1) & (actuals == 1)).sum()
        tn = ((predictions_at_threshold == 0) & (actuals == 0)).sum()
        fp = ((predictions_at_threshold == 1) & (actuals == 0)).sum()
        fn = ((predictions_at_threshold == 0) & (actuals == 1)).sum()
        
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Finance-specific metrics
        approved = (predictions_at_threshold == 0).sum()
        approval_rate = (approved / total * 100) if total > 0 else 0
        
        # False Approvals: approved customers who defaulted (FP count)
        false_approvals = fp
        false_approvals_pct = (fp / approved * 100) if approved > 0 else 0
        
        # False Rejections: rejected customers who didn't default (FN count)
        false_rejections = fn
        rejected = (predictions_at_threshold == 1).sum()
        false_rejections_pct = (fn / rejected * 100) if rejected > 0 else 0
        
        results.append({
            "threshold": threshold,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_approvals": false_approvals,
            "false_approvals_pct": round(false_approvals_pct, 2),
            "false_rejections": false_rejections,
            "false_rejections_pct": round(false_rejections_pct, 2),
            "approval_rate": round(approval_rate, 2),
            "approved_count": approved,
            "rejected_count": rejected
        })
    
    return pd.DataFrame(results)


def build_llm_prompt(summary_stats, metrics, portfolio_summary, model_comparison, selection_basis='default_accuracy', report_model_name=None):
    if portfolio_summary is None:
        portfolio_summary = {
            'avg_age': 'N/A', 'median_income': 'N/A', 'avg_loan_amount': 'N/A',
            'avg_debt_to_income': 'N/A', 'pct_high_loan_percent': 'N/A',
            'pct_prior_default': 'N/A', 'top_loan_intent': 'N/A', 'top_home_ownership': 'N/A'
        }
    # Build prompt with neutral model evaluation instructions. Provide the model comparison
    # table and ask the LLM to analyze and decide which model is best based on the
    # provided metrics rather than telling it which model to prefer.
    selection_note = ""
    baseline_note = ""
    if selection_basis.startswith('expected_loss') and report_model_name:
        selection_note = (
            "Note: the final model recommendation should be guided by expected financial loss after threshold tuning, "
            "not only by the default-threshold classification metrics in the comparison table."
        )
        baseline_note = (
            "Note: the default 0.50 threshold comparison is a baseline. Final model selection is based on each model's optimized threshold "
            "under the expected-loss objective and configured business constraints."
        )
    return f"""
You are writing a concise professional credit risk report for a financial analyst.

Use only the information provided below.

PREDICTION SUMMARY
- Total records analyzed: {summary_stats['total_records']}
- Average predicted risk probability: {summary_stats['avg_risk']}
- Low risk customers: {summary_stats['low_count']}
- Medium risk customers: {summary_stats['medium_count']}
- High risk customers: {summary_stats['high_count']}

MODEL EVALUATION
Use the model comparison table below to evaluate model performance. Do not assume one model is superior without referring to the metrics in the table.

{model_comparison}

{selection_note}

{baseline_note}

CUSTOMER PORTFOLIO SUMMARY
- Average age: {portfolio_summary['avg_age']}
- Median income: {portfolio_summary['median_income']}
- Average loan amount: {portfolio_summary['avg_loan_amount']}
- Average debt-to-income ratio: {portfolio_summary['avg_debt_to_income']}
- Percent of customers with high loan percent: {portfolio_summary['pct_high_loan_percent']}%
- Percent of customers with prior default history: {portfolio_summary.get('pct_prior_default', 'N/A')}%
- Most common loan intent: {portfolio_summary.get('top_loan_intent', 'N/A')}
- Most common home ownership category: {portfolio_summary.get('top_home_ownership', 'N/A')}

Write the report in this exact structure:

Executive Summary:
Write 1 short paragraph of 3 to 5 sentences explaining the overall risk profile of the dataset and the main takeaway.

Model Performance Comparison:
Analyze the model comparison table and identify the best performing model. Explain your choice in 2-3 sentences, focusing on why this model is superior based on the metrics provided.

Model Selection Rationale:
Write a single polished paragraph titled "Model Selection Rationale" using only the information provided above (prediction summary, model comparison table, and customer portfolio summary). The paragraph must be exactly one paragraph, 3–6 sentences long, professional and recruiter‑facing. When composing the paragraph: analyze the metrics and choose the most appropriate final model (do not automatically pick the model with the highest accuracy); decide which metrics matter most given the project goal and justify that choice; briefly explain what each major metric means in this lending context (one short phrase per metric); compare the strongest models and discuss concrete tradeoffs including accuracy, precision, recall, ROC‑AUC, PR‑AUC, calibration, interpretability, and expected financial loss when available; state if the recommended model choice depends on decision threshold or a specified business strategy and name the recommended operating threshold/strategy if applicable; note key drawbacks of the selected model and why the other models were not chosen; if the metrics are too close to decide, say so and recommend additional analyses (calibration plots, PR‑AUC, backtesting, cost‑sensitivity analysis, or more data). Do not invent numbers or facts, and reference only the information provided above.

Key Insights:
Write 3 bullet points highlighting important patterns in the customer portfolio and predicted risk.

Recommendations:
Write 3 numbered practical recommendations for a financial analyst or lender.

Rules:
- Be clear, professional, and non-technical.
- Do not invent numbers or facts.
- Do not mention missing information unless necessary.
- Focus on business interpretation, not coding details.
- Do not use exaggerated or dramatic language.
"""

def generate_llm_text(prompt):
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You write concise professional risk reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Warning: Could not generate AI report ({e}). Using template report instead.")
        return """
Executive Summary:
Provide a concise executive summary (3–5 sentences) describing the overall risk profile of the dataset and the main takeaway, using only the data supplied in the prompt.

Model Performance Comparison:
Analyze the provided model comparison table and identify the best-performing model(s). Explain your choice in 2–3 sentences, referencing the supplied metrics.

Key Insights:
Write 3 bullet points highlighting the most important patterns in the customer portfolio and predicted risk, based only on the provided summary and portfolio statistics.

Recommendations:
Provide 3 numbered, practical recommendations for a financial analyst or lender that follow from the analysis above.
"""


def generate_failure_group_narratives(failure_groups):
    prompt_lines = [
        "You are writing a concise business-focused credit risk report for a lender.",
        "For each group below, write 2–3 plain-English sentences explaining what kinds of borrowers the model struggles with and why that matters for lending policy.",
        "Use the actual summary statistics provided for each group. Do not explain the general concept of false approvals or false rejections unless it is directly tied to the numbers.",
        "Do not use technical jargon. Keep the answer brief and practical.",
        "",
        "Failure groups and group summaries:",
    ]
    for title, description, stats in failure_groups:
        prompt_lines.append(f"- {title}: {description}")
        prompt_lines.append(f"  Number of borrowers: {stats['count']}")
        prompt_lines.append(f"  Average loan amount: {stats['avg_loan_amount']}")
        prompt_lines.append(f"  Average income: {stats['avg_income']}")
        prompt_lines.append(f"  Average loan % of income: {stats['avg_loan_percent_income']}")
        prompt_lines.append(f"  Average predicted risk probability: {stats['avg_risk_probability']}")
        prompt_lines.append("")
    prompt_lines.append("Provide the response as three short paragraphs, one paragraph per group, starting each paragraph with the group name.")
    prompt = "\n".join(prompt_lines)
    try:
        response = generate_llm_text(prompt)
        return response.strip()
    except Exception as e:
        print(f"Warning: Could not generate failure group narratives ({e}).")
        return None

def select_sample_predictions(predictions_df, max_rows=10):
    sample_rows = pd.concat([
        predictions_df.sort_values("risk_probability", ascending=False).head(3),
        predictions_df.sort_values("risk_probability", ascending=True).head(3),
        predictions_df.sample(n=min(4, len(predictions_df)), random_state=42),
    ]).drop_duplicates().head(max_rows)
    return sample_rows


def create_pdf_report(summary_stats, llm_text, df, predictions_df, metrics=None, threshold_data=None, optimized_summary=None, config=None, selection_basis='default_accuracy', report_model_name=None, model_metrics=None, final_selection_info=None, shap_info=None, output_path="outputs/credit_risk_report.pdf"):
    os.makedirs("outputs", exist_ok=True)

    if threshold_data is None:
        threshold_data = load_threshold_tuning_data()

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    portfolio_summary = summarize_customer_profile(df, predictions_df)

    def fmt(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    def fmt_percent(x, decimals=1):
        try:
            return f"{float(x):.{decimals}f}%"
        except Exception:
            return str(x)

    def fmt_money(x):
        try:
            v = float(x)
        except Exception:
            return str(x)
        if abs(v) >= 1e6:
            return f"${v/1e6:,.2f}M"
        return f"${v:,.2f}"

    def summarize_failure_group(filter_mask):
        group = predictions_df.loc[filter_mask]
        loan_col = 'loan_amount' if 'loan_amount' in predictions_df.columns else 'loan_amnt' if 'loan_amnt' in predictions_df.columns else None
        income_col = 'income' if 'income' in predictions_df.columns else None
        pct_col = 'loan_percent_income' if 'loan_percent_income' in predictions_df.columns else None
        intent_col = 'loan_intent' if 'loan_intent' in predictions_df.columns else None

        stats = {
            'count': int(len(group)),
            'avg_loan_amount': fmt_money(group[loan_col].mean()) if loan_col and not group.empty else 'N/A',
            'avg_income': fmt_money(group[income_col].mean()) if income_col and not group.empty else 'N/A',
            'avg_loan_percent_income': f"{group[pct_col].mean():.2f}%" if pct_col and not group.empty else 'N/A',
            'avg_risk_probability': f"{group['risk_probability'].mean():.4f}" if not group.empty else 'N/A',
        }
        return stats

    def build_failure_table(stats):
        headers = [
            'Number of borrowers',
            'Average loan amount',
            'Average income',
            'Average loan % of income',
            'Avg predicted risk probability',
        ]
        values = [
            str(stats['count']),
            stats['avg_loan_amount'],
            stats['avg_income'],
            stats['avg_loan_percent_income'],
            stats['avg_risk_probability'],
        ]
        table = Table([headers, values], colWidths=[95, 95, 95, 95, 120])
        table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        return table

    def boxed_paragraph(text, style=None, bg_color=colors.whitesmoke):
        if style is None:
            style = styles['BodyText']
        box = Table([[Paragraph(text, style)]], colWidths=[480])
        box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        return box

    def page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        footer_text = 'Credit Risk Model Risk & Policy Report'
        canvas.drawString(40, 30, footer_text)
        canvas.drawRightString(560, 30, f'Page {doc.page}')
        canvas.restoreState()

    # Title and subtitle
    title = Paragraph("Credit Risk Model Risk & Policy Report", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))
    subtitle = Paragraph("Industry-style analytics deliverable: model performance, policy impact, and recommendations.", styles["Normal"])
    elements.append(subtitle)
    elements.append(Spacer(1, 12))

    model_order = ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression']

    # 1. Executive Decision Summary
    elements.append(Paragraph("1. Executive Decision Summary", styles["Heading1"]))
    elements.append(Spacer(1, 6))
    final_model = final_selection_info.get('report_model_name') if final_selection_info else report_model_name
    card_values = [
        ('Final Model', final_model or 'N/A'),
        ('Optimized Threshold', 'N/A'),
        ('Expected Loss', 'N/A'),
        ('Approval Rate', 'N/A'),
        ('Precision', 'N/A'),
        ('Recall', 'N/A')
    ]
    if optimized_summary and final_model:
        selected = next((r for r in optimized_summary if (r.get('Model') or '').lower() == (final_model or '').lower()), None)
        if selected:
            card_values = [
                ('Final Model', final_model),
                ('Optimized Threshold', f"{selected.get('Optimal_Threshold'):.3f}"),
                ('Expected Loss', fmt_money(selected.get('Expected Loss'))),
                ('Approval Rate', fmt_percent(selected.get('Approval Rate'))),
                ('Precision', fmt(selected.get('Precision'))),
                ('Recall', fmt(selected.get('Recall')))
            ]
    cards = []
    for i in range(0, len(card_values), 2):
        left = card_values[i]
        right = card_values[i+1] if i+1 < len(card_values) else ('', '')
        cards.append([
            Paragraph(f"<b>{left[0]}</b><br/>{left[1]}", styles['BodyText']),
            Paragraph(f"<b>{right[0]}</b><br/>{right[1]}", styles['BodyText'])
        ])
    card_table = Table(cards, colWidths=[240, 240], rowHeights=50)
    card_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(card_table)
    elements.append(Spacer(1, 12))

    # Executive decision sentence (dynamic)
    baseline_best = None
    if model_metrics:
        try:
            baseline_best = max(model_metrics, key=lambda name: float(model_metrics[name].get('Accuracy', 0)))
        except Exception:
            baseline_best = None
    decision = f"{baseline_best or 'Baseline model'} performs best on default-threshold metrics; {final_model or 'Selected model'} was chosen after cost-sensitive threshold optimization as it minimizes expected loss while meeting constraints."
    elements.append(Paragraph(decision, styles['BodyText']))
    elements.append(Spacer(1, 12))

    # 2. Business Objective and Cost Framework
    elements.append(Paragraph('2. Business Objective and Cost Framework', styles['Heading2']))
    elements.append(Spacer(1,6))
    objective_text = (
        'Threshold selection is treated as a lending policy decision: lower thresholds reduce risk exposure by rejecting more borrowers, while higher thresholds increase borrower access at the cost of additional expected loss.'
    )
    elements.append(boxed_paragraph(objective_text, styles['BodyText']))
    elements.append(Spacer(1,12))
    defs = [["Term", "Definition"], ["False Approval", "Actual=1 but Predicted=0 — approved risky borrower (financial loss)"], ["False Rejection", "Actual=0 but Predicted=1 — rejected safe borrower (lost profit)"]]
    defs_table = Table(defs, colWidths=[150, 350])
    defs_table.setStyle(TableStyle([("GRID", (0,0),(-1,-1),0.5,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),9)]))
    elements.append(defs_table)
    elements.append(Spacer(1,12))
    assumptions = (
        'Assumptions: the expected loss framework uses dataset loan amounts and measured false approval/rejection exposures; the configured business constraints preserve a minimum approval rate and classification reliability.'
    )
    elements.append(boxed_paragraph(assumptions, styles['BodyText'], bg_color=colors.HexColor('#eef6fb')))
    elements.append(Spacer(1,6))
    constraints = config.get('threshold_optimization', {}).get('constraints', {}) if config else {}
    elements.append(Paragraph(f"Configured constraints: approval_rate >= {constraints.get('min_approval_rate',0.8)*100:.1f}%, precision >= {constraints.get('min_precision',0.9)*100:.1f}%, recall >= {constraints.get('min_recall',0.7)*100:.1f}%", styles['BodyText']))
    elements.append(Spacer(1,12))

    # 3. Baseline Model Performance
    elements.append(Paragraph('3. Baseline Model Performance (Threshold = 0.50)', styles['Heading2']))
    elements.append(Spacer(1,6))
    elements.append(Paragraph('Baseline classification performance (for reference only).', styles['BodyText']))
    elements.append(Spacer(1,6))
    if model_metrics:
        rows = [["Model","Accuracy","Precision","Recall","ROC-AUC"]]
        for mname in model_order:
            md = model_metrics.get(mname, {})
            rows.append([mname, f"{float(md.get('Accuracy',0)):.3f}", f"{float(md.get('Precision',0)):.3f}", f"{float(md.get('Recall',0)):.3f}", f"{float(md.get('ROC-AUC',0)):.3f}"])
        t = Table(rows, colWidths=[120, 90, 90, 90, 90])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,1),(-1,-1),"CENTER")] ))
        elements.append(t)
        elements.append(Spacer(1,12))

    # 4. Threshold Optimization Results
    elements.append(Paragraph('4. Threshold Optimization Results', styles['Heading2']))
    elements.append(Spacer(1,6))
    # A. Performance metrics
    elements.append(Paragraph('A. Performance metrics', styles['Heading3']))
    elements.append(Spacer(1,6))
    perf_rows = [["Model","Threshold","Accuracy","Precision","Recall","F1","ROC-AUC"]]
    biz_rows = [["Model","Approval Rate","False Approvals","False Rejections","Expected Loss","Constraint Status"]]
    if optimized_summary:
        ordered = { (r.get('Model') or '').lower(): r for r in optimized_summary }
        for name in model_order:
            r = ordered.get(name.lower())
            if not r:
                continue
            perf_rows.append([name, f"{r['Optimal_Threshold']:.3f}", f"{float(r['Accuracy']):.3f}", f"{float(r['Precision']):.3f}", f"{float(r['Recall']):.3f}", f"{float(r['F1 Score']):.3f}", f"{float(r['ROC-AUC']):.3f}"])
            biz_rows.append([name, fmt_percent(r['Approval Rate']), str(int(r['False Approvals'])), str(int(r['False Rejections'])), fmt_money(r['Expected Loss']), r['Constraint Status']])
    perf_table = Table(perf_rows, colWidths=[120, 70, 70, 70, 70, 60, 70], hAlign='LEFT')
    perf_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,1),(-1,-1),"CENTER")]))
    elements.append(perf_table)
    elements.append(Spacer(1,8))
    # B. Business impact
    elements.append(Paragraph('B. Business impact', styles['Heading3']))
    elements.append(Spacer(1,6))
    biz_table = Table(biz_rows, colWidths=[120, 80, 80, 90, 90, 100], hAlign='LEFT')
    biz_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(1,1),(-1,-1),"CENTER")]))
    elements.append(biz_table)
    elements.append(Spacer(1,12))

    # 5. Recommended Lending Policy
    elements.append(Paragraph('5. Recommended Lending Policy', styles['Heading2']))
    elements.append(Spacer(1,6))
    rec = 'Recommendation not available.'
    if final_selection_info:
        selected_model = final_selection_info.get('report_model_name')
        selected_row = None
        if optimized_summary and selected_model:
            for r in optimized_summary:
                try:
                    name = r.get('Model') or r.get('model')
                except Exception:
                    name = None
                if name and selected_model and name.lower() == selected_model.lower():
                    selected_row = r
                    break

        if selected_row and selected_row.get('Optimal_Threshold') is not None:
            try:
                thr = float(selected_row.get('Optimal_Threshold'))
                rec = f"Recommend {selected_model} at threshold {thr:.3f} as it minimizes expected loss while meeting constraints."
            except Exception:
                rec = f"Recommend {selected_model} based on optimized expected-loss selection (threshold value unavailable)."
        else:
            rec = f"Recommend {selected_model} based on optimized expected-loss selection. (Optimal threshold details not available.)"
    elements.append(boxed_paragraph(rec, styles['BodyText'], bg_color=colors.HexColor('#fff4e5')))
    elements.append(Spacer(1,12))

    # 6. Visualizations
    elements.append(Paragraph('6. Visualizations', styles['Heading2']))
    elements.append(Spacer(1,6))
    if threshold_data:
        # Expected loss vs threshold
        fig, ax = plt.subplots(figsize=(7,3))
        ax.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        for name, df in threshold_data.items():
            if 'Threshold' in df.columns and 'Expected_Loss' in df.columns:
                ax.plot(df['Threshold'], df['Expected_Loss']/1e6, marker='o', linewidth=2, markersize=5, label=name)
        ax.set_xlabel('Threshold', fontsize=9)
        ax.set_ylabel('Expected Loss ($M)', fontsize=9)
        ax.set_title('Expected Loss vs Threshold', fontsize=11)
        ax.grid(True, color='#dddddd', linewidth=0.8)
        ax.set_facecolor('#f9f9f9')
        ax.legend(fontsize=8, frameon=False)
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        elements.append(RLImage(buf, width=520, height=220))
        elements.append(Paragraph('Figure 1: Cost-focused threshold selection shows the expected loss tradeoff as the policy becomes more conservative.', styles['BodyText']))
        elements.append(Spacer(1,8))

        # Approval rate vs threshold
        fig, ax = plt.subplots(figsize=(7,3))
        ax.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        for name, df in threshold_data.items():
            if 'Threshold' in df.columns and 'Approval_Rate' in df.columns:
                ax.plot(df['Threshold'], df['Approval_Rate'], marker='o', linewidth=2, markersize=5, label=name)
        ax.set_xlabel('Threshold', fontsize=9)
        ax.set_ylabel('Approval Rate (%)', fontsize=9)
        ax.set_title('Approval Rate vs Threshold', fontsize=11)
        ax.grid(True, color='#dddddd', linewidth=0.8)
        ax.set_facecolor('#f9f9f9')
        ax.legend(fontsize=8, frameon=False)
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        elements.append(RLImage(buf, width=520, height=220))
        elements.append(Paragraph('Figure 2: Higher thresholds increase approval rates while shifting the balance between borrower access and risk control.', styles['BodyText']))
        elements.append(Spacer(1,8))

    # Baseline ROC-AUC bar
    if model_metrics:
        fig, ax = plt.subplots(figsize=(7,3))
        names = model_order
        vals = [float(model_metrics.get(n, {}).get('ROC-AUC', 0)) for n in names]
        bars = ax.bar(names, vals, color=['#4c78a8', '#f58518', '#54a24b', '#e45756'])
        y_upper = min(1.08, max(vals) + 0.08)
        ax.set_ylim(0, y_upper)
        ax.set_title('Baseline ROC-AUC Comparison', fontsize=11)
        ax.grid(axis='y', color='#eeeeee', linewidth=0.8)
        ax.set_facecolor('#f9f9f9')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_upper * 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=8)
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        elements.append(RLImage(buf, width=520, height=220))
        elements.append(Paragraph('Figure 3: Baseline ROC-AUC comparison shows relative ranking at the default 0.50 threshold.', styles['BodyText']))
        elements.append(Spacer(1,12))

    # 7. Model Interpretability: Why the Model Flags Borrowers as Risky
    if shap_info:
        elements.append(Paragraph('7. Model Interpretability: Why the Model Flags Borrowers as Risky', styles['Heading2']))
        elements.append(Spacer(1,6))
        elements.append(Paragraph(shap_info.get('shap_note', ''), styles['BodyText']))
        elements.append(Spacer(1,6))

        if shap_info.get('feature_importance_path'):
            elements.append(RLImage(shap_info['feature_importance_path'], width=520, height=220))
            elements.append(Paragraph('Figure 4: SHAP feature importance ranks the most influential variables driving the XGBoost risk score.', styles['BodyText']))
            elements.append(Spacer(1,8))

        if shap_info.get('summary_path'):
            elements.append(RLImage(shap_info['summary_path'], width=520, height=220))
            elements.append(Paragraph('Figure 5: SHAP summary plot shows the overall direction and distribution of feature effects across the test sample.', styles['BodyText']))
            elements.append(Spacer(1,8))

        if shap_info.get('top_driver_text'):
            elements.append(Paragraph(shap_info['top_driver_text'], styles['BodyText']))
            elements.append(Spacer(1,6))

        if shap_info.get('local_main_plot'):
            elements.append(Paragraph('Local borrower explanation', styles['Heading3']))
            elements.append(RLImage(shap_info['local_main_plot'], width=520, height=220))
            elements.append(Paragraph('Figure 6: Local SHAP explanation for a representative borrower, showing how individual feature contributions drive the final risk score.', styles['BodyText']))
            elements.append(Spacer(1,8))

        if shap_info.get('error_analysis_text'):
            elements.append(Paragraph(shap_info['error_analysis_text'], styles['BodyText']))
            elements.append(Spacer(1,12))

    # 8. Where the Model Fails
    if predictions_df is not None:
        elements.append(Paragraph('8. Where the Model Fails', styles['Heading2']))
        elements.append(Spacer(1,6))
        elements.append(Paragraph('The final XGBoost model at threshold 0.40 struggles with three groups of borrowers: false approvals, false rejections, and borderline risk estimates.', styles['BodyText']))
        elements.append(Spacer(1,8))

        groups = [
            (
                'False approvals',
                (predictions_df['actual'] == 1) & (predictions_df['predicted'] == 0),
                'Borrowers who actually defaulted but were predicted as low risk and approved.',
            ),
            (
                'False rejections',
                (predictions_df['actual'] == 0) & (predictions_df['predicted'] == 1),
                'Borrowers who repaid successfully but were predicted as high risk and rejected.',
            ),
            (
                'Borderline cases',
                (predictions_df['risk_probability'] >= 0.35) & (predictions_df['risk_probability'] <= 0.45),
                'Borrowers with predicted risk close to the 0.40 decision threshold, where small changes can flip the outcome.',
            ),
        ]

        group_summaries = []
        for title, mask, description in groups:
            stats = summarize_failure_group(mask)
            group_summaries.append((title, description, stats))

        narratives_text = generate_failure_group_narratives(group_summaries)
        narrative_paragraphs = []
        if narratives_text:
            narrative_paragraphs = [p.strip() for p in narratives_text.split('\n\n') if p.strip()]

        for idx, (title, mask, _) in enumerate(groups):
            stats = summarize_failure_group(mask)
            elements.append(Paragraph(title, styles['Heading3']))
            elements.append(build_failure_table(stats))
            elements.append(Spacer(1,6))
            if idx < len(narrative_paragraphs):
                elements.append(Paragraph(narrative_paragraphs[idx], styles['BodyText']))
            else:
                elements.append(Paragraph('These borrowers represent a model weakness area that should be monitored with targeted policy actions.', styles['BodyText']))
            elements.append(Spacer(1,12))

    # 9. Appendix: full threshold tuning tables
    elements.append(Paragraph('9. Appendix: Full Threshold Tuning Tables', styles['Heading2']))
    elements.append(Spacer(1,6))
    if threshold_data:
        display_name_map = {
            'logistic_regression': 'Logistic Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM'
        }
        for name, df in threshold_data.items():
            normalized_name = name.lower().replace(' ', '_')
            display_name = display_name_map.get(normalized_name, name.replace('_', ' ').title())
            elements.append(Paragraph(f'{display_name} - Full Threshold Table', styles['Heading3']))
            cols = ['Threshold', 'Precision', 'Recall', 'Approval_Rate', 'False_Approvals', 'False_Rejections', 'Expected_Loss']
            label_map = {
                'Threshold': 'Threshold',
                'Precision': 'Precision',
                'Recall': 'Recall',
                'Approval_Rate': 'Approval Rate',
                'False_Approvals': 'False Approvals',
                'False_Rejections': 'False Rejections',
                'Expected_Loss': 'Expected Loss'
            }
            available = [c for c in cols if c in df.columns]
            headers = [label_map.get(c, c.replace('_', ' ')) for c in available]
            rows = [headers]
            for _, r in df.iterrows():
                rowvals = []
                for c in available:
                    if c == 'Expected_Loss':
                        rowvals.append(fmt_money(r[c]))
                    else:
                        rowvals.append(fmt(r[c]))
                rows.append(rowvals)
            num_cols = len(available)
            col_widths = [480 / num_cols] * num_cols
            t = LongTable(rows, colWidths=col_widths, repeatRows=1)
            t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,colors.black),("FONTSIZE",(0,0),(-1,-1),8),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("ALIGN",(0,0),(-1,-1),"CENTER")]))
            elements.append(t)
            elements.append(Spacer(1,12))

    if shap_info and shap_info.get('appendix_local_plots'):
        elements.append(Paragraph('10. Appendix: Additional SHAP Local Explanations', styles['Heading2']))
        elements.append(Spacer(1,6))
        for title, path in shap_info['appendix_local_plots']:
            elements.append(Paragraph(title, styles['Heading3']))
            elements.append(RLImage(path, width=520, height=220))
            elements.append(Spacer(1,6))
        elements.append(Spacer(1,12))

    doc.build(elements, onFirstPage=page_footer, onLaterPages=page_footer)
    return output_path