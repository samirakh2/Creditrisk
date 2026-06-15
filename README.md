# Credit Risk Modeling Pipeline

An end-to-end machine learning pipeline for credit default risk assessment. It loads raw applicant data, engineers risk-related features, trains multiple classification models, tunes decision thresholds against business constraints, generates predictions, and produces a comprehensive PDF report with AI-generated narrative and SHAP-based explainability.

## Features

- **Data preprocessing**: column renaming/standardization, missing value handling, and categorical encoding
- **Feature engineering**: 25+ derived risk indicators (debt-to-income, credit maturity, risk scores, interaction features, etc.)
- **Multi-model training**: Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Threshold tuning**: evaluates a range of classification thresholds per model and selects the optimal one based on expected financial loss and configurable business constraints (minimum approval rate, precision, recall)
- **Prediction enrichment**: risk levels, confidence scores, prediction margins, misclassification flags, and percentile rankings
- **Model evaluation**: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix, and full classification report
- **Automated reporting**: generates a PDF report combining model comparisons, threshold tuning results, AI-written narrative summary (via OpenAI), and SHAP explainability plots

## Project Structure

```
.
├── main.py                  # Pipeline entry point
├── src/
│   ├── loader.py            # CSV loading
│   ├── schema_inference.py  # Column renaming/mapping
│   ├── preprocessing.py      # Data cleaning and encoding
│   ├── feature_engineering.py # Derived feature creation
│   ├── train.py             # Model training (LR, RF, XGBoost, LightGBM)
│   ├── evaluate.py          # Metrics, threshold tuning/optimization
│   ├── predict.py           # Prediction generation and enrichment
│   ├── summary.py           # Customer profile summary stats
│   ├── report_generator.py  # PDF report + LLM narrative generation
│   └── util.py
├── data/
│   └── credit_risk_dataset.csv
├── outputs/                  # Generated CSVs, figures, and PDF report
└── config.yaml               # Business rules and threshold configuration (optional)
```

## Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib reportlab openai pyyaml shap
```

> `shap` is optional — if not installed, SHAP explainability sections are skipped automatically.

## Configuration

An optional `config.yaml` in the project root controls threshold tuning and the financial loss model:

```yaml
thresholds: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Alternative: define a range instead of an explicit list
threshold_optimization:
  threshold_min: 0.30
  threshold_max: 0.90
  threshold_step: 0.05
  constraints:
    min_approval_rate: 0.80   # fraction (e.g. 0.80 = 80%)
    min_precision: 0.90
    min_recall: 0.70

loss_given_default: 0.6   # fraction of loan amount lost on a missed default
profit_margin: 0.05       # fraction of loan amount earned on a correct approval
default_loan_amount: 10000  # fallback loan size if no loan amount column is found
```

If `config.yaml` is absent, sensible defaults are used for all values.

### OpenAI API key

The report generator uses the OpenAI API to write narrative summaries. Set your API key as an environment variable before running:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

1. Place your dataset at `data/credit_risk_dataset.csv` (expects the standard [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) column format, e.g. `person_age`, `person_income`, `loan_amnt`, `loan_status`, etc.)
2. (Optional) Create a `config.yaml` to customize thresholds and business constraints.
3. Run the pipeline:

```bash
python main.py
```

## Pipeline Stages

1. **Load & preprocess** — raw CSV is loaded, columns renamed to readable names, missing values dropped, and categorical fields one-hot encoded.
2. **Feature engineering** — derived features (debt-to-income, credit history ratios, risk scores, interaction terms, etc.) are added.
3. **Model training** — the dataset is split (80/20, stratified) and four models are trained.
4. **Threshold tuning (Phase 1)** — for each model, every candidate threshold is scored on accuracy/precision/recall, approval rate, and expected financial loss; the threshold that best satisfies business constraints (or minimizes expected loss as a fallback) is selected.
5. **Predictions (Phase 2)** — each model generates enriched predictions (risk levels, confidence, correctness flags, percentile ranks) using its tuned threshold.
6. **Evaluation** — standard classification metrics and confusion matrices are computed and saved per model.
7. **Report generation** — the best-performing model (by expected loss and constraint satisfaction) is selected for the final report, which includes:
   - Model comparison table across all four models
   - Optimized threshold summary and expected loss breakdown
   - AI-generated narrative analysis (OpenAI)
   - SHAP feature importance and local explanation plots (if SHAP is installed)

## Outputs

All artifacts are written to `outputs/`:

| File | Description |
|---|---|
| `processed_credit_risk.csv` | Fully cleaned and feature-engineered dataset |
| `threshold_tuning_<model>.csv` | Threshold sweep results per model |
| `predictions_<model>.csv` | Enriched predictions per model |
| `model_metrics_<model>.csv` | Evaluation metrics per model |
| `optimized_threshold_comparison.csv` | Side-by-side comparison of all models at their optimal thresholds |
| `figures/shap_*.png` | SHAP summary and local explanation plots (if SHAP enabled) |
| `*.pdf` | Final comprehensive credit risk report |

## Model Selection Logic

The final report is built using the model that:
1. Meets all configured business constraints (approval rate, precision, recall) **and** has the lowest expected financial loss among those that qualify, or
2. If no model meets all constraints, the model with the lowest expected loss overall (flagged as a fallback selection in the report).

## Notes

- The dataset is expected to follow the standard credit risk schema (see `COLUMN_MAPPING` in `src/schema_inference.py` for the full set of expected source columns).
- Network access is required at runtime for the OpenAI narrative generation step.
