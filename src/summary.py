def summarize_customer_profile(df, predictions_df):
    summary = {}

    summary["avg_age"] = round(df["age"].mean(), 2)
    summary["median_income"] = round(df["income"].median(), 2)
    summary["avg_loan_amount"] = round(df["loan_amount"].mean(), 2)
    summary["avg_debt_to_income"] = round(df["debt_to_income"].mean(), 4)
    summary["pct_high_loan_percent"] = round(df["high_loan_percent"].mean() * 100, 2)

    if "default_history_Y" in df.columns:
        summary["pct_prior_default"] = round(df["default_history_Y"].mean() * 100, 2)

    summary["top_loan_intent"] = "N/A"

    return summary