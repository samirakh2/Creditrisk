def engineer_features(df):
    import numpy as np
    # Debt to income ratio - primary risk indicator
    df["debt_to_income"] = df["loan_amount"] / df["income"].replace(0, 1)

    # Income per year employed - employment stability metric
    df["income_per_year_employed"] = df["income"] / (df["employment_length"] + 1)

    # Credit maturity - credit history relative to age
    df["credit_history_ratio"] = df["credit_history_length"] / df["age"].replace(0, 1)

    # High loan percent flag - loan risk indicator
    df["high_loan_percent"] = (df["loan_percent_income"] > 0.4).astype(int)

    # Interest rate features
    interest_rate_median = df["interest_rate"].median()
    df["high_interest_rate"] = (df["interest_rate"] > interest_rate_median).astype(int)
    df["interest_to_loan_ratio"] = df["interest_rate"] / (df["loan_amount"] + 1)

    # Age-based features
    df["young_borrower"] = (df["age"] < 25).astype(int)
    df["senior_borrower"] = (df["age"] > 40).astype(int)
    df["age_squared"] = df["age"] ** 2  # Capture non-linear age effects

    # Employment features
    df["experienced_worker"] = (df["employment_length"] > 10).astype(int)
    df["new_worker"] = (df["employment_length"] < 2).astype(int)
    df["employment_stability"] = df["employment_length"] / (df["employment_length"].max() + 1)

    # Income features
    income_q25 = df["income"].quantile(0.25)
    income_q75 = df["income"].quantile(0.75)
    df["low_income"] = (df["income"] < income_q25).astype(int)
    df["high_income"] = (df["income"] > income_q75).astype(int)
    df["income_to_loan_ratio"] = df["income"] / (df["loan_amount"] + 1)

    # Credit history features
    df["credit_maturity"] = df["credit_history_length"] / (df["credit_history_length"].max() + 1)
    df["short_credit_history"] = (df["credit_history_length"] < 3).astype(int)
    df["long_credit_history"] = (df["credit_history_length"] > 10).astype(int)

    # Interest and income burden features
    df["interest_income_burden"] = (df["interest_rate"] * df["loan_amount"]) / (df["income"] + 1)
    df["monthly_payment_burden"] = (df["loan_amount"] / 60) / (df["income"] / 12 + 1)  # Assuming 5-year loan

    # Composite risk features
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["total_debt_burden"] = df["debt_to_income"] + df["loan_percent_income"]
    df["risk_score"] = (
        df["high_interest_rate"] * 0.15
        + df["high_loan_percent"] * 0.20
        + df["young_borrower"] * 0.10
        + df["new_worker"] * 0.15
        + df["low_income"] * 0.15
        + df["short_credit_history"] * 0.15
    )

    # Default history impact
    df["has_default_history"] = df.get("default_history_Y", 0).astype(int)

    # Loan grade risk mapping (higher grades = riskier)
    grade_risk = {"B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
    df["loan_grade_risk_score"] = 0
    for grade, risk_value in grade_risk.items():
        df["loan_grade_risk_score"] += (df.get(f"loan_grade_{grade}", 0).astype(int) * risk_value)

    # Personal vs. high-risk loan purposes
    high_risk_intents = ["PERSONAL", "VENTURE"]
    df["high_risk_loan_purpose"] = 0
    for intent in high_risk_intents:
        df["high_risk_loan_purpose"] += df.get(f"loan_intent_{intent}", 0).astype(int)
    df["high_risk_loan_purpose"] = df["high_risk_loan_purpose"].astype(int)

    # Interaction features
    df["age_income_interaction"] = (df["age"] / (df["age"].max() + 1)) * (df["income"] / (df["income"].max() + 1))
    df["credit_employment_interaction"] = df["credit_maturity"] * df["employment_stability"]

    return df