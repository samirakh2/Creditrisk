COLUMN_MAPPING = {
    "person_age": "age",
    "person_income": "income",
    "person_home_ownership": "home_ownership",
    "person_emp_length": "employment_length",
    "loan_intent": "loan_intent",
    "loan_grade": "loan_grade",
    "loan_amnt": "loan_amount",
    "loan_int_rate": "interest_rate",
    "loan_status": "target",
    "loan_percent_income": "loan_percent_income",
    "cb_person_default_on_file": "default_history",
    "cb_person_cred_hist_length": "credit_history_length"
}

def rename_columns(df):
    return df.rename(columns=COLUMN_MAPPING)