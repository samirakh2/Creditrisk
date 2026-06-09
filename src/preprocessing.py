import pandas as pd

def clean_data(df):

    # 1. Handle missing values
    df = df.dropna()

    # 2. Encode categorical variables
    categorical_cols = [
        "home_ownership",
        "loan_intent",
        "loan_grade",
        "default_history"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df