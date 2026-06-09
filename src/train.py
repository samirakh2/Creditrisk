from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 


def train_model(df, target_col="target"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Logistic Regression model
    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Train LightGBM model
    lgbm_model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    lgbm_model.fit(X_train, y_train)

    models = {
        'logistic_regression': lr_model,
        'random_forest': rf_model,
        'xgboost': xgb_model,
        'lightgbm': lgbm_model
    }

    return models, X_train, X_test, y_train, y_test