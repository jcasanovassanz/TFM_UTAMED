import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def run_walk_forward_validation(df, features, start_test_year=2015):
    df = df.copy()
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    results = []

    for year in years:
        if year < start_test_year:
            continue

        train = df[df["year"] < year]
        test = df[df["year"] == year]

        if train.empty or test.empty:
            continue

        X_train = train[features]
        y_train = train["target"]

        X_test = test[features]
        y_test = test["target"]

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        if y_test.nunique() > 1:
            auc_value = roc_auc_score(y_test, y_prob)
        else:
            auc_value = None

        results.append({
            "test_year": year,
            "n_train": len(train),
            "n_test": len(test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": auc_value
        })

    return pd.DataFrame(results)

def run_walk_forward_validation_rf(df, features, start_test_year=2015):
    df = df.copy()
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    results = []

    for year in years:
        if year < start_test_year:
            continue

        train = df[df["year"] < year]
        test = df[df["year"] == year]

        if train.empty or test.empty:
            continue

        X_train = train[features]
        y_train = train["target"]

        X_test = test[features]
        y_test = test["target"]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        if y_test.nunique() > 1:
            auc_value = roc_auc_score(y_test, y_prob)
        else:
            auc_value = None

        results.append({
            "test_year": year,
            "n_train": len(train),
            "n_test": len(test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": auc_value
        })

    return pd.DataFrame(results)