import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.models import infer_signature

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "openrate.csv")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    target = "target_opened"
    feature_cols = [
        "user_id",
        "site",
        "campaign_type",
        "device_os",
        "hour_of_day",
        "day_of_week",
        "historical_open_rate",
        "historical_push_count",
        "days_since_last_open",
        "segment",
    ]

    X = df[feature_cols]
    y = df[target]

    num_cols = [
        "hour_of_day",
        "historical_open_rate",
        "historical_push_count",
        "days_since_last_open",
    ]

    cat_cols = [
        "user_id",
        "site",
        "campaign_type",
        "device_os",
        "segment",
        "day_of_week",
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("openrate-notifications")

    with mlflow.start_run(run_name="random_forest_baseline"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="notification_open_rf_model",
            signature=signature,
            input_example=X_train.head(3),
        )

        metrics_df = pd.DataFrame([{
            "model": "RandomForestClassifier",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
        }])
        metrics_df.to_csv(os.path.join(output_dir, "metrics_random_forest.csv"), index=False)

        with open(os.path.join(output_dir, "model_training_report.md"), "w", encoding="utf-8") as f:
            f.write("# Experimento Random Forest\n\n")
            f.write("Se entrenó un modelo RandomForestClassifier para predecir `target_opened`.\n\n")
            f.write("## Métricas\n\n")
            f.write(f"- Accuracy: {acc:.4f}\n")
            f.write(f"- Precision: {prec:.4f}\n")
            f.write(f"- Recall: {rec:.4f}\n")
            f.write(f"- F1-score: {f1:.4f}\n")
            f.write(f"- ROC AUC: {auc:.4f}\n")

        print(metrics_df)
        
if __name__ == "__main__":
    main()