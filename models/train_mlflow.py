import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from mlflow.models import infer_signature


def main():
    # Directorio base = raíz del repo (sube un nivel desde models)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "openrate.csv")

    # Cargar datos
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

    # Separar columnas numéricas y categóricas
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

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment("openrate-notifications")

    with mlflow.start_run(run_name="log_reg_baseline"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name="notification_open_model",
        )

        print(f"accuracy: {acc:.4f}")
        print(f"precision: {prec:.4f}")
        print(f"recall: {rec:.4f}")
        print(f"f1_score: {f1:.4f}")
        print(f"roc_auc: {auc:.4f}")


if __name__ == "__main__":
    main()