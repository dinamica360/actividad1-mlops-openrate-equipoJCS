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


def run_log_reg_experiment(C_value, class_weight, run_name, exp_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "openrate.csv")

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

    clf = LogisticRegression(
        max_iter=1000,
        C=C_value,
        class_weight=class_weight,
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("openrate-notifications")

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)
        mlflow.log_param("class_weight", str(class_weight))
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            name="notification_open_logreg",
            signature=signature,
            input_example=X_train.head(3),
        )

        print(
            f"{run_name} -> acc={acc:.4f}, "
            f"prec={prec:.4f}, rec={rec:.4f}, "
            f"f1={f1:.4f}, auc={auc:.4f}"
        )

        return {
            "experimento": exp_id,
            "modelo": "LogisticRegression",
            "C": C_value,
            "class_weight": str(class_weight),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
        }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    resultados = []

    # E2: C=0.1, sin balanceo de clases
    resultados.append(
        run_log_reg_experiment(
            C_value=0.1,
            class_weight=None,
            run_name="log_reg_C_0.1",
            exp_id="E2",
        )
    )

    # E3: C=10, sin balanceo de clases
    resultados.append(
        run_log_reg_experiment(
            C_value=10.0,
            class_weight=None,
            run_name="log_reg_C_10",
            exp_id="E3",
        )
    )

    # E4: C=1.0, con balanceo automático de clases
    resultados.append(
        run_log_reg_experiment(
            C_value=1.0,
            class_weight="balanced",
            run_name="log_reg_C_1_balanced",
            exp_id="E4",
        )
    )

    # CSV con resultados
    df_results = pd.DataFrame(resultados)
    csv_path = os.path.join(output_dir, "trainlogreg.csv")
    df_results.to_csv(csv_path, index=False)

    # MD con resumen
    md_path = os.path.join(output_dir, "trainlogreg.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experimentos de Logistic Regression (tuning de C y class_weight)\n\n")
        f.write(
            "Se ejecutaron tres experimentos adicionales de regresión logística "
            "para analizar el efecto de la regularización y del balanceo de "
            "clases en la predicción de `target_opened`.\n\n"
        )
        f.write("## Configuración de experimentos\n\n")
        for r in resultados:
            f.write(
                f"- {r['experimento']}: "
                f"LogisticRegression(C={r['C']}, "
                f"class_weight={r['class_weight']})\n"
            )
        f.write("\n## Resultados\n\n")
        for r in resultados:
            f.write(
                f"### {r['experimento']} — "
                f"C={r['C']}, class_weight={r['class_weight']}\n\n"
            )
            f.write(f"- Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"- Precision: {r['precision']:.4f}\n")
            f.write(f"- Recall: {r['recall']:.4f}\n")
            f.write(f"- F1-score: {r['f1_score']:.4f}\n")
            f.write(f"- ROC AUC: {r['roc_auc']:.4f}\n\n")

    print("Archivos generados en output/:")
    print(f"- {csv_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()