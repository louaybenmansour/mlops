from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "churn_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Le fichier de données '{data_path.name}' est introuvable. "
            "Exécutez d'abord generate_data.py."
        )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn-prediction")

    df = pd.read_csv(data_path)
    df["abonnement"] = df["abonnement"].map({
        "Basique": 0,
        "Standard": 1,
        "Premium": 2,
    })

    X = df[["age", "salaire", "abonnement", "anciennete", "consommation"]]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name="random-forest-champion") as run:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"Run ID : {run.info.run_id}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Model URI : runs:/{run.info.run_id}/model")


if __name__ == "__main__":
    main()
