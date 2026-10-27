from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd


def predict_churn():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_name = "churn-model"

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    except Exception as exc:
        raise RuntimeError(
            "Le modèle champion n'est pas encore enregistré. "
            "Exécutez d'abord train_mlflow.py puis register_model.py pour créer "
            "l'alias 'champion'."
        ) from exc

    print("Modèle chargé :", type(model).__name__)
    print("Variables attendues :", list(model.feature_names_in_))

    client = pd.DataFrame(
        [{
            "age": 35,
            "salaire": 2500,
            "abonnement": 1,
            "anciennete": 1,
            "consommation": 10,
        }]
    )

    client = client[list(model.feature_names_in_)]
    prediction = int(model.predict(client)[0])
    print("Le client va-t-il partir ? ->", "OUI" if prediction == 1 else "NON")
    return prediction


if __name__ == "__main__":
    predict_churn()