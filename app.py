from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="API Churn")
mlflow.set_tracking_uri("sqlite:///mlflow.db")

try:
    model = mlflow.sklearn.load_model("models:/churn-model@champion")
except Exception as exc:
    raise RuntimeError(
        "Le modèle champion n'est pas encore enregistré. "
        "Exécutez d'abord train_mlflow.py puis register_model.py."
    ) from exc


class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(client: Client):
    mapping = {"Basique": 0, "Standard": 1, "Premium": 2}

    data = pd.DataFrame([
        {
            "age": client.age,
            "salaire": client.salaire,
            "abonnement": mapping[client.abonnement],
            "anciennete": client.anciennete,
            "consommation": client.consommation,
        }
    ])

    data = data[list(model.feature_names_in_)]
    prediction = int(model.predict(data)[0])

    return {
        "churn": prediction,
        "message": "Le client va partir" if prediction == 1 else "Le client reste",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
