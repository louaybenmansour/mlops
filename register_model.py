import mlflow
from mlflow import MlflowClient


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    runs = mlflow.search_runs(
        experiment_names=["churn-prediction"],
        order_by=["metrics.f1 DESC"],
    )

    if runs.empty:
        raise ValueError(
            "Aucun run trouvé dans l'expérience 'churn-prediction'. "
            "Exécutez d'abord train_mlflow.py."
        )

    best = runs.iloc[0]
    run_id = best["run_id"]

    print(
        "Meilleur run :",
        best.get("tags.mlflow.runName", "Sans nom"),
        "| ID :",
        run_id,
    )
    print("Score F1 :", best["metrics.f1"])

    model_name = "churn-model"
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"Modèle enregistré : {model_name} | Version : {result.version}")

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=result.version,
    )
    print(f"Alias 'champion' attribué à la version {result.version}")

    model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    print("Le modèle champion est prêt à être utilisé.")
    print(type(model).__name__)


if __name__ == "__main__":
    main()