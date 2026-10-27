
# ============================================
# 1. Importer les bibliothèques
# ============================================
import mlflow
import mlflow.sklearn
import pandas as pd


# ============================================
# 2. Se connecter à la base de données MLflow
# ============================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ============================================
# 3. Charger le modèle champion avec son alias
# ============================================
model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

print("Modèle chargé :", type(model).__name__)

# Vérifier les variables attendues par le modèle
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "Le modèle ne fournit pas l'attribut feature_names_in_."
    )

features = list(model.feature_names_in_)
print("Variables attendues :", features)


# ============================================
# 4. Créer les données d'un nouveau client
# ============================================
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])


# ============================================
# 5. Vérifier les colonnes et respecter leur ordre
# ============================================
missing_features = [
    feature for feature in features
    if feature not in client.columns
]

if missing_features:
    raise ValueError(
        f"Variables manquantes : {missing_features}"
    )

client = client[features]


# ============================================
# 6. Effectuer la prédiction
# ============================================
prediction = model.predict(client)[0]

if prediction == 1:
    print("Le client va-t-il partir ? -> OUI")
elif prediction == 0:
    print("Le client va-t-il partir ? -> NON")
else:
    print("Classe prédite :", prediction)
