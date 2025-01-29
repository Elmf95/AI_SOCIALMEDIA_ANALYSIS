import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import re
import sys

# Charger les chemins depuis paths.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paths import (
    DB_PATH,
    CLEANED_DATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    VECTOR_DIR,
    VECTOR_FILE,
)

# Chemins complets des fichiers
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
VECTOR_PATH = os.path.join(VECTOR_DIR, VECTOR_FILE)


# Charger et tester le modèle
def test_model_on_data(data_file):
    try:
        # Charger les données nettoyées
        data = pd.read_csv(data_file)
        print(f"Données chargées depuis : {data_file}")

        # Charger le modèle
        model = joblib.load(MODEL_PATH)
        print(f"Modèle chargé depuis : {MODEL_PATH}")

        # Charger le vectorizer
        vectorizer = joblib.load(VECTOR_PATH)
        print(f"Vectorizer chargé depuis : {VECTOR_PATH}")

        # Nettoyage des textes
        def clean_text(text):
            text = re.sub(r"@\w+", "", text)  # Supprimer les mentions
            text = re.sub(r"#\w+", "", text)  # Supprimer les hashtags
            text = re.sub(r"http\S+", "", text)  # Supprimer les URLs
            text = re.sub(r"\s+", " ", text).strip()  # Nettoyer les espaces
            return text

        data["clean_text"] = data["clean_text"].apply(clean_text)

        # Transformation des données
        X = vectorizer.transform(data["clean_text"])
        y_true = data["target"]

        # Prédictions
        y_pred = model.predict(X)

        # Évaluation
        print("Rapport de classification :\n", classification_report(y_true, y_pred))
        print(f"Précision globale : {accuracy_score(y_true, y_pred):.2f}")

    except Exception as e:
        print(f"Erreur : {e}")


if __name__ == "__main__":
    test_model_on_data(CLEANED_DATA_FILE)
