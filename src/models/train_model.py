import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import PROCESSED_DATA_FILE, MODEL_DIR, MODEL_FILE

# Création des répertoires si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(file_path):
    """
    Charger le fichier CSV des tweets traités.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset chargé depuis {file_path}.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None


def train_and_save_model(data):
    """
    Entraîne un modèle de classification et sauvegarde le modèle.
    """
    # Séparer les features (X) et la cible (y)
    X = data["clean_text"]
    y = data["target"]

    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorisation des textes
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Entraîner un modèle de régression logistique
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Évaluer le modèle
    y_pred = model.predict(X_val_tfidf)
    print("Rapport de classification :\n", classification_report(y_val, y_pred))
    print("Précision :", accuracy_score(y_val, y_pred))

    # Sauvegarder le modèle et le vectorizer
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILE))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    print(f"Modèle et vectorizer sauvegardés dans {MODEL_DIR}.")


if __name__ == "__main__":
    # Charger les données traitées
    data = load_data(PROCESSED_DATA_FILE)

    if data is not None:
        train_and_save_model(data)
