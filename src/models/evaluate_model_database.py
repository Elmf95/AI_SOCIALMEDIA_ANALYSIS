import os
import joblib
import pandas as pd
import sqlite3
from sklearn.metrics import classification_report, accuracy_score
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Importer les chemins depuis votre fichier paths.py
from paths import (
    DB_PATH,
    CLEANED_DATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    VECTOR_DIR,
    VECTOR_FILE,
)

# Chemins des fichiers nécessaires
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
VECTORIZER_PATH = os.path.join(VECTOR_DIR, VECTOR_FILE)


# Charger le modèle avec pipeline
def load_model():
    try:
        # Charger le modèle complet via joblib (en supposant que c'est un pipeline)
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé avec succès.")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None


# Charger le vecteur de transformation (Tfidf, CountVectorizer, etc.)
def load_vectorizer():
    try:
        # Charger le vectorizer avec joblib
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Vectorizer chargé avec succès.")
        return vectorizer
    except Exception as e:
        print(f"Erreur lors du chargement du vectorizer : {e}")
        return None


# Charger les données depuis la base SQLite
def load_data_from_sqlite(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT text, airline_sentiment FROM tweets"  # Adapter la requête pour utiliser 'text' et 'airline_sentiment'
        data = pd.read_sql(query, conn)
        conn.close()
        print(f"Données chargées depuis la base de données SQLite.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données SQLite : {e}")
        return None


# Fonction de nettoyage des tweets
def clean_tweet(text):
    # Retirer les mentions (@user) et les hashtags (#hashtag)
    text = re.sub(r"@\w+", "", text)  # Enlever les mentions
    text = re.sub(r"#\w+", "", text)  # Enlever les hashtags
    text = re.sub(r"http\S+", "", text)  # Enlever les URLs
    text = re.sub(r"\s+", " ", text)  # Remplacer les espaces multiples par un seul
    text = text.strip()  # Supprimer les espaces au début et à la fin
    return text


# Tester le modèle sur de nouvelles données
def test_model_on_data(model, vectorizer, data):
    try:
        # Vérifier que les colonnes nécessaires sont présentes
        if "text" not in data.columns or "airline_sentiment" not in data.columns:
            print(
                "Les colonnes 'text' et 'airline_sentiment' sont nécessaires pour le test."
            )
            return

        # Extraire les features (X) et la cible réelle (y)
        X = data["text"].apply(clean_tweet)  # Appliquer le nettoyage à chaque tweet
        y_true = data["airline_sentiment"]

        # Vectorisation des tweets avant la prédiction
        X_vect = vectorizer.transform(
            X
        )  # Transformer les tweets en vecteurs numériques

        # Utiliser le modèle pour faire des prédictions
        y_pred = model.predict(
            X_vect
        )  # Le modèle prend maintenant les vecteurs de caractéristiques

        # Évaluer les performances
        print("Rapport de classification :\n", classification_report(y_true, y_pred))
        print("Précision globale :", accuracy_score(y_true, y_pred))
    except Exception as e:
        print(f"Erreur lors du test du modèle : {e}")


if __name__ == "__main__":
    # Charger le modèle
    model = load_model()

    # Vérifier que le modèle est valide
    if model is not None:
        # Charger le vectorizer
        vectorizer = load_vectorizer()

        # Vérifier que le vectorizer est valide
        if vectorizer is not None:
            # Charger les données depuis SQLite
            data_from_sqlite = load_data_from_sqlite(DB_PATH)

            if data_from_sqlite is not None:
                # Tester le modèle sur les données extraites
                test_model_on_data(model, vectorizer, data_from_sqlite)
