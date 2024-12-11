import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import sys

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import CLEANED_DATA_FILE, VECTOR_DIR, VECTOR_FILE

# Création du répertoire de sauvegarde si nécessaire
os.makedirs(VECTOR_DIR, exist_ok=True)


def vectorize_and_save(data, vectorizer_path, output_path):
    """
    Vectorise les textes et sauvegarde le vectorizer ainsi que les vecteurs.
    """
    # Filtrer les tweets vides
    data = data[data["clean_text"].str.strip() != ""]

    # Si le dataframe est vide après le filtre
    if data.empty:
        print("Erreur : Aucun tweet à vectoriser (données vides après nettoyage).")
        return

    # Vectorisation avec TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10000, stop_words=None)
    X_tfidf = vectorizer.fit_transform(data["clean_text"])
    print(f"Shape des vecteurs TF-IDF : {X_tfidf.shape}")

    # Sauvegarder le vectorizer et les vecteurs
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(X_tfidf, output_path)
    print(f"Vectorizer sauvegardé dans {vectorizer_path}.")
    print(f"Vecteurs sauvegardés dans {output_path}.")


if __name__ == "__main__":
    # Charger les données nettoyées
    try:
        data = pd.read_csv(CLEANED_DATA_FILE)
        print(f"Données nettoyées chargées depuis {CLEANED_DATA_FILE}.")
    except Exception as e:
        print(f"Erreur lors du chargement des données nettoyées : {e}")
        sys.exit(1)

    # Vérifier les premières lignes des tweets nettoyés
    print(data["clean_text"].head())

    # Vectoriser et sauvegarder
    vectorize_and_save(
        data,
        vectorizer_path=os.path.join(VECTOR_DIR, "vectorizer.joblib"),
        output_path=os.path.join(VECTOR_DIR, VECTOR_FILE),
    )
