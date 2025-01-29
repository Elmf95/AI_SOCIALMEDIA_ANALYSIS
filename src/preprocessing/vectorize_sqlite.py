import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Ajouter le dossier racine au chemin pour accéder à paths.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Importer les chemins dynamiques à partir de paths.py
from paths import CLEANED_DATA_FILE, MODEL_DIR, MODEL_FILE, VECTOR_DIR, VECTOR_FILE

# Construire les chemins complets pour le modèle et le vectorizer
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
VECTOR_PATH = os.path.join(VECTOR_DIR, VECTOR_FILE)

# Assurer que les répertoires nécessaires existent
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


def regenerate_model_and_vectorizer(data_file, model_path, vectorizer_path):
    print("Chargement des données...")
    data = pd.read_csv(data_file)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"], data["target"], test_size=0.2, random_state=42
    )

    print("Entraînement du vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Entraînement du modèle...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)

    print("Évaluation du modèle...")
    y_pred = model.predict(X_test_vec)
    print("Rapport de classification :\n", classification_report(y_test, y_pred))
    print(f"Précision globale : {accuracy_score(y_test, y_pred):.2f}")

    # Sauvegarder le modèle et le vectorizer
    print("Sauvegarde du modèle et du vectorizer...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Modèle sauvegardé dans : {model_path}")
    print(f"Vectorizer sauvegardé dans : {vectorizer_path}")


if __name__ == "__main__":
    regenerate_model_and_vectorizer(CLEANED_DATA_FILE, MODEL_PATH, VECTOR_PATH)
