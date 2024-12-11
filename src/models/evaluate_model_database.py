import os
import sqlite3
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Ajout du chemin dynamique
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paths import DB_PATH, MODEL_DIR, MODEL_FILE

# Charger le modèle et le vectorizer
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Modèle et vectorizer chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle/vectorizer : {e}")
    model = None
    vectorizer = None


# Charger les données depuis SQLite
def load_data(path, table_name="TweetsEnriched", sample_size=1000):
    conn = sqlite3.connect(path)
    query = f"SELECT text, airline_sentiment FROM {table_name} LIMIT {sample_size}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Évaluer le modèle
def evaluate_model(df):
    if model is None or vectorizer is None:
        raise ValueError("Le modèle ou le vectorizer n'a pas été chargé correctement.")

    # Prétraiter les données
    X = vectorizer.transform(df["text"])
    y_true = df["airline_sentiment"]

    # Mapper les sentiments en valeurs numériques
    sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}
    y_true_mapped = y_true.map(sentiment_mapping)

    # Faire des prédictions
    y_pred = model.predict(X)

    # Calculer les métriques
    print("\nRapport de classification :")
    print(
        classification_report(
            y_true_mapped, y_pred, target_names=sentiment_mapping.keys()
        )
    )

    # Matrice de confusion
    cm = confusion_matrix(y_true_mapped, y_pred, labels=[1, 0, -1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=sentiment_mapping.keys(),
        yticklabels=sentiment_mapping.keys(),
    )
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Matrice de confusion sauvegardée : confusion_matrix.png")


# Fonction principale
def main():
    print("Chargement des données...")
    df = load_data(DB_PATH, sample_size=1000)  # Taille de l'échantillon ajustable
    print(f"Nombre de tweets chargés : {len(df)}")
    print("Évaluation du modèle...")
    evaluate_model(df)


if __name__ == "__main__":
    main()
