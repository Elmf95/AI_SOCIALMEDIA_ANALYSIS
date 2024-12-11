import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import joblib

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import PROCESSED_DATA_FILE, MODEL_DIR, MODEL_FILE

# Création des répertoires si nécessaire
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(file_path):
    """
    Charger les données depuis le fichier CSV.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset chargé depuis {file_path}.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None


def load_model_and_vectorizer(model_dir, model_file):
    """
    Charger le modèle et le vectorizer sauvegardés.
    """
    try:
        model = joblib.load(os.path.join(model_dir, model_file))
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        print("Modèle et vectorizer chargés avec succès.")
        return model, vectorizer
    except Exception as e:
        print(f"Erreur lors du chargement du modèle/vectorizer : {e}")
        return None, None


def evaluate_model(data, model, vectorizer):
    """
    Évaluer le modèle et générer des visualisations.
    """
    # Séparer les features (X) et la cible (y)
    X = data["clean_text"]
    y = data["target"]

    # Vectoriser les données
    X_tfidf = vectorizer.transform(X)

    # Prédictions
    y_pred = model.predict(X_tfidf)
    y_proba = model.predict_proba(X_tfidf)[:, 1]

    # Rapport de classification
    print("Rapport de classification :\n", classification_report(y, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Classe 0", "Classe 1"],
        yticklabels=["Classe 0", "Classe 1"],
    )
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC-AUC
    roc_auc = roc_auc_score(y, y_proba)
    print(f"Score ROC-AUC : {roc_auc:.4f}")
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Courbe ROC")
    plt.xlabel("Taux de Faux Positifs (FPR)")
    plt.ylabel("Taux de Vrais Positifs (TPR)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MODEL_DIR, "roc_curve.png"))
    plt.close()


if __name__ == "__main__":
    # Charger les données traitées
    data = load_data(PROCESSED_DATA_FILE)

    # Charger le modèle et le vectorizer
    model, vectorizer = load_model_and_vectorizer(MODEL_DIR, MODEL_FILE)

    if data is not None and model is not None and vectorizer is not None:
        evaluate_model(data, model, vectorizer)
