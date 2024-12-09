import pandas as pd
import re
import os
import sys
from sklearn.model_selection import train_test_split  # Correction ici

# Ajouter le dossier racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import FILTERED_DATA_FILE, PROCESSED_DATA_FILE


# Fonction de nettoyage des tweets
def clean_tweet(text):
    text = re.sub(r"@\w+", "", text)  # Suppression des mentions (@user)
    text = re.sub(r"#\w+", "", text)  # Suppression des hashtags (#hashtag)
    text = re.sub(r"http\S+|www\S+", "", text)  # Suppression des URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Suppression des caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()  # Suppression des espaces multiples
    return text.lower()


# Charger les données filtrées
def load_filtered_data(file_path):
    try:
        dataset = pd.read_csv(file_path, encoding="utf-8")
        print(f"Dataset chargé avec succès depuis {file_path}.")
        return dataset
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        return None


# Préparer les données pour l'entraînement
def prepare_data(dataset):
    dataset["clean_text"] = dataset["text"].apply(clean_tweet)
    dataset = dataset[["clean_text", "target"]]  # Conserver les colonnes nécessaires
    dataset["target"] = dataset["target"].apply(
        lambda x: 1 if x == 4 else 0
    )  # Convertir en binaire
    return dataset


if __name__ == "__main__":
    # Charger les tweets filtrés
    dataset = load_filtered_data(FILTERED_DATA_FILE)
    if dataset is not None and not dataset.empty:
        # Nettoyer et préparer les données
        prepared_data = prepare_data(dataset)

        # Séparer en ensembles d'entraînement et de validation
        train_data, val_data = train_test_split(
            prepared_data,
            test_size=0.2,
            random_state=42,
            stratify=prepared_data["target"],
        )

        # Sauvegarder les ensembles dans un fichier CSV
        processed_data_dir = os.path.dirname(PROCESSED_DATA_FILE)
        os.makedirs(processed_data_dir, exist_ok=True)

        train_data.to_csv(
            PROCESSED_DATA_FILE.replace(".csv", "_train.csv"),
            index=False,
            encoding="utf-8",
        )
        val_data.to_csv(
            PROCESSED_DATA_FILE.replace(".csv", "_val.csv"),
            index=False,
            encoding="utf-8",
        )
        print(
            f"Données préparées enregistrées dans {PROCESSED_DATA_FILE}_train.csv et {PROCESSED_DATA_FILE}_val.csv."
        )
    else:
        print("Erreur : le dataset est vide ou introuvable.")
