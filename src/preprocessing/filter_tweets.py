import pandas as pd
import sys
import os

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import RAW_DATA_FILE, FILTERED_DATA_FILE


# Charger le dataset brut avec gestion des colonnes
def load_raw_dataset(file_path):
    try:
        # Charger sans en-tête et renommer les colonnes
        dataset = pd.read_csv(file_path, header=None, encoding="latin1")
        # Renommer les colonnes manuellement
        dataset.columns = ["target", "id", "date", "flag", "user", "text"]
        print("Colonnes du dataset :", dataset.columns)
        print(dataset.head())
        return dataset
    except UnicodeDecodeError as e:
        print(f"Erreur d'encodage lors du chargement de {file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        return None


# Filtrer les tweets contenant des mots-clés spécifiques
def filter_tweets_by_keywords(dataset, keywords):
    return dataset[
        dataset["text"].str.contains("|".join(keywords), case=False, na=False)
    ]


if __name__ == "__main__":
    # Mots-clés liés à l'e-commerce
    keywords = [
        "e-commerce",
        "achat",
        "livraison",
        "commande",
        "Amazon",
        "Prime",
        "marketplace",
    ]

    # Charger et filtrer les tweets
    dataset = load_raw_dataset(RAW_DATA_FILE)
    if dataset is not None and not dataset.empty:
        filtered_tweets = filter_tweets_by_keywords(dataset, keywords)
        if not filtered_tweets.empty:
            filtered_tweets.to_csv(FILTERED_DATA_FILE, index=False, encoding="utf-8")
            print(f"Tweets filtrés enregistrés dans {FILTERED_DATA_FILE}.")
        else:
            print("Aucun tweet correspondant n'a été trouvé.")
