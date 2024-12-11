import sqlite3
import pandas as pd
import re
import os
import sys

# Ajouter le dossier racine au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import DB_PATH, CLEANED_DATA_FILE


def clean_tweet(text):
    """
    Nettoie un tweet en supprimant les mentions, hashtags, URLs, caractères spéciaux, etc.
    """
    if not isinstance(text, str):  # Vérifie que le texte est bien une chaîne
        return ""

    text = re.sub(r"@\w+", "", text)  # Suppression des mentions (@user)
    text = re.sub(r"#\w+", "", text)  # Suppression des hashtags (#hashtag)
    text = re.sub(r"http\S+|www\S+", "", text)  # Suppression des URL
    text = re.sub(
        r"[^a-zA-Z0-9\s]", "", text
    )  # Suppression des caractères spéciaux, mais garde chiffres
    text = re.sub(r"\s+", " ", text).strip()  # Suppression des espaces multiples
    return text.lower()


def load_sqlite_data(db_path, table_name="tweets"):
    """
    Charge les données de la table SQLite spécifiée.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name};"
        data = pd.read_sql_query(query, conn)
        conn.close()
        print(
            f"Colonnes disponibles dans la table {table_name} : {data.columns.tolist()}"
        )
        print(f"Données chargées depuis {db_path}, table {table_name}.")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données SQLite : {e}")
        return None


def clean_sqlite_data(data):
    """
    Applique le nettoyage des tweets sur les textes.
    """
    if "text" not in data.columns:
        raise KeyError("La colonne 'text' est introuvable dans les données.")

    # Nettoyer les tweets
    data["clean_text"] = data["text"].apply(clean_tweet)

    # Supprimer les lignes où le texte nettoyé est vide
    data = data[data["clean_text"] != ""]

    # Vérifier et renommer la colonne cible si nécessaire
    if "airline_sentiment" in data.columns:
        data.rename(columns={"airline_sentiment": "target"}, inplace=True)

    # Vérifier que la colonne "target" existe maintenant
    if "target" not in data.columns:
        raise KeyError("La colonne 'target' est introuvable dans les données.")

    # Filtrer les lignes avec des valeurs valides dans `target`
    valid_targets = [
        "positif",
        "negatif",
        "neutre",
        "positive",  # Ajouter d'autres variantes si nécessaires
        "negative",
        "neutral",
    ]
    data = data[data["target"].isin(valid_targets)]

    return data[
        ["clean_text", "target"]
    ]  # Conserver uniquement les colonnes nécessaires


if __name__ == "__main__":
    # Charger les données SQLite
    data = load_sqlite_data(DB_PATH)

    if data is not None and not data.empty:
        try:
            # Nettoyer les données
            print("Avant nettoyage :", data.head())  # Log des données brutes pour debug
            cleaned_data = clean_sqlite_data(data)

            # Sauvegarder les données nettoyées dans un fichier CSV
            if not cleaned_data.empty:
                cleaned_data.to_csv(CLEANED_DATA_FILE, index=False, encoding="utf-8")
                print(f"Données nettoyées enregistrées dans {CLEANED_DATA_FILE}.")
            else:
                print("Erreur : toutes les données ont été supprimées après nettoyage.")
        except KeyError as e:
            print(f"Erreur : {e}")
    else:
        print("Erreur : aucun tweet trouvé ou base SQLite introuvable.")
