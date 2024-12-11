import sqlite3
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import DB_PATH


def prepare_data():
    """
    Prépare les données en nettoyant et enrichissant la base SQLite.
    """
    print("Connexion à la base de données...")

    # Vérifier que le fichier SQLite existe
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Le fichier SQLite spécifié n'existe pas : {DB_PATH}")

    # Connexion à la base de données SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Chargement des données existantes...")
    df = pd.read_sql_query("SELECT * FROM Tweets", conn)

    # Vérifier les colonnes disponibles
    print("Colonnes disponibles dans la base de données :")
    print(df.columns.tolist())

    # Nettoyage des données (supprimer les doublons, si applicable)
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(
        f"Nombre de lignes initiales : {initial_count}, après suppression des doublons : {final_count}"
    )

    # Ajouter une colonne pour les hashtags extraits (enrichissement)
    print("Ajout de colonnes enrichies...")
    df["hashtags"] = df["text"].str.findall(r"#\w+")  # Extraction des hashtags

    # Convertir les listes de hashtags en chaînes de caractères pour SQLite
    df["hashtags"] = df["hashtags"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )

    # Enregistrement des modifications dans une nouvelle table
    enriched_table = "TweetsEnriched"
    print(f"Création ou remplacement de la table enrichie : {enriched_table}")
    df.to_sql(enriched_table, conn, if_exists="replace", index=False)

    print("Préparation des données terminée.")

    # Fermer la connexion
    conn.close()


if __name__ == "__main__":
    prepare_data()
