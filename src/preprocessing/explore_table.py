import sqlite3
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import DB_PATH


def explore_table(db_path, table_name="Tweets"):
    """
    Affiche les premières lignes de la table spécifiée dans la base de données.
    """
    try:
        # Connexion à la base de données
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name} LIMIT 5"
        df = pd.read_sql_query(query, conn)
        print(f"Aperçu de la table '{table_name}':\n", df)
        print("Colonnes disponibles :\n", df.columns)
        conn.close()
        return df
    except Exception as e:
        print(f"Erreur lors de l'exploration de la table '{table_name}' : {e}")
        return None


if __name__ == "__main__":
    explore_table(DB_PATH)
