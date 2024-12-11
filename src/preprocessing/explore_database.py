import os
import sqlite3
import pandas as pd

from paths import DATA_DIR  # Assurez-vous que le chemin DATA_DIR est bien configuré

# Chemin vers la base de données SQLite
DB_PATH = os.path.join(DATA_DIR, "database.sqlite")


def explore_database(db_path):
    """
    Explorer les tables et les colonnes d'une base de données SQLite.
    """
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, conn)
        print("Tables disponibles :\n", tables)
        conn.close()
    except Exception as e:
        print(f"Erreur lors de l'exploration de la base de données : {e}")


if __name__ == "__main__":
    explore_database(DB_PATH)
