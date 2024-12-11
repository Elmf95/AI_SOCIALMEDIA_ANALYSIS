import os
import sqlite3
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from paths import DB_PATH


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
