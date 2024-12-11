import pandas as pd
import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paths import DB_PATH, CSV_PATH


def compare_sources(db_path, csv_path, table_name="Tweets"):
    """
    Compare les données entre la base SQLite et le fichier CSV.
    """
    try:
        # Charger les données SQLite
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        # Charger les données CSV
        df_csv = pd.read_csv(csv_path)

        # Comparaison
        same_data = df_db.equals(df_csv)
        print("Les données des deux sources sont-elles identiques ? :", same_data)

        if not same_data:
            print("Différences dans les colonnes ou les lignes.")
            # Comparaison des colonnes
            print(
                "Colonnes dans SQLite mais pas dans CSV :",
                set(df_db.columns) - set(df_csv.columns),
            )
            print(
                "Colonnes dans CSV mais pas dans SQLite :",
                set(df_csv.columns) - set(df_db.columns),
            )

            # Différences au niveau des données
            print("Différence au niveau des lignes :", len(df_db) - len(df_csv))

        return df_db, df_csv
    except Exception as e:
        print(f"Erreur lors de la comparaison : {e}")
        return None, None


if __name__ == "__main__":
    compare_sources(DB_PATH, CSV_PATH)
