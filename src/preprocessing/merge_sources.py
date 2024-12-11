import pandas as pd
import sqlite3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paths import DB_PATH, CSV_PATH
# Chemins des fichiers


# Charger les données du CSV
def load_csv_data(path):
    print("Chargement des données depuis le CSV...")
    return pd.read_csv(path)


# Charger les données de la base SQLite
def load_sqlite_data(path, table_name="Tweets"):
    print("Chargement des données depuis SQLite...")
    conn = sqlite3.connect(path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Fusionner les deux sources
def merge_dataframes(df_csv, df_sqlite):
    print("Fusion des deux sources...")
    merged_df = pd.concat([df_csv, df_sqlite]).drop_duplicates().reset_index(drop=True)
    return merged_df


# Sauvegarder les données fusionnées dans SQLite
def save_to_sqlite(df, path, table_name="Tweets"):
    print("Sauvegarde des données fusionnées dans SQLite...")
    conn = sqlite3.connect(path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    # Charger les données des deux sources
    df_csv = load_csv_data(CSV_PATH)
    df_sqlite = load_sqlite_data(DB_PATH)

    # Afficher un aperçu
    print("Aperçu des données CSV :\n", df_csv.head())
    print("Aperçu des données SQLite :\n", df_sqlite.head())

    # Fusionner les deux sources
    merged_df = merge_dataframes(df_csv, df_sqlite)

    # Aperçu des données fusionnées
    print("Aperçu des données fusionnées :\n", merged_df.head())
    print(f"Nombre de lignes fusionnées : {len(merged_df)}")

    # Sauvegarder dans SQLite
    save_to_sqlite(merged_df, DB_PATH)

    print("Données fusionnées et sauvegardées avec succès dans SQLite.")
