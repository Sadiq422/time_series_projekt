# Datei: retails/utils.py
# Zweck: Zentrale Hilfsfunktionen für Datenladen, -speichern und Modellverwaltung

# ============================================================
# Kernbibliotheken importieren
# ============================================================

# Betriebssystem-Funktionen für Dateipfad-Operationen
import os
# Input/Output-Operationen
import io
# HTTP-Anfragen für externe Datenquellen
import requests
# Numerische Berechnungen und Array-Operationen
import numpy as np
# Datenmanipulation und -analyse
import pandas as pd
# Objektorientierte Dateipfad-Handhabung
from pathlib import Path
# Anzeigefunktionen für Jupyter Notebooks
from IPython.display import display
# Serialisierung von Python-Objekten (speziell für ML-Modelle)
import joblib

# Projektinterne Pfadverwaltung importieren
from paths import get_path

# ============================================================
# CSV-Dateien laden (Basis-Lader)
# ============================================================

def load_csv(folder: str, filename: str) -> pd.DataFrame:
    """
    Lädt eine CSV-Datei. 'folder' kann ein Pfad-Schlüssel oder ein tatsächlicher Pfad sein.
    
    Args:
        folder (str): Pfad-Schlüssel (aus paths.py) oder tatsächlicher Ordnerpfad
        filename (str): Name der CSV-Datei
    
    Returns:
        pd.DataFrame: Geladenes DataFrame
    
    Raises:
        FileNotFoundError: Wenn die Datei nicht gefunden wird
    """
    # Dynamischer Import um Zirkelabhängigkeiten zu vermeiden
    from paths import get_path
    
    # Prüfen, ob 'folder' ein Pfad-Schlüssel ist
    try:
        # Versuche, es als Pfad-Schlüssel aus der zentralen Pfadverwaltung zu holen
        folder_path = get_path(folder)
        print(f"📁 Verwende Pfad-Schlüssel: '{folder}' -> {folder_path}")
    except ValueError:
        # Falls kein gültiger Schlüssel, wird es als direkter Pfad behandelt
        folder_path = folder
        print(f"📁 Verwende direkten Pfad: {folder_path}")
    
    # Vollständigen Dateipfad erstellen
    file_path = os.path.join(folder_path, filename)
    
    # Existenz der Datei überprüfen
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {file_path}")
    
    # Bestätigung der Dateiladung ausgeben
    print(f"📂 Geladen: {file_path}")
    
    try:
        # CSV-Datei in DataFrame laden
        df = pd.read_csv(file_path)
        # Statistiken zum geladenen Dataset ausgeben
        print(
            f"📊 Dataset geladen: '{filename}' | "
            f"Zeilen: {len(df):,} | Spalten: {df.shape[1]}\n"
        )
        # Zufällige Beispielzeilen anzeigen (maximal 10)
        display(df.sample(n=min(10, len(df))))
        
    except pd.errors.EmptyDataError:
        # Behandlung von leeren oder ungültigen CSV-Dateien
        print(f"⚠️ Warnung: '{filename}' ist leer oder konnte nicht geparst werden.")
        df = pd.DataFrame()  # Leeres DataFrame zurückgeben
    
    return df

# ============================================================
# Modelle speichern
# ============================================================

def save_model(model, model_name: str, model_type: str, model_dir: str) -> str:
    """
    Speichert ein trainiertes Modell mit joblib im angegebenen Modellverzeichnis.
    
    Args:
        model (object): Das zu speichernde trainierte Modell
        model_name (str): Basisname des Modells (z.B. "best_arima")
        model_type (str): Zusätzlicher Identifikator (z.B. "p3_d1_q2")
        model_dir (str): Verzeichnis, in dem das Modell gespeichert werden soll
    
    Returns:
        str: Vollständiger Pfad zur gespeicherten Modell-Datei
    """
    # Dateinamen aus Name und Typ erstellen
    filename = f"{model_name}_{model_type}.pkl"
    # Vollständigen Dateipfad erstellen
    filepath = os.path.join(model_dir, filename)
    
    # Modell mit joblib serialisieren und speichern
    joblib.dump(model, filepath)
    
    # Bestätigung der Speicherung ausgeben
    print(f"💾 Model saved: {filepath}")
    
    return filepath

# ============================================================
# CSV-Dateien speichern
# ============================================================

def save_csv(df: pd.DataFrame, folder: str, filename: str) -> None:
    """
    Speichert ein DataFrame als CSV-Datei.
    
    Args:
        df (pd.DataFrame): Zu speicherndes DataFrame
        folder (str): Zielordner
        filename (str): Name der CSV-Datei
    """
    # Vollständigen Dateipfad erstellen
    file_path = os.path.join(folder, filename)
    # DataFrame als CSV speichern (ohne Index-Spalte)
    df.to_csv(file_path, index=False)
    
    # Bestätigung der Speicherung ausgeben
    print(f"💾 Gespeichert: {file_path}")

# ============================================================
# Hilfsfunktion: Deterministischen Dateinamen für gefilterte Daten erstellen
# ============================================================

def build_filtered_filename(table_name: str, filters: dict) -> str:
    """
    Erstellt einen deterministischen Dateinamen basierend auf Filterkriterien.
    
    Args:
        table_name (str): Name der Originaltabelle
        filters (dict): Filterkriterien als Dictionary
    
    Returns:
        str: Generierter Dateiname mit Filterinformationen
    """
    # Basisnamen ohne .csv-Endung
    base = table_name.replace(".csv", "")
    # Liste für Namensbestandteile initialisieren
    parts = [base]
    
    # MAX_DATE-Filter hinzufügen (falls vorhanden)
    if "MAX_DATE" in filters:
        # Datum auf YYYY-MM-DD formatieren
        clean_date = str(filters["MAX_DATE"]).split(" ")[0]
        parts.append(f"MAXDATE-{clean_date}")
    
    # STORE_IDS-Filter hinzufügen (falls vorhanden)
    if "STORE_IDS" in filters:
        # Store-IDs als String mit Bindestrichen verknüpfen
        parts.append("STORE-" + "-".join(map(str, filters["STORE_IDS"])))
    
    # ITEM_IDS-Filter hinzufügen (falls vorhanden)
    if "ITEM_IDS" in filters:
        # Item-IDs als String mit Bindestrichen verknüpfen
        parts.append("ITEM-" + "-".join(map(str, filters["ITEM_IDS"])))
    
    # Alle Teile mit doppelten Unterstrichen verbinden und .csv anhängen
    return "__".join(parts) + ".csv"

# ============================================================
# Gefilterte CSV-Dateien laden mit Caching-Mechanismus
# ============================================================

def load_filtered_csv(
    folder_name: str,
    table_name: str,
    filters: dict,
    filter_folder: str = "filtered",
    force_recompute: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Lädt eine CSV-Datei, wendet Filter an und cached das gefilterte Ergebnis.
    
    Args:
        folder_name (str): Name des Quellordners (Pfad-Schlüssel)
        table_name (str): Name der CSV-Tabelle
        filters (dict): Filterkriterien als Dictionary
        filter_folder (str): Zielordner für gefilterte Dateien (Standard: "filtered")
        force_recompute (bool): Wenn True, wird Cache ignoriert und neu berechnet
    
    Returns:
        pd.DataFrame: Gefiltertes DataFrame
    
    Raises:
        TypeError: Wenn filter_folder kein String ist
    """
    # 1. Filterausgabeverzeichnis auflösen
    if not isinstance(filter_folder, str):
        raise TypeError("filter_folder must be a string")
    
    # Verzeichnis aus Pfadverwaltung holen (wird ggf. erstellt)
    filtered_dir = get_path(filter_folder)
    
    # Deterministischen Dateinamen generieren
    filtered_filename = build_filtered_filename(table_name, filters)
    # Vollständigen Pfad erstellen
    filtered_path = os.path.join(filtered_dir, filtered_filename)
    
    # 2. Gecachte Datei laden (falls vorhanden und nicht erzwungen)
    if os.path.exists(filtered_path) and not force_recompute:
        print(f"⚡ Loading existing filtered dataset: {filtered_filename}")
        df = pd.read_csv(filtered_path)
        
        # Dateibereich anzeigen (falls Daten vorhanden)
        if not df.empty and "date" in df.columns:
            print("\n📅 Date Range:")
            print(f"   Start: {df['date'].min()}")
            print(f"   End:   {df['date'].max()}")
            print(f"   Days:  {len(df['date'].unique())}")
        
        return df
    
    print("🔍 No cached file found. Computing filtering...")
    
    # 3. Roh-CSV laden
    df = load_csv(folder=folder_name, filename=table_name)
    
    # Dateibereich anzeigen (falls Daten vorhanden)
    if not df.empty and "date" in df.columns:
        print("\n📅 Date Range:")
        print(f"   Start: {df['date'].min()}")
        print(f"   End:   {df['date'].max()}")
        print(f"   Days:  {len(df['date'].unique())}")
    
    # 4. Filter anwenden
    print("🔎 Applying filters...")
    
    # Datumsspalte in datetime konvertieren (falls vorhanden)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # MAX_DATE-Filter anwenden (falls angegeben)
    if "MAX_DATE" in filters:
        df = df[df["date"] <= pd.to_datetime(filters["MAX_DATE"])]
    
    # STORE_IDS-Filter anwenden (falls angegeben)
    if "STORE_IDS" in filters:
        df = df[df["store_nbr"].isin(filters["STORE_IDS"])]
    
    # ITEM_IDS-Filter anwenden (falls angegeben)
    if "ITEM_IDS" in filters:
        df = df[df["item_nbr"].isin(filters["ITEM_IDS"])]
    
    print(f"✅ Filtered shape: {df.shape}")
    
    # 5. Gefiltertes Dataset speichern
    df.to_csv(filtered_path, index=False)
    print(f"💾 Saved filtered dataset to: {filtered_path}")
    
    return df

# ============================================================
# Gefilterte CSV-Dateien nur nach Max-Datum laden
# ============================================================

def load_data_filtered_by_date(
    folder_name: str,
    table_name: str,
    max_date: str,
    filter_folder: str = "filtered",
    force_recompute: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Lädt eine CSV-Datei und filtert sie nach einem maximalen Datum.
    
    Args:
        folder_name (str): Name des Quellordners
        table_name (str): Name der CSV-Tabelle
        max_date (str): Maximales Datum (inklusive)
        filter_folder (str): Zielordner für gefilterte Dateien
        force_recompute (bool): Wenn True, wird Cache ignoriert
    
    Returns:
        pd.DataFrame: Nach Datum gefiltertes DataFrame
    """
    # 0. Verzeichnisstruktur sicherstellen
    processed_folder = get_path("processed")
    filtered_dir = os.path.join(processed_folder, filter_folder)
    os.makedirs(filtered_dir, exist_ok=True)
    
    # Datum für Dateinamen bereinigen
    clean_date = str(max_date).split(" ")[0]
    # Basisnamen ohne .csv-Endung
    base_name = table_name.replace(".csv", "")
    # Dateinamen mit Datumsfilter generieren
    filtered_filename = f"{base_name}__MAXDATE-{clean_date}.csv"
    # Vollständigen Pfad erstellen
    filtered_path = os.path.join(filtered_dir, filtered_filename)
    
    # 1. Gecachte Version laden (falls vorhanden und nicht erzwungen)
    if os.path.exists(filtered_path) and not force_recompute:
        print(f"⚡ Loading existing date-filtered dataset: {filtered_filename}")
        df = pd.read_csv(filtered_path)
        
        # Dateibereich anzeigen (falls Daten vorhanden)
        if not df.empty and "date" in df.columns:
            print("\n📅 Date Range:")
            print(f"   Start: {df['date'].min()}")
            print(f"   End:   {df['date'].max()}")
            print(f"   Days:  {len(df['date'].unique())}")
        
        return df
    
    # 2. Rohdaten laden und filtern
    print(f"🔍 Filtering {table_name} by Max Date: {clean_date}...")
    df = load_csv(folder=folder_name, filename=table_name)
    
    # Datumsfilter anwenden (falls Datumsspalte existiert)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"] <= pd.to_datetime(max_date)]
    else:
        print(f"⚠️ Warning: 'date' column not found in {table_name}. Returning unfiltered.")
    
    # 3. Gefiltertes Dataset speichern
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    df.to_csv(filtered_path, index=False)
    print(f"💾 Saved date-filtered dataset (Shape: {df.shape})")
    
    return df