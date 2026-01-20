# Datei: retails/paths.py
# Zweck: Zentrale Verwaltung von Dateipfaden für das Retail Demand Analysis Projekt

# Betriebssystem-spezifische Funktionen für Dateipfad-Operationen importieren
import os

# 1. GRUNDLEGENDE PFADKONFIGURATION
# ---------------------------------

# Absoluten Pfad zu dieser Datei ermitteln
# __file__ enthält den relativen Pfad der aktuellen Datei, abspath() macht ihn absolut
current_file = os.path.abspath(__file__)

# Projekt-Root-Verzeichnis bestimmen (3 Ebenen über dieser Datei)
# os.path.dirname() entfernt die letzte Komponente eines Pfades
# Hier wird dreimal auf das übergeordnete Verzeichnis gewechselt
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

# Externes Daten-Root-Verzeichnis (gemountete Festplatte)
# Enthält die Rohdaten für das Grocery Sales Forecasting Projekt
external_data_root = os.path.join("/Volumes/Expansion/time_series_projekt_daten/corporación_favorita_grocery_sales_forecasting")

# 2. KERNVERZEICHNISSE DES PROJEKTS
# ---------------------------------

# Verzeichnis für Jupyter Notebooks
notebook_dir = os.path.join(project_root, "notebooks")

# Haupt-Datenverzeichnis (befindet sich auf externer Festplatte)
data_dir = os.path.join(external_data_root, "data")

# Verzeichnis für gespeicherte Modelle
models_dir = os.path.join(project_root, "models")

# Verzeichnis für Berichte und Auswertungen
reports_dir = os.path.join(project_root, "reports")

# Unterverzeichnis für Visualisierungen und Diagramme
figures_dir = os.path.join(reports_dir, "figures")

# Unterverzeichnis für numerische Ergebnisse und Metriken
results_dir = os.path.join(reports_dir, "results")

# 3. DATEN-UNTERVERZEICHNISSE
# ---------------------------

# Verzeichnis für Rohdaten (unveränderte Originaldaten)
raw_dir = os.path.join(data_dir, "raw")

# Verzeichnis für verarbeitete Daten
processed_dir = os.path.join(data_dir, "processed")

# Verzeichnis für bereinigte Daten (nach Data Cleaning)
cleaner_dir = os.path.join(processed_dir, "cleaner")

# Verzeichnis für erzeugte Features (Feature Engineering)
feature_dir = os.path.join(processed_dir, "features")

# Verzeichnis für gefilterte Daten (z.B. nach bestimmten Kriterien)
filtered_dir = os.path.join(processed_dir, "filtered")

# 4. MODELLSPEZIFISCHE VERZEICHNISSE
# ----------------------------------

# Verzeichnis für ARIMA-Modelle (klassisches Zeitreihenmodell)
arima_model_dir = os.path.join(models_dir, "arima")

# Verzeichnis für LSTM-Modelle (Recurrent Neural Network für Zeitreihen)
lstm_model_dir = os.path.join(models_dir, "lstm")

# Verzeichnis für XGBoost-Modelle (Gradient Boosting Algorithmus)
xgboost_model_dir = os.path.join(models_dir, "xgboost")

# 5. VISUALISIERUNGSVERZEICHNISSE PRO MODELLTYP
# ---------------------------------------------

# Verzeichnis für ARIMA-spezifische Diagramme
arima_figure_dir = os.path.join(figures_dir, "arima")

# Verzeichnis für LSTM-spezifische Diagramme
lstm_figure_dir = os.path.join(figures_dir, "lstm")

# Verzeichnis für XGBoost-spezifische Diagramme
xgboost_figure_dir = os.path.join(figures_dir, "xgboost")

# 6. ERGEBNISVERZEICHNISSE PRO MODELLTYP
# --------------------------------------

# Verzeichnis für ARIMA-Ergebnisse (Metriken, Vorhersagen)
arima_results_dir = os.path.join(results_dir, "arima")

# Verzeichnis für LSTM-Ergebnisse
lstm_results_dir = os.path.join(results_dir, "lstm")

# Verzeichnis für XGBoost-Ergebnisse
xgboost_results_dir = os.path.join(results_dir, "xgboost")

# 7. ZENTRALE PFADVERWALTUNG (DICTIONARY)
# ---------------------------------------

# Dictionary das Schlüsselwörter auf entsprechende Pfade abbildet
# Ermöglicht einfachen Zugriff auf alle Pfade über konsistente Namen
_path_dict = {
    # Grundlegende Verzeichnisse
    "root": project_root,
    "notebooks": notebook_dir,
    "data": data_dir,
    "models": models_dir,
    "reports": reports_dir,
    "figures": figures_dir,
    "results": results_dir,
    
    # Datenunterverzeichnisse
    "raw": raw_dir,
    "processed": processed_dir,
    "cleaner": cleaner_dir,
    "features": feature_dir,
    "filtered": filtered_dir,

    # Modellverzeichnisse
    "arima_model": arima_model_dir,
    "lstm_model": lstm_model_dir,
    "xgboost_model": xgboost_model_dir,

    # Visualisierungsverzeichnisse
    "arima_figures": arima_figure_dir,
    "lstm_figures": lstm_figure_dir,
    "xgboost_figures": xgboost_figure_dir,

    # Ergebnisverzeichnisse
    "arima_results": arima_results_dir,
    "lstm_results": lstm_results_dir,
    "xgboost_results": xgboost_results_dir,
}

# 8. HILFSFUNKTION FÜR PFADZUGRIFF
# --------------------------------

def get_path(path_name: str, mkdir: bool = True) -> str:
    """
    Ermittelt einen Projektpfad anhand seines Namens.
    
    Args:
        path_name (str): Name des Pfades aus dem _path_dict
        mkdir (bool): Wenn True, wird das Verzeichnis erstellt (falls nicht vorhanden)
    
    Returns:
        str: Der absolute Pfad als String
    
    Raises:
        ValueError: Wenn der angegebene Pfadname nicht existiert
    
    Beispiel:
        >>> data_path = get_path("data")
        >>> raw_data_path = get_path("raw")
    """
    # Prüfen ob der angefragte Pfadname im Dictionary existiert
    if path_name not in _path_dict:
        # Erstellen einer Liste aller gültigen Schlüssel für die Fehlermeldung
        valid = ", ".join(_path_dict.keys())
        raise ValueError(f"Invalid path name '{path_name}'. Valid options: {valid}")

    # Pfad aus dem Dictionary abrufen
    path = _path_dict[path_name]

    # Bei Bedarf Verzeichnis erstellen (rekursiv, falls Elternverzeichnisse fehlen)
    if mkdir:
        os.makedirs(path, exist_ok=True)

    # Pfad zurückgeben
    return path