README
---

# 🚀 **Time Series Analysis & Forecasting Dashboard**

---


Eine End-to-End Machine Learning Pipeline zur Analyse und Vorhersage komplexer Zeitreihen für den Einzelhandel. Dieses Projekt kombiniert modernste Deep-Learning-Verfahren (LSTM) mit klassischen statistischen Modellen und Gradient Boosting in einem professionellen, interaktiven Dashboard.

---

# 📌 **Highlights**
Enterprise-Ready Dashboard: Professionelle Dark Mode Benutzeroberfläche mit Echtzeit-Visualisierungen

---

# 📁 Projektstruktur & Navigation

time_series_projekt/
├── app/                          # Streamlit Dashboard & UI Logik
│   ├── app.py                      # Hauptanwendung (Dark Mode)
│   ├── app_backup.py               # Backup der ursprünglichen App
│   └── bootstrap.py                # UI-Komponenten & Styling
│
├── notebooks/                    # Forschungs- & Entwicklungs-Pipeline
│   ├── 01_preprocessing.ipynb      # Datenreinigung & Transformation
│   ├── 02_feature_engineering.ipynb# Feature-Generierung
│   ├── 03_data_management.ipynb    # I/O Prozesse
│   ├── 04_lstm_modeling.ipynb      # Deep Learning Modelle
│   ├── 05_xgboost_modeling.ipynb   # Gradient Boosting
│   └── 06_arima_analysis.ipynb     # Statistische Baseline
│
├── data/                         # Roh- und vorverarbeitete Datensätze
│   └── filtered/                   # Vorverarbeitete Daten
│
├── outputs/                      # Ergebnisse & Exporte
│   ├── forecasts/                  # Vorhersage-Ergebnisse
│   ├── visualizations/             # Automatisch generierte Plots
│   └── reports/                    # Analysen & Dokumentation
│
├── paths.py                     # Zentrale Pfadverwaltung
├── utils.py                     # Core Helper Functions
├── visualizer.py                # Plotting Engine
├── requirements.txt             # Hauptabhängigkeiten
├── environment.yml              # Conda Environment
└── README.md                    # Diese Dokumentation

---

Multi-Modell Ensemble: Kombiniert LSTM, XGBoost und ARIMA für robuste Vorhersagen

Automatisches Feature-Engineering: Lag-Features, Rolling Statistics, Saisonale Dekomposition

Produktionsreife Pipeline: Vollständige ML Pipeline von Datenvorbereitung bis Deployment

Interactive Analytics: Echtzeit-Analyse mit Konfidenzintervallen und Performance-Metriken

---

# 📊 **Dashboard Features**
🔗 Kernfunktionen
Echtzeit Forecasting: Historische und zukünftige Verkaufsprognosen

Performance Monitoring: MAE, RMSE, R² Metriken in Echtzeit

Residuen-Analyse: Detaillierte Fehleranalyse und Diagnostik

Konfidenzintervalle: Statistische Unsicherheitsquantifizierung

---

# 📈 **Visualisierungen**
Interactive Plots: Plotly-basierte interaktive Diagramme

Vergleichende Analysen: Tatsächliche vs. vorhergesagte Werte

Trend-Analyse: Saisonale Dekomposition und Trenderkennung

Fehlerverteilungen: Histogramme und Residuen-Plots

---

# ⚙️ **Konfiguration**
Store & Item Selection: Flexible Auswahl von Geschäften und Artikeln

Modell-Parameter: Anpassbare Forecast-Horizonte und Konfidenzniveaus

Export-Funktionen: CSV-Export und Report-Generierung

---

# 🛠️ **Technologiestack**

## Machine Learning & Data Science:

TensorFlow/Keras: LSTM Neural Networks für Sequenzvorhersagen

XGBoost: Gradient Boosting für tabulare Daten

Statsmodels: ARIMA und statistische Analysen

Scikit-learn: Feature Engineering und Preprocessing

Pandas & NumPy: Datenmanipulation und -analyse

## Dashboard & Visualisierung:

Streamlit: Interactive Web Application Framework

Plotly: Interaktive Visualisierungen

Matplotlib/Seaborn: Statische Plot-Generierung

## Entwicklung & Deployment:


Python 3.9+: Hauptprogrammiersprache

Git: Versionskontrolle

Conda/Pip: Paketverwaltung

---

# 🚀 **Installation**

## Voraussetzungen
Python 3.9 oder höher

pip oder conda

---

# 📖 **Verwendung**

Daten hochladen: Laden Sie Ihre Zeitreihendaten im CSV-Format

Modell konfigurieren: Wählen Sie Vorhersagehorizont und Konfidenzniveau

Training starten: Lassen Sie das Ensemble-Modell automatisch trainieren

Ergebnisse analysieren: Nutzen Sie die interaktiven Visualisierungen

Exportieren: Speichern Sie Vorhersagen und Berichte

---

# 📊 **Performance Metriken**

Das System berechnet folgende Metriken automatisch:

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

R² (Determinationskoeffizient)

MAPE (Mean Absolute Percentage Error)

---

# 🔧 **Konfiguration**

Anpassbare Parameter in config.py:

Forecast Horizon (1-52 Wochen)

Konfidenzintervalle (80%, 90%, 95%)

Modellgewichtungen (LSTM, XGBoost, ARIMA)

Feature Engineering Parameter

# 📄 **Lizenz**

---

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe LICENSE Datei für Details.

---

# **📞 Kontakt**

Für Fragen oder Support:
Claudia
E-mail: fotsoclaudia88@gmail.com
Sadiq
qais.sadiq422@gmail.com

