🚀 Advanced Time Series Analysis & Forecasting Dashboard
https://img.shields.io/badge/python-3.9+-blue.svg
https://img.shields.io/badge/Streamlit-1.28.0+-red.svg
https://img.shields.io/badge/TensorFlow-2.13.0+-orange.svg
https://img.shields.io/badge/License-MIT-green.svg
https://img.shields.io/badge/platform-macOS%2520%257C%2520Linux%2520%257C%2520Windows-lightgrey.svg

Eine End-to-End Machine Learning Pipeline zur Analyse und Vorhersage komplexer Zeitreihen für den Einzelhandel. Dieses Projekt kombiniert modernste Deep-Learning-Verfahren (LSTM) mit klassischen statistischen Modellen und Gradient Boosting in einem professionellen, interaktiven Dashboard.

🌟 Highlights
Enterprise-Ready Dashboard: Professionelle Dark Mode Benutzeroberfläche mit Echtzeit-Visualisierungen

Multi-Modell Ensemble: Kombiniert LSTM, XGBoost und ARIMA für robuste Vorhersagen

Automatische Feature-Engineering: Lag-Features, Rolling Statistics, Saisonale Dekomposition

Produktionsreife Pipeline: Vollständige ML Pipeline von Datenvorbereitung bis Deployment

Interactive Analytics: Echtzeit-Analyse mit Konfidenzintervallen und Performance-Metriken

📊 Dashboard Features
🎯 Core Features
Echtzeit Forecasting: Historische und zukünftige Verkaufsprognosen

Performance Monitoring: MAE, RMSE, R² Metriken in Echtzeit

Residuen-Analyse: Detailleirte Fehleranalyse und Diagnostik

Konfidenzintervalle: Statistische Unsicherheitsquantifizierung

📈 Visualisierungen
Interactive Plots: Plotly-basierte interaktive Diagramme

Vergleichende Analysen: Tatsächliche vs. vorhergesagte Werte

Trend-Analyse: Saisonale Dekomposition und Trenderkennung

Fehlerverteilungen: Histogramme und Residuen-Plots

⚙️ Konfiguration
Store & Item Selection: Flexible Auswahl von Geschäften und Artikeln

Modell-Parameter: Anpassbare Forecast-Horizonte und Konfidenzniveaus

Export-Funktionen: CSV-Export und Report-Generierung

🏗️ Projektstruktur
time_series_projekt/
├── 📂 app/                          # Streamlit Dashboard & UI
│   ├── app.py                      # Hauptanwendung (Dark Mode)
│   ├── app_backup.py               # Backup der ursprünglichen App
│   └── bootstrap.py                # UI-Komponenten & Styling
├── 📂 notebooks/                    # Forschungs- & Entwicklungs-Pipeline
│   ├── 01_preprocessing.ipynb      # Datenreinigung & Transformation
│   ├── 02_feature_engineering.ipynb# Feature-Generierung
│   ├── 03_data_management.ipynb    # I/O Prozesse
│   ├── 04_lstm_modeling.ipynb      # Deep Learning Modelle
│   ├── 05_xgboost_modeling.ipynb   # Gradient Boosting
│   └── 06_arima_analysis.ipynb     # Statistische Baseline
├── 📂 data/                         # Datensätze
│   └── filtered/                   # Vorverarbeitete Daten
├── 📂 models/                       # Trainierte Modelle
│   ├── lstm_model.h5               # LSTM Modellgewichte
│   └── scaler.pkl                  # Feature-Scaler
├── 📂 outputs/                      # Ergebnisse & Exporte
│   ├── forecasts/                  # Vorhersage-Ergebnisse
│   └── visualizations/             # Automatisch generierte Plots
├── 📂 reports/                      # Analysen & Dokumentation
│   └── lstm_metrics.csv            # Modell-Performance Metriken
├── 📜 paths.py                      # Zentrale Pfadverwaltung
├── 📜 utils.py                      # Core Helper Functions
├── 📜 visualizer.py                 # Plotting Engine
├── 📜 requirements.txt              # Hauptabhängigkeiten
├── 📜 requirements_app.txt          # Streamlit App Abhängigkeiten
├── 📜 environment.yml               # Conda Environment
└── 📜 README.md                     # Diese Dokumentation


Dokumentation
🛠️ Technologiestack
Machine Learning & Data Science
TensorFlow/Keras: LSTM Neural Networks für Sequenzvorhersagen

XGBoost: Gradient Boosting für Feature-Interaktionen

Scikit-learn: Preprocessing, Feature Engineering, Model Evaluation

Statsmodels: ARIMA, Saisonale Dekomposition, Zeitreihenanalyse

Data Processing & Visualization
Pandas/Numpy: Datenmanipulation und numerische Berechnungen

Plotly/Matplotlib: Interaktive und statische Visualisierungen

Darts: Zeitreihen-Bibliothek für Forecasting

Dashboard & UI
Streamlit: Interactive Web Dashboard Framework

Custom CSS: Professionelles Dark Mode Design

Plotly Graph Objects: Echtzeit-Updates und Interaktionen