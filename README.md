# Time Series Forecasting Dashboard — Absatzprognose Einzelhandel

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

End-to-End-Machine-Learning-Pipeline zur Verkaufsprognose im Einzelhandel, kombiniert mit einem interaktiven Streamlit-Dashboard (Dark Mode). Klassische Statistik (ARIMA), Gradient Boosting (XGBoost) und Deep Learning (LSTM) in einem System.

> 👥 Teamprojekt von **Sadiq Qais** und **Claudia Tagbo**.

---

## 🎯 Problemstellung

Verlässliche Absatzprognosen pro Geschäft und Artikel sind die Basis für Einkauf, Lagerhaltung und Planung. Ziel: robuste Vorhersagen über ein Modell-Ensemble plus ein Dashboard, das Prognosen und Unsicherheit verständlich macht.

## 🧪 Methoden

- **Feature Engineering:** Lag-Features, Rolling-Statistiken, saisonale Dekomposition.
- **Modelle:** ARIMA (statistische Baseline), XGBoost (tabellarisch), LSTM (Sequenzen) — als Ensemble kombiniert.
- **Evaluation:** MAE, RMSE, R², MAPE; Residuen-Analyse und Konfidenzintervalle.

## 🖥️ Dashboard

Streamlit-App mit:
- Auswahl von Geschäft und Artikel sowie Prognosehorizont (1–52 Wochen),
- Echtzeit-Forecasts mit Konfidenzintervallen,
- Performance-Metriken und Residuen-Diagnostik,
- CSV-Export der Prognosen.

## 🛠️ Setup

```bash
git clone https://github.com/Sadiq422/time_series_projekt.git
cd time_series_projekt
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Forschungs-Pipeline in `notebooks/`: Preprocessing → Feature Engineering → ARIMA / XGBoost / LSTM.

## 🏗️ Projektstruktur

```
app/         # Streamlit-Dashboard (app.py) + UI/Bootstrap
notebooks/   # preprocessing, feature-engineering, arima, xgboost, lstm
utils.py     # Helper-Funktionen
visualizer.py# Plotting-Engine
paths.py     # zentrale Pfadverwaltung
```

## 🧰 Tech Stack

Python, TensorFlow/Keras (LSTM), XGBoost, statsmodels (ARIMA), scikit-learn, Streamlit, Plotly, pandas/NumPy.

## 👥 Autoren

**Sadiq Qais** ([LinkedIn](https://www.linkedin.com/in/sadiq-qais)) · **Claudia Tagbo**

## 📄 Lizenz

MIT — siehe `LICENSE`.
