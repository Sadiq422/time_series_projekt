import bootstrap 
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Tuple, Any
from paths import get_path

# ==========================================================
# KONFIGURATION UND SETUP
# ==========================================================
st.set_page_config(
    page_title="📊 LSTM Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Globale Variablen
TARGET_COL = "unit_sales"
TIME_STEPS = 30

# NEUE KONTRASTREICHE FARBPALETTE
COLORS = {
    "actual": "#1E88E5",       # Kräftiges Blau für tatsächliche Werte
    "forecast": "#FF6B35",     # Kräftiges Orange für Vorhersagen
    "future": "#D32F2F",       # Kräftiges Rot für Zukunftsprognosen
    "residuals": "#43A047",    # Kräftiges Grün für Residuen
    "positive": "#4CAF50",     # Grün für positive Werte
    "negative": "#F44336",     # Rot für negative Werte
    "warning": "#FF9800",      # Orange für Warnungen
    "neutral": "#757575",      # Grau für neutrale Elemente
    "background": "#FFFFFF",   # Weißer Hintergrund
    "grid": "#E0E0E0",         # Helles Grau für Grid
    "text": "#212121"          # Dunkelgrau für Text
}

# ==========================================================
# LADE MODELL UND SCALER
# ==========================================================
@st.cache_resource
def load_models():
    """Lade alle trainierten Modelle mit Caching"""
    try:
        model_dir = get_path("lstm_model")
        metrics_dir = get_path("lstm_results")
        
        # LSTM Modell
        model_path = os.path.join(model_dir, "lstm_model.h5")
        lstm_model = tf.keras.models.load_model(model_path)
        
        # Scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        scaler_data = joblib.load(scaler_path)
        
        # Handle different scaler formats
        if isinstance(scaler_data, dict):
            scaler = scaler_data.get('scaler', scaler_data)
            feature_names = scaler_data.get('feature_names', None)
        else:
            scaler = scaler_data
            feature_names = None
        
        # Metriken
        metrics_path = os.path.join(metrics_dir, "lstm_metrics.csv")
        metrics_df = pd.read_csv(metrics_path)
        
        st.session_state['models_loaded'] = True
        return lstm_model, scaler, metrics_df, feature_names
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Modelle: {str(e)}")
        return None, None, None, None

# ==========================================================
# DATENVERARBEITUNG
# ==========================================================
def load_and_prepare_data():
    """Lade und bereite die Daten vor"""
    FEATURE_FILE = "/Volumes/Expansion/time_series_projekt_daten/corporación_favorita_grocery_sales_forecasting/data/processed/filtered/train_features__MAXDATE-2014-04-01__STORE-24__ITEM-105577.csv"
    
    if os.path.exists(FEATURE_FILE):
        try:
            df = pd.read_csv(FEATURE_FILE)
            
            # Überprüfe Spalten
            st.success(f"✅ Datei geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
            
            # Stelle sicher, dass wir die richtigen Spalten haben
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                # Prüfe ob wir die Zielvariable haben
                if TARGET_COL not in df.columns:
                    # Suche nach alternativen Spaltennamen
                    sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'unit' in col.lower()]
                    if sales_cols:
                        df[TARGET_COL] = df[sales_cols[0]]
                        st.warning(f"⚠️ Verwende '{sales_cols[0]}' als Zielvariable")
                    else:
                        st.error("❌ Keine Verkaufsdaten gefunden!")
                        return None
                
                return df
            else:
                st.error("❌ 'date' Spalte nicht gefunden!")
                return None
                
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Datei: {str(e)}")
            return None
    else:
        # Erstelle Beispieldaten für Demo
        st.warning("⚠️ Datei nicht gefunden. Erstelle Beispieldaten...")
        dates = pd.date_range(start='2014-01-01', periods=100, freq='D')
        
        # Realistischere Beispieldaten
        np.random.seed(42)
        base_sales = 50
        trend = np.linspace(0, 20, 100)
        seasonality = 10 * np.sin(2 * np.pi * np.arange(100) / 30)
        noise = np.random.normal(0, 5, 100)
        
        sales = base_sales + trend + seasonality + noise
        sales = np.maximum(sales, 0)
        
        df = pd.DataFrame({
            'date': dates,
            'unit_sales': sales,
            'store_nbr': 24,
            'item_nbr': 105577,
            'onpromotion': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })
        
        return df

def create_sequences(data, time_steps=TIME_STEPS):
    """Erstelle Sequenzen für LSTM"""
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ==========================================================
# VORHERSAGEFUNKTIONEN
# ==========================================================
def make_historical_predictions(df, model, scaler):
    """Mache historische Vorhersagen"""
    # Skaliere Daten
    sales_data = df[TARGET_COL].values.reshape(-1, 1)
    sales_scaled = scaler.transform(sales_data)
    
    # Erstelle Sequenzen
    X_seq, y_true = create_sequences(sales_scaled.flatten())
    X_seq = X_seq.reshape((X_seq.shape[0], TIME_STEPS, 1))
    
    # Mache Vorhersagen
    y_pred_scaled = model.predict(X_seq, verbose=0)
    
    # Rücktransformation
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    
    # Füge NaN für die ersten TIME_STEPS Tage hinzu
    predictions = np.full(len(df), np.nan)
    predictions[TIME_STEPS:] = y_pred
    
    return predictions

def make_future_predictions(df, model, scaler, days=30):
    """Mache Zukunftsprognosen"""
    # Letzte TIME_STEPS Tage als Basis
    sales_data = df[TARGET_COL].values.reshape(-1, 1)
    sales_scaled = scaler.transform(sales_data)
    
    last_sequence = sales_scaled[-TIME_STEPS:].flatten()
    future_predictions = []
    
    for _ in range(days):
        input_seq = last_sequence.reshape((1, TIME_STEPS, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
        future_predictions.append(pred_scaled)
        last_sequence = np.append(last_sequence[1:], pred_scaled)
    
    # Rücktransformation
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()
    
    return future_predictions

# ==========================================================
# VISUALISIERUNGSFUNKTIONEN MIT BESSEREN FARBEN
# ==========================================================
def plot_actual_vs_forecast(df):
    """Plot tatsächliche vs. vorhergesagte Werte mit kräftigen Farben"""
    fig = go.Figure()
    
    # Tatsächliche Werte (Kräftiges Blau)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[TARGET_COL],
        name='Tatsächliche Verkäufe',
        line=dict(color=COLORS["actual"], width=3, shape='linear'),
        mode='lines',
        opacity=0.9,
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Verkäufe:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Vorhersagen (Kräftiges Orange)
    mask = ~df['forecast'].isna()
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'date'],
            y=df.loc[mask, 'forecast'],
            name='Vorhersage',
            line=dict(color=COLORS["forecast"], width=3, dash='solid'),
            mode='lines',
            opacity=0.9,
            hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Vorhersage:</b> %{y:.0f}<extra></extra>'
        ))
    
    # Layout mit besserem Kontrast
    fig.update_layout(
        title=dict(
            text='📊 Tatsächliche vs. Vorhergesagte Verkäufe',
            font=dict(size=20, color=COLORS["text"])
        ),
        xaxis=dict(
            title='Datum',
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            linecolor=COLORS["grid"]
        ),
        yaxis=dict(
            title='Anzahl Verkäufe',
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"]
        ),
        hovermode='x unified',
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS["grid"],
            borderwidth=1
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def plot_residuals(df):
    """Plot Residuen mit verbesserten Farben"""
    mask = ~df['forecast'].isna()
    
    fig = go.Figure()
    
    # Residuen (Kräftiges Grün)
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'date'],
        y=df.loc[mask, 'residual'],
        name='Residuen',
        line=dict(color=COLORS["residuals"], width=2.5),
        mode='lines+markers',
        marker=dict(size=6, color=COLORS["residuals"]),
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Fehler:</b> %{y:.1f}<extra></extra>'
    ))
    
    # Null-Linie mit besserer Sichtbarkeit
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_width=2,
        line_color=COLORS["neutral"],
        opacity=0.8,
        annotation_text="Perfekte Vorhersage",
        annotation_font=dict(color=COLORS["text"])
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text='📉 Vorhersagefehler (Residuen)',
            font=dict(size=20, color=COLORS["text"])
        ),
        xaxis=dict(
            title='Datum',
            gridcolor=COLORS["grid"],
            linecolor=COLORS["grid"]
        ),
        yaxis=dict(
            title='Fehler (Tatsächlich - Vorhersage)',
            gridcolor=COLORS["grid"]
        ),
        hovermode='x unified',
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        height=400
    )
    
    return fig

def plot_error_distribution(df):
    """Plot Fehlerverteilung mit besseren Farben"""
    mask = ~df['forecast'].isna()
    
    fig = go.Figure()
    
    # Fehlerverteilung (Grün mit Transparenz)
    fig.add_trace(go.Histogram(
        x=df.loc[mask, 'residual'],
        nbinsx=30,
        name='Fehlerverteilung',
        marker_color=COLORS["residuals"],
        opacity=0.85,
        hovertemplate='<b>Fehlerbereich:</b> %{x:.1f}<br><b>Anzahl:</b> %{y}<extra></extra>'
    ))
    
    # Mittellinie für Durchschnitt
    mean_error = df.loc[mask, 'residual'].mean()
    fig.add_vline(
        x=mean_error,
        line_dash="dash",
        line_width=3,
        line_color=COLORS["warning"],
        annotation=dict(
            text=f"Durchschnitt: {mean_error:.2f}",
            font=dict(color=COLORS["warning"]),
            bgcolor="white",
            bordercolor=COLORS["warning"],
            borderwidth=1
        ),
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(
            text='📦 Verteilung der Vorhersagefehler',
            font=dict(size=20, color=COLORS["text"])
        ),
        xaxis=dict(
            title='Fehlerwert',
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title='Häufigkeit',
            gridcolor=COLORS["grid"]
        ),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        bargap=0.1,
        height=400
    )
    
    return fig

def plot_future_forecast(df, future_dates, future_predictions):
    """Plot Zukunftsprognose mit klaren Farben"""
    fig = go.Figure()
    
    # Historische Daten (Blau)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[TARGET_COL],
        name='Historische Verkäufe',
        line=dict(color=COLORS["actual"], width=3),
        mode='lines',
        opacity=0.9,
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Verkäufe:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Zukunftsprognose (Rot)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name='Zukunftsprognose',
        line=dict(color=COLORS["future"], width=3.5, dash='dash'),
        mode='lines',
        opacity=0.9,
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Prognose:</b> %{y:.0f}<extra></extra>'
    ))
    
    # Vertikale Linie für Übergang
    last_historical_date = df['date'].iloc[-1]
    fig.add_vline(
        x=last_historical_date,
        line_dash="dash",
        line_width=3,
        line_color=COLORS["neutral"],
        opacity=0.9,
        annotation=dict(
            text="Heute",
            font=dict(color=COLORS["text"], size=14),
            bgcolor="white",
            bordercolor=COLORS["neutral"]
        ),
        annotation_position="top left"
    )
    
    # Bereich für Zukunft markieren
    fig.add_vrect(
        x0=last_historical_date,
        x1=future_dates[-1],
        fillcolor="rgba(211, 47, 47, 0.1)",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation=dict(
            text="Zukunftsprognose",
            font=dict(color=COLORS["future"], size=12)
        ),
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=dict(
            text=f'🔮 Zukunftsprognose für nächste {len(future_dates)} Tage',
            font=dict(size=20, color=COLORS["text"])
        ),
        xaxis=dict(
            title='Datum',
            gridcolor=COLORS["grid"]
        ),
        yaxis=dict(
            title='Anzahl Verkäufe',
            gridcolor=COLORS["grid"]
        ),
        hovermode='x unified',
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        font=dict(color=COLORS["text"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=COLORS["grid"],
            borderwidth=1
        ),
        height=500
    )
    
    return fig

# ==========================================================
# MODERNES LAYOUT FÜR METRIKEN
# ==========================================================
def create_metric_card(title, value, delta=None, color="primary"):
    """Erstelle eine schöne Metrik-Karte"""
    colors = {
        "primary": COLORS["actual"],
        "secondary": COLORS["forecast"],
        "accent": COLORS["residuals"],
        "warning": COLORS["warning"]
    }
    
    delta_color = "normal"
    if delta:
        if delta.startswith('-'):
            delta_color = "inverse"
        elif float(delta.replace('-', '').replace('+', '')) > 0:
            delta_color = "normal"
    
    return st.metric(
        label=title,
        value=value,
        delta=delta,
        delta_color=delta_color
    )

def display_metrics(metrics_df, df=None):
    """Zeige Metriken in einem modernen Layout"""
    st.markdown("### 📊 Modell Performance")
    
    if df is not None and 'forecast' in df.columns:
        mask = ~df['forecast'].isna()
        if mask.any():
            # Aktuelle Metriken berechnen
            residuals = df.loc[mask, 'residual']
            current_mae = np.abs(residuals).mean()
            current_rmse = np.sqrt((residuals**2).mean())
            current_mape = np.mean(np.abs(residuals / df.loc[mask, TARGET_COL].replace(0, 1))) * 100
            
            # Trainingsmetriken
            train_mae = metrics_df.loc[metrics_df.Metric == 'MAE', 'Value'].values[0]
            train_rmse = metrics_df.loc[metrics_df.Metric == 'RMSE', 'Value'].values[0]
            train_r2 = metrics_df.loc[metrics_df.Metric == 'R2', 'Value'].values[0]
            
            # Metriken in 4 Spalten
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_metric_card(
                    "MAE",
                    f"{train_mae:.3f}",
                    f"{(current_mae - train_mae):+.3f}" if current_mae else None,
                    "primary"
                )
            
            with col2:
                create_metric_card(
                    "RMSE",
                    f"{train_rmse:.3f}",
                    f"{(current_rmse - train_rmse):+.3f}" if current_rmse else None,
                    "secondary"
                )
            
            with col3:
                create_metric_card(
                    "R² Score",
                    f"{train_r2:.3f}",
                    None,
                    "accent"
                )
            
            with col4:
                create_metric_card(
                    "MAPE",
                    f"{current_mape:.1f}%",
                    None,
                    "warning"
                )
        else:
            # Nur Trainingsmetriken anzeigen
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_metric_card("MAE", f"{metrics_df.loc[metrics_df.Metric == 'MAE', 'Value'].values[0]:.3f}")
            
            with col2:
                create_metric_card("RMSE", f"{metrics_df.loc[metrics_df.Metric == 'RMSE', 'Value'].values[0]:.3f}")
            
            with col3:
                create_metric_card("R² Score", f"{metrics_df.loc[metrics_df.Metric == 'R2', 'Value'].values[0]:.3f}")
    else:
        # Nur Trainingsmetriken anzeigen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("MAE", f"{metrics_df.loc[metrics_df.Metric == 'MAE', 'Value'].values[0]:.3f}")
        
        with col2:
            create_metric_card("RMSE", f"{metrics_df.loc[metrics_df.Metric == 'RMSE', 'Value'].values[0]:.3f}")
        
        with col3:
            create_metric_card("R² Score", f"{metrics_df.loc[metrics_df.Metric == 'R2', 'Value'].values[0]:.3f}")

# ==========================================================
# MODERNE SIDEBAR
# ==========================================================
def create_sidebar():
    """Erstelle eine moderne Sidebar"""
    with st.sidebar:
        # Logo und Titel
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: {COLORS["actual"]};'>📈 LSTM Forecast</h1>
            <p style='color: {COLORS["text"]};'>Predictive Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Store und Item Konfiguration
        st.markdown("### 🏪 Store & Item")
        col1, col2 = st.columns(2)
        with col1:
            store_id = st.number_input(
                "Store ID",
                min_value=1,
                value=24,
                help="ID des Geschäfts"
            )
        with col2:
            item_id = st.number_input(
                "Item ID",
                min_value=1,
                value=105577,
                help="ID des Artikels"
            )
        
        st.divider()
        
        # Datei Upload mit besserem Styling
        st.markdown("### 📁 Datenquelle")
        uploaded_file = st.file_uploader(
            "CSV-Datei hochladen",
            type=["csv"],
            help="Lade eine CSV-Datei mit Verkaufsdaten hoch"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
        
        st.divider()
        
        # Prognose-Einstellungen
        st.markdown("### 🔮 Prognose")
        forecast_days = st.slider(
            "Tage für Zukunftsprognose",
            min_value=7,
            max_value=90,
            value=30,
            help="Anzahl der Tage, die in die Zukunft prognostiziert werden sollen"
        )
        
        st.divider()
        
        # Farblegende
        st.markdown("### 🎨 Farblegende")
        st.markdown(f"""
        <div style='background-color: {COLORS["background"]}; padding: 15px; border-radius: 10px;'>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["actual"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Tatsächliche Verkäufe</span>
            </div>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["forecast"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Historische Vorhersagen</span>
            </div>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["future"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Zukunftsprognosen</span>
            </div>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["residuals"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Vorhersagefehler</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Footer
        st.markdown(f"""
        <div style='text-align: center; color: {COLORS["neutral"]}; font-size: 0.8em; padding-top: 20px;'>
            <p>LSTM Forecast Dashboard v1.0</p>
            <p>© {datetime.now().year} - Alle Rechte vorbehalten</p>
        </div>
        """, unsafe_allow_html=True)
        
        return store_id, item_id, forecast_days

# ==========================================================
# MODERNES DATEN-PREVIEW
# ==========================================================
def display_data_preview(df):
    """Zeige Datenvorschau in einem modernen Design"""
    with st.expander("📋 Datenübersicht", expanded=True):
        # Statistiken in Karten
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS["actual"]}, #64B5F6); 
                        padding: 20px; border-radius: 10px; color: white;'>
                <h3 style='margin: 0; font-size: 14px;'>Zeitraum</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold;'>
                    {df['date'].min().date()} - {df['date'].max().date()}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS["forecast"]}, #FFB74D); 
                        padding: 20px; border-radius: 10px; color: white;'>
                <h3 style='margin: 0; font-size: 14px;'>Anzahl Tage</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold;'>
                    {len(df)}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_sales = df[TARGET_COL].mean()
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS["residuals"]}, #81C784); 
                        padding: 20px; border-radius: 10px; color: white;'>
                <h3 style='margin: 0; font-size: 14px;'>Durchschnitt</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold;'>
                    {avg_sales:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            std_sales = df[TARGET_COL].std()
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {COLORS["future"]}, #E57373); 
                        padding: 20px; border-radius: 10px; color: white;'>
                <h3 style='margin: 0; font-size: 14px;'>Standardabw.</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold;'>
                    {std_sales:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Datenvorschau
        st.markdown("#### Datenvorschau (erste 10 Zeilen)")
        st.dataframe(
            df.head(10).style.background_gradient(
                subset=[TARGET_COL],
                cmap='Blues'
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn("Datum", format="DD.MM.YYYY"),
                TARGET_COL: st.column_config.NumberColumn("Verkäufe", format="%.0f"),
                "store_nbr": st.column_config.NumberColumn("Store"),
                "item_nbr": st.column_config.NumberColumn("Item")
            }
        )

# ==========================================================
# MODERNE APP
# ==========================================================
# ==========================================================
# MODERNE APP (Überarbeitet für Screenshot-Layout)
# ==========================================================
def main():
    """Hauptfunktion der App - Überarbeitet für Screenshot-Layout"""
    
    # Header mit modernem Design
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {COLORS["actual"]}, #0D47A1); 
                padding: 40px 20px; border-radius: 0 0 20px 20px; margin-bottom: 30px;'>
        <div style='max-width: 1200px; margin: 0 auto; text-align: center; color: white;'>
            <h1 style='font-size: 2.5em; margin-bottom: 10px;'>LSTM Forecast</h1>
            <p style='font-size: 1.2em; opacity: 0.9;'>Predictive Analytics Dashboard</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar erstellen
    store_id, item_id, forecast_days = create_sidebar()
    
    # Hauptinhalt
    main_container = st.container()
    
    with main_container:
        # Lade Modelle
        lstm_model, scaler, metrics_df, _ = load_models()
        
        if lstm_model is None or scaler is None:
            st.error("❌ Modelle konnten nicht geladen werden. Bitte überprüfe die Pfade.")
            return
        
        # Lade Daten
        with st.spinner("📂 Lade und verarbeite Daten..."):
            df = load_and_prepare_data()
        
        if df is None:
            st.error("❌ Daten konnten nicht geladen werden.")
            return
        
        # Metriken anzeigen (kompakter)
        col1, col2, col3 = st.columns(3)
        with col1:
            if metrics_df is not None:
                create_metric_card("MAE", f"{metrics_df.loc[metrics_df.Metric == 'MAE', 'Value'].values[0]:.3f}")
        with col2:
            if metrics_df is not None:
                create_metric_card("RMSE", f"{metrics_df.loc[metrics_df.Metric == 'RMSE', 'Value'].values[0]:.3f}")
        with col3:
            if metrics_df is not None:
                create_metric_card("R² Score", f"{metrics_df.loc[metrics_df.Metric == 'R2', 'Value'].values[0]:.3f}")
        
        st.divider()
        
        # Historische Vorhersage - WIE IM SCREENSHOT
        st.markdown("## Historische Vorhersage")
        
        # Container für Vorhersage-Button und Erfolgsmeldung
        forecast_container = st.container()
        with forecast_container:
            st.info("Klicke auf 'Vorhersage starten' um historische Vorhersagen zu generieren.")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Größerer, zentrierter Button
                if st.button("🚀 **Vorhersage starten**", 
                           type="primary", 
                           use_container_width=True,
                           key="forecast_button_main"):
                    st.session_state['run_forecast'] = True
        
        # Vorhersage-Logik
        if st.session_state.get('run_forecast', False):
            with st.spinner("Berechne Vorhersagen..."):
                try:
                    # Mache Vorhersagen
                    predictions = make_historical_predictions(df, lstm_model, scaler)
                    
                    # Füge Vorhersagen zum DataFrame hinzu
                    df['forecast'] = predictions
                    
                    # Berechne Residuen
                    mask = ~df['forecast'].isna()
                    forecast_days_count = mask.sum()
                    df.loc[mask, 'residual'] = df.loc[mask, TARGET_COL] - df.loc[mask, 'forecast']
                    
                    # Erfolgsmeldung WIE IM SCREENSHOT
                    st.success(f"✅ **Vorhersage erfolgreich! ({forecast_days_count} Tage prognostiziert)**")
                    
                    # Zwei Spalten für Diagramme wie im Screenshot
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Verkaufsverlauf (vereinfacht)
                        st.markdown("#### Verkaufsverlauf")
                        fig_simple = go.Figure()
                        fig_simple.add_trace(go.Scatter(
                            x=df['date'],
                            y=df[TARGET_COL],
                            name='Verkäufe',
                            line=dict(color=COLORS["actual"], width=2),
                            mode='lines'
                        ))
                        fig_simple.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20),
                            plot_bgcolor=COLORS["background"],
                            paper_bgcolor=COLORS["background"]
                        )
                        st.plotly_chart(fig_simple, use_container_width=True)
                    
                    with col2:
                        # Tatsächliche vs. Vorhergesagte Verkäufe WIE IM SCREENSHOT
                        st.markdown("#### Tatsächliche vs. Vorhergesagte Verkäufe")
                        
                        # Erstelle einen einfachen Plot wie im Screenshot
                        fig_comparison = go.Figure()
                        
                        # Tatsächliche Verkäufe
                        fig_comparison.add_trace(go.Scatter(
                            x=df['date'],
                            y=df[TARGET_COL],
                            name='Tatsächliche Verkäufe',
                            line=dict(color=COLORS["actual"], width=3),
                            mode='lines'
                        ))
                        
                        # Vorhersagen
                        if 'forecast' in df.columns:
                            mask = ~df['forecast'].isna()
                            fig_comparison.add_trace(go.Scatter(
                                x=df.loc[mask, 'date'],
                                y=df.loc[mask, 'forecast'],
                                name='Vorhersage',
                                line=dict(color=COLORS["forecast"], width=3, dash='dash'),
                                mode='lines'
                            ))
                        
                        # Layout wie im Screenshot
                        fig_comparison.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20),
                            plot_bgcolor=COLORS["background"],
                            paper_bgcolor=COLORS["background"],
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        # Y-Achse wie im Screenshot (mit festen Werten)
                        fig_comparison.update_yaxes(
                            range=[df[TARGET_COL].min() * 0.8, df[TARGET_COL].max() * 1.2],
                            tickvals=[60, 80, 100]  # Beispielwerte wie im Screenshot
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Tabs für detaillierte Analyse (unterhalb)
                    st.divider()
                    st.markdown("### 📊 Detaillierte Analyse")
                    
                    tab1, tab2, tab3 = st.tabs(["Vollständiger Verlauf", "Fehleranalyse", "Daten"])
                    
                    with tab1:
                        st.plotly_chart(plot_actual_vs_forecast(df), use_container_width=True)
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(plot_residuals(df), use_container_width=True)
                        with col2:
                            st.plotly_chart(plot_error_distribution(df), use_container_width=True)
                    
                    with tab3:
                        st.dataframe(
                            df[['date', TARGET_COL, 'forecast', 'residual']].tail(20),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    st.divider()
                    
                    # Zukunftsprognose
                    st.markdown("## 🔮 Zukunftsprognose")
                    
                    future_container = st.container()
                    with future_container:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if st.button("🚀 **Zukunftsprognose generieren**", 
                                       type="secondary", 
                                       use_container_width=True,
                                       key="future_forecast_button"):
                                st.session_state['run_future_forecast'] = True
                    
                    if st.session_state.get('run_future_forecast', False):
                        with st.spinner("Berechne Zukunftsprognose..."):
                            future_predictions = make_future_predictions(
                                df, lstm_model, scaler, days=forecast_days
                            )
                            
                            # Zukünftige Datumsliste
                            last_date = df['date'].iloc[-1]
                            future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
                            
                            # Plot
                            st.plotly_chart(
                                plot_future_forecast(df, future_dates, future_predictions),
                                use_container_width=True
                            )
                            
                            # Prognose-Tabelle
                            with st.expander("📋 Detaillierte Prognosetabelle"):
                                st.dataframe(
                                    pd.DataFrame({
                                        'Datum': [d.strftime('%d.%m.%Y') for d in future_dates],
                                        'Prognose': future_predictions.round(1)
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                
                except Exception as e:
                    st.error(f"❌ Fehler bei der Vorhersage: {str(e)}")
        
        # Datenübersicht (am Ende)
        st.divider()
        display_data_preview(df)
    
    # Footer
    st.markdown(f"""
    <div style='background-color: {COLORS["background"]}; padding: 20px; border-radius: 10px; margin-top: 50px;'>
        <div style='text-align: center; color: {COLORS["text"]};'>
            <p style='font-weight: bold;'>LSTM Forecast Dashboard | Store {store_id} | Item {item_id}</p>
            <p style='font-size: 0.9em; opacity: 0.7;'>
                Letzte Aktualisierung: {datetime.now().strftime("%d.%m.%Y %H:%M")}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# CSS VERBESSERUNGEN FÜR BUTTONS
# ==========================================================
def apply_custom_css():
    """Wende benutzerdefiniertes CSS an - Verbessert für Screenshot"""
    st.markdown(f"""
    <style>
    /* Hauptcontainer */
    .main .block-container {{
        padding-top: 0;
        padding-bottom: 2rem;
    }}
    
    /* Primäre Buttons (wie im Screenshot) */
    div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, {COLORS["forecast"]}, #E65100);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }}
    
    div[data-testid="stButton"] > button[kind="primary"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #E65100, {COLORS["forecast"]});
    }}
    
    /* Sekundäre Buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {{
        background: linear-gradient(135deg, {COLORS["actual"]}, #1565C0);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 3px 5px rgba(0,0,0,0.1);
        width: 100%;
    }}
    
    /* Erfolgsmeldungen hervorheben */
    .stSuccess {{
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid {COLORS["positive"]};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Info-Boxen */
    .stInfo {{
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 4px solid {COLORS["actual"]};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Trennlinien */
    .stDivider {{
        margin: 2rem 0;
    }}
    
    /* Metriken Karten */
    [data-testid="stMetric"] {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {COLORS["grid"]};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Diagramm-Container */
    .js-plotly-plot {{
        border-radius: 10px;
        border: 1px solid {COLORS["grid"]};
        padding: 10px;
        background-color: white;
    }}
    
    /* Sidebar Anpassungen */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["background"]};
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS["background"]};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS["neutral"]};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS["actual"]};
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# CSS FÜR MODERNES DESIGN
# ==========================================================
def apply_custom_css():
    """Wende benutzerdefiniertes CSS an"""
    st.markdown(f"""
    <style>
    /* Hauptcontainer */
    .main .block-container {{
        padding-top: 0;
        padding-bottom: 2rem;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS["actual"]}, #0D47A1);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #0D47A1, {COLORS["actual"]});
    }}
    
    /* DataFrames */
    .dataframe {{
        border-radius: 10px;
        border: 1px solid {COLORS["grid"]};
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["background"]};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLORS["background"]};
        padding: 10px;
        border-radius: 10px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 8px;
        padding: 12px 24px;
        border: 1px solid {COLORS["grid"]};
        transition: all 0.3s;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS["actual"]};
        color: white !important;
        border-color: {COLORS["actual"]};
    }}
    
    /* Metriken */
    [data-testid="stMetric"] {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid {COLORS["grid"]};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {COLORS["background"]};
        border-radius: 10px;
        border: 1px solid {COLORS["grid"]};
        font-weight: bold;
        color: {COLORS["text"]};
    }}
    
    /* Slider */
    .stSlider > div > div > div {{
        background-color: {COLORS["actual"]};
    }}
    
    /* Input Felder */
    .stNumberInput > div > div > input {{
        border: 2px solid {COLORS["grid"]};
        border-radius: 8px;
    }}
    
    .stNumberInput > div > div > input:focus {{
        border-color: {COLORS["actual"]};
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================================
# APP STARTEN
# ==========================================================
if __name__ == "__main__":
    # Session State initialisieren
    if 'run_forecast' not in st.session_state:
        st.session_state.run_forecast = False
    
    # CSS anwenden
    apply_custom_css()
    
    # App ausführen
    main()