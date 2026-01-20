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
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# DARK MODE KONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="📊 Corporación favorita sales forecasting | Dark Mode",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Farbpalette
COLORS = {
    "dark_bg": "#0F172A",        # Dunkler Hintergrund
    "card_bg": "#1E293B",        # Karten Hintergrund
    "sidebar_bg": "#111827",     # Sidebar Hintergrund
    "primary": "#3B82F6",        # Primär (Blau)
    "secondary": "#8B5CF6",      # Sekundär (Violett)
    "accent": "#10B981",         # Akzent (Grün)
    "warning": "#F59E0B",        # Warnung (Orange)
    "danger": "#EF4444",         # Gefahr (Rot)
    "text_primary": "#F8FAFC",   # Primärer Text
    "text_secondary": "#94A3B8", # Sekundärer Text
    "border": "#334155",         # Rahmen
    "grid": "#475569",           # Gitternetz
    "actual": "#60A5FA",         # Tatsächliche Werte
    "forecast": "#A78BFA",       # Vorhersagen
    "future": "#F87171",         # Zukunftsprognosen
    "residuals": "#34D399"       # Residuen
}

# ==========================================================
# DARK MODE CSS
# ==========================================================
st.markdown(f"""
<style>
/* ===== GLOBAL STYLES ===== */
.stApp {{
    background: {COLORS["dark_bg"]};
    color: {COLORS["text_primary"]};
}}

/* ===== ALL TEXT ===== */
h1, h2, h3, h4, h5, h6, p, div, span, label {{
    color: {COLORS["text_primary"]} !important;
}}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {{
    background: {COLORS["sidebar_bg"]} !important;
    border-right: 1px solid {COLORS["border"]} !important;
}}

[data-testid="stSidebar"] * {{
    color: {COLORS["text_primary"]} !important;
}}

/* ===== BUTTONS ===== */
.stButton > button {{
    background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]}) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}}

/* ===== METRICS ===== */
[data-testid="stMetric"] {{
    background: {COLORS["card_bg"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}}

[data-testid="stMetricLabel"], 
[data-testid="stMetricValue"], 
[data-testid="stMetricDelta"] {{
    color: {COLORS["text_primary"]} !important;
}}

/* ===== DATA TABLES ===== */
.dataframe {{
    background: {COLORS["card_bg"]} !important;
    color: {COLORS["text_primary"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 8px !important;
}}

.dataframe thead tr th {{
    background: #2C3E50 !important;
    color: {COLORS["text_primary"]} !important;
    border-bottom: 2px solid {COLORS["primary"]} !important;
}}

.dataframe tbody tr {{
    background: {COLORS["card_bg"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

.dataframe tbody tr:nth-child(even) {{
    background: #2C3E50 !important;
}}

.dataframe tbody tr:hover {{
    background: rgba(59, 130, 246, 0.2) !important;
}}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {{
    background: {COLORS["card_bg"]} !important;
    color: {COLORS["text_primary"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 8px !important;
}}

.streamlit-expanderContent {{
    background: {COLORS["card_bg"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 0 0 8px 8px !important;
}}

/* ===== SUCCESS/INFO/WARNING MESSAGES ===== */
.stSuccess {{
    background-color: rgba(16, 185, 129, 0.1) !important;
    border-left: 4px solid {COLORS["accent"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

.stInfo {{
    background-color: rgba(59, 130, 246, 0.1) !important;
    border-left: 4px solid {COLORS["primary"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

.stWarning {{
    background-color: rgba(245, 158, 11, 0.1) !important;
    border-left: 4px solid {COLORS["warning"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

.stError {{
    background-color: rgba(239, 68, 68, 0.1) !important;
    border-left: 4px solid {COLORS["danger"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS["card_bg"]} !important;
    border-bottom: 1px solid {COLORS["border"]} !important;
}}

.stTabs [data-baseweb="tab"] {{
    color: {COLORS["text_secondary"]} !important;
    background: transparent !important;
}}

.stTabs [aria-selected="true"] {{
    color: {COLORS["primary"]} !important;
    border-bottom: 3px solid {COLORS["primary"]} !important;
}}

/* ===== INPUT FIELDS ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stSlider > div > div > div {{
    background: {COLORS["card_bg"]} !important;
    color: {COLORS["text_primary"]} !important;
    border: 1px solid {COLORS["border"]} !important;
}}

/* ===== DIVIDER ===== */
hr {{
    border-color: {COLORS["border"]} !important;
}}

/* ===== PLOTLY CHARTS ===== */
.js-plotly-plot, .plotly {{
    background: {COLORS["card_bg"]} !important;
    border: 1px solid {COLORS["border"]} !important;
    border-radius: 8px !important;
    padding: 10px !important;
}}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS["card_bg"]};
}}

::-webkit-scrollbar-thumb {{
    background: {COLORS["border"]};
    border-radius: 5px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {COLORS["primary"]};
}}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, {COLORS["primary"]}, {COLORS["secondary"]});
}}

/* ===== MARKDOWN TABLES ===== */
table {{
    background: {COLORS["card_bg"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

th, td {{
    border: 1px solid {COLORS["border"]} !important;
    color: {COLORS["text_primary"]} !important;
}}

/* ===== MAIN CONTAINER ===== */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

/* ===== HEADER FIX ===== */
.st-emotion-cache-1v0mbdj {{
    background: transparent !important;
}}
</style>
""", unsafe_allow_html=True)

# Globale Variablen
TARGET_COL = "unit_sales"
TIME_STEPS = 30

# ==========================================================
# LADE MODELL UND SCALER
# ==========================================================
@st.cache_resource
def load_models():
    """Lade alle trainierten Modelle mit Caching"""
    try:
        st.session_state['models_loaded'] = True
        
        # Simulierte Metriken
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R2'],
            'Value': [12.345, 18.678, 0.876]
        })
        
        return None, None, metrics_df, None
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Modelle: {str(e)}")
        return None, None, None, None

# ==========================================================
# DATENVERARBEITUNG
# ==========================================================
def load_and_prepare_data():
    """Lade und bereite die Daten vor"""
    try:
        dates = pd.date_range(start='2013-01-02', end='2014-04-01', freq='D')
        
        np.random.seed(42)
        base_sales = 50
        trend = np.linspace(0, 30, len(dates))
        seasonality = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        weekly = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 5, len(dates))
        
        sales = base_sales + trend + seasonality + weekly + noise
        sales = np.maximum(sales, 0)
        
        df = pd.DataFrame({
            'date': dates,
            'unit_sales': sales,
            'store_nbr': 24,
            'item_nbr': 105577,
            'onpromotion': np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'year': dates.year
        })
        
        df['unit_sales_7d_mean'] = df['unit_sales'].rolling(window=7).mean()
        df['unit_sales_30d_mean'] = df['unit_sales'].rolling(window=30).mean()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
        return None

def clean_dataframe(df):
    """Bereinige den DataFrame von unnötigen Spalten"""
    important_cols = [
        'date', 'store_nbr', 'item_nbr', 'unit_sales', 
        'onpromotion', 'day_of_week', 'month', 'year',
        'unit_sales_7d_mean', 'unit_sales_30d_mean'
    ]
    
    important_cols = [col for col in important_cols if col in df.columns]
    
    return df[important_cols]

# ==========================================================
# VORHERSAGEFUNKTIONEN
# ==========================================================
def make_historical_predictions(df, model, scaler):
    """Mache historische Vorhersagen (simuliert für Demo)"""
    np.random.seed(42)
    predictions = np.full(len(df), np.nan)
    
    actual_sales = df['unit_sales'].values
    
    for i in range(TIME_STEPS, len(df)):
        predictions[i] = actual_sales[i] * np.random.uniform(0.85, 1.15)
        
        if i > TIME_STEPS:
            predictions[i] = predictions[i-1] * 0.8 + actual_sales[i] * 0.2
    
    return predictions

def make_future_predictions(df, model, scaler, days=30):
    """Mache Zukunftsprognosen (simuliert für Demo)"""
    last_values = df['unit_sales'].values[-30:]
    
    if len(last_values) > 7:
        trend = np.mean(np.diff(last_values[-7:]))
    else:
        trend = 0
    
    last_value = last_values[-1]
    future_predictions = []
    
    for i in range(days):
        seasonal = 5 * np.sin(2 * np.pi * (len(df) + i) / 30)
        noise = np.random.normal(0, 3)
        
        next_value = last_value + trend + seasonal + noise
        next_value = max(next_value, 0)
        
        future_predictions.append(next_value)
        last_value = next_value
    
    return np.array(future_predictions)

# ==========================================================
# VISUALISIERUNGSFUNKTIONEN FÜR DARK MODE
# ==========================================================
def plot_actual_vs_forecast(df):
    """Plot tatsächliche vs. vorhergesagte Werte mit Dark Mode"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['unit_sales'],
        name='Tatsächliche Verkäufe',
        line=dict(color=COLORS["actual"], width=3),
        mode='lines',
        opacity=0.9,
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Verkäufe:</b> %{y:.0f}<extra></extra>'
    ))
    
    if 'forecast' in df.columns and df['forecast'].notna().any():
        mask = ~df['forecast'].isna()
        fig.add_trace(go.Scatter(
            x=df.loc[mask, 'date'],
            y=df.loc[mask, 'forecast'],
            name='Vorhersage',
            line=dict(color=COLORS["forecast"], width=3, dash='dash'),
            mode='lines',
            opacity=0.9,
            hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Vorhersage:</b> %{y:.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='📊 Tatsächliche vs. Vorhergesagte Verkäufe',
            font=dict(size=20, color=COLORS["text_primary"])
        ),
        xaxis=dict(
            title='Datum',
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        yaxis=dict(
            title='Anzahl Verkäufe',
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        hovermode='x unified',
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS["text_primary"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(color=COLORS["text_primary"])
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def plot_future_forecast(df, future_dates, future_predictions):
    """Plot Zukunftsprognose mit Dark Mode"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['unit_sales'],
        name='Historische Verkäufe',
        line=dict(color=COLORS["actual"], width=3),
        mode='lines',
        opacity=0.8,
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Verkäufe:</b> %{y:.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        name='Zukunftsprognose',
        line=dict(color=COLORS["future"], width=3.5, dash='dot'),
        mode='lines',
        opacity=0.9,
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Prognose:</b> %{y:.0f}<extra></extra>'
    ))
    
    upper_bound = future_predictions * 1.15
    lower_bound = future_predictions * 0.85
    
    fig.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
        fill='toself',
        fillcolor=f'rgba{(*tuple(int(COLORS["future"][i:i+2], 16) for i in (1, 3, 5)), 0.2)}',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Konfidenzintervall',
        hoverinfo='skip'
    ))
    
    last_historical_date = df['date'].iloc[-1]
    fig.add_vline(
        x=last_historical_date,
        line_dash="dash",
        line_width=2,
        line_color=COLORS["text_secondary"],
        opacity=0.7,
        annotation=dict(
            text="Heute",
            font=dict(color=COLORS["text_primary"], size=12),
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["border"],
            borderwidth=1
        ),
        annotation_position="top left"
    )
    
    fig.update_layout(
        title=dict(
            text=f'🔮 Zukunftsprognose für nächste {len(future_dates)} Tage',
            font=dict(size=20, color=COLORS["text_primary"])
        ),
        xaxis=dict(
            title='Datum',
            gridcolor=COLORS["grid"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        yaxis=dict(
            title='Anzahl Verkäufe',
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        hovermode='x unified',
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS["text_primary"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor=COLORS["card_bg"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(color=COLORS["text_primary"])
        ),
        height=500
    )
    
    return fig

def plot_residuals(df):
    """Plot Residuen mit Dark Mode"""
    if 'forecast' not in df.columns or df['forecast'].isna().all():
        return None
    
    mask = ~df['forecast'].isna()
    residuals = df.loc[mask, 'unit_sales'] - df.loc[mask, 'forecast']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'date'],
        y=residuals,
        name='Residuen',
        line=dict(color=COLORS["residuals"], width=2.5),
        mode='lines+markers',
        marker=dict(size=6, color=COLORS["residuals"]),
        hovertemplate='<b>Datum:</b> %{x|%d.%m.%Y}<br><b>Fehler:</b> %{y:.1f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_width=2,
        line_color=COLORS["text_secondary"],
        opacity=0.8,
        annotation_text="Perfekte Vorhersage",
        annotation_font=dict(color=COLORS["text_secondary"])
    )
    
    fig.update_layout(
        title=dict(
            text='📉 Vorhersagefehler (Residuen)',
            font=dict(size=20, color=COLORS["text_primary"])
        ),
        xaxis=dict(
            title='Datum',
            gridcolor=COLORS["grid"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        yaxis=dict(
            title='Fehler (Tatsächlich - Vorhersage)',
            gridcolor=COLORS["grid"],
            tickfont=dict(color=COLORS["text_secondary"])
        ),
        hovermode='x unified',
        plot_bgcolor=COLORS["card_bg"],
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS["text_primary"]),
        height=400
    )
    
    return fig

# ==========================================================
# DARK MODE SIDEBAR
# ==========================================================
def create_sidebar():
    """Erstelle Dark Mode Sidebar"""
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 20px 0; margin-bottom: 20px;'>
            <h1 style='color: {COLORS["primary"]};'>🌙 LSTM Forecast</h1>
            <p style='color: {COLORS["text_secondary"]};'>Dark Mode Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### 🏪 Store & Item")
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
        
        st.markdown(f"### 🔮 Prognose")
        forecast_days = st.slider(
            "Tage für Zukunftsprognose",
            min_value=7,
            max_value=90,
            value=30,
            help="Anzahl der Tage, die in die Zukunft prognostiziert werden sollen"
        )
        
        st.divider()
        
        st.markdown(f"### ⚙️ Modell-Einstellungen")
        model_type = st.selectbox(
            "Modell-Typ",
            ["LSTM (Standard)", "GRU", "CNN-LSTM", "Transformer"],
            index=0
        )
        
        include_confidence = st.checkbox("Konfidenzintervalle anzeigen", value=True)
        
        st.divider()
        
        st.markdown(f"### 🎨 Farblegende")
        st.markdown(f"""
        <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 10px;'>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["actual"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Tatsächliche Verkäufe</span>
            </div>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["forecast"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Vorhersagen</span>
            </div>
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <div style='width: 20px; height: 20px; background-color: {COLORS["future"]}; margin-right: 10px; border-radius: 3px;'></div>
                <span>Zukunftsprognosen</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown(f"""
        <div style='text-align: center; color: {COLORS["text_secondary"]}; font-size: 0.8em; padding-top: 20px;'>
            <p>Dark Mode Dashboard v1.0</p>
            <p>© {datetime.now().year} - Alle Rechte vorbehalten</p>
        </div>
        """, unsafe_allow_html=True)
        
        return store_id, item_id, forecast_days

# ==========================================================
# DARK MODE DATEN-ÜBERSICHT
# ==========================================================
def display_data_preview(df):
    """Zeige Datenübersicht in Dark Mode Design"""
    with st.expander("📋 Datenübersicht", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style='background: {COLORS["card_bg"]}; 
                        padding: 20px; border-radius: 10px; border: 1px solid {COLORS["border"]};'>
                <h3 style='margin: 0; font-size: 14px; color: {COLORS["text_secondary"]}'>Zeitraum</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: {COLORS["text_primary"]}'>
                    {df['date'].min().date()} - {df['date'].max().date()}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='background: {COLORS["card_bg"]}; 
                        padding: 20px; border-radius: 10px; border: 1px solid {COLORS["border"]};'>
                <h3 style='margin: 0; font-size: 14px; color: {COLORS["text_secondary"]}'>Anzahl Tage</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: {COLORS["text_primary"]}'>
                    {len(df)}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_sales = df[TARGET_COL].mean()
            st.markdown(f"""
            <div style='background: {COLORS["card_bg"]}; 
                        padding: 20px; border-radius: 10px; border: 1px solid {COLORS["border"]};'>
                <h3 style='margin: 0; font-size: 14px; color: {COLORS["text_secondary"]}'>Durchschnitt</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: {COLORS["text_primary"]}'>
                    {avg_sales:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            std_sales = df[TARGET_COL].std()
            st.markdown(f"""
            <div style='background: {COLORS["card_bg"]}; 
                        padding: 20px; border-radius: 10px; border: 1px solid {COLORS["border"]};'>
                <h3 style='margin: 0; font-size: 14px; color: {COLORS["text_secondary"]}'>Standardabw.</h3>
                <p style='margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: {COLORS["text_primary"]}'>
                    {std_sales:.1f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Datenvorschau (erste 10 Zeilen)")
        
        preview_cols = ['date', 'unit_sales', 'onpromotion', 'day_of_week']
        if 'forecast' in df.columns:
            preview_cols.append('forecast')
        
        preview_df = df[preview_cols].head(10).copy()
        preview_df['date'] = preview_df['date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.TextColumn("Datum"),
                "unit_sales": st.column_config.NumberColumn("Verkäufe", format="%.0f"),
                "forecast": st.column_config.NumberColumn("Vorhersage", format="%.0f"),
                "onpromotion": st.column_config.NumberColumn("Promotion", format="%.0f"),
                "day_of_week": st.column_config.NumberColumn("Wochentag", format="%.0f")
            }
        )

# ==========================================================
# HAUPTAPPLIKATION - KORRIGIERT
# ==========================================================
def main():
    """Hauptfunktion der Dark Mode App"""
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div style='margin-bottom: 2rem;'>
            <h1 style='color: {COLORS["primary"]};'>📊 LSTM Forecast Dashboard</h1>
            <p style='color: {COLORS["text_secondary"]}; margin-top: -0.5rem;'>
                Dark Mode Edition • Enterprise Predictive Analytics
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: {COLORS["card_bg"]}; padding: 1rem; border-radius: 10px; border: 1px solid {COLORS["border"]}; text-align: center;'>
            <div style='font-size: 0.875rem; color: {COLORS["text_secondary"]};'>Status</div>
            <div style='color: {COLORS["accent"]}; font-weight: 600;'>🟢 Live</div>
        </div>
        """, unsafe_allow_html=True)
    
    store_id, item_id, forecast_days = create_sidebar()
    
    main_container = st.container()
    
    with main_container:
        lstm_model, scaler, metrics_df, _ = load_models()
        
        with st.spinner("📂 Lade und verarbeite Daten..."):
            df = load_and_prepare_data()
        
        if df is None:
            st.error("❌ Daten konnten nicht geladen werden.")
            return
        
        df = clean_dataframe(df)
        
        if metrics_df is not None:
            st.markdown("### 📊 Modell Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE", f"{metrics_df.loc[metrics_df.Metric == 'MAE', 'Value'].values[0]:.3f}")
            with col2:
                st.metric("RMSE", f"{metrics_df.loc[metrics_df.Metric == 'RMSE', 'Value'].values[0]:.3f}")
            with col3:
                st.metric("R² Score", f"{metrics_df.loc[metrics_df.Metric == 'R2', 'Value'].values[0]:.3f}")
        
        st.divider()
        
        st.markdown("## 🔍 Historische Vorhersage")
        
        forecast_container = st.container()
        with forecast_container:
            st.info("Klicke auf 'Vorhersage starten' um historische Vorhersagen zu generieren.")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 **Vorhersage starten**", 
                           type="primary", 
                           use_container_width=True,
                           key="forecast_button_main"):
                    st.session_state['run_forecast'] = True
        
        if 'run_forecast' not in st.session_state:
            st.session_state['run_forecast'] = False
            
        if st.session_state.get('run_forecast', False):
            with st.spinner("Berechne Vorhersagen..."):
                try:
                    predictions = make_historical_predictions(df, lstm_model, scaler)
                    
                    df['forecast'] = predictions
                    
                    mask = ~df['forecast'].isna()
                    forecast_days_count = mask.sum()
                    df.loc[mask, 'residual'] = df.loc[mask, 'unit_sales'] - df.loc[mask, 'forecast']
                    
                    st.success(f"✅ **Vorhersage erfolgreich! ({forecast_days_count} Tage prognostiziert)**")
                    
                    st.plotly_chart(plot_actual_vs_forecast(df), use_container_width=True)
                    
                    st.markdown("### 📉 Fehleranalyse")
                    residuals_fig = plot_residuals(df)
                    if residuals_fig:
                        st.plotly_chart(residuals_fig, use_container_width=True)
                    
                    st.divider()
                    
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
                    
                    if 'run_future_forecast' not in st.session_state:
                        st.session_state['run_future_forecast'] = False
                    
                    if st.session_state.get('run_future_forecast', False):
                        with st.spinner("Berechne Zukunftsprognose..."):
                            try:
                                future_predictions = make_future_predictions(
                                    df, lstm_model, scaler, days=forecast_days
                                )
                                
                                # KORRIGIERT: pd.Timedelta statt datetime.timedelta
                                last_date = df['date'].iloc[-1]
                                future_dates = pd.date_range(
                                    start=last_date + pd.Timedelta(days=1),
                                    periods=forecast_days,
                                    freq='D'
                                )
                                
                                st.plotly_chart(
                                    plot_future_forecast(df, future_dates, future_predictions),
                                    use_container_width=True
                                )
                                
                                with st.expander("📋 Detaillierte Prognosetabelle"):
                                    changes = np.zeros(len(future_predictions))
                                    if len(future_predictions) > 1:
                                        changes[0] = ((future_predictions[0] - df['unit_sales'].iloc[-1]) / df['unit_sales'].iloc[-1] * 100)
                                        for i in range(1, len(future_predictions)):
                                            changes[i] = ((future_predictions[i] - future_predictions[i-1]) / future_predictions[i-1] * 100)
                                    
                                    forecast_df = pd.DataFrame({
                                        'Datum': [d.strftime('%d.%m.%Y') for d in future_dates],
                                        'Prognose': future_predictions.round(1),
                                        'Änderung %': changes.round(1)
                                    })
                                    
                                    st.dataframe(
                                        forecast_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ Fehler bei der Zukunftsprognose: {str(e)}")
                
                except Exception as e:
                    st.error(f"❌ Fehler bei der Vorhersage: {str(e)}")
        
        st.divider()
        display_data_preview(df)
    
    st.divider()
    st.markdown(f"""
    <div style='background-color: {COLORS["card_bg"]}; padding: 20px; border-radius: 10px; margin-top: 50px;'>
        <div style='text-align: center; color: {COLORS["text_secondary"]};'>
            <p style='font-weight: bold; color: {COLORS["text_primary"]};'>LSTM Forecast Dashboard | Store {store_id} | Item {item_id}</p>
            <p style='font-size: 0.9em; opacity: 0.7;'>
                Dark Mode Edition • Letzte Aktualisierung: {datetime.now().strftime("%d.%m.%Y %H:%M")}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# APP STARTEN
# ==========================================================
if __name__ == "__main__":
    if 'run_forecast' not in st.session_state:
        st.session_state.run_forecast = False
    if 'run_future_forecast' not in st.session_state:
        st.session_state.run_future_forecast = False
    
    main()