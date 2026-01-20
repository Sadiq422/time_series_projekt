import matplotlib.pyplot as plt  # Für grundlegende Diagramme und Grafiken
import seaborn as sns  # Für statistische Visualisierungen (schönere Grafiken)
import numpy as np  # Für numerische Berechnungen und Arrays

# Datenverarbeitung und Visualisierung
import pandas as pd  # Für Datenanalyse und -manipulation (Tabellen wie Excel)


# Farbpalette für dunkles Design definieren
# Jede Farbe hat einen spezifischen Zweck im Dashboard

dark_palette = {
    "background": "#020a3d",   # Tiefer dunkelblauer Hintergrund (GitHub-Dark Style)
    "axes": "#161b22",         # Etwas helleres Dunkelblau für Diagrammachsen
    "text": "#c9d1d9",         # Weiches hellgrau-blau für Text (gut lesbar auf dunklem Hintergrund)
    "accent": "#58a6ff",       # Helles Blau für Hauptakzente (ersetzt traditionelles Rot)
    "accent2": "#39c5bb",      # Cyan/Türkis für sekundäre Akzente
    "accent3": "#8b6cff",      # Weiches Violett für zusätzliche Akzentfarben
    "grid": "#21262d",         # Gedämpftes blau-grau für Gitternetzlinien
    "bar": "#58a6ff",          # Blaue Farbe für Balkendiagramme
    "bar2": "#39c5bb",         # Türkise Farbe für alternative Balken
}
def apply_dark_theme():
    """
    Diese Funktion wendet ein dunkles Design auf alle Diagramme an.
    Sie wird vor dem Erstellen von Visualisierungen aufgerufen,
    um ein einheitliches, augenschonendes Erscheinungsbild zu gewährleisten.
    """

    # 1. Setze das vordefinierte Matplotlib-Dark-Theme
    #    (stellt grundlegende dunkle Farben für Diagramme bereit)
    plt.style.use("dark_background")

    # 2. Setze das Seaborn-Theme auf "darkgrid"
    #    (fügt ein dunkles Hintergrundgitter hinzu, das die Lesbarkeit verbessert)
    sns.set_theme(style="darkgrid")

    # 3. Aktualisiere die individuellen Design-Einstellungen von Matplotlib
    #    (überschreibt die Standardfarben mit unserer eigenen Farbpalette)
    plt.rcParams.update({
        # Hintergrundfarbe der gesamten Figure (des gesamten Bildes)
        "figure.facecolor": dark_palette["background"],

        # Hintergrundfarbe der Zeichenfläche (innerhalb der Achsen)
        "axes.facecolor": dark_palette["axes"],

        # Farbe der Achsenränder (der Linien, die die Zeichenfläche begrenzen)
        "axes.edgecolor": dark_palette["text"],

        # Farbe der Achsenbeschriftungen (xlabel, ylabel)
        "axes.labelcolor": dark_palette["text"],

        # Farbe der Skalenstriche und -beschriftungen auf der X-Achse
        "xtick.color": dark_palette["text"],

        # Farbe der Skalenstriche und -beschriftungen auf der Y-Achse
        "ytick.color": dark_palette["text"],

        # Standardfarbe für alle Text-Elemente (Titel, Legenden etc.)
        "text.color": dark_palette["text"],

        # Farbe des Gitternetzes (grid) in den Diagrammen
        "grid.color": dark_palette["grid"],

        # Hintergrundfarbe beim Speichern der Abbildung als Datei
        "savefig.facecolor": dark_palette["background"],

        # Randfarbe beim Speichern der Abbildung als Datei
        "savefig.edgecolor": dark_palette["background"],
    })

def plot_time_series(
    df,
    date_col="date",
    value_col="unit_sales",
    title="Time Series Plot",
    xlabel="Date",
    ylabel="Value",
    figsize=(12, 6),
    rotation=45,
    state_name=None
):
    """
    Aggregiert eine Metrik nach Datum und erstellt einen Zeitreihen-Plot.

    Parameter
    ----------
    df : pd.DataFrame
        Eingabe-DataFrame mit einer Datumsspalte und einer numerischen Metrik.
    date_col : str
        Name der Datumsspalte.
    value_col : str
        Spalte, die aggregiert und geplottet werden soll (z.B. 'unit_sales').
    title : str
        Titel des Plots.
    xlabel : str
        Beschriftung der X-Achse.
    ylabel : str
        Beschriftung der Y-Achse.
    figsize : tuple
        Größe der Abbildung (Breite, Höhe).
    rotation : int
        Rotationswinkel für die X-Achsen-Beschriftungen.
    state_name : str or None
        Optionaler Regions-/Bundesland-Name, der an den Titel angehängt wird.

    Rückgabe
    -------
    None
    """

    print("📊 Starte Zeitreihen-Aggregation und Plotting...")
    print(f"   ➤ Datumsspalte: {date_col}")
    print(f"   ➤ Wertespalte: {value_col}")

    # 1. Datumsspalte in datetime-Format konvertieren
    #    pd.to_datetime() wandelt verschiedene Datumsformate in einheitliches Format um
    df[date_col] = pd.to_datetime(df[date_col])
    print("   ✔ Datumsspalte in datetime konvertiert.")

    # 2. Aggregation der Werte pro Datum
    #    groupby() gruppiert nach Datum, sum() summiert die Werte
    print("📅 Aggregiere Werte nach Datum...")
    aggregated = df.groupby(date_col)[value_col].sum()
    print(f"   ✔ Aggregation abgeschlossen. Anzahl Tage: {len(aggregated)}")

    # 3. Dynamischen Titel erstellen
    #    Wenn state_name angegeben ist, wird es an den Titel angehängt
    if state_name:
        full_title = f"{title} in {state_name}"
    else:
        full_title = title

    # 4. Zeitreihen-Plot erstellen
    print("📈 Erstelle Zeitreihen-Plot...")

    # Neue Figure mit angegebener Größe erstellen
    plt.figure(figsize=figsize)

    # Linienplot erstellen: x = Datum, y = aggregierte Werte
    plt.plot(aggregated.index, aggregated.values)

    # 5. Plot-Elemente formatieren
    plt.title(full_title, fontsize=20, fontweight="bold")  # Titel
    plt.xlabel(xlabel, fontsize=16)                        # X-Achsen-Beschriftung
    plt.ylabel(ylabel, fontsize=16)                        # Y-Achsen-Beschriftung
    plt.xticks(fontsize=14, rotation=rotation)             # X-Achsen-Ticks drehen
    plt.yticks(fontsize=14)                                # Y-Achsen-Ticks

    # 6. Layout optimieren (vermeidet Überlappungen)
    plt.tight_layout()

    # 7. Plot anzeigen
    plt.show()

    print("🎉 Zeitreihen-Plot erfolgreich erstellt.\n")

def plot_year_month_heatmap(
    df,
    year_col="year",
    month_col="month",
    value_col="unit_sales",
    title="Monthly Sales Trends Over Years",
    cmap="coolwarm",
    figsize=(15, 10)
):
    """
    Erstellt eine Jahr-Monat-Heatmap für eine aggregierte Metrik.

    Parameter
    ----------
    df : pd.DataFrame
        Eingabe-DataFrame mit Jahr-, Monats- und Metrik-Spalten.
    year_col : str
        Spalte, die das Jahr repräsentiert.
    month_col : str
        Spalte, die den Monat repräsentiert.
    value_col : str
        Metrik, die aggregiert und visualisiert werden soll (z.B. 'unit_sales').
    title : str
        Titel der Heatmap.
    cmap : str
        Farbkarte für die Heatmap.
    figsize : tuple
        Größe der Abbildung.

    Rückgabe
    -------
    None
    """

    print("📊 Starte Jahr-Monat-Heatmap-Erstellung...")
    print(f"   ➤ Jahr-Spalte: {year_col}")
    print(f"   ➤ Monat-Spalte: {month_col}")
    print(f"   ➤ Wertespalte: {value_col}")

    # Schritt 1: Aggregation nach Jahr und Monat
    # Gruppiert die Daten nach Jahr und Monat, summiert die Werte
    print("📅 Aggregiere Werte nach Jahr und Monat...")
    pivot = df.groupby([year_col, month_col])[value_col].sum().unstack()

    print(f"   ✔ Aggregation abgeschlossen. Shape: {pivot.shape}")
    print("   Beispiel der aggregierten Daten:")
    print(pivot.head(), "\n")

    # Schritt 2: Heatmap plotten
    print("📈 Erstelle Heatmap...")
    plt.figure(figsize=figsize)

    # Erstelle die Heatmap mit seaborn
    sns.heatmap(
        pivot,                      # Die pivot-Tabelle mit aggregierten Daten
        cmap=cmap,                  # Farbkarte (coolwarm = blau-rot)
        linewidths=0.5,             # Dünne Linien zwischen den Zellen
        linecolor="white",          # Weiße Trennlinien
        cbar_kws={"label": value_col.replace("_", " ").title()}  # Farbleiste-Beschriftung
    )

    # Titel und Achsenbeschriftungen
    plt.title(title, fontsize=22, fontweight="bold")
    plt.xlabel("Monat", fontsize=18, labelpad=10)   # X-Achse = Monate
    plt.ylabel("Jahr", fontsize=18, labelpad=10)    # Y-Achse = Jahre

    # Achsenbeschriftungen formatieren
    plt.xticks(fontsize=14, rotation=45)  # Monatsbeschriftungen drehen
    plt.yticks(fontsize=14)               # Jahresbeschriftungen

    # Layout optimieren und Plot anzeigen
    plt.tight_layout()
    plt.show()

    print("🎉 Heatmap erfolgreich erstellt.\n")

def plot_holiday_impact(
    df,
    value_col="unit_sales",
    holiday_col="type",
    title="Impact of Holidays on Sales",
    figsize=(8, 5)
):
    """
    Zeigt den durchschnittlichen Umsatz für jeden Feiertagstyp in einem Balkendiagramm.

    Parameter
    ----------
    df : pd.DataFrame
        Zusammengeführter Datensatz aus Verkaufs- und Feiertagsdaten.
    value_col : str
        Spalte mit der zu analysierenden Metrik (z.B. Umsatz).
    holiday_col : str
        Spalte mit den Feiertagstypen.
    title : str
        Titel des Plots.
    figsize : tuple
        Größe der Abbildung (Breite, Höhe).
    """

    # 1. Durchschnittlichen Umsatz pro Feiertagstyp berechnen
    print("📊 Berechne durchschnittlichen Umsatz pro Feiertagstyp...")
    holiday_sales = df.groupby(holiday_col)[value_col].mean()
    print("   ✔ Aggregation abgeschlossen.")
    print("   Beispiel:\n", holiday_sales.head(), "\n")

    # 2. Balkendiagramm erstellen
    print("📈 Erstelle Feiertagseinfluss-Diagramm...")

    # Neue Figure mit angegebener Größe erstellen
    plt.figure(figsize=figsize)

    # Balkendiagramm zeichnen:
    # - kind="bar": Erstellt ein Balkendiagramm
    # - color="lightgreen": Hellgrüne Balken
    # - edgecolor="black": Schwarze Kanten für bessere Sichtbarkeit
    holiday_sales.plot(kind="bar", color="lightgreen", edgecolor="black")

    # 3. Diagramm formatieren
    plt.title(title, fontsize=20, fontweight="bold")   # Titel
    plt.ylabel("Durchschnittlicher Umsatz", fontsize=16)  # Y-Achsen-Beschriftung
    plt.xlabel("")  # X-Achse ohne Beschriftung (Feiertagstypen sind in Balken)
    plt.xticks(fontsize=14)  # Feiertagsnamen in Größe 14
    plt.yticks(fontsize=14)  # Y-Achsen-Werte in Größe 14

    # 4. Layout optimieren und Diagramm anzeigen
    plt.tight_layout()
    plt.show()

    print("🎉 Feiertagseinfluss-Diagramm erfolgreich erstellt.\n")

def plot_perishable_sales(
    df,
    perishable_col="perishable",
    value_col="unit_sales",
    title="Sales of Perishable vs Non-Perishable Items",
    figsize=(12, 6)
):
    """
    Zeigt den Gesamtumsatz für verderbliche vs. nicht-verderbliche Artikel in einem Balkendiagramm.

    Parameter
    ----------
    df : pd.DataFrame
        Datensatz mit Verderblichkeits-Flag und Umsatzwerten.
    perishable_col : str
        Spalte, die den Verderblichkeits-Status anzeigt (boolean: 0/1 oder True/False).
    value_col : str
        Spalte mit den Umsatzwerten.
    title : str
        Titel des Plots.
    figsize : tuple
        Größe der Abbildung (Breite, Höhe).

    Rückgabe
    -------
    None
    """

    # 1. Gesamtumsatz nach Verderblichkeits-Kategorie berechnen
    print("📊 Berechne Gesamtumsatz nach Verderblichkeits-Kategorie...")
    perishable_sales = df.groupby(perishable_col)[value_col].sum()

    print("   ✔ Aggregation abgeschlossen.")
    print("   Umsatz-Übersicht:")
    print(perishable_sales, "\n")

    # 2. Balkendiagramm erstellen
    print("📈 Erstelle Vergleichsdiagramm verderblich vs. nicht-verderblich...")

    # Neue Figure mit angegebener Größe erstellen
    plt.figure(figsize=figsize)

    # Balkendiagramm zeichnen:
    # - kind="bar": Erstellt ein Balkendiagramm
    # - color=["orange", "green"]: Farben für die zwei Kategorien
    # - edgecolor="black": Schwarze Kanten für bessere Sichtbarkeit
    perishable_sales.plot(
        kind="bar",
        color=["orange", "green"],
        edgecolor="black"
    )

    # 3. Diagramm formatieren
    plt.title(title, fontsize=18, fontweight="bold")   # Titel
    plt.ylabel("Gesamtumsatz", fontsize=16)           # Y-Achsen-Beschriftung
    plt.xlabel("")  # X-Achse ohne Beschriftung (wird separat gesetzt)

    # 4. X-Achsen-Beschriftungen anpassen
    #    ticks=[0, 1]: Positionen der Balken auf der X-Achse
    #    labels: Beschriftungen für die beiden Kategorien
    #    rotation=0: Keine Drehung der Beschriftungen
    plt.xticks(
        ticks=[0, 1],
        labels=["Nicht-verderblich", "Verderblich"],
        fontsize=16,
        rotation=0
    )
    plt.yticks(fontsize=14)  # Y-Achsen-Werte in Größe 14

    # 5. Layout optimieren und Diagramm anzeigen
    plt.tight_layout()
    plt.show()

    print("🎉 Verderblichkeits-Diagramm erfolgreich erstellt.\n")