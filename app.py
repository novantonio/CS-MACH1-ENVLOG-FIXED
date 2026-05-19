"""
app.py
------
CS-MACH EnvLogger pipeline — Streamlit single-file app.

File richiesti nella stessa cartella:
  • cs_mach1_theme.py
  • logo.png
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# ── Constants ─────────────────────────────────────────────────────────────────

CORA_URL_TEMPLATE = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = [
    "Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
    "Lug", "Ago", "Set", "Ott", "Nov", "Dic"
]

TMAX = 32


def _year_marker(year: int) -> str:
    return {2025: "*", 2026: "^", 2027: "s"}.get(year, "o")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


# ── Parser ────────────────────────────────────────────────────────────────────

def extract_metadata(
    df: pd.DataFrame,
    default_lat: float,
    default_lon: float,
) -> LoggerMetadata:
    serial = str(df.iloc[9, 1]).strip()
    custom_name = str(df.iloc[10, 1]).strip()
    sampling_frequency = str(df.iloc[13, 1]).strip()

    # Rilevamento posizione lat/lon
    has_latitude = "lat" in str(df.iloc[15, 0]).lower() if len(df) > 15 else False
    if has_latitude:
        latitude = df.iloc[15, 1]
        longitude = df.iloc[16, 1]
    else:
        latitude = df.iloc[16, 1]
        longitude = df.iloc[17, 1]

    latitude = pd.to_numeric(latitude, errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")

    if pd.isna(latitude) or pd.isna(longitude):
        latitude, longitude = default_lat, default_lon

    return LoggerMetadata(
        serial=serial,
        custom_name=custom_name,
        sampling_frequency=sampling_frequency,
        latitude=float(latitude),
        longitude=float(longitude),
    )


def parse_envlog_csv(
    df: pd.DataFrame,
    default_lat: float,
    default_lon: float,
) -> pd.DataFrame:
    metadata = extract_metadata(df, default_lat, default_lon)

    clean_df = df.iloc[21:, :].copy()
    clean_df = clean_df.dropna(how="all").reset_index(drop=True)
    clean_df.columns = ["time", "temperature"]

    clean_df["time"] = pd.to_datetime(clean_df["time"], errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"], errors="coerce")

    clean_df["serial"] = metadata.serial
    clean_df["custom_name"] = metadata.custom_name
    clean_df["sampling_frequency"] = metadata.sampling_frequency
    clean_df["latitude"] = metadata.latitude
    clean_df["longitude"] = metadata.longitude

    return clean_df.dropna(subset=["time", "temperature"]).reset_index(drop=True)


def add_rolling_mean(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    result = df.copy()
    result["temperature_rolling_mean"] = (
        result["temperature"].rolling(window=window_size, min_periods=1).mean()
    )
    return result


# ── CORA API ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Scaricamento dati CORA…")
def fetch_cora_data(latitude: float, longitude: float) -> pd.DataFrame | None:
    url = CORA_URL_TEMPLATE.format(lat=round(latitude, 4), lon=round(longitude, 4))
    try:
        response = requests.get(url, verify=False, timeout=60)
        response.raise_for_status()
        if "<html" in response.text.lower():
            raise ValueError("CORA ha restituito HTML invece di CSV")
        
        df = pd.read_csv(io.StringIO(response.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna(subset=["time", "TEMP"])
    except Exception as exc:
        st.warning(f"Impossibile recuperare i dati CORA: {exc}")
        return None


# ── Plotting Functions ────────────────────────────────────────────────────────

def plot_series_and_doy(
    sdata: pd.DataFrame,
    cora_df: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.28})
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    label = sdata["custom_name"].iloc[0]
    yr = sdata["time"].iloc[0].year
    t_mean = sdata["temperature"].mean()
    t_med = sdata["temperature"].median()
    marker = _year_marker(yr)
    m_month = sdata["time"].iloc[0].month
    d_doy = sdata["time"].iloc[0].timetuple().tm_yday

    # CORA monthly
    cora_m = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()

    # Colours for years
    years = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    # Time series
    ax1.plot(sdata["time"], sdata["temperature"], alpha=0.4, linewidth=0.8,
             color="steelblue", label="Temperatura raw")
    if "temperature_rolling_mean" in sdata.columns:
        ax1.plot(sdata["time"], sdata["temperature_rolling_mean"],
                 linewidth=2, color="tomato", label="Media mobile")
    ax1.axhline(t_mean, color="crimson", linewidth=1.4, linestyle="--",
                label=f"Media {t_mean:.2f} °C")
    ax1.axhline(t_med, color="darkorange", linewidth=1.4, linestyle="--",
                label=f"Mediana {t_med:.2f} °C")
    ax1.legend(fontsize=9)
    ax1.set_xlabel("Tempo")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title(f"Serie temporale — {label} ({yr})")
    ax1.grid(True, alpha=0.3)

    # Monthly CORA
    ax2.scatter(cora_monthly["month"], cora_monthly["mean"],
                color="steelblue", zorder=3, label="CORA media mensile")
    ax2.errorbar(cora_monthly["month"], cora_monthly["mean"],
                 yerr=cora_monthly["std"], fmt="o", color="steelblue",
                 capsize=3, alpha=0.5, label="± std")
    ax2.plot(m_month, t_mean, marker=marker, markersize=12, linestyle="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} media")
    ax2.plot(m_month, t_med, marker=marker, markersize=12, linestyle="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} mediana")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax2.set_xlabel("Mese")
    ax2.set_ylabel("Temperatura [°C]")
    ax2.set_ylim(top=TMAX)
    ax2.set_title("CORA Media Mensile ± Std vs Logger")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    def _draw_cora_doy(ax):
        for colour, (_, year_data) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
            doy = year_data["time"].dt.dayofyear
            ax.plot(doy, year_data["TEMP"], marker=".", markersize=4,
                    linestyle="--", color=colour, alpha=0.6)

    # DOY Mean
    _draw_cora_doy(ax3)
    ax3.plot(d_doy, t_mean, marker=marker, markersize=22, linestyle="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax3.set_xlabel("Giorno dell'Anno")
    ax3.set_ylabel("Temperatura [°C]")
    ax3.set_title(f"DOY — Marker Media | ({latitude:.2f}, {longitude:.2f})")
    ax3.grid(True, alpha=0.3)

    # DOY Median
    _draw_cora_doy(ax4)
    ax4.plot(d_doy, t_med, marker=marker, markersize=22, linestyle="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax4.set_xlabel("Giorno dell'Anno")
    ax4.set_ylabel("Temperatura [°C]")
    ax4.set_title(f"DOY — Marker Mediana | ({latitude:.2f}, {longitude:.2f})")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f"{label} ({yr})", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_doy_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    years = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    for colour, (_, year_data) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        doy = year_data["time"].dt.dayofyear
        ax.plot(doy, year_data["TEMP"], marker=".", markersize=4,
                linestyle="--", color=colour, alpha=0.5)

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        d = sdata["time"].iloc[0].timetuple().tm_yday
        t_mean = sdata["temperature"].mean()
        t_med = sdata["temperature"].median()
        label = sdata["custom_name"].iloc[0]
        yr = sdata["time"].iloc[0].year
        marker = _year_marker(yr)

        ax.plot(d, t_mean, marker=marker, markersize=12, linestyle="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8)
        ax.plot(d, t_med, marker=marker, markersize=12, linestyle="None",
                color="white", markeredgecolor=sc, markeredgewidth=2)
        ax.plot([d, d], [t_mean, t_med], color="grey", linewidth=1, linestyle=":")

    ax.set_xlabel("Giorno dell'Anno")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title(f"Variabilità Interannuale — Tutti i logger\n({latitude:.2f}, {longitude:.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    cora_m = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()

    ax.scatter(cora_monthly["month"], cora_monthly["mean"],
               label="CORA media mensile", color="steelblue")
    ax.errorbar(cora_monthly["month"], cora_monthly["mean"],
                yerr=cora_monthly["std"], fmt="o", color="steelblue",
                capsize=3, alpha=0.6, label="± std")

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        month = sdata["time"].iloc[0].month
        t_mean = sdata["temperature"].mean()
        t_med = sdata["temperature"].median()
        label = sdata["custom_name"].iloc[0]
        yr = sdata["time"].iloc[0].year
        marker = _year_marker(yr)

        ax.plot(month, t_mean, marker=marker, markersize=12, linestyle="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8)
        ax.plot(month, t_med, marker=marker, markersize=12, linestyle="None",
                color="white", markeredgecolor=sc, markeredgewidth=2)
        ax.plot([month, month], [t_mean, t_med], color="grey", linewidth=1, linestyle=":")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlabel("Mese")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title("Confronto Mensile — Tutti i logger vs CORA")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────

apply_cs_mach1_theme(
    page_title="CS-MACH EnvLogger Pipeline",
    page_icon="logo.png",
    main_title="🌊 CS-MACH: Cosa dice il mio envlogger sulla temperatura dell'acqua? 🌡",
    subtitle="Piattaforma di confronto temperatura oceanica (logger in-situ vs rianalisi CORA)",
    logo_path="logo.png",
    logo_width=220,
)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Impostazioni")
    window_size = st.slider("Finestra media mobile", 1, 20, 5)

    st.divider()
    st.markdown("#### 📍 Coordinate di default")
    default_lat = st.number_input("Latitudine default", value=44.376290, format="%.6f")
    default_lon = st.number_input("Longitudine default", value=9.071358, format="%.6f")

    st.divider()
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# File uploader
uploaded_files = st.file_uploader(
    "Carica uno o più file CSV envlog",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# Processing
if st.button("▶️ Avvia Elaborazione", type="primary", use_container_width=True) and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    logger_dfs: dict[str, pd.DataFrame] = {}

    pbar = st.progress(0)

    for i, file in enumerate(raw_files):
        pbar.progress((i + 1) / len(raw_files), text=f"Elaborazione: {file.name}")
        try:
            raw_df = pd.read_csv(file, header=None)
            clean_df = parse_envlog_csv(raw_df, default_lat, default_lon)
            proc_df = add_rolling_mean(clean_df, window_size)
            logger_dfs[file.name] = proc_df
        except Exception as e:
            st.error(f"Errore su {file.name}: {e}")

    pbar.progress(1.0, text="✅ Elaborazione completata!")

    if logger_dfs:
        st.session_state["logger_dfs"] = logger_dfs
        st.success(f"{len(logger_dfs)} file elaborati con successo!")

# Display results
if "logger_dfs" in st.session_state:
    logger_dfs = st.session_state["logger_dfs"]
    first_df = next(iter(logger_dfs.values()))
    latitude = float(first_df["latitude"].iloc[0])
    longitude = float(first_df["longitude"].iloc[0])

    with st.spinner("Caricamento dati CORA..."):
        cora_df = fetch_cora_data(latitude, longitude)

    if cora_df is None or cora_df.empty:
        st.error("Impossibile scaricare i dati CORA.")
        st.stop()

    # Per singolo file
    for fname, sdata in logger_dfs.items():
        st.subheader(f"📄 {fname}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Media", f"{sdata['temperature'].mean():.2f} °C")
        col2.metric("Mediana", f"{sdata['temperature'].median():.2f} °C")
        col3.metric("Std", f"{sdata['temperature'].std():.2f} °C")
        col4.metric("Coordinate", f"{latitude:.4f}, {longitude:.4f}")

        fig = plot_series_and_doy(sdata, cora_df, latitude, longitude)
        st.pyplot(fig)
        plt.close(fig)
        st.divider()

    # Riepilogo globale
    st.header("📊 Riepilogo — Tutti i Logger")
    summary = []
    for fname, sdata in logger_dfs.items():
        summary.append({
            "File": fname,
            "Logger": sdata["custom_name"].iloc[0],
            "Media (°C)": round(sdata["temperature"].mean(), 2),
            "Mediana (°C)": round(sdata["temperature"].median(), 2),
            "Std (°C)": round(sdata["temperature"].std(), 2),
            "Campioni": len(sdata),
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    fig_doy = plot_doy_all(cora_df, logger_dfs, latitude, longitude)
    st.pyplot(fig_doy)
    plt.close(fig_doy)

    fig_month = plot_monthly_all(cora_df, logger_dfs, latitude, longitude)
    st.pyplot(fig_month)
    plt.close(fig_month)

    st.info("**Marker pieno** = media | **Marker aperto** = mediana")

    cs_mach1_footer(
        text="CS-MACH · EnvLogger Analysis Pipeline"
    )
