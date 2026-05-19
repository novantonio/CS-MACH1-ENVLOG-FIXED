"""
app.py
------
CS-MACH EnvLogger pipeline — Streamlit single-file app.

File di progetto richiesti nella stessa cartella:
  • cs_mach1_theme.py   – branding CS-MACH1
  • logo.png            – logo CS-MACH1

Per ogni CSV caricato:
  • Metriche: mean, median, std, coordinate
  • Plot 1  – Time-series (raw + rolling mean + linee mean/median)
  • Plot 2  – CORA monthly mean ± std + marker logger (mean & median)
  • Plot 3  – CORA interannuale DOY scatter + marker MEAN (crimson)
  • Plot 4  – CORA interannuale DOY scatter + marker MEDIAN (darkorange)

Dopo tutti i file:
  • Summary table
  • Plot globale DOY vs CORA (tutti i logger)
  • Plot globale mensile vs CORA (tutti i logger)
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
    "Lug", "Ago", "Set", "Ott", "Nov", "Dic",
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
    serial             = df.iloc[9, 1]
    custom_name        = df.iloc[10, 1]
    sampling_frequency = df.iloc[13, 1]

    has_latitude = "lat" in str(df.iloc[15, 0]).lower()
    if has_latitude:
        latitude  = df.iloc[15, 1]
        longitude = df.iloc[16, 1]
    else:
        latitude  = df.iloc[16, 1]
        longitude = df.iloc[17, 1]

    latitude  = pd.to_numeric(latitude,  errors="coerce")
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

    clean_df = df.iloc[21:, :].dropna().reset_index(drop=True)
    clean_df.columns = ["time", "temperature"]
    clean_df["time"]        = pd.to_datetime(clean_df["time"],        errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"],  errors="coerce")

    clean_df["serial"]             = metadata.serial
    clean_df["custom_name"]        = metadata.custom_name
    clean_df["sampling_frequency"] = metadata.sampling_frequency
    clean_df["latitude"]           = metadata.latitude
    clean_df["longitude"]          = metadata.longitude

    return clean_df.dropna()


def add_rolling_mean(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    result = df.copy()
    result["temperature_rolling_mean"] = (
        result["temperature"].rolling(window=window_size).mean()
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
            raise ValueError("CORA ha restituito una pagina HTML invece di CSV.")
        df = pd.read_csv(io.StringIO(response.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna(subset=["time", "TEMP"])
    except Exception as exc:
        st.warning(f"Impossibile recuperare i dati CORA: {exc}")
        return None


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_series_and_doy(
    sdata: pd.DataFrame,
    cora_df: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """Figura 2×2 per singolo logger."""

    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    label  = sdata["custom_name"].iloc[0]
    yr     = sdata["time"].iloc[0].year
    t_mean = sdata["temperature"].mean()
    t_med  = sdata["temperature"].median()
    marker = _year_marker(yr)
    m_month = sdata["time"].iloc[0].month
    d_doy   = sdata["time"].iloc[0].timetuple().tm_yday

    # CORA monthly stats
    cora_m = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()

    # CORA DOY colours
    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    # ── ax1: Time-series ──────────────────────────────────────────────────
    ax1.plot(
        sdata["time"], sdata["temperature"],
        alpha=0.4, linewidth=0.8, color="steelblue", label="Temperatura raw",
    )
    if "temperature_rolling_mean" in sdata.columns:
        ax1.plot(
            sdata["time"], sdata["temperature_rolling_mean"],
            linewidth=2, color="tomato", label="Media mobile",
        )
    ax1.axhline(t_mean, color="crimson",    linewidth=1.4, linestyle="--",
                label=f"Media {t_mean:.2f} °C")
    ax1.axhline(t_med,  color="darkorange", linewidth=1.4, linestyle="--",
                label=f"Mediana {t_med:.2f} °C")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("Tempo")
    ax1.set_ylabel("Temperatura (°C)")
    ax1.set_title(f"Serie temporale — {label} ({yr})")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=25)

    # ── ax2: CORA mensile ± std + marker logger ───────────────────────────
    ax2.scatter(cora_monthly["month"], cora_monthly["mean"],
                color="steelblue", zorder=3, label="CORA media mensile")
    ax2.errorbar(cora_monthly["month"], cora_monthly["mean"],
                 yerr=cora_monthly["std"],
                 fmt="o", color="steelblue", capsize=3, alpha=0.5, label="± std")
    ax2.plot(m_month, t_mean,
             marker=marker, markersize=12, linestyle="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} media {t_mean:.2f} °C")
    ax2.plot(m_month, t_med,
             marker=marker, markersize=12, linestyle="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} mediana {t_med:.2f} °C")
    ax2.plot([m_month, m_month], [t_mean, t_med],
             color="grey", linewidth=1.2, linestyle=":", zorder=4)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax2.set_xlabel("Mese")
    ax2.set_ylabel("Temperatura [°C]")
    ax2.set_ylim(top=TMAX)
    ax2.set_title("CORA Media Mensile ± Std vs Logger")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Helper: sfondo CORA DOY
    def _draw_cora_doy(ax):
        for colour, (year, year_data) in zip(
            colours, cora_df.groupby(cora_df["time"].dt.year)
        ):
            doy = year_data["time"].dt.dayofyear
            ax.plot(doy, year_data["TEMP"],
                    marker=".", markersize=4, linestyle="--",
                    color=colour, alpha=0.6)
        ax.set_xlabel("Giorno dell'Anno")
        ax.set_ylabel("Temperatura [°C]")
        ax.grid(True, alpha=0.3)

    # ── ax3: DOY — marker MEDIA (crimson) ────────────────────────────────
    _draw_cora_doy(ax3)
    ax3.plot(d_doy, t_mean,
             marker=marker, markersize=22, linestyle="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"media {t_mean:.2f} °C")
    ax3.annotate(
        f"media {t_mean:.2f} °C",
        xy=(d_doy, t_mean), xytext=(d_doy + 4, t_mean + 0.3),
        fontsize=8, color="crimson", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8),
    )
    ax3.set_title(f"DOY — Marker Media | ({latitude:.2f}, {longitude:.2f})")

    # ── ax4: DOY — marker MEDIANA (darkorange) ───────────────────────────
    _draw_cora_doy(ax4)
    ax4.plot(d_doy, t_med,
             marker=marker, markersize=22, linestyle="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"mediana {t_med:.2f} °C")
    ax4.annotate(
        f"mediana {t_med:.2f} °C",
        xy=(d_doy, t_med), xytext=(d_doy + 4, t_med - 0.4),
        fontsize=8, color="darkorange", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="darkorange", lw=0.8),
    )
    ax4.set_title(f"DOY — Marker Mediana | ({latitude:.2f}, {longitude:.2f})")

    fig.suptitle(f"{label} ({yr})", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_doy_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """DOY interannuale CORA + tutti i logger (pieno = media, aperto = mediana)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    for colour, (year, year_data) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        doy = year_data["time"].dt.dayofyear
        ax.plot(doy, year_data["TEMP"],
                marker=".", markersize=4, linestyle="--",
                color=colour, alpha=0.5)

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        d      = sdata["time"].iloc[0].timetuple().tm_yday
        t_mean = sdata["temperature"].mean()
        t_med  = sdata["temperature"].median()
        label  = sdata["custom_name"].iloc[0]
        yr     = sdata["time"].iloc[0].year
        marker = _year_marker(yr)

        ax.plot(d, t_mean, marker=marker, markersize=12, linestyle="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8,
                label=f"{label} ({yr}) media")
        ax.plot(d, t_med,  marker=marker, markersize=12, linestyle="None",
                color="white", markeredgecolor=sc, markeredgewidth=2,
                label=f"{label} ({yr}) mediana")
        ax.plot([d, d], [t_mean, t_med], color="grey", linewidth=1, linestyle=":")

    ax.set_xlabel("Giorno dell'Anno")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title(
        f"Variabilità Interannuale della Temperatura ({latitude:.2f}, {longitude:.2f})\n"
        "— Tutti i logger — pieno = media · aperto = mediana —"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """Media mensile CORA ± std + tutti i logger."""
    fig, ax = plt.subplots(figsize=(12, 6))

    cora_m = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()

    ax.scatter(cora_monthly["month"], cora_monthly["mean"],
               label="CORA media mensile", color="steelblue")
    ax.errorbar(cora_monthly["month"], cora_monthly["mean"],
                yerr=cora_monthly["std"],
                fmt="o", color="steelblue", capsize=3,
                label="± std", alpha=0.6)

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        month  = sdata["time"].iloc[0].month
        t_mean = sdata["temperature"].mean()
        t_med  = sdata["temperature"].median()
        label  = sdata["custom_name"].iloc[0]
        yr     = sdata["time"].iloc[0].year
        marker = _year_marker(yr)

        ax.plot(month, t_mean, marker=marker, markersize=12, linestyle="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8,
                label=f"{label} ({yr}) media")
        ax.plot(month, t_med,  marker=marker, markersize=12, linestyle="None",
                color="white", markeredgecolor=sc, markeredgewidth=2,
                label=f"{label} ({yr}) mediana")
        ax.plot([month, month], [t_mean, t_med],
                color="grey", linewidth=1, linestyle=":")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=9)
    ax.set_xlabel("Mese")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title("CORA vs Logger — Confronto Mensile (tutti i logger)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

apply_cs_mach1_theme(
    page_title="CS-MACH EnvLogger Pipeline",
    page_icon="logo.png",
    main_title="🌊 CS-MACH: Cosa dice il mio envlogger sulla temperatura dell'acqua? 🌡",
    subtitle="Piattaforma di confronto temperatura oceanica (logger in-situ vs ranalisi CORA)",
    logo_path="logo.png",
    logo_width=220,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Impostazioni")

    window_size = st.slider("Finestra media mobile", min_value=1, max_value=20, value=5)

    st.divider()
    st.markdown("#### 📍 Coordinate di default")
    st.caption(
        "Usate quando latitudine/longitudine non sono valorizzate nel CSV."
    )
    default_lat = st.number_input(
        "Latitudine default", value=44.376290, format="%.6f", step=0.001
    )
    default_lon = st.number_input(
        "Longitudine default", value=9.071358, format="%.6f", step=0.001
    )

    st.divider()
    n_files = len(st.session_state.get("uploaded_files", []))
    if n_files > 0:
        st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} caricati")
    else:
        st.info("📂 Nessun file caricato")

    st.divider()
    sidebar_progress = st.empty()
    sidebar_status   = st.empty()
    st.divider()

    start_button = st.button(
        "▶️ Avvia Elaborazione", type="primary", use_container_width=True
    )
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── File uploader ─────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Carica uno o più file CSV envlog, poi premi **Avvia Elaborazione**",
    type=["csv"],
    accept_multiple_files=True,
)
if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

# ── Processing ────────────────────────────────────────────────────────────────

if start_button and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    total     = len(raw_files)

    logger_dfs: dict[str, pd.DataFrame] = {}
    pbar = sidebar_progress.progress(0, text="Avvio…")

    for i, file in enumerate(raw_files):
        pct  = int((i / total) * 100)
        text = f"Elaborazione {i + 1}/{total}: {file.name}"
        pbar.progress(pct, text=text)
        sidebar_status.caption(text)

        try:
            raw_df   = pd.read_csv(file, header=None)
            clean_df = parse_envlog_csv(raw_df, default_lat, default_lon)
            proc_df  = add_rolling_mean(clean_df, window_size=window_size)
            logger_dfs[file.name] = proc_df
        except Exception as exc:
            st.warning(f"Errore nel file **{file.name}**: {exc}")

    pbar.progress(100, text="✅ Fatto!")
    sidebar_status.caption(
        f"Elaborati {len(logger_dfs)}/{total} file con successo."
    )

    if not logger_dfs:
        st.error("Nessun dataset logger valido trovato.")
        st.stop()

    st.session_state["logger_dfs"] = logger_dfs
    st.session_state["default_lat"] = default_lat
    st.session_state["default_lon"] = default_lon

# ── Display ───────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state:
    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    d_lat = st.session_state.get("default_lat", default_lat)
    d_lon = st.session_state.get("default_lon", default_lon)

    # Coordinate: dal primo logger (già validate/di-default)
    first_df  = next(iter(logger_dfs.values()))
    latitude  = float(first_df["latitude"].iloc[0])
    longitude = float(first_df["longitude"].iloc[0])

    # Fetch CORA una volta sola
    with st.spinner("Caricamento dati CORA…"):
        cora_df = fetch_cora_data(latitude, longitude)

    if cora_df is None or cora_df.empty:
        st.error(
            "I dati CORA non sono disponibili. "
            "Controlla la connessione e riprova."
        )
        st.stop()

    # ── Sezione per-file ──────────────────────────────────────────────────

    for fname, sdata in logger_dfs.items():
        st.subheader(f"📄 {fname}")

        # Metriche
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Temperatura media",   f"{sdata['temperature'].mean():.2f} °C")
        col_b.metric("Temperatura mediana", f"{sdata['temperature'].median():.2f} °C")
        col_c.metric("Dev. standard",       f"{sdata['temperature'].std():.2f} °C")
        col_d.metric(
            "Coordinate",
            f"{sdata['latitude'].iloc[0]:.4f}, {sdata['longitude'].iloc[0]:.4f}",
        )

        # Tabella dati grezzi (collassabile)
        with st.expander("📋 Dati grezzi (prime 50 righe)"):
            st.dataframe(
                sdata[["time", "temperature", "temperature_rolling_mean"]]
                .head(50)
                .reset_index(drop=True),
                use_container_width=True,
            )

        # Figura 2×2
        fig = plot_series_and_doy(sdata, cora_df, latitude, longitude)
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

    # ── Sezione summary ───────────────────────────────────────────────────

    st.header("📊 Riepilogo — Tutti i Logger vs CORA")

    rows = []
    for fname, sdata in logger_dfs.items():
        rows.append({
            "File":         fname,
            "Logger":       sdata["custom_name"].iloc[0],
            "Mese":         sdata["time"].iloc[0].strftime("%B %Y"),
            "Media (°C)":   round(sdata["temperature"].mean(),   2),
            "Mediana (°C)": round(sdata["temperature"].median(), 2),
            "Std (°C)":     round(sdata["temperature"].std(),    2),
            "N campioni":   len(sdata),
            "Lat":          round(float(sdata["latitude"].iloc[0]),  4),
            "Lon":          round(float(sdata["longitude"].iloc[0]), 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Plot DOY globale
    fig3 = plot_doy_all(cora_df, logger_dfs, latitude, longitude)
    st.pyplot(fig3)
    plt.close(fig3)

    st.divider()

    # Plot mensile globale
    fig4 = plot_monthly_all(cora_df, logger_dfs, latitude, longitude)
    st.pyplot(fig4)
    plt.close(fig4)

    st.info(
        "⭐ stelle = 2025 · ▲ triangoli = 2026 · ■ quadrati = 2027 · ● cerchi = altri anni\n\n"
        "**Marker pieno** = media · **Marker aperto** = mediana"
    )

    st.divider()

cs_mach1_footer(
    text="CS-MACH · EnvLogger Analysis Pipeline · Ocean temperature comparison: in-situ loggers vs CORA reanalysis"
)
