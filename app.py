"""
app.py
------
CS-MACH EnvLogger Pipeline — Simplified Version
Only one main DOY plot as requested.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

apply_cs_mach1_theme(
    page_title="CS-MACH1 fixed envlogger pipeline",
    main_title="CS-MACH1: What does a fixed envlogger say about Sea Water Temperature?",
    subtitle="Ocean temperature comparison platform (in-situ loggers vs CORA reanalysis)",
    page_icon="logo.png",
)

# ── Constants ─────────────────────────────────────────────────────────────────

CORA_URL_TEMPLATE = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)


# ── Metadata & Parser ─────────────────────────────────────────────────────────

@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


def extract_metadata(df: pd.DataFrame, default_lat: float, default_lon: float) -> LoggerMetadata:
    try:
        # Flexible row detection
        serial_row = 8 if "serial" in str(df.iloc[8, 0]).lower() else 9
        name_row = serial_row + 1
        samples_row = 13 if "samples" in str(df.iloc[13, 0]).lower() else 14

        serial = str(df.iloc[serial_row, 1]).strip()
        custom_name = str(df.iloc[name_row, 1]).strip()
        sampling = str(df.iloc[samples_row, 1]).strip()

        # Find latitude row
        lat_row = None
        for i in range(14, 22):
            cell = str(df.iloc[i, 0]).lower()
            if "lat" in cell:
                lat_row = i
                break

        if lat_row is not None:
            lat = pd.to_numeric(df.iloc[lat_row, 1], errors="coerce")
            lon = pd.to_numeric(df.iloc[lat_row + 1, 1], errors="coerce")
        else:
            lat = lon = None

        if pd.isna(lat) or pd.isna(lon):
            lat, lon = default_lat, default_lon

        return LoggerMetadata(serial, custom_name or "Unknown", sampling, float(lat), float(lon))
    except:
        return LoggerMetadata("Unknown", "Unknown", "Unknown", default_lat, default_lon)


def parse_envlog_csv(df: pd.DataFrame, default_lat: float, default_lon: float) -> pd.DataFrame:
    metadata = extract_metadata(df, default_lat, default_lon)

    # Find where data starts
    start_row = 20
    for i in range(15, 30):
        if "time" in str(df.iloc[i, 0]).lower():
            start_row = i + 1
            break

    clean_df = df.iloc[start_row:, :].copy()
    clean_df = clean_df.dropna(how="all").reset_index(drop=True)
    clean_df.columns = ["time", "temperature"]

    clean_df["time"] = pd.to_datetime(clean_df["time"], errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"], errors="coerce")

    clean_df["serial"] = metadata.serial
    clean_df["custom_name"] = metadata.custom_name
    clean_df["latitude"] = metadata.latitude
    clean_df["longitude"] = metadata.longitude

    return clean_df.dropna(subset=["time", "temperature"]).reset_index(drop=True)


# ── CORA ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading CORA data...")
def fetch_cora_data(lat: float, lon: float) -> pd.DataFrame | None:
    url = CORA_URL_TEMPLATE.format(lat=round(lat, 4), lon=round(lon, 4))
    try:
        resp = requests.get(url, verify=False, timeout=60)
        resp.raise_for_status()
        if "<html" in resp.text.lower():
            raise ValueError("HTML returned")
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna(subset=["time", "TEMP"])
    except Exception as e:
        st.warning(f"Could not download CORA data: {e}")
        return None


# ── Main Plot (exactly as in your notebook) ───────────────────────────────────

def plot_interannual_variability(cora_df: pd.DataFrame, logger_dfs: dict, lat: float, lon: float):
    fig, ax = plt.subplots(figsize=(12, 6))

    cora_plotted = False
    logger_plotted = False

    # CORA
    if cora_df is not None and not cora_df.empty:
        cora_data = cora_df.copy()
        cora_data['day_of_year'] = cora_data['time'].dt.dayofyear
        daily_cora = cora_data.groupby('day_of_year')['TEMP'].agg(['mean', 'std']).reset_index()

        ax.plot(daily_cora['day_of_year'], daily_cora['mean'],
                color='blue', linewidth=2, label='CORA Interannual Mean')
        ax.fill_between(daily_cora['day_of_year'],
                        daily_cora['mean'] - daily_cora['std'].fillna(0),
                        daily_cora['mean'] + daily_cora['std'].fillna(0),
                        color='lightblue', alpha=0.4, label='CORA ± Std')
        cora_plotted = True

    # Loggers combined
    if logger_dfs:
        all_logger_df = pd.concat(logger_dfs.values())
        if not all_logger_df.empty:
            all_logger_df['day_of_year'] = all_logger_df['time'].dt.dayofyear
            daily_logger = all_logger_df.groupby('day_of_year')['temperature'].agg(['mean', 'max', 'min']).reset_index()

            ax.plot(daily_logger['day_of_year'], daily_logger['mean'],
                    color='red', label='Loggers Mean', linestyle='-', marker='o', markersize=3)
            ax.plot(daily_logger['day_of_year'], daily_logger['max'],
                    color='darkorange', label='Loggers Max', linestyle=':', marker='^', markersize=3)
            ax.plot(daily_logger['day_of_year'], daily_logger['min'],
                    color='green', label='Loggers Min', linestyle='--', marker='v', markersize=3)
            logger_plotted = True

    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title(f'Interannual Temperature Variability at ({lat:.2f}, {lon:.2f})')
    ax.grid(True)

    if cora_plotted or logger_plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout()
    return fig


# ── Streamlit App ─────────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("### Settings")
    default_lat = st.number_input("Default Latitude", value=44.376, format="%.4f")
    default_lon = st.number_input("Default Longitude", value=9.071, format="%.4f")

uploaded_files = st.file_uploader(
    "Upload one or more EnvLogger CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if st.button("🚀 Process Files", type="primary", use_container_width=True) and uploaded_files:
    logger_dfs = {}
    progress = st.progress(0)

    for i, file in enumerate(uploaded_files):
        progress.progress((i + 1) / len(uploaded_files), text=f"Processing {file.name}")
        try:
            raw_df = pd.read_csv(file, header=None)
            clean_df = parse_envlog_csv(raw_df, default_lat, default_lon)
            logger_dfs[file.name] = clean_df
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if logger_dfs:
        st.session_state.logger_dfs = logger_dfs
        st.success(f"✅ {len(logger_dfs)} file(s) processed!")

# Display
if "logger_dfs" in st.session_state and st.session_state.logger_dfs:
    logger_dfs = st.session_state.logger_dfs
    first_df = next(iter(logger_dfs.values()))
    lat = float(first_df["latitude"].iloc[0])
    lon = float(first_df["longitude"].iloc[0])

    cora_df = fetch_cora_data(lat, lon)

    st.header("📊 Interannual Variability (All Loggers vs CORA)")
    fig = plot_interannual_variability(cora_df, logger_dfs, lat, lon)
    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    summary = []
    for fname, df in logger_dfs.items():
        summary.append({
            "File": fname,
            "Logger Name": df["custom_name"].iloc[0],
            "Mean (°C)": round(df["temperature"].mean(), 2),
            "Max (°C)": round(df["temperature"].max(), 2),
            "Min (°C)": round(df["temperature"].min(), 2),
            "Samples": len(df),
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    cs_mach1_footer("CS-MACH · EnvLogger Analysis Pipeline")
