"""
app.py
------
CS-MACH1 EnvLogger Pipeline — Con download PDF
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
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


# ── Parser (invariato) ────────────────────────────────────────────────────────

@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


def extract_metadata(df: pd.DataFrame, default_lat: float, default_lon: float) -> LoggerMetadata:
    try:
        serial_row = 8 if "serial" in str(df.iloc[8, 0]).lower() else 9
        serial = str(df.iloc[serial_row, 1]).strip()
        custom_name = str(df.iloc[serial_row + 1, 1]).strip()
        sampling = str(df.iloc[13 if "samples" in str(df.iloc[13, 0]).lower() else 14, 1]).strip()

        lat_row = None
        for i in range(14, 22):
            if "lat" in str(df.iloc[i, 0]).lower():
                lat_row = i
                break

        if lat_row:
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
            raise ValueError("HTML error")
        df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna(subset=["time", "TEMP"])
    except Exception as e:
        st.warning(f"CORA download failed: {e}")
        return None


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_individual_logger(sdata: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sdata["time"], sdata["temperature"], color="steelblue", linewidth=1.6, alpha=0.85, label="Raw Temperature")
    rolling = sdata["temperature"].rolling(window=7, min_periods=1).mean()
    ax.plot(sdata["time"], rolling, color="red", linewidth=2.8, label="7-day Rolling Mean")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"Logger: {sdata['custom_name'].iloc[0]}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_interannual_variability(cora_df: pd.DataFrame | None, logger_dfs: dict, lat: float, lon: float):
    fig, ax = plt.subplots(figsize=(12, 6))

    if cora_df is not None and not cora_df.empty:
        cora_data = cora_df.copy()
        cora_data['day_of_year'] = cora_data['time'].dt.dayofyear
        daily_cora = cora_data.groupby('day_of_year')['TEMP'].agg(['mean', 'std']).reset_index()

        ax.plot(daily_cora['day_of_year'], daily_cora['mean'], color='blue', linewidth=2, label='CORA Interannual Mean')
        ax.fill_between(daily_cora['day_of_year'],
                        daily_cora['mean'] - daily_cora['std'].fillna(0),
                        daily_cora['mean'] + daily_cora['std'].fillna(0),
                        color='lightblue', alpha=0.4, label='CORA ± Std')

    if logger_dfs:
        all_logger = pd.concat(logger_dfs.values())
        if not all_logger.empty:
            all_logger['day_of_year'] = all_logger['time'].dt.dayofyear
            daily_logger = all_logger.groupby('day_of_year')['temperature'].agg(['mean', 'max', 'min']).reset_index()

            ax.plot(daily_logger['day_of_year'], daily_logger['mean'], color='red', label='Loggers Mean', marker='o', markersize=3)
            ax.plot(daily_logger['day_of_year'], daily_logger['max'], color='darkorange', label='Loggers Max', linestyle=':', marker='^', markersize=3)
            ax.plot(daily_logger['day_of_year'], daily_logger['min'], color='green', label='Loggers Min', linestyle='--', marker='v', markersize=3)

    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title(f'Interannual Temperature Variability at ({lat:.2f}, {lon:.2f})')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


# ── Download PDF Helper ───────────────────────────────────────────────────────

def download_plot_as_pdf(fig, filename="plot.pdf"):
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf


# ── App ───────────────────────────────────────────────────────────────────────

apply_cs_mach1_theme(
    page_title="CS-MACH1 fixed envlogger pipeline",
    main_title="CS-MACH1: What does a fixed envlogger say about Sea Water Temperature?",
    subtitle="Ocean temperature comparison platform (in-situ loggers vs CORA reanalysis)",
    page_icon="logo.png",
)

with st.sidebar:
    st.markdown("### Settings")
    default_lat = st.number_input("Default Latitude", value=44.376, format="%.4f")
    default_lon = st.number_input("Default Longitude", value=9.071, format="%.4f")

uploaded_files = st.file_uploader("Upload EnvLogger CSV files", type=["csv"], accept_multiple_files=True)

if st.button("🚀 Process Files", type="primary", use_container_width=True) and uploaded_files:
    logger_dfs = {}
    for i, file in enumerate(uploaded_files):
        try:
            raw_df = pd.read_csv(file, header=None)
            clean_df = parse_envlog_csv(raw_df, default_lat, default_lon)
            logger_dfs[file.name] = clean_df
        except Exception as e:
            st.error(f"Error with {file.name}: {e}")

    if logger_dfs:
        st.session_state.logger_dfs = logger_dfs
        st.success(f"✅ {len(logger_dfs)} files processed successfully!")

# ── Results ───────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state and st.session_state.logger_dfs:
    logger_dfs = st.session_state.logger_dfs
    first_df = next(iter(logger_dfs.values()))
    lat = float(first_df["latitude"].iloc[0])
    lon = float(first_df["longitude"].iloc[0])

    cora_df = fetch_cora_data(lat, lon)

    # Plot Combinato + Download
    st.header("📊 Interannual Variability — All Loggers vs CORA")
    fig_combined = plot_interannual_variability(cora_df, logger_dfs, lat, lon)
    st.pyplot(fig_combined)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📥 Download Combined Plot as PDF", use_container_width=True):
            pdf_bytes = download_plot_as_pdf(fig_combined, "CS_MACH1_Combined_Plot.pdf")
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="CS_MACH1_Combined_Plot.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # Summary Table
    st.header("📋 Summary Table")
    summary = []
    for fname, df in logger_dfs.items():
        summary.append({
            "File": fname,
            "Logger Name": df["custom_name"].iloc[0],
            "Mean (°C)": round(df["temperature"].mean(), 2),
            "Max (°C)": round(df["temperature"].max(), 2),
            "Min (°C)": round(df["temperature"].min(), 2),
            "Std (°C)": round(df["temperature"].std(), 2),
            "Samples": len(df),
        })
    st.dataframe(pd.DataFrame(summary), use_container_width=True)

    # Plots Individuali + Download
    st.header("📈 Individual Logger Analysis")
    for fname, sdata in logger_dfs.items():
        with st.expander(f"🔍 {fname} — {sdata['custom_name'].iloc[0]}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{sdata['temperature'].mean():.2f} °C")
            col2.metric("Max", f"{sdata['temperature'].max():.2f} °C")
            col3.metric("Min", f"{sdata['temperature'].min():.2f} °C")
            col4.metric("Samples", len(sdata))

            fig_ind = plot_individual_logger(sdata)
            st.pyplot(fig_ind)

            if st.button(f"📥 Download PDF — {sdata['custom_name'].iloc[0]}", key=f"dl_{fname}"):
                pdf_bytes = download_plot_as_pdf(fig_ind, f"CS_MACH1_{sdata['custom_name'].iloc[0]}.pdf")
                st.download_button(
                    label="⬇️ Download this plot as PDF",
                    data=pdf_bytes,
                    file_name=f"CS_MACH1_{sdata['custom_name'].iloc[0]}.pdf",
                    mime="application/pdf"
                )

    cs_mach1_footer("CS-MACH1 · Fixed EnvLogger Analysis Pipeline")
