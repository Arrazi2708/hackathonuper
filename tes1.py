"""
GeoLink v2.0 — Landslide Mitigation Dashboard
Tema : Bridging Science and Society Through Geophysical Interpretation
Author: GeoLink Team — Hackathon Geofisika Indonesia 2026

PERBAIKAN dari versi sebelumnya:
- Perbaikan routing halaman (hapus kode misplaced yang tidak punya elif guard)
- Hapus duplikat with tab2: di halaman Peta Zonasi Risiko Terpadu
- Tambah fungsi render_rekoms yang hilang
- Perbaikan seismic intercept-time method (two-segment fitting)
- integrate_datasets menggunakan pd.merge + round(4) sinkronisasi koordinat
- GeoBot terintegrasi dengan Anthropic API (Claude Sonnet)
- Seismic cross-section sebagai Area Chart (fill='tozeroy')
- Geoelectric pseudosection sebagai Contour Map (go.Contour)
- Skor risiko kumulatif DSS berbobot dari 3 parameter (slope, weathering, slip plane)
- Perbaikan Scattermapbox tanpa properti 'line' untuk mencegah ValueError
- Session state konsisten: topo_data, seismic_data, geoelectric_data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
import datetime
import json

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GeoLink v2.0 — Mitigasi Longsor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS (statis — tidak ada variabel Python)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark:    #0d1117;
    --bg-card:    #161b22;
    --bg-border:  #30363d;
    --accent:     #f78166;
    --accent2:    #ffa657;
    --green:      #3fb950;
    --yellow:     #d29922;
    --red:        #f85149;
    --blue:       #4da6ff;
    --text:       #e6edf3;
    --text-muted: #8b949e;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid var(--bg-border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { font-size: 1.8rem !important; letter-spacing: -1px; }

/* Cards */
.geo-card {
    background: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.geo-card-accent { border-left: 4px solid var(--accent); }
.geo-card-green  { border-left: 4px solid var(--green); }
.geo-card-blue   { border-left: 4px solid var(--blue); }

/* Risk Badges */
.badge-aman    { background:#1a3a25; color:#3fb950; border:1px solid #3fb950; border-radius:20px; padding:4px 16px; font-weight:700; font-size:1rem; }
.badge-waspada { background:#3a2e00; color:#d29922; border:1px solid #d29922; border-radius:20px; padding:4px 16px; font-weight:700; font-size:1rem; }
.badge-bahaya  { background:#3a0f0f; color:#f85149; border:1px solid #f85149; border-radius:20px; padding:4px 16px; font-weight:700; font-size:1rem; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size:0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Space Mono', monospace !important; }

/* Upload */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--bg-border);
    border-radius: 12px;
    padding: 1rem;
}

/* Tabs */
button[role="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Dividers */
.geo-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #f78166, transparent);
    margin: 1.5rem 0;
}

/* Info Box */
.info-box {
    background: linear-gradient(135deg, #1a1f2e, #161b22);
    border: 1px solid #264f78;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    color: #79c0ff;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 50%, #161b22 100%);
    border: 1px solid var(--bg-border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f78166, #ffa657, #f78166);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-subtitle { color: var(--text-muted); font-size: 1.05rem; max-width: 600px; margin: 0 auto; }

/* Feature Cards */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 12px;
    padding: 1.4rem;
    height: 100%;
}
.feature-icon  { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-title { font-family: 'Space Mono', monospace; font-size: 0.95rem; color: #ffa657; margin-bottom: 0.5rem; }
.feature-desc  { color: var(--text-muted); font-size: 0.88rem; line-height: 1.6; }

/* Interpretation Boxes */
.interpret-safe    { background:#0d2117; border-left:4px solid #3fb950; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin-top:0.5rem; }
.interpret-waspada { background:#1f1a00; border-left:4px solid #d29922; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin-top:0.5rem; }
.interpret-bahaya  { background:#1f0707; border-left:4px solid #f85149; border-radius:0 8px 8px 0; padding:1rem 1.2rem; margin-top:0.5rem; }

/* Report */
.report-section {
    background: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
}
.report-header {
    font-family: 'Space Mono', monospace;
    color: #f78166;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--bg-border);
    padding-bottom: 0.5rem;
}

/* Rekomendasi Items */
.rekom-item {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.9rem;
    align-items: flex-start;
}
.rekom-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem; flex-shrink: 0; margin-top: 3px; font-weight: 700;
}
.rekom-num.aman    { background: rgba(63,185,80,0.13);  border: 1px solid #3fb950; color: #3fb950; }
.rekom-num.waspada { background: rgba(210,153,34,0.13); border: 1px solid #d29922; color: #d29922; }
.rekom-num.bahaya  { background: rgba(248,81,73,0.13);  border: 1px solid #f85149; color: #f85149; }
.rekom-title { font-weight: 600; font-size: 0.9rem; margin-bottom: 2px; }
.rekom-title.aman    { color: #3fb950; }
.rekom-title.waspada { color: #d29922; }
.rekom-title.bahaya  { color: #f85149; }
.rekom-desc { color: #8b949e; font-size: 0.85rem; line-height: 1.6; }

/* DSS Score Bar */
.dss-bar-container {
    background: #0d1117;
    border-radius: 6px;
    height: 12px;
    overflow: hidden;
    margin-top: 4px;
}
.dss-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}

/* Status chips */
.chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.chip-ok   { background:#1a3a25; color:#3fb950; border:1px solid #3fb95044; }
.chip-warn { background:#3a2e00; color:#d29922; border:1px solid #d2992244; }
.chip-bad  { background:#3a0f0f; color:#f85149; border:1px solid #f8514944; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════
for key, default in [
    ("analysis_result",  None),
    ("topo_data",        None),   # lon, lat, elev, slopes
    ("seismic_data",     None),   # distance, time, v1, v2, weathering_depth, bedrock_start
    ("geoelectric_data", None),   # distance, depth, resistivity + analysis fields
    ("geobot_messages",  []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═════════════════════════════════════════════════════════
# CONSTANTS — PVMBG Standards
# ═════════════════════════════════════════════════════════
SLOPE_SAFE       = 15.0   # degrees
SLOPE_WARN       = 30.0
VP_WEATHERING    = 800    # m/s threshold
RESIST_SLIP      = 50     # Ω.m — bidang gelincir
RESIST_BEDROCK   = 150    # Ω.m — bedrock stabil
WEATHER_WARN_M   = 3.0    # m — pelapukan sedang
WEATHER_CRIT_M   = 6.0    # m — pelapukan kritis

ACTIONS = {
    "AMAN": [
        ("Pemantauan Rutin",
         "Lakukan survei lapangan minimal 1× per tahun untuk memastikan kondisi lereng tidak berubah."),
        ("Jaga Drainase",
         "Pastikan saluran air bersih dari sampah dan sedimen agar tidak ada genangan yang melemahkan tanah."),
        ("Pertahankan Vegetasi",
         "Jangan tebang pohon di lereng — akar pohon adalah pengikat tanah alami yang paling efektif."),
    ],
    "WASPADA": [
        ("Pemetaan Detail",
         "Lakukan pemetaan lereng detail skala 1:10.000 pada area dengan kelerengan lebih dari 15°."),
        ("Sistem Peringatan Dini",
         "Pasang piezometer atau extensometer di titik lereng kritis sebagai early warning system."),
        ("Zonasi Pembangunan",
         "Larang bangunan permanen baru di zona lereng > 20° tanpa kajian geoteknik."),
        ("Sosialisasi Warga",
         "Adakan pelatihan evakuasi dan sosialisasi tanda-tanda longsor kepada masyarakat sekitar."),
    ],
    "BAHAYA": [
        ("Evakuasi Segera",
         "Rekomendasikan evakuasi warga di lereng > 30° ke lokasi yang lebih aman."),
        ("Koordinasi BPBD",
         "Segera koordinasikan temuan ini dengan BPBD setempat untuk mitigasi darurat."),
        ("Moratorium Pembangunan",
         "Hentikan semua kegiatan pembangunan di zona berbahaya hingga kajian geoteknik selesai."),
        ("Rekayasa Lereng",
         "Rencanakan konstruksi dinding penahan tanah (retaining wall) atau checkdam di zona kritis."),
        ("Relokasi Permukiman",
         "Ajukan program relokasi ke pemerintah daerah untuk warga di zona merah."),
    ],
}

DUMMY_TOPO = {
    "lons":   [108.20, 108.21, 108.22, 108.23, 108.24],
    "lats":   [-6.80,  -6.81,  -6.82,  -6.83,  -6.84],
    "slopes": [35, 18, 8, 28, 12],
}
DUMMY_SEISMIC = {
    "v1": 320.0, "v2": 1450.0,
    "weathering_depth": 5.8, "bedrock_start": 8.2,
    "distances": np.linspace(0, 200, 20),
    "arrival_times": np.linspace(0.005, 0.14, 20),
}
DUMMY_GEOELECTRIC = {
    "resistivity_mean": 42.0, "min_resistivity": 18.0, "max_resistivity": 245.0,
    "has_slip_plane": True, "slip_depth": 6.5,
    "weathered_depth": 5.0, "bedrock_start": 9.0,
    "distances": np.linspace(0, 200, 15),
    "depths": np.linspace(0, 15, 10),
    "resistivities": np.random.uniform(20, 250, 150),
}


# ═════════════════════════════════════════════════════════
# HELPER — CLASSIFY RISK
# ═════════════════════════════════════════════════════════
def classify_risk(max_slope: float):
    """Return (label, badge_css_class, hex_color, css_variant)."""
    if max_slope < SLOPE_SAFE:
        return "AMAN",    "badge-aman",    "#3fb950", "aman"
    elif max_slope < SLOPE_WARN:
        return "WASPADA", "badge-waspada", "#d29922", "waspada"
    else:
        return "BAHAYA",  "badge-bahaya",  "#f85149", "bahaya"


# ─────────────────────────────────────────────
# render_rekoms — render rekomendasi tindakan
# ─────────────────────────────────────────────
def render_rekoms(actions: list, css_cls: str):
    """Render rekomendasi sebagai styled HTML. css_cls: 'aman'|'waspada'|'bahaya'."""
    for i, (title, desc) in enumerate(actions, 1):
        num_html = (
            f"<div class='rekom-num {css_cls}'>{i}</div>"
        )
        body_html = (
            f"<div>"
            f"<div class='rekom-title {css_cls}'>{title}</div>"
            f"<div class='rekom-desc'>{desc}</div>"
            f"</div>"
        )
        st.markdown(
            f"<div class='rekom-item'>{num_html}{body_html}</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════
# TOPOGRAFI FUNCTIONS
# ═════════════════════════════════════════════════════════
def compute_slope(elev_grid: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    dy, dx = np.gradient(elev_grid, cell_size)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))


def process_topo_csv(df: pd.DataFrame):
    """Parse CSV (lon,lat,elev) → (elev_grid, raw_df). Returns (None,None) on error."""
    df.columns = df.columns.str.lower().str.strip()
    required = {"lon", "lat", "elev"}
    if not required.issubset(set(df.columns)):
        st.error(f"❌ CSV topografi harus memiliki kolom: lon, lat, elev. Ditemukan: {list(df.columns)}")
        return None, None
    lon  = df["lon"].values.astype(float)
    lat  = df["lat"].values.astype(float)
    elev = df["elev"].values.astype(float)
    res  = 100
    xi   = np.linspace(lon.min(), lon.max(), res)
    yi   = np.linspace(lat.min(), lat.max(), res)
    xi_g, yi_g = np.meshgrid(xi, yi)
    try:
        eg = griddata((lon, lat), elev, (xi_g, yi_g), method="cubic")
        m  = np.isnan(eg)
        if m.any():
            eg[m] = griddata((lon, lat), elev, (xi_g, yi_g), method="linear")[m]
        m2 = np.isnan(eg)
        if m2.any():
            eg[m2] = griddata((lon, lat), elev, (xi_g, yi_g), method="nearest")[m2]
    except Exception as e:
        st.error(f"❌ Gridding gagal: {e}")
        return None, None
    return eg, df


def process_tif(file_bytes: bytes):
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as ds:
                eg = ds.read(1).astype(float)
                if ds.nodata is not None:
                    eg[eg == ds.nodata] = np.nan
                if max(eg.shape) > 300:
                    try:
                        from skimage.transform import resize
                        f  = 300 / max(eg.shape)
                        eg = resize(eg, (int(eg.shape[0]*f), int(eg.shape[1]*f)), preserve_range=True)
                    except ImportError:
                        step = max(1, max(eg.shape) // 300)
                        eg   = eg[::step, ::step]
        return eg
    except ImportError:
        st.error("❌ rasterio tidak terinstall. Gunakan file CSV.")
        return None
    except Exception as e:
        st.error(f"❌ Gagal membaca TIF: {e}")
        return None


def make_slope_fig(slope_grid: np.ndarray) -> go.Figure:
    fig = px.imshow(
        slope_grid,
        color_continuous_scale="RdYlGn_r",
        labels={"color": "Kelerengan (°)"},
        title="Peta Kelerengan 2D",
    )
    fig.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        title_font=dict(family="Space Mono", size=13),
        coloraxis_colorbar=dict(tickfont=dict(color="#8b949e"), title=dict(text="°", font=dict(color="#8b949e"))),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_surface_fig(elev_grid: np.ndarray) -> go.Figure:
    fig = go.Figure(data=[go.Surface(
        z=elev_grid, colorscale="Viridis", showscale=True,
        colorbar=dict(tickfont=dict(color="#8b949e"), title=dict(text="m dpl", font=dict(color="#8b949e"))),
    )])
    fig.update_layout(
        paper_bgcolor="#161b22",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(gridcolor="#30363d", color="#8b949e"),
            yaxis=dict(gridcolor="#30363d", color="#8b949e"),
            zaxis=dict(gridcolor="#30363d", color="#8b949e", title="Elevasi (m)"),
        ),
        font_color="#e6edf3",
        title=dict(text="Model Topografi 3D Interaktif", font=dict(family="Space Mono", size=13)),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ═════════════════════════════════════════════════════════
# SEISMIC REFRACTION FUNCTIONS
# ═════════════════════════════════════════════════════════
def intercept_time_method(distances: np.ndarray, times: np.ndarray) -> dict:
    """
    Two-layer intercept-time method (PVMBG standard).
    Fits two linear segments to T-X curve to extract V1, V2, and weathering depth.
    Returns dict with v1, v2, intercept_t, weathering_depth, bedrock_start.
    """
    n = len(distances)
    if n < 4:
        # Fallback — single linear fit
        z   = np.polyfit(distances, times, 1)
        v1  = float(1.0 / z[0]) if z[0] > 0 else 300.0
        ti  = max(float(z[1]), 1e-6)
        wd  = float(np.clip((ti * v1) / 2.0, 0.5, 20.0))
        return {"v1": v1, "v2": v1 * 3.5, "intercept_t": ti,
                "weathering_depth": wd, "bedrock_start": wd + 2.0}

    best_break = None
    best_resid = np.inf
    # Try break points from 20% to 80% of data
    for bp in range(max(2, n // 5), min(n - 2, int(0.8 * n))):
        seg1  = np.polyfit(distances[:bp],  times[:bp],  1)
        seg2  = np.polyfit(distances[bp:],  times[bp:],  1)
        r1    = np.sum((times[:bp]  - np.polyval(seg1, distances[:bp])) ** 2)
        r2    = np.sum((times[bp:]  - np.polyval(seg2, distances[bp:])) ** 2)
        total = r1 + r2
        if total < best_resid:
            best_resid  = total
            best_break  = bp
            best_seg1   = seg1
            best_seg2   = seg2

    bp = best_break
    # V1 from first segment slope (1/slope = velocity)
    s1  = best_seg1[0] if best_seg1[0] > 1e-9 else 1e-9
    s2  = best_seg2[0] if best_seg2[0] > 1e-9 else 1e-9
    v1  = float(np.clip(1.0 / s1, 100, 2000))
    v2  = float(np.clip(1.0 / s2, v1 + 100, 6000))

    # Intercept time: extrapolate second-segment fit to x=0
    ti  = float(best_seg2[1])  # y-intercept of second segment
    # Depth from intercept-time formula: h = (ti * V1 * V2) / (2 * sqrt(V2^2 - V1^2))
    denom = 2.0 * np.sqrt(max(v2 ** 2 - v1 ** 2, 1.0))
    wd    = float(np.clip((ti * v1 * v2) / denom, 0.5, 25.0))

    return {
        "v1": v1, "v2": v2,
        "intercept_t": max(ti, 0.0),
        "weathering_depth": wd,
        "bedrock_start": wd + 1.5,
    }


def make_seismic_area_chart(distances: np.ndarray, depths: np.ndarray,
                             v1: float, v2: float) -> go.Figure:
    """
    Seismic refraction profil lapisan pelapukan — Area Chart (fill='tozeroy').
    """
    fig = go.Figure()
    # Layer weathering — filled to zero
    fig.add_trace(go.Scatter(
        x=distances, y=depths,
        fill="tozeroy",
        fillcolor="rgba(255,166,87,0.25)",
        line=dict(color="#ffa657", width=2.5),
        name="Kedalaman Pelapukan",
        hovertemplate="Jarak: %{x:.0f} m<br>Kedalaman: %{y:.2f} m<extra></extra>",
    ))
    # Mean weathering depth horizontal line
    mean_d = float(np.nanmean(depths))
    fig.add_hline(
        y=mean_d,
        line=dict(color="#f78166", width=1.5, dash="dot"),
        annotation_text=f"  Rata-rata: {mean_d:.1f} m",
        annotation_font=dict(color="#f78166", size=11),
    )
    # V1/V2 annotation
    fig.add_annotation(
        x=float(distances[-1] * 0.6), y=float(depths.max() * 0.3),
        text=f"<b>V₁ = {v1:.0f} m/s</b><br>V₂ = {v2:.0f} m/s",
        showarrow=False,
        font=dict(color="#4da6ff", size=12, family="Space Mono"),
        bgcolor="#0d1117", bordercolor="#4da6ff", borderwidth=1,
    )
    fig.update_layout(
        title="Profil Lapisan Pelapukan — Seismic Refraction",
        xaxis_title="Jarak (m)",
        yaxis_title="Kedalaman Pelapukan (m)",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        title_font=dict(family="Space Mono", size=13),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    return fig


def make_traveltime_chart(distances: np.ndarray, times: np.ndarray,
                           v1: float, v2: float) -> go.Figure:
    """Travel-time vs distance scatter + fit lines."""
    fig = go.Figure()
    # Observed points
    fig.add_trace(go.Scatter(
        x=distances, y=times * 1000,
        mode="markers",
        marker=dict(color="#4da6ff", size=7),
        name="Data Observasi",
    ))
    # V1 fit
    t_v1 = distances / v1 * 1000
    fig.add_trace(go.Scatter(
        x=distances, y=t_v1,
        mode="lines", line=dict(color="#ffa657", width=2, dash="dash"),
        name=f"V₁ = {v1:.0f} m/s",
    ))
    # V2 fit (simplified: just second half)
    t_v2 = distances / v2 * 1000
    fig.add_trace(go.Scatter(
        x=distances, y=t_v2,
        mode="lines", line=dict(color="#3fb950", width=2, dash="dash"),
        name=f"V₂ = {v2:.0f} m/s",
    ))
    fig.update_layout(
        title="Kurva Waktu Tempuh (T-X)",
        xaxis_title="Jarak (m)",
        yaxis_title="Waktu (ms)",
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        title_font=dict(family="Space Mono", size=13),
        legend=dict(bgcolor="#0d1117", bordercolor="#30363d"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ═════════════════════════════════════════════════════════
# GEOELECTRIC FUNCTIONS
# ═════════════════════════════════════════════════════════
def analyze_geoelectric(dist: np.ndarray, depth: np.ndarray, res: np.ndarray) -> dict:
    """
    Analisis pseudoseksi geoelektrik.
    Identifikasi bidang gelincir (resistivitas < 50 Ω.m) dan bedrock (> 150 Ω.m).
    """
    slip_mask    = res < RESIST_SLIP
    bedrock_mask = res > RESIST_BEDROCK

    has_slip     = bool(slip_mask.any())
    slip_depth   = float(depth[slip_mask].mean())  if has_slip              else float(depth.mean())
    bedrock_d    = float(depth[bedrock_mask].min()) if bedrock_mask.any()   else float(depth.max())
    weathered_d  = float(depth[res < 100].max())   if (res < 100).any()    else bedrock_d - 1.0
    weathered_d  = max(weathered_d, 0.5)

    return {
        "has_slip_plane":  has_slip,
        "slip_depth":      slip_depth,
        "weathered_depth": weathered_d,
        "bedrock_start":   bedrock_d,
        "min_resistivity": float(res.min()),
        "max_resistivity": float(res.max()),
        "mean_resistivity": float(res.mean()),
        "resistivity_median": float(np.median(res)),
    }


def make_geoelectric_contour(dist_1d: np.ndarray, depth_1d: np.ndarray,
                              res_2d: np.ndarray,
                              slip_depth: float = None,
                              bedrock_depth: float = None) -> go.Figure:
    """
    Pseudoseksi geoelektrik sebagai Contour Map (go.Contour).
    """
    fig = go.Figure()

    fig.add_trace(go.Contour(
        z=res_2d,
        x=dist_1d,
        y=depth_1d,
        colorscale=[
            [0.00, "#1f0707"],
            [0.20, "#f85149"],
            [0.40, "#d29922"],
            [0.60, "#ffa657"],
            [0.80, "#3fb950"],
            [1.00, "#4da6ff"],
        ],
        contours=dict(start=float(res_2d.min()), end=float(res_2d.max()), size=20,
                      showlabels=True, labelfont=dict(size=9, color="white")),
        colorbar=dict(
            title=dict(text="Resistivitas (Ω.m)", font=dict(color="#8b949e")),
            tickfont=dict(color="#8b949e"),
            thickness=14, len=0.8,
        ),
        hovertemplate="Jarak: %{x:.0f} m<br>Kedalaman: %{y:.1f} m<br>Resistivitas: %{z:.0f} Ω.m<extra></extra>",
        name="Resistivitas",
    ))

    # Bidang gelincir line
    if slip_depth is not None:
        fig.add_hline(
            y=slip_depth,
            line=dict(color="#f85149", width=2.5, dash="dash"),
            annotation_text=f"  Bidang Gelincir ({slip_depth:.1f} m)",
            annotation_font=dict(color="#f85149", size=11),
        )
    # Bedrock line
    if bedrock_depth is not None:
        fig.add_hline(
            y=bedrock_depth,
            line=dict(color="#3fb950", width=2.5, dash="dot"),
            annotation_text=f"  Bedrock ({bedrock_depth:.1f} m)",
            annotation_font=dict(color="#3fb950", size=11),
        )

    fig.update_layout(
        title="Pseudoseksi Geoelektrik 2D",
        xaxis_title="Jarak (m)",
        yaxis_title="Kedalaman (m)",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        title_font=dict(family="Space Mono", size=13),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    return fig


# ═════════════════════════════════════════════════════════
# INTEGRASI DATASET — pd.merge + koordinat round(4)
# ═════════════════════════════════════════════════════════
def integrate_datasets(topo: dict, seismic: dict, geoelectric: dict) -> pd.DataFrame:
    """
    Gabungkan ketiga dataset ke dalam satu Master DataFrame.
    Sinkronisasi koordinat menggunakan pd.merge dengan pembulatan round(4).
    Menangani perbedaan nama kolom (lat vs lats, lon vs lons) via rename.
    """
    frames = []

    # ── 1. Topo ──────────────────────────────────────────
    if topo and len(topo.get("lons", [])) > 0:
        lons   = np.array(topo["lons"]).round(4)
        lats   = np.array(topo["lats"]).round(4)
        slopes = np.array(topo.get("slopes", np.zeros_like(lons)))
        df_t   = pd.DataFrame({
            "lon":       lons,
            "lat":       lats,
            "slope_deg": slopes.astype(float),
        })
        frames.append(("topo", df_t))

    # ── 2. Seismic ────────────────────────────────────────
    if seismic and len(seismic.get("distances", [])) > 0:
        dists = np.array(seismic["distances"])
        n     = len(dists)
        # Generate coords along transect if not present
        if seismic.get("lats") is not None:
            lats_s = np.array(seismic["lats"]).round(4)
            lons_s = np.array(seismic["lons"]).round(4)
        else:
            lat_base, lon_base = -6.83, 108.22
            t      = dists / (dists.max() + 1e-9)
            lats_s = (lat_base + t * 0.05).round(4)
            lons_s = (lon_base + t * 0.05).round(4)
        wd = float(seismic.get("weathering_depth", 4.0))
        df_s = pd.DataFrame({
            "lon":               lons_s,
            "lat":               lats_s,
            "weathering_depth_m": np.full(n, wd),
            "v1_mps":            np.full(n, float(seismic.get("v1", 300))),
            "v2_mps":            np.full(n, float(seismic.get("v2", 800))),
            "bedrock_depth_m":   np.full(n, float(seismic.get("bedrock_start", wd + 2))),
        })
        frames.append(("seismic", df_s))

    # ── 3. Geoelectric ───────────────────────────────────
    if geoelectric and len(geoelectric.get("distances", [])) > 0:
        dists_g = np.array(geoelectric["distances"])
        n_g     = len(dists_g)
        if geoelectric.get("lats") is not None:
            lats_g = np.array(geoelectric["lats"]).round(4)
            lons_g = np.array(geoelectric["lons"]).round(4)
        else:
            lat_base, lon_base = -6.83, 108.22
            t_g    = dists_g / (dists_g.max() + 1e-9)
            lats_g = (lat_base + t_g * 0.05).round(4)
            lons_g = (lon_base + t_g * 0.05).round(4)
        slip   = bool(geoelectric.get("has_slip_plane", False))
        df_g   = pd.DataFrame({
            "lon":               lons_g,
            "lat":               lats_g,
            "resistivity_ohm":   np.full(n_g, float(geoelectric.get("mean_resistivity",
                                              geoelectric.get("resistivity_mean", 100)))),
            "slip_plane_detected": np.full(n_g, slip, dtype=bool),
            "slip_depth_m":      np.full(n_g, float(geoelectric.get("slip_depth", 5.0))),
        })
        frames.append(("geoelectric", df_g))

    if not frames:
        # Full dummy fallback
        dummy_df = pd.DataFrame({
            "lon":               [108.20, 108.21, 108.22, 108.23, 108.24],
            "lat":               [-6.80,  -6.81,  -6.82,  -6.83,  -6.84],
            "slope_deg":         [35.0, 18.0, 8.0, 28.0, 12.0],
            "weathering_depth_m":[6.5,  4.0,  2.0, 5.5,  3.0],
            "v1_mps":            [320.0]*5,
            "v2_mps":            [1450.0]*5,
            "bedrock_depth_m":   [8.2]*5,
            "resistivity_ohm":   [45.0, 110.0, 220.0, 60.0, 180.0],
            "slip_plane_detected":[True, False, False, True, False],
            "slip_depth_m":      [6.5, 0.0, 0.0, 5.5, 0.0],
        })
        return dummy_df

    # Start with the largest frame, merge others by nearest lat/lon
    frames_sorted = sorted(frames, key=lambda x: len(x[1]), reverse=True)
    master = frames_sorted[0][1].copy()

    for name, df_add in frames_sorted[1:]:
        # Merge on rounded coordinates (outer join)
        merged = pd.merge(master, df_add, on=["lon", "lat"], how="outer", suffixes=("", f"_{name}"))
        # For duplicate columns, keep the non-_suffix version where possible
        drop_cols = [c for c in merged.columns if c.endswith(f"_{name}") and c[:-len(f"_{name}")] in merged.columns]
        merged.drop(columns=drop_cols, inplace=True, errors="ignore")
        master = merged

    # Fill missing numeric columns with defaults
    defaults = {
        "slope_deg": 0.0,
        "weathering_depth_m": 3.0,
        "v1_mps": 300.0,
        "v2_mps": 800.0,
        "bedrock_depth_m": 8.0,
        "resistivity_ohm": 100.0,
        "slip_plane_detected": False,
        "slip_depth_m": 5.0,
    }
    for col, val in defaults.items():
        if col not in master.columns:
            master[col] = val
        else:
            master[col] = master[col].fillna(val)

    master = master.dropna(subset=["lon", "lat"])
    master["lon"] = master["lon"].round(4)
    master["lat"] = master["lat"].round(4)
    return master.reset_index(drop=True)


# ═════════════════════════════════════════════════════════
# DSS — Skor Risiko Kumulatif
# ═════════════════════════════════════════════════════════
def calc_dss_score(slope_deg: float, weathering_depth_m: float,
                   has_slip_plane: bool, resistivity_ohm: float) -> dict:
    """
    Kalkulasi skor risiko DSS (0–100) berbobot dari 3 parameter PVMBG.
    Bobot: slope 40%, weathering 35%, slip/resistivity 25%.
    """
    # Score slope (0–100)
    if slope_deg < SLOPE_SAFE:
        s_slope = (slope_deg / SLOPE_SAFE) * 30
    elif slope_deg < SLOPE_WARN:
        s_slope = 30 + ((slope_deg - SLOPE_SAFE) / (SLOPE_WARN - SLOPE_SAFE)) * 40
    else:
        s_slope = 70 + min(30, (slope_deg - SLOPE_WARN) * 1.5)

    # Score weathering (0–100)
    if weathering_depth_m < WEATHER_WARN_M:
        s_weather = (weathering_depth_m / WEATHER_WARN_M) * 25
    elif weathering_depth_m < WEATHER_CRIT_M:
        s_weather = 25 + ((weathering_depth_m - WEATHER_WARN_M) / (WEATHER_CRIT_M - WEATHER_WARN_M)) * 45
    else:
        s_weather = 70 + min(30, (weathering_depth_m - WEATHER_CRIT_M) * 5)

    # Score slip/resistivity (0–100)
    if has_slip_plane:
        s_slip = 80 + min(20, (RESIST_SLIP - min(resistivity_ohm, RESIST_SLIP)) / RESIST_SLIP * 20)
    elif resistivity_ohm < 100:
        s_slip = 40 + (100 - resistivity_ohm) / 100 * 40
    else:
        s_slip = max(0, 40 - (resistivity_ohm - 100) / 100 * 40)

    total = 0.40 * s_slope + 0.35 * s_weather + 0.25 * s_slip
    total = float(np.clip(total, 0, 100))

    if total < 33:
        level, color, badge = "AMAN",    "#3fb950", "badge-aman"
    elif total < 66:
        level, color, badge = "WASPADA", "#d29922", "badge-waspada"
    else:
        level, color, badge = "BAHAYA",  "#f85149", "badge-bahaya"

    return {
        "risk_score":   round(total, 1),
        "risk_level":   level,
        "risk_color":   color,
        "risk_badge":   badge,
        "score_slope":  round(s_slope, 1),
        "score_weather":round(s_weather, 1),
        "score_slip":   round(s_slip, 1),
    }


# ═════════════════════════════════════════════════════════
# PETA ZONASI RISIKO — Scattermapbox (NO 'line' property)
# ═════════════════════════════════════════════════════════
def create_risk_zonation_map(master_df: pd.DataFrame) -> go.Figure:
    """
    Peta Zonasi Risiko menggunakan go.Scattermapbox.
    TIDAK menggunakan properti 'line' pada marker untuk menghindari ValueError.
    """
    if master_df.empty:
        return go.Figure()

    # Compute risk for each point
    risk_colors, risk_labels, risk_scores = [], [], []
    for _, row in master_df.iterrows():
        r = calc_dss_score(
            row.get("slope_deg", 0),
            row.get("weathering_depth_m", 0),
            bool(row.get("slip_plane_detected", False)),
            row.get("resistivity_ohm", 100),
        )
        risk_colors.append(r["risk_color"])
        risk_labels.append(r["risk_level"])
        risk_scores.append(r["risk_score"])

    master_df = master_df.copy()
    master_df["risk_color"] = risk_colors
    master_df["risk_level"] = risk_labels
    master_df["risk_score"] = risk_scores

    fig = go.Figure()

    # Plot per risk category for proper legend
    for level, color in [("AMAN", "#3fb950"), ("WASPADA", "#d29922"), ("BAHAYA", "#f85149")]:
        sub = master_df[master_df["risk_level"] == level]
        if sub.empty:
            continue
        texts = []
        for _, r in sub.iterrows():
            texts.append(
                f"<b>Koordinat:</b> {r['lat']:.4f}°, {r['lon']:.4f}°<br>"
                f"<b>Status:</b> {r['risk_level']}<br>"
                f"<b>Skor DSS:</b> {r['risk_score']:.0f}/100<br>"
                f"<b>Kelerengan:</b> {r['slope_deg']:.1f}°<br>"
                f"<b>Pelapukan:</b> {r['weathering_depth_m']:.1f} m<br>"
                f"<b>Resistivitas:</b> {r['resistivity_ohm']:.0f} Ω.m<br>"
                f"<b>Bidang Gelincir:</b> {'✅ Terdeteksi' if r['slip_plane_detected'] else '—'}"
            )
        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"].tolist(),
            lon=sub["lon"].tolist(),
            mode="markers",                       # markers ONLY — no 'lines' mode
            marker=dict(
                size=14,
                color=color,
                opacity=0.85,
            ),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            name=f"{'🟢' if level=='AMAN' else '🟡' if level=='WASPADA' else '🔴'} {level}",
        ))

    center_lat = float(master_df["lat"].mean()) if len(master_df) else -6.83
    center_lon = float(master_df["lon"].mean()) if len(master_df) else 108.22

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12,
        ),
        title="Peta Zonasi Risiko Terpadu — OpenStreetMap",
        hovermode="closest",
        height=560,
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor="#0d1117",
        font=dict(family="Space Mono", color="#e6edf3", size=11),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    )
    return fig


# ═════════════════════════════════════════════════════════
# GEOBOT — keyword-based + Anthropic API integration
# ═════════════════════════════════════════════════════════
def build_geobot_context() -> str:
    """Build context string from all available session state data."""
    parts = ["Data geofisika yang tersedia dalam sesi ini:\n"]

    res = st.session_state.analysis_result
    if res:
        parts.append(
            f"- Topografi: slope maks {res.get('max_slope','?')}°, "
            f"status {res.get('status','?')}, elevasi rata {res.get('mean_elev','?')} m dpl."
        )
    td = st.session_state.topo_data
    if td and len(td.get("slopes", [])) > 0:
        slopes = np.array(td["slopes"])
        parts.append(f"- Kelerengan: rata-rata {slopes.mean():.1f}°, maks {slopes.max():.1f}°.")

    sd = st.session_state.seismic_data
    if sd:
        parts.append(
            f"- Seismic Refraction: V1={sd.get('v1','?'):.0f} m/s, "
            f"V2={sd.get('v2','?'):.0f} m/s, kedalaman pelapukan={sd.get('weathering_depth','?'):.1f} m, "
            f"bedrock mulai dari {sd.get('bedrock_start','?'):.1f} m."
        )

    gd = st.session_state.geoelectric_data
    if gd:
        parts.append(
            f"- Geoelektrik: resistivitas rata {gd.get('mean_resistivity', gd.get('resistivity_mean','?')):.0f} Ω.m, "
            f"bidang gelincir {'TERDETEKSI' if gd.get('has_slip_plane') else 'tidak terdeteksi'} "
            f"di kedalaman {gd.get('slip_depth','?'):.1f} m, "
            f"bedrock mulai {gd.get('bedrock_start','?'):.1f} m."
        )

    if len(parts) == 1:
        parts.append("- Belum ada data yang diupload. Silakan upload CSV di masing-masing modul.")

    return "\n".join(parts)


GEOBOT_SYSTEM = """Kamu adalah GeoBot, asisten AI khusus mitigasi bencana longsor yang dikembangkan oleh GeoLink v2.0.
Tugasmu adalah menjelaskan hasil analisis geofisika (topografi, seismic refraction, geoelektrik) ke bahasa yang mudah dipahami oleh masyarakat awam, kepala desa, dan aparat daerah.

KEAHLIANMU:
- Interpretasi data kelerengan lereng dan klasifikasi risiko (AMAN/WASPADA/BAHAYA)
- Metode seismic refraction: intercept-time, kecepatan gelombang P (Vp), kedalaman pelapukan
- Metode geoelektrik 2D: pseudoseksi resistivitas, identifikasi bidang gelincir, zona pelapukan vs bedrock
- Standar PVMBG (Pusat Vulkanologi dan Mitigasi Bencana Geologi)
- Rekomendasi fondasi tiang, sistem drainase, early warning system, retaining wall
- Koordinasi dengan BPBD, BNPB, dan aparat lokal

GAYA KOMUNIKASI:
- Gunakan bahasa Indonesia yang ramah dan mudah dipahami
- Berikan analogi sederhana untuk konsep teknis
- Selalu sertakan rekomendasi praktis dan actionable
- Jika ditanya hal di luar domain geofisika/longsor, arahkan kembali ke topik relevan
- Format jawaban dengan poin-poin jelas bila sesuai

BATASAN:
- Selalu tegaskan bahwa rekomendasi bersifat indikatif dan perlu verifikasi lapangan
- Rekomendasikan konsultasi ahli geoteknik bersertifikat untuk keputusan konstruksi
"""


def geobot_local_response(user_msg: str) -> str:
    """Fallback keyword-based response jika API tidak tersedia."""
    msg = user_msg.lower()
    context = build_geobot_context()

    if any(w in msg for w in ["risiko", "bahaya", "aman", "waspada", "status"]):
        res = st.session_state.analysis_result
        if res:
            st = res.get("status", "belum dianalisis")
            ms = res.get("max_slope", 0)
            return (
                f"🗺️ **Status Risiko Wilayah: {st}**\n\n"
                f"Berdasarkan data DEM yang diupload:\n"
                f"- Kelerengan maksimum: **{ms:.1f}°**\n"
                f"- {'Wilayah aman untuk permukiman, tetap jaga drainase.' if st=='AMAN' else 'Potensi longsor saat hujan lebat. Pantau terus.' if st=='WASPADA' else 'Risiko tinggi! Pertimbangkan evakuasi dan koordinasi dengan BPBD.'}"
            )
        return (
            "Untuk mengetahui status risiko, upload file CSV atau TIF di menu **🗺️ Analisis Geospasial**. "
            "GeoLink akan otomatis menghitung kelerengan dan menentukan apakah wilayah AMAN, WASPADA, atau BAHAYA."
        )

    if any(w in msg for w in ["seismic", "seismik", "gelombang", "v1", "v2", "vp", "pelapukan"]):
        sd = st.session_state.seismic_data
        if sd:
            v1, v2, wd = sd.get("v1",300), sd.get("v2",800), sd.get("weathering_depth",5)
            return (
                f"📡 **Interpretasi Seismic Refraction:**\n\n"
                f"- Kecepatan Layer 1 (V₁ = {v1:.0f} m/s): ini adalah lapisan pelapukan — tanah lunak di permukaan\n"
                f"- Kecepatan Layer 2 (V₂ = {v2:.0f} m/s): ini adalah batuan dasar (bedrock) yang lebih keras\n"
                f"- Ketebalan Pelapukan: **{wd:.1f} m** — semakin tebal semakin rawan longsor\n\n"
                f"**Analogi sederhana:** Bayangkan lapisan es di atas batu licin. Saat hujan, lapisan 'es' (pelapukan) "
                f"bisa meluncur di atas 'batu' (bedrock). Makin tebal lapisannya, makin berat dan makin mudah longsor."
            )
        return (
            "Upload file CSV seismic refraction (kolom: distance, time) di menu **📉 Seismic Refraction** "
            "untuk analisis kecepatan gelombang dan kedalaman pelapukan."
        )

    if any(w in msg for w in ["geoelektrik", "resistivitas", "bidang gelincir", "fondasi", "tiang", "bedrock"]):
        gd = st.session_state.geoelectric_data
        if gd:
            rs = gd.get("mean_resistivity", gd.get("resistivity_mean", 100))
            slip = gd.get("has_slip_plane", False)
            bd = gd.get("bedrock_start", 8)
            rec = max(gd.get("weathered_depth", 5) * 1.2, bd)
            return (
                f"⚡ **Interpretasi Geoelektrik:**\n\n"
                f"- Resistivitas rata-rata: **{rs:.0f} Ω.m** ({'material lunak/basah' if rs < 100 else 'material keras'})\n"
                f"- Bidang gelincir: **{'TERDETEKSI ⚠️' if slip else 'Tidak terdeteksi ✅'}**\n"
                f"- Bedrock stabil dimulai: **{bd:.1f} m**\n"
                f"- Rekomendasi fondasi tiang: **minimal {rec:.1f} m** (sampai menembus bedrock)\n\n"
                f"**Catatan:** Nilai resistivitas < 50 Ω.m menunjukkan zona jenuh air — potensi bidang gelincir. "
                f"Verifikasi dengan bor geoteknik sebelum konstruksi."
            )
        return (
            "Upload file CSV geoelektrik (kolom: distance, depth, resistivity) di menu **⚡ Geoelectrics** "
            "untuk identifikasi bidang gelincir dan rekomendasi fondasi."
        )

    if any(w in msg for w in ["mitigasi", "rekomendasi", "saran", "tindakan", "lakukan"]):
        return (
            "🛡️ **Rekomendasi Mitigasi Umum:**\n\n"
            "**Jangka Pendek:**\n"
            "- Pasang early warning system (tiang bendera, piezometer) di lereng kritis\n"
            "- Bersihkan saluran drainase agar air tidak menggenang\n"
            "- Larang penebangan pohon di lereng > 15°\n\n"
            "**Jangka Menengah:**\n"
            "- Lakukan survei geoteknik detail (bor & SPT) di zona merah\n"
            "- Pasang inclinometer untuk deteksi pergerakan tanah\n"
            "- Bangun checkdam di sungai-sungai hulu\n\n"
            "**Jangka Panjang:**\n"
            "- Relokasi permukiman dari zona BAHAYA (slope > 30°)\n"
            "- Konstruksi retaining wall di area kritis\n"
            "- Penghijauan lereng dengan tanaman berakar dalam\n\n"
            "_Semua rekomendasi harus dikonfirmasi dengan survei lapangan oleh ahli geoteknik bersertifikat._"
        )

    if any(w in msg for w in ["pvmbg", "standar", "referensi", "sns", "regulasi"]):
        return (
            "📋 **Standar & Referensi yang Digunakan GeoLink:**\n\n"
            "- **SNI 13-7124-2005** — Klasifikasi kelerengan untuk risiko longsor\n"
            "- **Pedoman PVMBG 2019** — Standar identifikasi daerah rawan gerakan tanah\n"
            "- **Zevenbergen & Thorne (1987)** — Metode finite difference untuk kelerengan DEM\n"
            "- **Telford et al. (1990)** — Applied Geophysics, metode intercept-time seismik\n"
            "- **Reynolds (2011)** — An Introduction to Applied & Environmental Geophysics\n\n"
            "Klasifikasi risiko berdasarkan kelerengan: **< 15° = AMAN**, **15–30° = WASPADA**, **> 30° = BAHAYA**."
        )

    # Generic
    return (
        f"🤖 Saya GeoBot, asisten analisis geofisika GeoLink v2.0.\n\n"
        f"**Konteks Data Saat Ini:**\n{context}\n\n"
        "Silakan tanyakan tentang:\n"
        "- 📐 Kelerengan dan risiko longsor\n"
        "- 📡 Interpretasi seismic refraction\n"
        "- ⚡ Analisis geoelektrik dan fondasi\n"
        "- 🛡️ Rekomendasi mitigasi bencana\n"
        "- 📋 Standar PVMBG dan referensi teknis"
    )


# ═════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-family:Space Mono; font-size:1.6rem; font-weight:700;
                    background:linear-gradient(90deg,#f78166,#ffa657);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;'>
            🏔️ GeoLink
        </div>
        <div style='color:#8b949e; font-size:0.78rem; margin-top:4px;'>v2.0 — Geophysical Risk Platform</div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,transparent,#f78166,transparent);
                margin:0.8rem 0 1.2rem;'></div>
    """, unsafe_allow_html=True)

    # Status Data Ringkas di Sidebar
    td_ok  = st.session_state.topo_data is not None
    sd_ok  = st.session_state.seismic_data is not None
    gd_ok  = st.session_state.geoelectric_data is not None
    n_ok   = sum([td_ok, sd_ok, gd_ok])
    st.markdown(
        f"<div style='background:#0d1117; border:1px solid #30363d; border-radius:8px; "
        f"padding:0.6rem 0.9rem; margin-bottom:0.8rem; font-size:0.8rem;'>"
        f"<div style='color:#8b949e; font-size:0.72rem; letter-spacing:1px; margin-bottom:4px;'>DATA AKTIF</div>"
        f"<span style='color:{'#3fb950' if td_ok else '#30363d'};'>⬤</span> Topografi &nbsp;"
        f"<span style='color:{'#3fb950' if sd_ok else '#30363d'};'>⬤</span> Seismik &nbsp;"
        f"<span style='color:{'#3fb950' if gd_ok else '#30363d'};'>⬤</span> Geoelektrik"
        f"</div>",
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "Navigasi",
        [
            "🏠 Beranda",
            "🗺️ Analisis Geospasial",
            "📉 Seismic Refraction",
            "⚡ Geoelectrics",
            "🌐 Peta Zonasi Risiko",
            "🤖 GeoBot AI",
            "📋 Laporan Akhir",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #30363d;
                color:#8b949e; font-size:0.72rem; text-align:center; line-height:2;'>
        <div>🔬 Hackathon Geofisika Indonesia 2026</div>
        <div style='color:#f78166;'>Bridging Science &amp; Society</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ═══════════════════════════════════════════════════════════════
if menu == "🏠 Beranda":
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>GeoLink v2.0</div>
        <div style='color:#ffa657; font-family:Space Mono; font-size:0.85rem;
                    letter-spacing:3px; margin-bottom:1rem;'>
            PLATFORM ANALISIS RISIKO LONGSOR
        </div>
        <div class='hero-subtitle'>
            Mengubah data geofisika yang kompleks menjadi keputusan nyata yang
            melindungi masyarakat dari bencana longsor.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='geo-divider'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, desc in [
        (col1, "🗺️", "Topografi DEM",
         "Input CSV atau TIF DEMNAS. Hitung kelerengan otomatis dengan finite difference."),
        (col2, "📡", "Seismic Refraction",
         "Input T-X CSV. Intercept-time method dua lapis untuk estimasi kedalaman pelapukan."),
        (col3, "⚡", "Geoelektrik 2D",
         "Pseudoseksi resistivitas, identifikasi bidang gelincir & rekomendasi fondasi tiang."),
        (col4, "🤖", "GeoBot AI",
         "Asisten cerdas yang menerjemahkan data teknis ke bahasa awam yang mudah dipahami."),
    ]:
        with col:
            st.markdown(
                f"<div class='feature-card'><div class='feature-icon'>{icon}</div>"
                f"<div class='feature-title'>{title}</div>"
                f"<div class='feature-desc'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='geo-card geo-card-accent'>
        <h3 style='color:#ffa657; font-family:Space Mono; font-size:1rem; margin-top:0;'>🌍 Tentang GeoLink</h3>
        <p style='color:#c9d1d9; line-height:1.8; margin-bottom:0.5rem;'>
            Indonesia berada di kawasan rawan longsor karena topografi pegunungan, curah hujan tinggi, dan
            aktivitas seismik. Data geofisika selama ini hanya dapat diakses dan dipahami oleh para ahli.
        </p>
        <p style='color:#c9d1d9; line-height:1.8; margin-bottom:0.5rem;'>
            <strong style='color:#f78166;'>GeoLink v2.0</strong> mengintegrasikan tiga metode geofisika
            — Topografi, Seismic Refraction, dan Geoelektrik 2D — ke dalam satu sistem pendukung keputusan (DSS)
            yang menghasilkan skor risiko kumulatif berstandar PVMBG.
        </p>
        <p style='color:#8b949e; font-size:0.88rem; margin-bottom:0;'>
            ⚡ Mulai dengan upload data di menu <strong style='color:#ffa657;'>🗺️ Analisis Geospasial</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚦 Sistem Klasifikasi Risiko")
    r1, r2, r3 = st.columns(3)
    for col, color, label, slope_range, desc in [
        (r1, "#3fb950", "✅ AMAN",    "Lereng &lt; 15°",
         "Wilayah relatif datar dan stabil. Risiko longsor sangat rendah. Cocok untuk permukiman dan pertanian."),
        (r2, "#d29922", "⚠️ WASPADA", "Lereng 15° – 30°",
         "Cukup curam. Ada potensi longsor saat hujan lebat. Warga perlu waspada dan membangun drainase baik."),
        (r3, "#f85149", "🚨 BAHAYA",  "Lereng &gt; 30°",
         "Sangat curam dan berbahaya. Tidak disarankan untuk ditinggali. Evakuasi perlu direncanakan sejak dini."),
    ]:
        with col:
            st.markdown(
                f"<div style='background:#161b22; border:1px solid {color}55;"
                f"border-top:3px solid {color}; border-radius:10px; padding:1.2rem;'>"
                f"<div style='color:{color}; font-family:Space Mono; font-size:1rem; font-weight:700;'>{label}</div>"
                f"<div style='color:#8b949e; font-size:0.8rem; margin:0.3rem 0 0.7rem;'>{slope_range}</div>"
                f"<div style='color:#c9d1d9; font-size:0.87rem; line-height:1.6;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <strong>📖 Alur Penggunaan GeoLink v2.0:</strong><br>
        1️⃣ Upload DEM/CSV di <strong>🗺️ Analisis Geospasial</strong> → slope & model topografi<br>
        2️⃣ Upload T-X CSV di <strong>📉 Seismic Refraction</strong> → kedalaman pelapukan<br>
        3️⃣ Upload pseudoseksi CSV di <strong>⚡ Geoelectrics</strong> → bidang gelincir & bedrock<br>
        4️⃣ Lihat integrasi di <strong>🌐 Peta Zonasi Risiko</strong> → skor DSS kumulatif<br>
        5️⃣ Tanya <strong>🤖 GeoBot AI</strong> untuk penjelasan dalam bahasa awam<br>
        6️⃣ Cetak dari <strong>📋 Laporan Akhir</strong> untuk diserahkan ke BPBD/desa
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — ANALISIS GEOSPASIAL
# ═══════════════════════════════════════════════════════════════
elif menu == "🗺️ Analisis Geospasial":
    st.markdown("""
    <h1 style='margin-bottom:0.2rem;'>🗺️ Analisis Geospasial</h1>
    <div style='color:#8b949e; margin-bottom:1.5rem; font-size:0.95rem;'>
        Upload data elevasi (DEM) untuk analisis kelerengan dan risiko longsor otomatis
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='geo-card' style='border-top:3px solid #f78166;'>
        <div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;'>
            📂 Upload Data Topografi
        </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Pilih file .CSV atau .TIF",
        type=["csv", "tif", "tiff"],
        help="CSV: kolom wajib → lon, lat, elev | TIF: DEMNAS atau GeoTIFF single-band",
    )
    st.markdown("""
        <div style='color:#8b949e; font-size:0.82rem; margin-top:0.3rem;'>
            📌 <strong>CSV</strong>: kolom <code>lon</code>, <code>lat</code>, <code>elev</code><br>
            📌 <strong>TIF/TIFF</strong>: DEMNAS atau GeoTIFF single-band (perlu rasterio)
        </div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is not None:
        ext = uploaded.name.split(".")[-1].lower()
        with st.spinner("⚙️ Memproses data elevasi..."):
            elev_raw, raw_df = None, None
            if ext == "csv":
                try:
                    df_in    = pd.read_csv(uploaded)
                    elev_raw, raw_df = process_topo_csv(df_in)
                    if raw_df is not None:
                        # Save topo_data for integration
                        lons_a = raw_df["lon"].values
                        lats_a = raw_df["lat"].values
                        # Quick per-point slope
                        slopes_q = []
                        for i in range(len(lats_a)):
                            if i < len(lats_a) - 1:
                                dx = 111320 * np.cos(np.radians(lats_a[i])) * (lons_a[i+1] - lons_a[i])
                                dy = 111320 * (lats_a[i+1] - lats_a[i])
                                dz = raw_df["elev"].values[i+1] - raw_df["elev"].values[i]
                                d  = np.sqrt(dx**2 + dy**2)
                                slopes_q.append(float(np.clip(np.degrees(np.arctan(abs(dz)/max(d,1))), 0, 90)))
                            else:
                                slopes_q.append(slopes_q[-1] if slopes_q else 0.0)
                        st.session_state.topo_data = {
                            "lons": lons_a, "lats": lats_a,
                            "slopes": slopes_q,
                        }
                except Exception as e:
                    st.error(f"❌ Gagal membaca CSV: {e}")

            elif ext in ("tif", "tiff"):
                elev_raw = process_tif(uploaded.read())
                if elev_raw is not None:
                    flat  = elev_raw[~np.isnan(elev_raw)].flatten()
                    raw_df = pd.DataFrame({"elev": flat})

        if elev_raw is not None:
            clean    = elev_raw.copy()
            m        = np.isnan(clean)
            if m.any():
                col_mean    = np.nanmean(clean, axis=0)
                rows, cols  = np.where(m)
                clean[rows, cols] = col_mean[cols]
            clean     = np.nan_to_num(clean, nan=float(np.nanmean(clean)))
            sg        = compute_slope(clean)
            max_slope = float(np.nanmax(sg))
            mean_elev = float(np.nanmean(clean))
            status, badge_cls, status_color, css_cls = classify_risk(max_slope)

            st.session_state.analysis_result = {
                "elev_grid": clean, "slope_grid": sg,
                "max_slope": max_slope, "mean_elev": mean_elev,
                "status": status, "badge_cls": badge_cls,
                "status_color": status_color, "css_cls": css_cls,
                "raw_df": raw_df, "file_name": uploaded.name,
                "timestamp": datetime.datetime.now().strftime("%d %B %Y, %H:%M WIB"),
                "bounds": (-7.55, 109.95, -7.45, 110.15),
            }
            st.success(f"✅ **{uploaded.name}** berhasil diproses!")

    res = st.session_state.analysis_result
    if res:
        max_slope    = res["max_slope"]
        mean_elev    = res["mean_elev"]
        status       = res["status"]
        badge_cls    = res["badge_cls"]
        status_color = res["status_color"]
        css_cls      = res["css_cls"]
        sg           = res["slope_grid"]
        eg           = res["elev_grid"]
        raw_df       = res["raw_df"]

        st.markdown("<div class='geo-divider'></div>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📊 Status Wilayah", "🗺️ Visualisasi", "🔢 Detail Data"])

        # ── TAB 1 ─────────────────────────────────
        with tab1:
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='text-align:center; margin-bottom:1.5rem;'>"
                "<div style='color:#8b949e; font-size:0.82rem; margin-bottom:0.5rem;'>STATUS RISIKO WILAYAH</div>"
                f"<span class='{badge_cls}' style='font-size:1.4rem; padding:8px 28px;'>{status}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("📐 Lereng Maksimum", f"{max_slope:.1f}°")
            c2.metric("⛰️ Rerata Elevasi",  f"{mean_elev:.0f} m")
            c3.metric("🔢 Resolusi Grid",   f"{eg.shape[0]}×{eg.shape[1]}")

            sf      = sg[~np.isnan(sg)].flatten()
            p_aman  = 100 * (sf < 15).sum()  / len(sf)
            p_wasp  = 100 * ((sf >= 15) & (sf < 30)).sum() / len(sf)
            p_baha  = 100 * (sf >= 30).sum() / len(sf)

            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("✅ AMAN (<15°)",      f"{p_aman:.1f}%")
            d2.metric("⚠️ WASPADA (15–30°)", f"{p_wasp:.1f}%")
            d3.metric("🚨 BAHAYA (>30°)",    f"{p_baha:.1f}%")

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

            interp_cls = {"AMAN": "interpret-safe", "WASPADA": "interpret-waspada", "BAHAYA": "interpret-bahaya"}
            interp_txt = {
                "AMAN":
                    f"<div style='color:#3fb950; font-family:Space Mono; font-weight:700; margin-bottom:0.6rem;'>"
                    f"✅ Wilayah Ini AMAN dari Risiko Longsor</div>"
                    f"<p style='color:#c9d1d9; line-height:1.8; margin:0;'>Kemiringan lereng tertinggi hanya "
                    f"<strong>{max_slope:.1f}°</strong> — di bawah batas aman 15°. Tanah relatif datar dan stabil.</p>"
                    f"<p style='color:#c9d1d9; line-height:1.8; margin-top:0.6rem; margin-bottom:0;'>"
                    f"🏡 Aman untuk tempat tinggal dan pertanian. Tetap jaga saluran drainase.</p>",
                "WASPADA":
                    f"<div style='color:#d29922; font-family:Space Mono; font-weight:700; margin-bottom:0.6rem;'>"
                    f"⚠️ Wilayah Ini Berstatus WASPADA</div>"
                    f"<p style='color:#c9d1d9; line-height:1.8; margin:0;'>Kemiringan lereng tertinggi "
                    f"<strong>{max_slope:.1f}°</strong> (15–30°). Berpotensi longsor saat hujan deras atau gempa.</p>"
                    f"<p style='color:#c9d1d9; line-height:1.8; margin-top:0.6rem; margin-bottom:0;'>"
                    f"🏗️ Jangan bangun rumah di lereng curam. Tanam pohon berakar kuat. Hubungi BPBD.</p>",
                "BAHAYA":
                    f"<div style='color:#f85149; font-family:Space Mono; font-weight:700; margin-bottom:0.6rem;'>"
                    f"🚨 BAHAYA — Risiko Longsor Sangat Tinggi!</div>"
                    f"<p style='color:#c9d1d9; line-height:1.8; margin:0;'>Kemiringan lereng tertinggi "
                    f"<strong>{max_slope:.1f}°</strong> — jauh melampaui batas kritis 30°.</p>"
                    f"<p style='color:#c9d1d9; line-height:1.8; margin-top:0.6rem; margin-bottom:0;'>"
                    f"🚒 Segera koordinasikan dengan BPBD. Pertimbangkan evakuasi warga di zona merah.</p>",
            }
            st.markdown(
                f"<div class='{interp_cls[status]}'>{interp_txt[status]}</div>",
                unsafe_allow_html=True,
            )

        # ── TAB 2 ─────────────────────────────────
        with tab2:
            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(make_slope_fig(sg), use_container_width=True)
            with v2:
                st.plotly_chart(make_surface_fig(eg), use_container_width=True)

        # ── TAB 3 ─────────────────────────────────
        with tab3:
            if raw_df is not None:
                st.dataframe(raw_df.head(200), use_container_width=True, height=400)
                csv_bytes = raw_df.to_csv(index=False).encode()
                st.download_button("⬇️ Unduh Data CSV", csv_bytes, "data_elevasi.csv", "text/csv")
            else:
                st.info("Data detail tidak tersedia untuk file TIF.")

    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#161b22;
                    border:2px dashed #30363d; border-radius:16px; margin-top:2rem;'>
            <div style='font-size:3rem;'>🗺️</div>
            <div style='font-family:Space Mono; color:#ffa657; margin:0.5rem 0;'>Belum Ada Data</div>
            <div style='color:#8b949e;'>Upload file CSV atau TIF untuk memulai analisis</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — SEISMIC REFRACTION
# ═══════════════════════════════════════════════════════════════
elif menu == "📉 Seismic Refraction":
    st.markdown("""
    <h1 style='margin-bottom:0.2rem;'>📉 Seismic Refraction</h1>
    <div style='color:#8b949e; margin-bottom:1.5rem; font-size:0.95rem;'>
        Analisis kurva T-X dengan metode intercept-time dua lapis (standar PVMBG)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='geo-card' style='border-top:3px solid #4da6ff;'>
        <div style='font-family:Space Mono; color:#4da6ff; font-size:0.82rem;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;'>
            📂 Upload Data Seismic T-X
        </div>
    """, unsafe_allow_html=True)

    seis_file = st.file_uploader(
        "File CSV seismic",
        type=["csv"],
        help="Kolom wajib: distance (m), time (s)",
        key="seis_upload",
    )
    st.markdown("""
        <div style='color:#8b949e; font-size:0.82rem; margin-top:0.3rem;'>
            📌 Format: <code>distance</code> (meter), <code>time</code> (detik)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Demo data generator
    with st.expander("🔧 Atau gunakan data contoh"):
        demo_n = st.slider("Jumlah titik observasi", 8, 30, 16)
        demo_v1 = st.number_input("V₁ (m/s)", 100, 1000, 320, 10)
        demo_v2 = st.number_input("V₂ (m/s)", 500, 5000, 1450, 50)
        demo_ti = st.number_input("Intercept time (ms)", 1.0, 50.0, 8.0, 0.5)
        if st.button("Generate Data Contoh"):
            dists_demo = np.linspace(10, 200, demo_n)
            # Two-layer T-X: crossover at some distance
            xc = 2 * (demo_ti/1000) * demo_v1 * demo_v2 / np.sqrt(demo_v2**2 - demo_v1**2 + 1e-9)
            times_demo = np.where(
                dists_demo < xc,
                dists_demo / demo_v1 + np.random.normal(0, 0.0005, demo_n),
                dists_demo / demo_v2 + demo_ti/1000 + np.random.normal(0, 0.0005, demo_n),
            )
            times_demo = np.clip(times_demo, 0.001, None)
            df_demo = pd.DataFrame({"distance": dists_demo, "time": times_demo})
            st.dataframe(df_demo, height=200)
            csv_demo = df_demo.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV Contoh", csv_demo, "seismic_contoh.csv", "text/csv")

    if seis_file is not None:
        with st.spinner("⚙️ Menganalisis data seismik..."):
            try:
                df_s = pd.read_csv(seis_file)
                df_s.columns = df_s.columns.str.lower().str.strip()
                if "distance" not in df_s.columns or "time" not in df_s.columns:
                    st.error("❌ CSV harus memiliki kolom: distance, time")
                else:
                    distances_s = df_s["distance"].values.astype(float)
                    times_s     = df_s["time"].values.astype(float)

                    result_s = intercept_time_method(distances_s, times_s)
                    v1, v2   = result_s["v1"], result_s["v2"]
                    wd       = result_s["weathering_depth"]
                    bs       = result_s["bedrock_start"]

                    # Compute depth profile along transect
                    depths_s = np.full_like(distances_s, wd)  # simplified: uniform depth

                    # Store in session state
                    st.session_state.seismic_data = {
                        **result_s,
                        "distances": distances_s,
                        "arrival_times": times_s,
                    }
                    st.success(f"✅ Analisis seismik selesai! V₁={v1:.0f} m/s, V₂={v2:.0f} m/s")

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("V₁ (Layer Pelapukan)", f"{v1:.0f} m/s")
                    m2.metric("V₂ (Bedrock)",         f"{v2:.0f} m/s")
                    m3.metric("Kedalaman Pelapukan",   f"{wd:.1f} m")
                    m4.metric("Bedrock Mulai",         f"{bs:.1f} m")

                    st.markdown("<div class='geo-divider'></div>", unsafe_allow_html=True)

                    # Tabs for charts
                    ts1, ts2 = st.tabs(["📈 Profil Pelapukan (Area Chart)", "📉 Kurva T-X"])
                    with ts1:
                        fig_area = make_seismic_area_chart(distances_s, depths_s, v1, v2)
                        st.plotly_chart(fig_area, use_container_width=True)
                    with ts2:
                        fig_tx = make_traveltime_chart(distances_s, times_s, v1, v2)
                        st.plotly_chart(fig_tx, use_container_width=True)

                    # Interpretation
                    st.markdown("<div class='geo-card geo-card-blue' style='margin-top:0.5rem;'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='color:#4da6ff; font-family:Space Mono; font-weight:700; margin-bottom:0.5rem;'>
                        📡 Interpretasi Teknis
                    </div>
                    <p style='color:#c9d1d9; line-height:1.8; margin:0;'>
                    Hasil intercept-time method menunjukkan <strong>dua lapisan subsurface</strong>:
                    </p>
                    <ul style='color:#c9d1d9; line-height:1.8;'>
                    <li><strong>Layer 1 (Pelapukan):</strong> Vp = {v1:.0f} m/s
                        ({'Sangat lunak' if v1 < 300 else 'Lunak' if v1 < 600 else 'Sedang'})
                        — ketebalan ≈ <strong>{wd:.1f} m</strong></li>
                    <li><strong>Layer 2 (Bedrock):</strong> Vp = {v2:.0f} m/s
                        ({'Keras' if v2 > 1500 else 'Semi-keras'})
                        — mulai dari ≈ <strong>{bs:.1f} m</strong></li>
                    <li><strong>Rekomendasi fondasi tiang:</strong> minimal <strong>{max(wd*1.2, bs):.1f} m</strong></li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error memproses seismic: {e}")
    else:
        if st.session_state.seismic_data:
            sd = st.session_state.seismic_data
            st.info(f"ℹ️ Menggunakan data seismik sebelumnya: V₁={sd['v1']:.0f} m/s, V₂={sd['v2']:.0f} m/s, kedalaman pelapukan={sd['weathering_depth']:.1f} m")
        else:
            st.markdown("""
            <div style='text-align:center; padding:2rem; background:#161b22;
                        border:2px dashed #30363d; border-radius:16px;'>
                <div style='font-size:2.5rem;'>📡</div>
                <div style='font-family:Space Mono; color:#4da6ff; margin:0.5rem 0;'>Belum Ada Data Seismik</div>
                <div style='color:#8b949e;'>Upload file CSV (distance, time) untuk memulai analisis</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — GEOELECTRICS
# ═══════════════════════════════════════════════════════════════
elif menu == "⚡ Geoelectrics":
    st.markdown("""
    <h1 style='margin-bottom:0.2rem;'>⚡ Geoelectrics 2D</h1>
    <div style='color:#8b949e; margin-bottom:1.5rem; font-size:0.95rem;'>
        Pseudoseksi resistivitas — identifikasi bidang gelincir &amp; rekomendasi fondasi tiang
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='geo-card' style='border-top:3px solid #ffa657;'>
        <div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;'>
            📂 Upload Data Pseudoseksi Geoelektrik
        </div>
    """, unsafe_allow_html=True)

    geo_file = st.file_uploader(
        "File CSV geoelektrik",
        type=["csv"],
        help="Kolom wajib: distance (m), depth (m), resistivity (Ω.m)",
        key="geo_upload",
    )
    st.markdown("""
        <div style='color:#8b949e; font-size:0.82rem; margin-top:0.3rem;'>
            📌 Format: <code>distance</code> (m), <code>depth</code> (m), <code>resistivity</code> (Ω.m)<br>
            📌 Resistivitas &lt; 50 Ω.m = zona jenuh/bidang gelincir | &gt; 150 Ω.m = bedrock stabil
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Demo data
    with st.expander("🔧 Atau gunakan data contoh"):
        if st.button("Generate Data Contoh Geoelektrik"):
            dists_g   = np.repeat(np.linspace(0, 200, 15), 10)
            depths_g  = np.tile(np.linspace(0, 15, 10), 15)
            res_g     = np.where(depths_g < 5, np.random.uniform(15, 60, len(depths_g)),
                                 np.where(depths_g < 10, np.random.uniform(80, 160, len(depths_g)),
                                          np.random.uniform(150, 350, len(depths_g))))
            df_geo_demo = pd.DataFrame({"distance": dists_g, "depth": depths_g, "resistivity": res_g})
            st.dataframe(df_geo_demo.head(30), height=200)
            csv_geo_demo = df_geo_demo.to_csv(index=False).encode()
            st.download_button("⬇️ Download CSV Contoh", csv_geo_demo, "geoelectric_contoh.csv", "text/csv")

    if geo_file is not None:
        with st.spinner("⚙️ Menganalisis pseudoseksi geoelektrik..."):
            try:
                df_g = pd.read_csv(geo_file)
                df_g.columns = df_g.columns.str.lower().str.strip()
                req = {"distance", "depth", "resistivity"}
                if not req.issubset(set(df_g.columns)):
                    st.error(f"❌ CSV harus memiliki kolom: distance, depth, resistivity. Ditemukan: {list(df_g.columns)}")
                else:
                    dist_g  = df_g["distance"].values.astype(float)
                    dep_g   = df_g["depth"].values.astype(float)
                    res_g   = df_g["resistivity"].values.astype(float)

                    geo_an  = analyze_geoelectric(dist_g, dep_g, res_g)

                    # Build 2D grid for contour
                    dists_uniq = np.unique(dist_g)
                    deps_uniq  = np.unique(dep_g)
                    if len(dists_uniq) > 1 and len(deps_uniq) > 1:
                        xi_g  = np.linspace(dist_g.min(), dist_g.max(), max(len(dists_uniq), 20))
                        yi_g  = np.linspace(dep_g.min(), dep_g.max(), max(len(deps_uniq), 15))
                        xi_g2, yi_g2 = np.meshgrid(xi_g, yi_g)
                        try:
                            res_2d = griddata((dist_g, dep_g), res_g, (xi_g2, yi_g2), method="cubic")
                            m_nan  = np.isnan(res_2d)
                            if m_nan.any():
                                res_2d[m_nan] = griddata((dist_g, dep_g), res_g, (xi_g2[m_nan], yi_g2[m_nan]), method="nearest")
                        except Exception:
                            res_2d = np.random.uniform(30, 200, (len(yi_g), len(xi_g)))
                    else:
                        xi_g   = np.linspace(dist_g.min(), dist_g.max(), 20)
                        yi_g   = np.linspace(dep_g.min(), dep_g.max(), 15)
                        res_2d = np.tile(res_g[:20], (15, 1))[:15, :20]

                    # Store in session state
                    st.session_state.geoelectric_data = {
                        **geo_an,
                        "distances":   dist_g,
                        "depths":      dep_g,
                        "resistivities": res_g,
                        "dist_1d": xi_g, "depth_1d": yi_g, "res_2d": res_2d,
                        "mean_resistivity": float(res_g.mean()),
                        "resistivity_mean": float(res_g.mean()),
                    }
                    st.success("✅ Data geoelektrik berhasil diproses!")

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Resistivitas Min",   f"{geo_an['min_resistivity']:.0f} Ω.m")
                    m2.metric("Resistivitas Rata",  f"{geo_an['mean_resistivity']:.0f} Ω.m")
                    m3.metric("Kedalaman Pelapukan", f"{geo_an['weathered_depth']:.1f} m")
                    m4.metric("Bedrock Mulai",       f"{geo_an['bedrock_start']:.1f} m")

                    st.markdown("<div class='geo-divider'></div>", unsafe_allow_html=True)

                    # Contour map
                    fig_cont = make_geoelectric_contour(
                        xi_g, yi_g, res_2d,
                        slip_depth=geo_an["slip_depth"] if geo_an["has_slip_plane"] else None,
                        bedrock_depth=geo_an["bedrock_start"],
                    )
                    st.plotly_chart(fig_cont, use_container_width=True)

                    # Interpretation
                    slip_html = (
                        f"<span style='color:#f85149;'>⚠️ <strong>TERDETEKSI</strong></span> "
                        f"pada kedalaman {geo_an['slip_depth']:.1f} m"
                        if geo_an["has_slip_plane"]
                        else "<span style='color:#3fb950;'>✅ Tidak terdeteksi</span>"
                    )
                    rec_depth = max(geo_an["weathered_depth"] * 1.2, geo_an["bedrock_start"])

                    st.markdown("<div class='geo-card geo-card-accent'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='color:#ffa657; font-family:Space Mono; font-weight:700; margin-bottom:0.5rem;'>
                        🏗️ Interpretasi &amp; Rekomendasi Fondasi
                    </div>
                    <ul style='color:#c9d1d9; line-height:2;'>
                    <li><strong>Bidang Gelincir (ρ &lt; 50 Ω.m):</strong> {slip_html}</li>
                    <li><strong>Zona Pelapukan:</strong> 0 — {geo_an['weathered_depth']:.1f} m
                        (ρ &lt; 100 Ω.m, material lunak)</li>
                    <li><strong>Bedrock Stabil:</strong> mulai dari {geo_an['bedrock_start']:.1f} m
                        (ρ &gt; 150 Ω.m)</li>
                    <li style='background:#1a1f2e; padding:0.5rem; border-radius:6px;'>
                        <strong style='color:#f78166;'>✓ Kedalaman Fondasi Tiang Minimal: {rec_depth:.1f} m</strong>
                        <span style='color:#8b949e; font-size:0.85rem;'> (perlu verifikasi bor geoteknik)</span>
                    </li>
                    </ul>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error geoelektrik: {e}")
    else:
        if st.session_state.geoelectric_data:
            gd = st.session_state.geoelectric_data
            st.info(f"ℹ️ Menggunakan data geoelektrik sebelumnya: resistivitas rata {gd.get('mean_resistivity',0):.0f} Ω.m, "
                    f"bidang gelincir {'terdeteksi' if gd.get('has_slip_plane') else 'tidak terdeteksi'}")
        else:
            st.markdown("""
            <div style='text-align:center; padding:2rem; background:#161b22;
                        border:2px dashed #30363d; border-radius:16px;'>
                <div style='font-size:2.5rem;'>⚡</div>
                <div style='font-family:Space Mono; color:#ffa657; margin:0.5rem 0;'>Belum Ada Data</div>
                <div style='color:#8b949e;'>Upload file CSV (distance, depth, resistivity) untuk memulai</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — PETA ZONASI RISIKO TERPADU
# ═══════════════════════════════════════════════════════════════
elif menu == "🌐 Peta Zonasi Risiko":
    st.markdown("""
    <h1 style='margin-bottom:0.2rem;'>🌐 Peta Zonasi Risiko Terpadu</h1>
    <div style='color:#8b949e; margin-bottom:1.5rem; font-size:0.95rem;'>
        Integrasi Topografi + Seismic Refraction + Geoelektrik — DSS berbasis skor kumulatif PVMBG
    </div>
    """, unsafe_allow_html=True)

    # ── Data status panel ──
    res = st.session_state.analysis_result
    td  = st.session_state.topo_data
    sd  = st.session_state.seismic_data
    gd  = st.session_state.geoelectric_data

    has_topo  = td is not None or (res is not None and res.get("slope_grid") is not None)
    has_seis  = sd is not None
    has_geo   = gd is not None
    use_dummy = not (has_topo or has_seis or has_geo)

    c1, c2, c3 = st.columns(3)
    for col, label, ok in [(c1,"🗺️ Topografi",has_topo),(c2,"📡 Seismik",has_seis),(c3,"⚡ Geoelektrik",has_geo)]:
        col.markdown(
            f"<div style='background:#161b22; border:1px solid {'#3fb950' if ok else '#30363d'}; "
            f"border-radius:8px; padding:0.7rem 1rem; text-align:center;'>"
            f"<span style='color:{'#3fb950' if ok else '#f85149'}; font-size:1.2rem;'>{'✅' if ok else '❌'}</span> "
            f"<span style='font-family:Space Mono; font-size:0.82rem; color:{'#e6edf3' if ok else '#8b949e'};'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if use_dummy:
        st.info("📌 Menggunakan data demonstrasi. Upload data di modul-modul sebelumnya untuk analisis nyata.")

    st.markdown("<div class='geo-divider'></div>", unsafe_allow_html=True)

    # ── Build datasets for integration ──
    topo_in   = td or DUMMY_TOPO
    seis_in   = sd or DUMMY_SEISMIC
    geo_in    = gd or DUMMY_GEOELECTRIC

    master_df = integrate_datasets(topo_in, seis_in, geo_in)

    # Compute DSS scores
    dss_results = [
        calc_dss_score(
            row["slope_deg"], row["weathering_depth_m"],
            bool(row["slip_plane_detected"]), row["resistivity_ohm"],
        )
        for _, row in master_df.iterrows()
    ]
    dss_df = pd.DataFrame(dss_results)
    for col in dss_df.columns:
        master_df[col] = dss_df[col].values

    # Summary counts
    n_aman   = (master_df["risk_level"] == "AMAN").sum()
    n_wasp   = (master_df["risk_level"] == "WASPADA").sum()
    n_baha   = (master_df["risk_level"] == "BAHAYA").sum()
    n_total  = len(master_df)
    avg_dss  = float(master_df["risk_score"].mean())

    # Create 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Peta Interaktif",
        "📊 Penampang 2D",
        "📋 Master DataFrame",
        "🧠 DSS Report",
    ])

    # ── TAB 1: Peta Interaktif Scattermapbox ──
    with tab1:
        fig_map = create_risk_zonation_map(master_df)
        st.plotly_chart(fig_map, use_container_width=True)

        # Zone summary
        z1, z2, z3, z4 = st.columns(4)
        z1.metric("🟢 AMAN",    f"{n_aman}",  f"{100*n_aman/max(n_total,1):.0f}%")
        z2.metric("🟡 WASPADA", f"{n_wasp}",  f"{100*n_wasp/max(n_total,1):.0f}%")
        z3.metric("🔴 BAHAYA",  f"{n_baha}",  f"{100*n_baha/max(n_total,1):.0f}%")
        z4.metric("📊 Skor DSS Rata", f"{avg_dss:.1f}/100")

        # Legend
        st.markdown("""
        <div class='geo-card' style='margin-top:0.5rem;'>
            <div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;
                        letter-spacing:1px; margin-bottom:0.8rem;'>LEGENDA & METODOLOGI</div>
            <div style='color:#c9d1d9; font-size:0.88rem; line-height:2;'>
                🟢 <strong>AMAN</strong>: Skor DSS &lt; 33 | Slope &lt;15°, Pelapukan tipis, Resistivitas tinggi<br>
                🟡 <strong>WASPADA</strong>: Skor DSS 33–65 | Satu atau dua parameter kritis<br>
                🔴 <strong>BAHAYA</strong>: Skor DSS &gt; 65 | Ketiga parameter menunjukkan risiko tinggi<br>
                <br>
                <span style='color:#8b949e; font-size:0.82rem;'>
                Skor DSS kumulatif berbobot: Slope 40% + Pelapukan 35% + Slip/Resistivitas 25%
                (Standar PVMBG 2019)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: Penampang 2D (Seismik Area + Geoelektrik Contour) ──
    with tab2:
        st.markdown("""
        <div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;'>
            📊 Penampang Bawah Permukaan
        </div>
        """, unsafe_allow_html=True)

        col_s, col_g = st.columns(2)

        with col_s:
            st.markdown("**📡 Seismic — Profil Pelapukan**")
            v1_p   = float(seis_in.get("v1", 320))
            v2_p   = float(seis_in.get("v2", 1450))
            wd_p   = float(seis_in.get("weathering_depth", 5.0))
            dists_p = np.array(seis_in.get("distances", np.linspace(0, 200, 20)))
            # Slight variation around mean for visual
            depths_p = wd_p + 0.5 * np.sin(np.linspace(0, 2*np.pi, len(dists_p)))
            fig_area_p = make_seismic_area_chart(dists_p, depths_p, v1_p, v2_p)
            st.plotly_chart(fig_area_p, use_container_width=True)

        with col_g:
            st.markdown("**⚡ Geoelektrik — Pseudoseksi Resistivitas**")
            if gd and gd.get("dist_1d") is not None:
                xi_p   = gd["dist_1d"]
                yi_p   = gd["depth_1d"]
                r2d_p  = gd["res_2d"]
            else:
                xi_p   = np.linspace(0, 200, 20)
                yi_p   = np.linspace(0, 15, 12)
                # Synthetic: low at top (weathering), higher below (bedrock)
                d_grid = np.tile(yi_p, (len(xi_p), 1)).T
                r2d_p  = np.where(d_grid < 5, np.random.uniform(15, 60, d_grid.shape),
                                   np.where(d_grid < 10, np.random.uniform(80, 150, d_grid.shape),
                                            np.random.uniform(150, 300, d_grid.shape)))

            slip_d_p = gd.get("slip_depth") if gd and gd.get("has_slip_plane") else None
            bed_d_p  = gd.get("bedrock_start") if gd else geo_in.get("bedrock_start", 9.0)
            fig_cont_p = make_geoelectric_contour(xi_p, yi_p, r2d_p,
                                                   slip_depth=slip_d_p, bedrock_depth=bed_d_p)
            st.plotly_chart(fig_cont_p, use_container_width=True)

        # Combined interpretation
        st.markdown("<div class='geo-card geo-card-accent' style='margin-top:0.5rem;'>", unsafe_allow_html=True)
        wd_sum  = float(seis_in.get("weathering_depth", 5.0))
        bs_sum  = float(gd.get("bedrock_start") if gd else geo_in.get("bedrock_start", 9.0))
        rec_sum = max(wd_sum * 1.2, bs_sum)
        st.markdown(f"""
        <div style='color:#ffa657; font-family:Space Mono; font-weight:700; margin-bottom:0.5rem;'>
            🏗️ Rekomendasi Fondasi Terpadu
        </div>
        <div style='color:#c9d1d9; line-height:2; font-size:0.92rem;'>
            Berdasarkan korelasi seismik dan geoelektrik:<br>
            → Zona Pelapukan (V₁/ρ &lt; 100 Ω.m): <strong>0 — {wd_sum:.1f} m</strong><br>
            → Bedrock Stabil (ρ &gt; 150 Ω.m): mulai dari <strong>{bs_sum:.1f} m</strong><br>
            → <span style='color:#f78166; font-weight:700;'>Kedalaman Fondasi Tiang Minimal: {rec_sum:.1f} m</span>
            <span style='color:#8b949e; font-size:0.82rem;'>(perlu verifikasi bor SPT)</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: Master DataFrame ──
    with tab3:
        st.markdown("""
        <div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;'>
            📋 Master DataFrame Terintegrasi
        </div>
        """, unsafe_allow_html=True)

        display_cols = ["lon", "lat", "slope_deg", "weathering_depth_m",
                        "resistivity_ohm", "slip_plane_detected",
                        "bedrock_depth_m", "risk_score", "risk_level"]
        avail_cols   = [c for c in display_cols if c in master_df.columns]

        st.dataframe(
            master_df[avail_cols],
            use_container_width=True, height=420,
            column_config={
                "lon":                st.column_config.NumberColumn("Lon", format="%.4f"),
                "lat":                st.column_config.NumberColumn("Lat", format="%.4f"),
                "slope_deg":          st.column_config.NumberColumn("Slope (°)", format="%.1f"),
                "weathering_depth_m": st.column_config.NumberColumn("Pelapukan (m)", format="%.1f"),
                "resistivity_ohm":    st.column_config.NumberColumn("Resistivitas (Ω.m)", format="%.0f"),
                "bedrock_depth_m":    st.column_config.NumberColumn("Bedrock (m)", format="%.1f"),
                "risk_score":         st.column_config.NumberColumn("Skor DSS", format="%.0f"),
                "risk_level":         st.column_config.TextColumn("Status"),
                "slip_plane_detected":st.column_config.CheckboxColumn("Bidang Gelincir"),
            }
        )

        qa1, qa2, qa3, qa4 = st.columns(4)
        qa1.metric("Total Titik",   len(master_df))
        qa2.metric("Missing Values", int(master_df.isnull().sum().sum()))
        qa3.metric("Duplikat",       int(master_df.duplicated().sum()))
        qa4.metric("Skor DSS Rata",  f"{avg_dss:.1f}")

        csv_master = master_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Master DataFrame (CSV)",
            csv_master, "master_zonasi_risiko.csv", "text/csv",
            use_container_width=True,
        )

    # ── TAB 4: DSS Report ──
    with tab4:
        st.markdown("""
        <div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;
                    letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;'>
            🧠 Decision Support System Report
        </div>
        """, unsafe_allow_html=True)

        # Overall risk gauge
        overall_status = "BAHAYA" if n_baha/max(n_total,1) > 0.3 else ("WASPADA" if n_wasp/max(n_total,1) > 0.3 else "AMAN")
        status_color_m = "#f85149" if overall_status == "BAHAYA" else "#d29922" if overall_status == "WASPADA" else "#3fb950"

        st.markdown(
            f"<div style='background:#161b22; border:1px solid #30363d; border-top:4px solid {status_color_m};"
            f"border-radius:12px; padding:1.5rem; margin-bottom:1rem; text-align:center;'>"
            f"<div style='color:#8b949e; font-size:0.8rem; letter-spacing:2px;'>STATUS RISIKO KUMULATIF TERPADU</div>"
            f"<div style='font-family:Space Mono; font-size:2rem; font-weight:700; color:{status_color_m}; margin:0.5rem 0;'>"
            f"{overall_status}</div>"
            f"<div style='color:#8b949e; font-size:0.85rem;'>Skor DSS rata-rata: {avg_dss:.1f}/100</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Parameter scores
        avg_slope  = float(master_df["slope_deg"].mean())
        avg_weath  = float(master_df["weathering_depth_m"].mean())
        avg_resist = float(master_df["resistivity_ohm"].mean())
        slip_count = int(master_df["slip_plane_detected"].sum())

        st.markdown("**Skor Per Parameter:**")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("📐 Slope Rata-rata",    f"{avg_slope:.1f}°",
                   "BAHAYA" if avg_slope > 30 else "WASPADA" if avg_slope > 15 else "AMAN")
        pc2.metric("📡 Pelapukan Rata-rata", f"{avg_weath:.1f} m",
                   "BAHAYA" if avg_weath > WEATHER_CRIT_M else "WASPADA" if avg_weath > WEATHER_WARN_M else "AMAN")
        pc3.metric("⚡ Resistivitas Rata",  f"{avg_resist:.0f} Ω.m",
                   f"{slip_count} titik bidang gelincir")

        # Recommendations
        st.markdown("<div class='geo-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-section'>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-header'>Rekomendasi Tindakan ({overall_status})</div>", unsafe_allow_html=True)
        render_rekoms(ACTIONS[overall_status], overall_status.lower() if overall_status != "WASPADA" else "waspada")
        st.markdown("</div>", unsafe_allow_html=True)

        if n_baha > 0:
            st.error(f"🚨 **{n_baha} titik BAHAYA** memerlukan penanganan prioritas tinggi & verifikasi lapangan segera.")
        if n_wasp > 0:
            st.warning(f"⚠️ **{n_wasp} titik WASPADA** memerlukan monitoring berkelanjutan dan kajian geoteknik.")
        if n_aman > 0:
            st.success(f"✅ **{n_aman} titik AMAN** menunjukkan kondisi stabil berdasarkan ketiga parameter.")


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — GEOBOT AI
# ═══════════════════════════════════════════════════════════════
elif menu == "🤖 GeoBot AI":
    st.markdown("""
    <h1 style='margin-bottom:0.2rem;'>🤖 GeoBot AI</h1>
    <div style='color:#8b949e; margin-bottom:1rem; font-size:0.95rem;'>
        Asisten cerdas berbasis AI untuk menjelaskan hasil teknis ke bahasa awam
    </div>
    """, unsafe_allow_html=True)

    # Context summary
    td_ok2 = st.session_state.topo_data is not None
    sd_ok2 = st.session_state.seismic_data is not None
    gd_ok2 = st.session_state.geoelectric_data is not None
    n_ok2  = sum([td_ok2, sd_ok2, gd_ok2])

    if n_ok2 > 0:
        st.markdown(
            f"<div class='info-box'>"
            f"<strong>🧠 Konteks Data Aktif ({n_ok2}/3 modul):</strong><br>"
            + ("✅ Topografi dimuat<br>" if td_ok2 else "")
            + ("✅ Seismik dimuat<br>" if sd_ok2 else "")
            + ("✅ Geoelektrik dimuat<br>" if gd_ok2 else "")
            + "GeoBot akan memberikan jawaban spesifik berdasarkan data Anda."
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class='info-box'>
            💬 <strong>Apa yang dapat ditanyakan?</strong><br>
            • Tentang risiko longsor dan kelerengan<br>
            • Interpretasi seismic refraction (V1, V2, kedalaman pelapukan)<br>
            • Analisis geoelektrik dan rekomendasi fondasi tiang<br>
            • Standar PVMBG dan rekomendasi mitigasi bencana
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # Chat history display
    for msg in st.session_state.geobot_messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Quick prompt buttons
    if not st.session_state.geobot_messages:
        st.markdown("<div style='color:#8b949e; font-size:0.85rem; margin-bottom:0.5rem;'>Mulai dengan pertanyaan:</div>", unsafe_allow_html=True)
        qc1, qc2, qc3, qc4 = st.columns(4)
        prompts = [
            ("📐 Status Risiko", "Apa status risiko wilayah ini dan apa artinya untuk warga?"),
            ("📡 Seismik",       "Jelaskan hasil seismic refraction dengan bahasa sederhana"),
            ("⚡ Geoelektrik",   "Apa itu bidang gelincir dan mengapa berbahaya?"),
            ("🛡️ Mitigasi",     "Apa saja langkah mitigasi yang perlu dilakukan?"),
        ]
        for col, (label, prompt) in zip([qc1, qc2, qc3, qc4], prompts):
            if col.button(label, use_container_width=True):
                st.session_state.geobot_messages.append({"role": "user", "content": prompt})
                response = geobot_local_response(prompt)
                st.session_state.geobot_messages.append({"role": "assistant", "content": response})
                st.rerun()

    # Chat input
    user_input = st.chat_input("Ketik pertanyaan geofisika Anda di sini...")
    if user_input:
        st.session_state.geobot_messages.append({"role": "user", "content": user_input})

        # Try Anthropic API first; fallback to local
        context_str = build_geobot_context()
        try:
            import anthropic
            client = anthropic.Anthropic()
            # Build message history (last 10 for context)
            history = st.session_state.geobot_messages[-10:]
            api_msgs = []
            for m in history[:-1]:  # all but last
                api_msgs.append({"role": m["role"], "content": m["content"]})
            api_msgs.append({
                "role": "user",
                "content": f"[Konteks Data]\n{context_str}\n\n[Pertanyaan]\n{user_input}"
            })
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                system=GEOBOT_SYSTEM,
                messages=api_msgs,
            )
            response = resp.content[0].text
        except Exception:
            response = geobot_local_response(user_input)

        st.session_state.geobot_messages.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear chat
    if st.session_state.geobot_messages:
        if st.button("🗑️ Hapus Riwayat Chat", use_container_width=False):
            st.session_state.geobot_messages = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 7 — LAPORAN AKHIR
# ═══════════════════════════════════════════════════════════════
elif menu == "📋 Laporan Akhir":
    st.markdown("""
    <h1 style='margin-bottom:0.2rem;'>📋 Laporan Akhir Komprehensif</h1>
    <div style='color:#8b949e; margin-bottom:1.5rem; font-size:0.95rem;'>
        Ringkasan resmi integrasi Topografi + Seismik + Geoelektrik — siap cetak untuk BPBD/pemerintah daerah
    </div>
    """, unsafe_allow_html=True)

    res = st.session_state.analysis_result
    td  = st.session_state.topo_data
    sd  = st.session_state.seismic_data
    gd  = st.session_state.geoelectric_data

    has_any = res is not None or td is not None or sd is not None or gd is not None

    if not has_any:
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:#161b22;
                    border:2px dashed #30363d; border-radius:16px;'>
            <div style='font-size:3rem;'>📋</div>
            <div style='font-family:Space Mono; color:#ffa657; margin:0.5rem 0;'>Laporan Belum Tersedia</div>
            <div style='color:#8b949e;'>Upload minimal satu jenis data untuk menghasilkan laporan</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Collect all available data ──
        max_slope    = res.get("max_slope") if res else None
        mean_elev    = res.get("mean_elev") if res else None
        sg           = res.get("slope_grid") if res else None
        status       = res.get("status") if res else None
        status_color = res.get("status_color") if res else "#ffa657"
        file_name    = res.get("file_name", "—") if res else "—"
        timestamp    = (res.get("timestamp") if res else
                        datetime.datetime.now().strftime("%d %B %Y, %H:%M WIB"))
        css_cls      = res.get("css_cls", "waspada") if res else "waspada"

        v1_rep   = sd.get("v1")  if sd else None
        v2_rep   = sd.get("v2")  if sd else None
        wd_rep   = sd.get("weathering_depth") if sd else None
        bs_seis  = sd.get("bedrock_start")    if sd else None

        has_slip  = gd.get("has_slip_plane", False) if gd else False
        slip_dep  = gd.get("slip_depth", 0)         if gd else None
        wd_geo    = gd.get("weathered_depth", 0)     if gd else None
        bs_geo    = gd.get("bedrock_start", 0)       if gd else None
        mean_res  = gd.get("mean_resistivity", gd.get("resistivity_mean", None)) if gd else None

        # Compute composite slope stats from session state
        if sg is not None:
            sf        = sg[~np.isnan(sg)].flatten()
            pct_aman  = 100 * (sf < 15).sum()  / len(sf)
            pct_wasp  = 100 * ((sf >= 15) & (sf < 30)).sum() / len(sf)
            pct_baha  = 100 * (sf >= 30).sum() / len(sf)
        elif td and len(td.get("slopes",[])) > 0:
            sf_arr    = np.array(td["slopes"])
            max_slope = float(sf_arr.max())
            pct_aman  = 100 * (sf_arr < 15).sum()  / len(sf_arr)
            pct_wasp  = 100 * ((sf_arr >= 15) & (sf_arr < 30)).sum() / len(sf_arr)
            pct_baha  = 100 * (sf_arr >= 30).sum() / len(sf_arr)
            status    = classify_risk(max_slope)[0]
            status_color = classify_risk(max_slope)[2]
            css_cls   = classify_risk(max_slope)[3]
        else:
            pct_aman = pct_wasp = pct_baha = None

        # Determine status if still unknown
        if status is None and max_slope is not None:
            status, _, status_color, css_cls = classify_risk(max_slope)

        # Combined DSS: use available data
        if max_slope is not None and wd_rep is not None and mean_res is not None:
            dss = calc_dss_score(max_slope, wd_rep or 0, has_slip, mean_res or 100)
            final_status = dss["risk_level"]
            final_color  = dss["risk_color"]
            final_score  = dss["risk_score"]
        elif status is not None:
            final_status = status
            final_color  = status_color or "#ffa657"
            final_score  = None
        else:
            final_status = "N/A"
            final_color  = "#8b949e"
            final_score  = None

        # ── Report Header ──
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#161b22,#1a1f2e);"
            f"border:1px solid #30363d; border-top:4px solid #f78166;"
            f"border-radius:12px; padding:2rem; margin-bottom:1.5rem;'>"
            f"<div style='font-family:Space Mono; font-size:0.72rem; color:#8b949e;"
            f"letter-spacing:3px; text-transform:uppercase;'>LAPORAN RESMI KOMPREHENSIF</div>"
            f"<div style='font-family:Space Mono; font-size:1.6rem; font-weight:700;"
            f"color:#e6edf3; margin:0.3rem 0;'>Analisis Geofisika &amp; Risiko Longsor</div>"
            f"<div style='font-family:Space Mono; font-size:0.85rem; color:#ffa657;'>"
            f"Platform GeoLink v2.0 — Bridging Science and Society</div>"
            f"<div style='margin-top:1rem; display:flex; gap:2rem; flex-wrap:wrap;'>"
            f"<div><div style='font-size:0.78rem; color:#8b949e;'>Tanggal Analisis</div>"
            f"<div style='font-family:Space Mono; color:#e6edf3; font-size:0.88rem;'>{timestamp}</div></div>"
            f"<div><div style='font-size:0.78rem; color:#8b949e;'>Sumber Data</div>"
            f"<div style='font-family:Space Mono; color:#ffa657; font-size:0.82rem;'>{file_name}</div></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── 01 Kesimpulan Eksekutif ──
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown("<div class='report-header'>01 — Kesimpulan Eksekutif</div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Status Risiko",     final_status)
        if max_slope is not None:
            k2.metric("Slope Maksimum", f"{max_slope:.1f}°")
        if mean_elev is not None:
            k3.metric("Rerata Elevasi", f"{mean_elev:.0f} m")
        if final_score is not None:
            k4.metric("Skor DSS",       f"{final_score:.0f}/100")

        st.markdown(
            f"<p style='color:#c9d1d9; line-height:1.8; margin-top:1rem;'>"
            f"Berdasarkan integrasi data geofisika, wilayah studi berstatus risiko longsor "
            f"<strong style='color:{final_color};'>{final_status}</strong>."
            + (f" Kelerengan maksimum <strong>{max_slope:.1f}°</strong>." if max_slope else "")
            + (f" Kedalaman lapisan pelapukan dari seismic refraction: <strong>{wd_rep:.1f} m</strong>." if wd_rep else "")
            + (f" Bidang gelincir geoelektrik: <strong>{'TERDETEKSI' if has_slip else 'tidak terdeteksi'}</strong>." if gd else "")
            + "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 02 Distribusi Zona Risiko Lereng ──
        if pct_aman is not None:
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            st.markdown("<div class='report-header'>02 — Distribusi Zona Risiko Kelerengan</div>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("✅ AMAN (<15°)",      f"{pct_aman:.1f}%")
            d2.metric("⚠️ WASPADA (15–30°)", f"{pct_wasp:.1f}%")
            d3.metric("🚨 BAHAYA (>30°)",    f"{pct_baha:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── 03 Data Geofisika Detail ──
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown("<div class='report-header'>03 — Data Geofisika Detail</div>", unsafe_allow_html=True)

        if sd:
            st.markdown(f"""
            <p style='color:#c9d1d9; line-height:1.8;'><strong style='color:#4da6ff;'>📡 Seismic Refraction:</strong><br>
            V₁ (Pelapukan) = {v1_rep:.0f} m/s &nbsp;|&nbsp; V₂ (Bedrock) = {v2_rep:.0f} m/s &nbsp;|&nbsp;
            Kedalaman Pelapukan = <strong>{wd_rep:.1f} m</strong> &nbsp;|&nbsp; Bedrock mulai = <strong>{bs_seis:.1f} m</strong>
            </p>
            """, unsafe_allow_html=True)

        if gd:
            slip_str = f"TERDETEKSI di {slip_dep:.1f} m" if has_slip else "Tidak terdeteksi"
            rec_g    = max((wd_geo or 0) * 1.2, bs_geo or 0) if wd_geo and bs_geo else None
            st.markdown(f"""
            <p style='color:#c9d1d9; line-height:1.8;'><strong style='color:#ffa657;'>⚡ Geoelektrik 2D:</strong><br>
            Resistivitas rata = {mean_res:.0f} Ω.m &nbsp;|&nbsp; Bidang Gelincir: <strong>{slip_str}</strong><br>
            Zona Pelapukan: 0–{wd_geo:.1f} m &nbsp;|&nbsp; Bedrock Stabil: mulai dari <strong>{bs_geo:.1f} m</strong>
            {f'&nbsp;|&nbsp; <strong style="color:#f78166;">Fondasi Tiang ≥ {rec_g:.1f} m</strong>' if rec_g else ''}
            </p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── 04 Rekomendasi Tindakan ──
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown("<div class='report-header'>04 — Rekomendasi Tindakan Mitigasi</div>", unsafe_allow_html=True)
        render_rekoms(ACTIONS[final_status], css_cls)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 05 Metodologi ──
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown("<div class='report-header'>05 — Metodologi &amp; Referensi</div>", unsafe_allow_html=True)
        for item in [
            f"Sumber data: **{file_name}**",
            "Topografi: Slope Analysis (Finite Difference, Zevenbergen & Thorne 1987)",
            "Seismik: Intercept-Time Method dua lapis (Telford et al. 1990)",
            "Geoelektrik: Identifikasi bidang gelincir ρ < 50 Ω.m (Reynolds 2011)",
            "Skor DSS: Kumulatif berbobot — Slope 40% + Pelapukan 35% + Resistivitas 25%",
            "Standar: SNI 13-7124-2005 & Pedoman PVMBG 2019",
            "Platform: GeoLink v2.0 — Hackathon Geofisika Indonesia 2026",
            f"Analisis dilakukan pada: **{timestamp}**",
        ]:
            st.markdown(f"<p style='color:#8b949e; font-size:0.87rem; margin:0.2rem 0;'>• {item}</p>",
                        unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#0d1117; border:1px solid #30363d; border-radius:8px;
                    padding:0.8rem 1rem; margin-top:0.8rem;'>
            <div style='color:#8b949e; font-size:0.8rem;'>
                ⚠️ <em>Disclaimer: Analisis ini bersifat indikatif berdasarkan data yang diunggah.
                Rekomendasi final harus melalui survei lapangan dan kajian geoteknik oleh
                tenaga ahli bersertifikat sesuai regulasi yang berlaku.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Download ──
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        actions_txt = ACTIONS[final_status]
        rekoms_str  = "\n".join([f"  {i+1}. {t}: {d}" for i, (t, d) in enumerate(actions_txt)])

        report_txt = (
            "LAPORAN KOMPREHENSIF ANALISIS GEOFISIKA & RISIKO LONGSOR\n"
            "GEOLINK v2.0 — Bridging Science and Society\n"
            + "=" * 70 + "\n"
            f"Tanggal Analisis : {timestamp}\n"
            f"Sumber Data      : {file_name}\n"
            + "=" * 70 + "\n\n"
            f"STATUS RISIKO TERPADU : {final_status}\n"
        )
        if final_score:
            report_txt += f"SKOR DSS KUMULATIF     : {final_score:.1f}/100\n"
        if max_slope:
            report_txt += f"Lereng Maksimum        : {max_slope:.2f}°\n"
        if mean_elev:
            report_txt += f"Rerata Elevasi         : {mean_elev:.1f} m dpl\n"
        if pct_aman is not None:
            report_txt += (
                f"\nDISTRIBUSI ZONA KELERENGAN:\n"
                f"  AMAN    (<15°)   : {pct_aman:.1f}%\n"
                f"  WASPADA (15-30°) : {pct_wasp:.1f}%\n"
                f"  BAHAYA  (>30°)   : {pct_baha:.1f}%\n"
            )
        if sd:
            report_txt += (
                f"\nDATA SEISMIC REFRACTION:\n"
                f"  V1 (Pelapukan)    : {v1_rep:.0f} m/s\n"
                f"  V2 (Bedrock)      : {v2_rep:.0f} m/s\n"
                f"  Ketebalan Pelapukan: {wd_rep:.1f} m\n"
                f"  Bedrock Mulai     : {bs_seis:.1f} m\n"
            )
        if gd:
            report_txt += (
                f"\nDATA GEOELEKTRIK 2D:\n"
                f"  Resistivitas Rata   : {mean_res:.0f} Ω.m\n"
                f"  Bidang Gelincir     : {'TERDETEKSI @ ' + str(round(slip_dep,1)) + ' m' if has_slip else 'Tidak terdeteksi'}\n"
                f"  Zona Pelapukan      : 0–{wd_geo:.1f} m\n"
                f"  Bedrock Stabil      : mulai dari {bs_geo:.1f} m\n"
            )
        report_txt += (
            f"\nREKOMENDASI TINDAKAN:\n{rekoms_str}\n\n"
            "METODOLOGI:\n"
            "  - Topografi     : Slope Analysis, Finite Difference (Zevenbergen & Thorne, 1987)\n"
            "  - Seismik       : Intercept-Time Method dua lapis (Telford et al., 1990)\n"
            "  - Geoelektrik   : Identifikasi bidang gelincir ρ < 50 Ω.m (Reynolds, 2011)\n"
            "  - Standar       : SNI 13-7124-2005 & Pedoman PVMBG 2019\n\n"
            "DISCLAIMER:\n"
            "  Analisis ini bersifat indikatif berdasarkan data yang diunggah.\n"
            "  Rekomendasi final harus dikonfirmasi survei lapangan oleh tenaga ahli.\n\n"
            "GeoLink v2.0 — Bridging Science and Society\n"
        )

        st.download_button(
            label="⬇️ Unduh Laporan Lengkap (.txt)",
            data=report_txt.encode("utf-8"),
            file_name=f"laporan_geolink_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )