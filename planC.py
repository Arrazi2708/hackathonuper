import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io

from scipy.ndimage import uniform_filter1d
try:
    from scipy.interpolate import griddata as scipy_griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if "geo_data"    not in st.session_state: st.session_state.geo_data    = None
if "df_grav"     not in st.session_state: st.session_state.df_grav     = None
if "mag_profile" not in st.session_state: st.session_state.mag_profile = None
if "mag_slicing" not in st.session_state: st.session_state.mag_slicing = None

st.set_page_config(
    page_title="GeoSight — Geophysics Dashboard",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injected CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #1c2333;
  --border: #30363d;
  --accent: #00d4aa;
  --accent2: #7c5ef5;
  --accent3: #f0a500;
  --text: #e6edf3;
  --muted: #8b949e;
}

html, body {
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}

.stApp { background-color: var(--bg); }

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}

.block-container {
  background: var(--bg) !important;
  padding-top: 1.5rem !important;
  padding-bottom: 2rem !important;
}

.geo-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 24px;
  margin-bottom: 12px;
  transition: border-color .2s;
}
.geo-card:hover { border-color: var(--accent); }

.geo-card .label {
  font-size: 0.75rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}
.geo-card .value {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent);
  line-height: 1;
}
.geo-card .unit {
  font-size: 0.85rem;
  color: var(--muted);
  margin-left: 4px;
}
.geo-card .delta {
  font-size: 0.8rem;
  color: #3fb950;
  margin-top: 4px;
}

.page-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-top: 10px;
  margin-bottom: 0;
  line-height: 1.3;
}
.page-sub {
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: 6px;
  margin-bottom: 28px;
}

.section-head {
  font-family: 'Syne', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text);
  border-left: 3px solid var(--accent);
  padding-left: 12px;
  margin: 24px 0 16px;
}

.badge {
  display: inline-block;
  background: rgba(0,212,170,.15);
  color: var(--accent);
  border: 1px solid rgba(0,212,170,.3);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 0.75rem;
  font-weight: 600;
  margin-right: 6px;
}
.badge-purple {
  background: rgba(124,94,245,.15);
  color: #a589f7;
  border-color: rgba(124,94,245,.3);
}
.badge-orange {
  background: rgba(240,165,0,.15);
  color: var(--accent3);
  border-color: rgba(240,165,0,.3);
}

.info-box {
  background: rgba(0,212,170,.07);
  border: 1px solid rgba(0,212,170,.2);
  border-radius: 10px;
  padding: 14px 18px;
  margin: 12px 0;
  font-size: 0.9rem;
  color: var(--text);
}

/* Upload area styling */
.upload-zone {
  background: rgba(0,212,170,.05);
  border: 1.5px dashed rgba(0,212,170,.4);
  border-radius: 12px;
  padding: 16px 20px;
  margin-bottom: 18px;
}
.upload-zone .uz-title {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  color: var(--accent);
  font-size: 0.95rem;
  margin-bottom: 4px;
}
.upload-zone .uz-sub {
  font-size: 0.78rem;
  color: var(--muted);
}

.stDataFrame { border-radius: 10px; overflow: hidden; }

.stButton > button {
  background: var(--accent) !important;
  color: #0d1117 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.5rem 1.4rem !important;
}
.stButton > button:hover { background: #00b894 !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border-radius: 10px;
  padding: 4px;
  border: 1px solid var(--border);
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 7px;
  color: var(--muted);
  font-family: 'Syne', sans-serif;
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #0d1117 !important;
}

label { color: var(--muted) !important; font-size: 0.85rem !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# HELPER – Plotly dark theme
# ═══════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,1)",
    font=dict(family="DM Sans", color="#e6edf3"),
    xaxis=dict(gridcolor="#30363d", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#30363d", linecolor="#30363d", zerolinecolor="#30363d"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def hex_to_rgba(hex_color, alpha=0.8):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def apply_theme(fig, title=""):
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=15, family="Syne", color="#e6edf3")))
    return fig

def page_penampang(
    geo_data: dict | None = None,
    df_grav: pd.DataFrame | None = None,
    mag_profile: pd.DataFrame | None = None,
    mag_slicing: pd.DataFrame | None = None,
):
    """
    Render halaman Penampang Terpadu.
 
    Parameters
    ----------
    geo_data   : dict keluaran parse_geolistrik()  — {sheet_name: DataFrame}
    df_grav    : DataFrame keluaran parse_gravity()
    mag_profile: DataFrame keluaran parse_magnetik()[0]
    mag_slicing: DataFrame keluaran parse_magnetik()[1]
 
    Jika parameter tidak disediakan, fungsi akan meminta upload ulang
    atau menggunakan data sintetis sebagai fallback.
    """
    
    # ── 1. KONTROL PANEL ────────────────────────────────────────────────
    with st.expander("⚙️ Pengaturan Penampang", expanded=True):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            v_exag = st.slider("Vertical Exaggeration", 1.0, 5.0, 2.0, 0.5,
                               help="Perbesaran vertikal untuk memperjelas perbedaan kedalaman lapisan.")
            max_depth = st.number_input("Kedalaman Maksimum (m)", 50, 500, 120, 10, key="penampang_max_depth")
        with col_s2:
            show_geo_layer = st.checkbox("Tampilkan Pseudo-Section Geolistrik", True, key="penampang_geo")
            show_grav_layer = st.checkbox("Tampilkan Profil CBA Gravitasi", True, key="penampang_grav_layer")
            show_mag_layer = st.checkbox("Tampilkan Profil ΔT Magnetik", True, key="penampang_mag_layer")
        with col_s3:
            show_boreholes = st.checkbox("Tampilkan Posisi Bor (Sintetis)", True, key="penampang_boreholes")
            show_litho_labels = st.checkbox("Tampilkan Label Litologi", True, key="penampang_litho_labels")
            section_lintasan = st.selectbox("Lintasan Referensi", list(geo_data.keys()) if geo_data else ["Lintasan Sintetis"], key="penampang_lintasan")
    
    # ── 2. SIAPKAN DATA ─────────────────────────────────────────────────
    if geo_data and section_lintasan in geo_data:
        df_geo = geo_data[section_lintasan]
        x_positions = df_geo["Datum_Point"].values
        rho_values  = df_geo["Rho"].values
        spasi_values = df_geo["Spasi"].values
        x_min, x_max_val = float(x_positions.min()), float(x_positions.max())
        has_real_geo = True
    else:
        # Sintetis geolistrik
        x_min, x_max_val = 0.0, 144.0
        nx_syn = 60
        x_positions = np.linspace(x_min, x_max_val, nx_syn)
        rng = np.random.default_rng(42)
        rho_values  = (20 + 15 * np.sin(x_positions / 30)
                       + rng.normal(0, 4, nx_syn)
                       - 8 * ((x_positions > 60) & (x_positions < 100)).astype(float))
        rho_values  = np.clip(rho_values, 5, 200)
        spasi_values = np.ones(nx_syn) * 8.0
        has_real_geo = False
 
    profile_length = x_max_val - x_min
    
    # Gravitasi profil sepanjang lintasan
    if df_grav is not None and not df_grav.empty:
        grav_x_raw = np.sqrt(
            (df_grav["X"].to_numpy(dtype=float) - float(df_grav["X"].iloc[0]))**2 +
            (df_grav["Y"].to_numpy(dtype=float) - float(df_grav["Y"].iloc[0]))**2
        )

        grav_cba = df_grav["CBA"].to_numpy(dtype=float)

        idx = np.argsort(grav_x_raw)
        
        xp = grav_x_raw[idx]
        fp = grav_cba[idx]

        x_target = np.linspace(0, float(xp.max()), len(x_positions))
        
        cba_interp = np.interp(x_target, xp, fp)
        has_real_grav = True
    else:
        rng2 = np.random.default_rng(7)
        cba_interp = (7.8 + 4 * np.sin(x_positions / 40)
                      - 2 * np.cos(x_positions / 15)
                      + rng2.normal(0, 0.4, len(x_positions)))
        has_real_grav = False
    
    # Magnetik profil sepanjang lintasan
    if mag_profile is not None and not mag_profile.empty:
        df_mp = mag_profile.dropna(subset=["UTM_X", "Anomali"]).sort_values("UTM_X").reset_index(drop=True)
        if len(df_mp) > 1:
            utm_x = df_mp["UTM_X"].to_numpy(dtype=float)
            utm_y = df_mp["UTM_Y"].fillna(0).to_numpy(dtype=float)
            anom  = df_mp["Anomali"].to_numpy(dtype=float)
            dist_m = np.concatenate([[0], np.cumsum(np.sqrt(dx_m**2 + dy_m**2))])
        else:
            dist_m = np.array([0.0])
            anom   = np.array([0.0])
        x_target = np.linspace(0,
            float(dist_m.max()) if dist_m.max() > 0 else 1,
            len(x_positions)
        )

        mag_interp = np.interp(x_target, dist_m, anom)
        has_real_mag = True
    else:
        rng3 = np.random.default_rng(13)
        mag_interp = (20 * np.sin(x_positions / 25)
                      - 15 * np.cos(x_positions / 18 + 1)
                      + rng3.normal(0, 4, len(x_positions)))
        has_real_mag = False
    
     # ── 3. MODEL BAWAH PERMUKAAN (lapisan dari resistivitas) ─────────────
    rho_norm = (rho_values - rho_values.min()) / (np.ptp(rho_values) + 1e-6)
 
    # Estimasi kedalaman batas lapisan dari resistivitas (smoothed)
    from scipy.ndimage import uniform_filter1d
    rho_smooth = uniform_filter1d(rho_values, size=max(3, len(rho_values)//10))
 
    # Tiga batas lapisan: dangkal, menengah, dalam
    depth_boundary_1 = (10 + 6 * np.sin(x_positions / 25)
                         + 3 * (rho_smooth - rho_smooth.min()) / (np.ptp(rho_smooth) + 1e-6) * 8)
    depth_boundary_2 = depth_boundary_1 + (18 + 10 * np.cos(x_positions / 35)
                                            + 5 * rho_norm * 4)
    depth_boundary_3 = depth_boundary_2 + (20 + 8 * np.sin(x_positions / 45 + 0.5)
                                            + 4 * rho_norm * 6)
    depth_boundary_4 = np.clip(depth_boundary_3 + 15 + 5 * np.cos(x_positions / 30), 0, max_depth * 0.85)
    
    LAYER_DEFS = [
        {"name": "Lapisan 1 — Lapukan / Tanah",      "color": "#d4a96a", "bot": depth_boundary_1},
        {"name": "Lapisan 2 — Sedimen Basah / Akuifer","color": "#7c9e72", "bot": depth_boundary_2},
        {"name": "Lapisan 3 — Batupasir / Lanau",     "color": "#8b7355", "bot": depth_boundary_3},
        {"name": "Lapisan 4 — Batuan Lapuk",          "color": "#5a6e80", "bot": depth_boundary_4},
        {"name": "Lapisan 5 — Batuan Dasar",          "color": "#3d4f5c", "bot": np.full_like(x_positions, max_depth)},
    ]
 
    LITHO_TABLE = {
        "Lapisan 1 — Lapukan / Tanah":       "5–40 Ω·m",
        "Lapisan 2 — Sedimen Basah / Akuifer":"5–25 Ω·m",
        "Lapisan 3 — Batupasir / Lanau":      "40–200 Ω·m",
        "Lapisan 4 — Batuan Lapuk":           "200–500 Ω·m",
        "Lapisan 5 — Batuan Dasar":           ">500 Ω·m",
    }
    
    # ── 4. BUAT FIGUR SUBPLOT ────────────────────────────────────────────
    n_rows = 1
    row_heights = [3]
    subplot_titles = ["Penampang Geologi Bawah Permukaan"]
    if show_geo_layer:
        n_rows += 1; row_heights.append(1); subplot_titles.append("Resistivitas ρₐ (Ω·m)")
    if show_grav_layer:
        n_rows += 1; row_heights.append(1); subplot_titles.append("CBA Gravitasi (mGal)")
    if show_mag_layer:
        n_rows += 1; row_heights.append(1); subplot_titles.append("ΔT Magnetik (nT)")
 
    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=row_heights,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=subplot_titles,
    )
    
    # ── 4a. PENAMPANG UTAMA ──────────────────────────────────────────────
    prev_tops = np.zeros(len(x_positions))
    for layer in LAYER_DEFS:
        bot = layer["bot"] * v_exag
        top = prev_tops
 
        # Isi polygon lapisan (toself)
        x_fill = np.concatenate([x_positions, x_positions[::-1]])
        y_fill = np.concatenate([-bot, -top[::-1]])
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            fill="toself",
            fillcolor="rgba(212,169,106,0.8)",
            line=dict(width=0),
            name=layer["name"],
            hoverinfo="skip",
            showlegend=True,
            legendgroup=layer["name"],
        ), row=1, col=1)
 
        # Garis batas lapisan
        fig.add_trace(go.Scatter(
            x=x_positions, y=-bot,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=1),
            showlegend=False,
            hovertemplate=(
                f"<b>{layer['name']}</b><br>"
                "Posisi: %{x:.0f} m<br>"
                "Kedalaman batas: %{customdata:.1f} m<br>"
                f"Rentang ρ: {LITHO_TABLE[layer['name']]}"
                "<extra></extra>"
            ),
            customdata=layer["bot"],
        ), row=1, col=1)
 
        prev_tops = bot
        
    # Label litologi di penampang
    if show_litho_labels:
        label_positions = [
            (0.15, LAYER_DEFS[0], 0),
            (0.4,  LAYER_DEFS[1], 0),
            (0.65, LAYER_DEFS[2], 0),
            (0.5,  LAYER_DEFS[3], 0),
            (0.8,  LAYER_DEFS[4], 0),
        ]
        for frac, ldef, _ in label_positions:
            xi = int(frac * (len(x_positions) - 1))
            if xi >= len(x_positions):
                continue
            bot_d = ldef["bot"][xi]
            top_d = LAYER_DEFS[max(0, LAYER_DEFS.index(ldef) - 1)]["bot"][xi] if LAYER_DEFS.index(ldef) > 0 else 0
            mid_d = (top_d + bot_d) / 2 * v_exag
            short_name = ldef["name"].split("—")[1].strip() if "—" in ldef["name"] else ldef["name"]
            fig.add_annotation(
                x=x_positions[xi], y=-mid_d,
                text=short_name,
                showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.7)", family="DM Sans"),
                row=1, col=1,
            )
    
    # Posisi bor sintetis
    if show_boreholes:
        borehole_xpos = [profile_length * 0.2, profile_length * 0.5, profile_length * 0.78]
        for i, bx in enumerate(borehole_xpos):
            fig.add_vline(
                x=bx,
                line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1.5,
                annotation_text=f"BH-{i+1:02d}",
                annotation_position="top",
                annotation_font=dict(size=10, color="#e6edf3"),
                row=1, col=1,
            )
 
    row_idx = 2
    
    # ── 4b. PSEUDO-SECTION / PROFIL GEOLISTRIK ───────────────────────────
    if show_geo_layer:
        fig.add_trace(go.Scatter(
            x=x_positions, y=rho_values,
            mode="lines+markers",
            name="ρₐ" + ("" if has_real_geo else " (sintetis)"),
            line=dict(color="#4fa3e0", width=1.8),
            marker=dict(size=4, color="#4fa3e0"),
            fill="tozeroy", fillcolor="rgba(79,163,224,0.12)",
            hovertemplate="Posisi: %{x:.0f} m<br>ρₐ: %{y:.1f} Ω·m<extra></extra>",
        ), row=row_idx, col=1)
        if not has_real_geo:
            fig.add_annotation(
                x=x_positions[len(x_positions)//2], y=rho_values.max() * 0.9,
                text="⚠ Data sintetis — upload file untuk data lapangan",
                showarrow=False, font=dict(size=9, color="#8b949e"),
                row=row_idx, col=1,
            )
        row_idx += 1
    
    # ── 4c. PROFIL GRAVITASI CBA ─────────────────────────────────────────
    if show_grav_layer:
        fig.add_trace(go.Scatter(
            x=x_positions, y=cba_interp,
            mode="lines",
            name="CBA" + ("" if has_real_grav else " (sintetis)"),
            line=dict(color="#e8923a", width=1.8),
            fill="tozeroy", fillcolor="rgba(232,146,58,0.10)",
            hovertemplate="Posisi: %{x:.0f} m<br>CBA: %{y:.3f} mGal<extra></extra>",
        ), row=row_idx, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1, row=row_idx, col=1)
        if not has_real_grav:
            fig.add_annotation(
                x=x_positions[len(x_positions)//2], y=cba_interp.max() * 0.85,
                text="⚠ Data sintetis", showarrow=False,
                font=dict(size=9, color="#8b949e"), row=row_idx, col=1,
            )
        row_idx += 1
        
    # ── 4d. PROFIL MAGNETIK ΔT ──────────────────────────────────────────
    if show_mag_layer:
        fig.add_trace(go.Scatter(
            x=x_positions, y=mag_interp,
            mode="lines",
            name="ΔT" + ("" if has_real_mag else " (sintetis)"),
            line=dict(color="#b87dd4", width=1.8),
            fill="tozeroy", fillcolor="rgba(184,125,212,0.10)",
            hovertemplate="Posisi: %{x:.0f} m<br>ΔT: %{y:.1f} nT<extra></extra>",
        ), row=row_idx, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1, row=row_idx, col=1)
        if not has_real_mag:
            fig.add_annotation(
                x=x_positions[len(x_positions)//2], y=mag_interp.max() * 0.85,
                text="⚠ Data sintetis", showarrow=False,
                font=dict(size=9, color="#8b949e"), row=row_idx, col=1,
            )
    
    # ── 5. LAYOUT AKHIR ──────────────────────────────────────────────────
    fig.update_layout(
        **PLOT_LAYOUT,
        height=180 * n_rows + 80,
        title=dict(
            text=(
                f"Penampang Terpadu — {section_lintasan} "
                f"| V.Exag {v_exag:.1f}× | Panjang {profile_length:.0f} m"
            ),
            font=dict(size=14, family="Syne", color="#e6edf3"),
        ),
        legend=dict(
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor="#30363d",
            borderwidth=1,
            font=dict(size=11),
            orientation="v",
            x=1.01, y=1,
        ),
        hovermode="x unified",
    )
 
    fig.update_xaxes(
        title_text="Jarak / Posisi (m)",
        gridcolor="#30363d", linecolor="#30363d",
        row=n_rows, col=1,
    )
    fig.update_yaxes(
        title_text=f"Kedalaman (m × {v_exag})",
        gridcolor="#30363d", linecolor="#30363d",
        row=1, col=1,
    )
    for a in fig.layout.annotations:
        if a.text in subplot_titles:
            a.font = dict(size=12, family="Syne", color="#e6edf3")
 
    st.plotly_chart(fig, use_container_width=True)
    
    # ── 6. TABEL INTERPRETASI LAPISAN ────────────────────────────────────
    st.markdown('<div class="section-head">Tabel Interpretasi Lapisan</div>', unsafe_allow_html=True)
 
    interp_rows = []
    prev_d = 0
    for i, layer in enumerate(LAYER_DEFS):
        mean_d = float(layer["bot"].mean())
        interp_rows.append({
            "Lapisan": f"L-{i+1}",
            "Deskripsi": layer["name"].split("—")[1].strip() if "—" in layer["name"] else layer["name"],
            "Kedalaman Top (m)": f"{prev_d:.1f}",
            "Kedalaman Dasar rata-rata (m)": f"{mean_d:.1f}",
            "Rentang Resistivitas": LITHO_TABLE[layer["name"]],
            "Interpretasi Geologi": _geo_interp(i),
        })
        prev_d = mean_d
 
    st.dataframe(pd.DataFrame(interp_rows), use_container_width=True, hide_index=True)
    
    # ── 7. KORELASI ANTAR METODE ─────────────────────────────────────────
    st.markdown('<div class="section-head">Korelasi Anomali vs Resistivitas</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
 
    with col_c1:
        fig_corr1 = go.Figure(go.Scatter(
            x=rho_values, y=cba_interp,
            mode="markers",
            marker=dict(
                size=7, color=x_positions,
                colorscale=[[0,"#4fa3e0"],[0.5,"#00d4aa"],[1,"#e8923a"]],
                colorbar=dict(
                    title=dict(text="Posisi (m)", font=dict(color="#e6edf3")),
                    tickfont=dict(color="#e6edf3"), len=0.8,
                ),
                opacity=0.75,
            ),
            text=[f"x={x:.0f}m, ρ={r:.1f} Ω·m, CBA={c:.2f} mGal"
                  for x, r, c in zip(x_positions, rho_values, cba_interp)],
            hovertemplate="%{text}<extra></extra>",
        ))
        apply_theme(fig_corr1, "Resistivitas vs CBA Gravitasi")
        fig_corr1.update_layout(
            height=300,
            xaxis_title="ρₐ (Ω·m)",
            yaxis_title="CBA (mGal)",
        )
        st.plotly_chart(fig_corr1, use_container_width=True)
 
    with col_c2:
        fig_corr2 = go.Figure(go.Scatter(
            x=rho_values, y=mag_interp,
            mode="markers",
            marker=dict(
                size=7, color=x_positions,
                colorscale=[[0,"#4fa3e0"],[0.5,"#7c5ef5"],[1,"#b87dd4"]],
                colorbar=dict(
                    title=dict(text="Posisi (m)", font=dict(color="#e6edf3")),
                    tickfont=dict(color="#e6edf3"), len=0.8,
                ),
                opacity=0.75,
            ),
            text=[f"x={x:.0f}m, ρ={r:.1f} Ω·m, ΔT={m:.1f} nT"
                  for x, r, m in zip(x_positions, rho_values, mag_interp)],
            hovertemplate="%{text}<extra></extra>",
        ))
        apply_theme(fig_corr2, "Resistivitas vs ΔT Magnetik")
        fig_corr2.update_layout(
            height=300,
            xaxis_title="ρₐ (Ω·m)",
            yaxis_title="ΔT (nT)",
        )
        st.plotly_chart(fig_corr2, use_container_width=True)
    
    # ── 8. CATATAN STATUS DATA ───────────────────────────────────────────
    badges_html = ""
    for label, real, color in [
        ("Geolistrik", has_real_geo, "#00d4aa"),
        ("Gravitasi",  has_real_grav,"#f0a500"),
        ("Magnetik",   has_real_mag, "#7c5ef5"),
    ]:
        status = "✅ Data lapangan" if real else "⚠ Data sintetis"
        badges_html += (
            f"<span style='background:rgba(255,255,255,.06);border:1px solid {color}55;"
            f"border-radius:8px;padding:6px 12px;font-size:.82rem;color:#e6edf3;'>"
            f"<b style='color:{color};'>{label}</b> · {status}</span>"
        )
 
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;'>{badges_html}</div>",
        unsafe_allow_html=True,
    )
    
    # ── 9. EKSPOR ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Ekspor Data Penampang</div>', unsafe_allow_html=True)
    export_df = pd.DataFrame({
        "Posisi (m)":            np.round(x_positions, 1),
        "ρₐ (Ω·m)":             np.round(rho_values, 2),
        "CBA (mGal)":            np.round(cba_interp, 3),
        "ΔT (nT)":               np.round(mag_interp, 1),
        "Batas L1 (m)":          np.round(LAYER_DEFS[0]["bot"], 1),
        "Batas L2 (m)":          np.round(LAYER_DEFS[1]["bot"], 1),
        "Batas L3 (m)":          np.round(LAYER_DEFS[2]["bot"], 1),
        "Batas L4 (m)":          np.round(LAYER_DEFS[3]["bot"], 1),
    })
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            "⬇️ Download CSV Penampang",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"penampang_{section_lintasan.replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="penampang_download_csv"
        )
    with col_e2:
        with st.expander("📋 Pratinjau Data Penampang"):
            st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)
 
 
def _geo_interp(idx: int) -> str:
    interps = [
        "Zona lapukan, material tidak terkonsolidasi, potensi kontaminasi tinggi",
        "Kemungkinan zona akuifer / air tanah, target sumur bor prioritas",
        "Sedimen terkonsolidasi, batupasir, kondisi teknik relatif stabil",
        "Batuan yang telah mengalami pelapukan, zona transisi",
        "Batuan dasar (basement), kecepatan seismik tinggi (>2000 m/s)",
    ]
    return interps[min(idx, len(interps) - 1)]
# ═══════════════════════════════════════════════════════════
# DATA PARSERS
# ═══════════════════════════════════════════════════════════

@st.cache_data
def parse_geolistrik(file_bytes):
    """Parse Pengolahan_DATA_GEOLISTRIK_2D_2024.xlsx — Wenner 2D pseudo-section."""
    bio = io.BytesIO(file_bytes)
    xl = pd.ExcelFile(bio)
    results = {}
    for sheet in xl.sheet_names:
        bio.seek(0)
        df = pd.read_excel(bio, sheet_name=sheet, header=None)
        # Data starts at row 82; columns 14=Datum_Point, 15=Spasi(a), 16=Rho
        data = df.iloc[82:, [14, 15, 16]].copy()
        data.columns = ["Datum_Point", "Spasi", "Rho"]
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.dropna()
        data = data[data["Rho"] > 0]
        if not data.empty:
            results[sheet] = data.reset_index(drop=True)
    return results


@st.cache_data
def parse_gravity(file_bytes):
    """Parse DATA_GRAV_FIKS_CIHUY.xlsx — Sheet3 CBA data."""
    bio = io.BytesIO(file_bytes)
    df = pd.read_excel(bio, sheet_name="Sheet3")
    df = df[["X", "Y", "Z GPS (m)", "CBA"]].copy()
    df.columns = ["X", "Y", "Z", "CBA"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df.reset_index(drop=True)


@st.cache_data
def parse_magnetik(file_bytes):
    """Parse MAGNET_FIKS_BANGET.xlsx — Olah sheet anomaly profile + Slicing 2D."""
    bio = io.BytesIO(file_bytes)
    # Olah sheet — anomaly profile
    df_olah = pd.read_excel(bio, sheet_name="Olah", header=0)
    df_olah = df_olah.iloc[1:].copy()  # skip sub-header row
    profile = pd.DataFrame({
        "Stasiun": df_olah["Stasiun"],
        "Longitude": pd.to_numeric(df_olah["Posisi"], errors="coerce"),
        "Latitude": pd.to_numeric(df_olah["Unnamed: 2"], errors="coerce"),
        "Elevasi": pd.to_numeric(df_olah["Elevasi (m)"], errors="coerce"),
        "Anomali": pd.to_numeric(df_olah["Anomali Magnet \n(nT)"], errors="coerce"),
        "Jarak": pd.to_numeric(df_olah["Jarak (m)"], errors="coerce"),
        "UTM_X": pd.to_numeric(df_olah["UTM"], errors="coerce"),
        "UTM_Y": pd.to_numeric(df_olah["Unnamed: 24"], errors="coerce"),
    })
    profile = profile.dropna(subset=["Anomali"]).reset_index(drop=True)

    # Slicing sheet — 2D map + profile
    bio.seek(0)
    df_slc = pd.read_excel(bio, sheet_name="Slicing")
    keep = [c for c in ["X", "Y", "GRTP", "RFHD", "Jarak", "GSVDZ",
                         "Normalisasi FHD", "Normalisasi SVD"] if c in df_slc.columns]
    slicing = df_slc[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["X", "Y", "GRTP"])

    return profile, slicing.reset_index(drop=True)


def upload_box(label, filename_hint, key, types=["xlsx", "csv"]):
    """Render a styled file uploader."""
    st.markdown(f"""
    <div class="upload-zone">
      <div class="uz-title">📂 {label}</div>
      <div class="uz-sub">File: <code>{filename_hint}</code> &nbsp;|&nbsp; Format: {', '.join('.' + t for t in types)}</div>
    </div>""", unsafe_allow_html=True)
    return st.file_uploader(
        label, type=types, key=key, label_visibility="collapsed"
    )


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:14px 0 8px'>
      <div style='font-family:Syne;font-size:1.5rem;font-weight:800;
                  background:linear-gradient(135deg,#00d4aa,#7c5ef5);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🌏 GeoSight
      </div>
      <div style='color:#8b949e;font-size:.78rem;margin-top:2px;'>
        Geophysics Intelligence Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠  Beranda", "🔊  Seismik Refraksi", "⚡  Geolistrik", "🧲  Gravitasi & Magnetik", "🪨  Penampang Terpadu", "🗺️  Peta Spasial", "📋  Laporan"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.75rem;color:#8b949e;line-height:1.8;'>
    <b style='color:#e6edf3;'>Tentang GeoSight</b><br>
    Platform analisis data geofisika terintegrasi untuk eksplorasi, lingkungan, dan mitigasi bencana.<br><br>
    <span style='color:#00d4aa;'>●</span> Seismik Refraksi<br>
    <span style='color:#7c5ef5;'>●</span> Geolistrik<br>
    <span style='color:#f0a500;'>●</span> Gravitasi & Magnetik<br>
    <span style='color:#3fb950;'>●</span> Visualisasi Spasial
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: BERANDA
# ═══════════════════════════════════════════════════════════
if page == "🏠  Beranda":
    st.markdown('<div class="page-title">Selamat Datang di GeoSight</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Dashboard geofisika terpadu — dari data mentah hingga keputusan nyata.</div>', unsafe_allow_html=True)

    kpi = [
        ("Total Lintasan", "48", "km", "↑ 12% bulan ini", "#00d4aa"),
        ("Titik Pengukuran", "1,240", "pts", "↑ 8% bulan ini", "#7c5ef5"),
        ("Metode Aktif", "4", "", "Seismik, Geolistrik, Grav, Mag", "#f0a500"),
        ("Kedalaman Maks.", "350", "m", "Hasil interpretasi terdalam", "#3fb950"),
    ]
    cols = st.columns(4)
    for col, (label, val, unit, delta, color) in zip(cols, kpi):
        with col:
            st.markdown(f"""
            <div class="geo-card">
              <div class="label">{label}</div>
              <div class="value" style="color:{color};">{val}<span class="unit">{unit}</span></div>
              <div class="delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Ikhtisar Metode Geofisika</div>', unsafe_allow_html=True)

    method_data = {
        "Metode": ["Seismik Refraksi", "Geolistrik (Wenner 2D)", "Gravitasi", "Magnetik"],
        "Kegunaan Utama": ["Batuan dasar, akuifer", "Resistivitas lapisan 2D", "Struktur kerak (CBA)", "Anomali Total Field"],
        "Kedalaman (m)": [150, 200, 5000, 300],
        "Lintasan": [18, 14, 8, 8],
        "Status": ["✅ Aktif", "✅ Aktif", "✅ Aktif", "✅ Aktif"],
    }
    df_methods = pd.DataFrame(method_data)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure(go.Bar(
            x=df_methods["Metode"], y=df_methods["Lintasan"],
            marker_color=["#00d4aa", "#7c5ef5", "#f0a500", "#3fb950"],
            text=df_methods["Lintasan"], textposition="outside",
            textfont=dict(family="Syne", color="#e6edf3"),
        ))
        apply_theme(fig, "Jumlah Lintasan per Metode")
        fig.update_layout(showlegend=False, height=300, yaxis_title="Lintasan")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Pie(
            labels=df_methods["Metode"], values=df_methods["Lintasan"], hole=0.58,
            marker=dict(colors=["#00d4aa", "#7c5ef5", "#f0a500", "#3fb950"],
                        line=dict(color="#0d1117", width=2)),
            textfont=dict(family="DM Sans", color="#e6edf3"),
        ))
        apply_theme(fig2, "Distribusi Metode")
        fig2.update_layout(height=300, showlegend=True, legend=dict(orientation="v", x=1.0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-head">Ringkasan Kondisi Lapisan Bawah Permukaan</div>', unsafe_allow_html=True)

    depths = np.linspace(0, 350, 200)
    v1 = np.where(depths < 30, 400 + np.random.normal(0, 20, 200), np.nan)
    v2 = np.where((depths >= 30) & (depths < 120), 1200 + np.random.normal(0, 80, 200), np.nan)
    v3 = np.where(depths >= 120, 3200 + np.random.normal(0, 150, 200), np.nan)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=v1, y=-depths, mode="lines", name="Lapisan 1 (Tanah Permukaan)", line=dict(color="#f0a500", width=2)))
    fig3.add_trace(go.Scatter(x=v2, y=-depths, mode="lines", name="Lapisan 2 (Batuan Sedimen)", line=dict(color="#7c5ef5", width=2)))
    fig3.add_trace(go.Scatter(x=v3, y=-depths, mode="lines", name="Lapisan 3 (Batuan Dasar)", line=dict(color="#00d4aa", width=2)))
    apply_theme(fig3, "Model Kecepatan Gelombang Seismik vs Kedalaman")
    fig3.update_layout(height=360, xaxis_title="Kecepatan (m/s)", yaxis_title="Kedalaman (m)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-head">Referensi Cepat</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-box">
          <b>🔊 Seismik Refraksi</b><br>
          Mengukur waktu tiba gelombang seismik untuk menentukan kedalaman dan kecepatan lapisan bawah permukaan.
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-box" style="border-color:rgba(124,94,245,.3);background:rgba(124,94,245,.07);">
          <b>⚡ Geolistrik Wenner 2D</b><br>
          Konfigurasi Wenner untuk pemetaan resistivitas 2D. Data dari lintasan 28E (144m, spasi 8m).
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="info-box" style="border-color:rgba(240,165,0,.3);background:rgba(240,165,0,.07);">
          <b>🧲 Gravitasi & Magnetik</b><br>
          CBA dari koreksi Bouguer lengkap, dan anomali total field magnetik dengan koreksi harian & IGRF.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: SEISMIK REFRAKSI
# ═══════════════════════════════════════════════════════════
elif page == "🔊  Seismik Refraksi":
    st.markdown('<div class="page-title">Analisis Seismik Refraksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Interpretasi waktu-jarak, kecepatan lapisan, dan kedalaman refraktor.</div>', unsafe_allow_html=True)

    with st.expander("⚙️ Parameter Input", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Lapisan 1 (Permukaan)**")
            v1 = st.number_input("Kecepatan V₁ (m/s)", 200, 2000, 500, 50, key="seismik_v1")
        with col2:
            st.markdown("**Lapisan 2**")
            v2 = st.number_input("Kecepatan V₂ (m/s)", 500, 5000, 1800, 100, key="seismik_v2")
        with col3:
            st.markdown("**Lapisan 3 (Batuan Dasar)**")
            v3_val = st.number_input("Kecepatan V₃ (m/s)", 1000, 8000, 3500, 100, key="seismik_v3")

        col4, col5 = st.columns(2)
        with col4:
            h1 = st.number_input("Ketebalan Lapisan 1, h₁ (m)", 1.0, 100.0, 25.0, 1.0)
            h2 = st.number_input("Ketebalan Lapisan 2, h₂ (m)", 1.0, 200.0, 80.0, 5.0)
        with col5:
            x_max = st.number_input("Jarak Offset Maks. (m)", 50, 1000, 300, 10, key="seismik_xmax")
            noise_level = st.slider("Level Noise Data (%)", 0, 10, 3, key="seismik_noise")

    x = np.linspace(1, x_max, 200)
    noise = 1 + (noise_level / 100) * np.random.randn(len(x))
    t_direct = x / v1
    ic1 = np.arcsin(v1 / v2)
    t_refr2 = x / v2 + 2 * h1 * np.cos(ic1) / v1
    ic2 = np.arcsin(v2 / v3_val)
    ic12 = np.arcsin(v1 / v3_val)
    t_refr3 = x / v3_val + 2 * h1 * np.cos(ic12) / v1 + 2 * h2 * np.cos(ic2) / v2
    t_obs = np.minimum.reduce([t_direct, t_refr2, t_refr3]) * noise
    xco1 = 2 * h1 * np.sqrt((v2 + v1) / (v2 - v1))
    ti1 = 2 * h1 * np.cos(ic1) / v1
    ti2 = 2 * h1 * np.cos(ic12) / v1 + 2 * h2 * np.cos(ic2) / v2

    kpi_row = [
        ("Crossover Dist. 1", f"{xco1:.1f}", "m"),
        ("Intercept Time T₁", f"{ti1*1000:.1f}", "ms"),
        ("Intercept Time T₂", f"{ti2*1000:.1f}", "ms"),
        ("Kedalaman h₁", f"{h1:.1f}", "m"),
        ("Kedalaman h₂", f"{h1+h2:.1f}", "m"),
    ]
    cols = st.columns(5)
    colors = ["#00d4aa", "#7c5ef5", "#f0a500", "#3fb950", "#fa7970"]
    for col, (lbl, val, unit), color in zip(cols, kpi_row, colors):
        with col:
            st.markdown(f"""
            <div class="geo-card">
              <div class="label">{lbl}</div>
              <div class="value" style="color:{color};font-size:1.6rem;">{val}<span class="unit"> {unit}</span></div>
            </div>""", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=t_obs*1000, mode="markers", name="Data Observasi",
                             marker=dict(size=4, color="#e6edf3", opacity=0.5)))
    fig.add_trace(go.Scatter(x=x, y=t_direct*1000, mode="lines", name=f"Gelombang Langsung (V₁={v1} m/s)",
                             line=dict(color="#f0a500", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=t_refr2*1000, mode="lines", name=f"Refraksi Lap.2 (V₂={v2} m/s)",
                             line=dict(color="#7c5ef5", width=2)))
    fig.add_trace(go.Scatter(x=x, y=t_refr3*1000, mode="lines", name=f"Refraksi Lap.3 (V₃={v3_val} m/s)",
                             line=dict(color="#00d4aa", width=2)))
    fig.add_vline(x=xco1, line_dash="dot", line_color="#f0a500",
                  annotation_text=f"Xco₁={xco1:.0f}m", annotation_font_color="#f0a500")
    apply_theme(fig, "Kurva Waktu-Jarak (T-X)")
    fig.update_layout(height=380, xaxis_title="Jarak Offset (m)", yaxis_title="Waktu Tempuh (ms)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-head">Model Bawah Permukaan</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        x_model = [0, x_max, x_max, 0, 0]
        fig2.add_trace(go.Scatter(x=x_model, y=[0, 0, -h1, -h1, 0], fill="toself",
                                  name="Lapisan 1 (Tanah)", fillcolor="rgba(240,165,0,0.3)",
                                  line=dict(color="#f0a500", width=1.5)))
        fig2.add_trace(go.Scatter(x=x_model, y=[-h1, -h1, -(h1+h2), -(h1+h2), -h1], fill="toself",
                                  name="Lapisan 2 (Sedimen)", fillcolor="rgba(124,94,245,0.3)",
                                  line=dict(color="#7c5ef5", width=1.5)))
        fig2.add_trace(go.Scatter(x=x_model, y=[-(h1+h2), -(h1+h2), -(h1+h2+100), -(h1+h2+100), -(h1+h2)],
                                  fill="toself", name="Lapisan 3 (Batuan Dasar)",
                                  fillcolor="rgba(0,212,170,0.3)", line=dict(color="#00d4aa", width=1.5)))
        apply_theme(fig2, "Penampang Bawah Permukaan")
        fig2.update_layout(height=360, xaxis_title="Jarak (m)", yaxis_title="Kedalaman (m)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">Tabel Interpretasi</div>', unsafe_allow_html=True)
        df_interp = pd.DataFrame({
            "Lapisan": ["Lapisan 1", "Lapisan 2", "Lapisan 3"],
            "Kecepatan (m/s)": [v1, v2, v3_val],
            "Ketebalan (m)": [f"{h1:.1f}", f"{h2:.1f}", "—"],
            "Kedalaman Top (m)": [0, h1, h1 + h2],
            "Interpretasi": ["Tanah permukaan / lapukan", "Sedimen terkonsolidasi", "Batuan dasar / basement"],
        })
        st.dataframe(df_interp, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="info-box">
          <b>Catatan Interpretasi:</b><br>
          • V₁ &lt; 800 m/s → Tanah tidak terkonsolidasi / lapukan<br>
          • 800–2000 m/s → Sedimen terkonsolidasi atau batupasir<br>
          • &gt; 2000 m/s → Batuan keras (granit, basalt, limestone)
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE: GEOLISTRIK
# ═══════════════════════════════════════════════════════════
elif page == "⚡  Geolistrik":
    st.markdown('<div class="page-title">Analisis Geolistrik 2D (Wenner)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Pengolahan data resistivitas 2D konfigurasi Wenner — pseudo-section dan interpretasi litologi.</div>', unsafe_allow_html=True)

    # ── FILE IMPORT ──────────────────────────────────────
    uploaded_geo = upload_box(
        "Import Data Geolistrik 2D",
        "Pengolahan_DATA_GEOLISTRIK_2D.xlsx",
        key="geo_upload",
        types=["xlsx", "csv"],
    )

    geo_data = None
    if uploaded_geo is not None:
        try:
            geo_data = parse_geolistrik(uploaded_geo.read())
            sheets_loaded = [s for s in geo_data]
            st.success(f"✅ Data berhasil dimuat — {len(sheets_loaded)} lintasan: {', '.join(sheets_loaded)}")
            st.session_state.geo_data = geo_data
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

    # ── LINTASAN SELECTOR (only if real data) ───────────
    active_sheet = None
    df_geo = None
    if geo_data:
        available = list(geo_data.keys())
        active_sheet = st.selectbox("Pilih Lintasan", available, key="geo_sheet")
        df_geo = geo_data[active_sheet]

        # KPI cards from real data
        st.markdown('<div class="section-head">Statistik Data Lapangan</div>', unsafe_allow_html=True)
        kpi_cols = st.columns(5)
        kpi_vals = [
            ("Jumlah Data Point", f"{len(df_geo)}", "pts"),
            ("Datum Point Min", f"{df_geo['Datum_Point'].min():.0f}", "m"),
            ("Datum Point Max", f"{df_geo['Datum_Point'].max():.0f}", "m"),
            ("Rho Min", f"{df_geo['Rho'].min():.1f}", "Ω·m"),
            ("Rho Max", f"{df_geo['Rho'].max():.1f}", "Ω·m"),
        ]
        kpi_colors = ["#00d4aa", "#7c5ef5", "#f0a500", "#3fb950", "#fa7970"]
        for col, (lbl, val, unit), color in zip(kpi_cols, kpi_vals, kpi_colors):
            with col:
                col.markdown(f"""
                <div class="geo-card">
                  <div class="label">{lbl}</div>
                  <div class="value" style="color:{color};font-size:1.5rem;">{val}
                    <span class="unit">{unit}</span></div>
                </div>""", unsafe_allow_html=True)

        # ── PSEUDO-SECTION from real data ──────────────
        st.markdown('<div class="section-head">Penampang Resistivitas 2D (Pseudo-Section) — Data Lapangan</div>', unsafe_allow_html=True)

        fig_ps = go.Figure(go.Scatter(
            x=df_geo["Datum_Point"],
            y=df_geo["Spasi"],
            mode="markers",
            marker=dict(
                size=14,
                color=df_geo["Rho"],
                colorscale=[
                    [0.0, "#2563eb"], [0.2, "#0ea5e9"], [0.4, "#22c55e"],
                    [0.6, "#f0a500"], [0.8, "#ef4444"], [1.0, "#7c3aed"],
                ],
                colorbar=dict(
                    title=dict(text="ρₐ (Ω·m)", font=dict(color="#e6edf3")),
                    tickfont=dict(color="#e6edf3"),
                ),
                symbol="square",
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=[f"Datum: {dp}m<br>Spasi: {sp}m<br>ρₐ: {rho:.2f} Ω·m"
                  for dp, sp, rho in zip(df_geo["Datum_Point"], df_geo["Spasi"], df_geo["Rho"])],
            hovertemplate="%{text}<extra></extra>",
        ))
        apply_theme(fig_ps, f"Pseudo-Section Wenner 2D — Lintasan {active_sheet}")
        fig_ps.update_layout(
            height=380,
            xaxis_title="Posisi Datum Point (m)",
            yaxis_title="Spasi Elektroda a (m) — Proksi Kedalaman",
            yaxis=dict(autorange="reversed", gridcolor="#30363d"),
        )
        st.plotly_chart(fig_ps, use_container_width=True)

        # ── RESISTIVITY PROFILE ──────────────────────
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown('<div class="section-head">Profil Resistivitas per Level Spasi</div>', unsafe_allow_html=True)
            fig_prof = go.Figure()
            colors_level = ["#f0a500", "#7c5ef5", "#00d4aa", "#3fb950", "#fa7970"]
            for i, (spasi_val, grp) in enumerate(df_geo.groupby("Spasi", sort=True)):
                fig_prof.add_trace(go.Scatter(
                    x=grp["Datum_Point"], y=grp["Rho"],
                    mode="lines+markers",
                    name=f"a = {spasi_val:.0f} m",
                    line=dict(color=colors_level[i % len(colors_level)], width=2),
                    marker=dict(size=6),
                ))
            apply_theme(fig_prof, "Profil ρₐ vs Posisi Datum Point")
            fig_prof.update_layout(height=300, xaxis_title="Datum Point (m)", yaxis_title="ρₐ (Ω·m)")
            st.plotly_chart(fig_prof, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-head">Klasifikasi Resistivitas</div>', unsafe_allow_html=True)
            class_data = [
                ("< 10", "Air asin / lempung", "#fa7970"),
                ("10 – 50", "Air tanah / sedimen basah", "#f0a500"),
                ("50 – 200", "Batupasir / lanau", "#3fb950"),
                ("200 – 1000", "Batuan lapuk", "#7c5ef5"),
                ("> 1000", "Batuan keras / kering", "#00d4aa"),
            ]
            for r, interp, color in class_data:
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;padding:8px 12px;
                             border-radius:8px;background:rgba(255,255,255,.04);margin-bottom:6px;'>
                  <div style='width:12px;height:12px;border-radius:3px;background:{color};flex-shrink:0'></div>
                  <div>
                    <span style='font-family:Syne;font-weight:700;color:{color};font-size:.9rem;'>{r}</span>
                    <span style='color:#8b949e;font-size:.8rem;margin-left:8px;'>{interp}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── DATA TABLE ────────────────────────────────
        with st.expander("📋 Tabel Data Lengkap", expanded=False):
            st.dataframe(
                df_geo.rename(columns={"Datum_Point": "Datum Point (m)", "Spasi": "Spasi a (m)", "Rho": "ρₐ (Ω·m)"}),
                use_container_width=True, hide_index=True,
            )
            csv_geo = df_geo.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv_geo,
                               file_name=f"geolistrik_{active_sheet or 'default' .replace(' ', '_')}.csv",
                               mime="text/csv")

    else:
        # ── FALLBACK: Synthetic VES + pseudo-section ──
        st.markdown("""
        <div class="info-box">
          ⬆️ Upload file <code>Pengolahan_DATA_GEOLISTRIK_2D_2024.xlsx</code> di atas
          untuk menampilkan data lapangan. Saat ini menampilkan contoh data sintetis.
        </div>""", unsafe_allow_html=True)

        with st.expander("⚙️ Parameter Lapisan (Contoh)", expanded=True):
            n_layers = st.slider("Jumlah Lapisan", 2, 5, 3)
            cols_p = st.columns(n_layers)
            res_defaults = [50, 8, 250, 15, 1200]
            thick_defaults = [3, 10, 25, 60, None]
            layer_params = []
            for i, col in enumerate(cols_p):
                with col:
                    col.markdown(f"**Lapisan {i+1}**")
                    rho = col.number_input(f"ρ{i+1} (Ω·m)", 1.0, 10000.0, float(res_defaults[i]), 1.0, key=f"rho_{i}")
                    h = col.number_input(f"h{i+1} (m)", 0.5, 200.0, float(thick_defaults[i]), 0.5, key=f"h_{i}") if i < n_layers-1 else None
                    layer_params.append((rho, h))

        ab2 = np.logspace(-0.3, 2.5, 100)
        rho_list = [p[0] for p in layer_params]
        h_list = [p[1] for p in layer_params if p[1] is not None]

        def schlumberger_forward(ab2, rho_list, h_list):
            rho_a = []
            for a in ab2:
                rho_eff = rho_list[0]
                depth_acc = 0
                for i, (rho_i, h_i) in enumerate(zip(rho_list, h_list)):
                    depth_acc += h_i
                    influence = np.exp(-depth_acc / a) if a > 0 else 0
                    rho_eff = rho_eff * (1 - influence) + rho_list[min(i+1, len(rho_list)-1)] * influence
                rho_a.append(rho_eff)
            return np.array(rho_a)

        rho_a = schlumberger_forward(ab2, rho_list, h_list)
        rho_obs = rho_a * (1 + 0.03 * np.random.randn(len(rho_a)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ab2, y=rho_obs, mode="markers", name="Data Lapangan",
                                 marker=dict(size=6, color="#e6edf3", symbol="circle-open", line=dict(width=1.5))))
        fig.add_trace(go.Scatter(x=ab2, y=rho_a, mode="lines", name="Kurva Teoritis",
                                 line=dict(color="#00d4aa", width=2.5)))
        apply_theme(fig, "Kurva Resistivitas Semu VES (Contoh Sintetis)")
        fig.update_layout(height=380, xaxis_type="log", yaxis_type="log",
                          xaxis_title="AB/2 (m)", yaxis_title="Resistivitas Semu ρₐ (Ω·m)")
        st.plotly_chart(fig, use_container_width=True)

        # Synthetic pseudo-section
        st.markdown('<div class="section-head">Penampang Resistivitas 2D — Contoh Sintetis</div>', unsafe_allow_html=True)
        nx, nz = 60, 30
        x_ps = np.linspace(0, 500, nx)
        z_ps = np.linspace(0, 100, nz)
        XX, ZZ = np.meshgrid(x_ps, z_ps)
        rho_2d = np.full((nz, nx), rho_list[0])
        cum = 0
        for i, (rho_i, h_i) in enumerate(zip(rho_list, h_list)):
            cum += h_i
            rho_2d[ZZ >= cum] = rho_list[min(i+1, len(rho_list)-1)]
        cx, cz, cr = 250, 55, 25
        anomaly = np.exp(-((XX - cx)**2 + (ZZ - cz)**2) / (2 * cr**2))
        rho_2d += anomaly * rho_list[-1] * 1.5 + np.random.randn(nz, nx) * rho_list[0] * 0.05
        fig3 = go.Figure(go.Heatmap(
            x=x_ps, y=-z_ps, z=np.log10(rho_2d),
            colorscale=[[0, "#2563eb"], [0.2, "#0ea5e9"], [0.4, "#22c55e"],
                        [0.6, "#f0a500"], [0.8, "#ef4444"], [1, "#7c3aed"]],
            colorbar=dict(title=dict(text="log₁₀ρ (Ω·m)", font=dict(color="#e6edf3")),
                          tickfont=dict(color="#e6edf3")),
            zsmooth="best",
        ))
        apply_theme(fig3, "Penampang 2D Resistivitas (Sintetis)")
        fig3.update_layout(height=320, xaxis_title="Jarak (m)", yaxis_title="Kedalaman (m)")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: GRAVITASI & MAGNETIK
# ═══════════════════════════════════════════════════════════
elif page == "🧲  Gravitasi & Magnetik":
    st.markdown('<div class="page-title">Analisis Gravitasi & Magnetik</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Anomali Bouguer (CBA) dan Total Field Magnetik — pemetaan struktur geologi bawah permukaan.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌍 Anomali Gravitasi", "🧲 Anomali Magnetik"])

    # ═══════════ GRAVITY TAB ═══════════
    with tab1:
        # ── FILE IMPORT ──────────────────────────────
        uploaded_grav = upload_box(
            "Import Data Gravitasi",
            "DATA_GRAV_FIKS_CIHUY.xlsx",
            key="grav_upload",
            types=["xlsx", "csv"],
        )

        df_grav = None
        if uploaded_grav is not None:
            try:
                df_grav = parse_gravity(uploaded_grav.read())
                st.success(f"✅ Data CBA berhasil dimuat — {len(df_grav)} titik pengukuran")
                st.session_state.df_grav = df_grav
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

        if df_grav is not None:
            # ── KPI ──────────────────────────────────
            kpi_g = [
                ("Titik Pengukuran", f"{len(df_grav)}", "pts"),
                ("CBA Min", f"{df_grav['CBA'].min():.2f}", "mGal"),
                ("CBA Max", f"{df_grav['CBA'].max():.2f}", "mGal"),
                ("CBA Rata-rata", f"{df_grav['CBA'].mean():.2f}", "mGal"),
                ("Elevasi Rata-rata", f"{df_grav['Z'].mean():.0f}", "m"),
            ]
            cols_g = st.columns(5)
            colors_g = ["#00d4aa", "#fa7970", "#3fb950", "#7c5ef5", "#f0a500"]
            for col, (lbl, val, unit), color in zip(cols_g, kpi_g, colors_g):
                with col:
                    col.markdown(f"""
                    <div class="geo-card">
                      <div class="label">{lbl}</div>
                      <div class="value" style="color:{color};font-size:1.4rem;">{val}
                        <span class="unit">{unit}</span></div>
                    </div>""", unsafe_allow_html=True)

            # ── CBA PROFILE ───────────────────────────
            st.markdown('<div class="section-head">Profil Complete Bouguer Anomaly (CBA) — Data Lapangan</div>', unsafe_allow_html=True)
            df_sorted = df_grav.sort_values("X").reset_index(drop=True)
            df_sorted["Dist"] = np.sqrt(
                (df_sorted["X"] - df_sorted["X"].iloc[0])**2 +
                (df_sorted["Y"] - df_sorted["Y"].iloc[0])**2
            )

            fig_cba = go.Figure()
            fig_cba.add_trace(go.Scatter(
                x=df_sorted["Dist"], y=df_sorted["CBA"],
                mode="markers+lines",
                name="CBA Lapangan",
                line=dict(color="#00d4aa", width=2),
                marker=dict(size=7, color="#00d4aa",
                            line=dict(width=1, color="#0d1117")),
                text=[f"X: {x:.0f} m<br>Y: {y:.0f} m<br>Z: {z:.0f} m<br>CBA: {c:.3f} mGal"
                      for x, y, z, c in zip(df_sorted["X"], df_sorted["Y"], df_sorted["Z"], df_sorted["CBA"])],
                hovertemplate="%{text}<extra></extra>",
            ))
            # Zero line
            fig_cba.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1)
            apply_theme(fig_cba, "Profil CBA vs Jarak Kumulatif")
            fig_cba.update_layout(height=340, xaxis_title="Jarak Kumulatif (m)", yaxis_title="CBA (mGal)")
            st.plotly_chart(fig_cba, use_container_width=True)

            # ── 2D SCATTER MAP ────────────────────────
            col_map, col_stat = st.columns([3, 1])
            with col_map:
                st.markdown('<div class="section-head">Peta Distribusi Anomali CBA 2D</div>', unsafe_allow_html=True)
                fig_map = go.Figure(go.Scatter(
                    x=df_grav["X"], y=df_grav["Y"],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=df_grav["CBA"],
                        colorscale="RdBu_r",
                        colorbar=dict(
                            title=dict(text="CBA (mGal)", font=dict(color="#e6edf3")),
                            tickfont=dict(color="#e6edf3"),
                        ),
                        line=dict(width=1, color="rgba(0,0,0,0.4)"),
                    ),
                    text=[f"X: {x:.0f}<br>Y: {y:.0f}<br>Z: {z:.0f} m<br>CBA: {c:.3f} mGal"
                          for x, y, z, c in zip(df_grav["X"], df_grav["Y"], df_grav["Z"], df_grav["CBA"])],
                    hovertemplate="%{text}<extra></extra>",
                ))
                apply_theme(fig_map, "Peta Titik Pengukuran Gravitasi (CBA)")
                fig_map.update_layout(
                    height=400,
                    xaxis_title="UTM Easting (m)",
                    yaxis_title="UTM Northing (m)",
                    xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor="#30363d"),
                )
                st.plotly_chart(fig_map, use_container_width=True)

            with col_stat:
                st.markdown('<div class="section-head">CBA vs Elevasi</div>', unsafe_allow_html=True)
                fig_xz = go.Figure(go.Scatter(
                    x=df_grav["Z"], y=df_grav["CBA"],
                    mode="markers",
                    marker=dict(size=8, color=df_grav["CBA"], colorscale="RdBu_r",
                                line=dict(width=0.5, color="#0d1117")),
                    text=[f"Z={z:.0f}m, CBA={c:.2f} mGal"
                          for z, c in zip(df_grav["Z"], df_grav["CBA"])],
                    hovertemplate="%{text}<extra></extra>",
                ))
                apply_theme(fig_xz, "CBA vs Elevasi")
                fig_xz.update_layout(height=400, xaxis_title="Elevasi (m)", yaxis_title="CBA (mGal)")
                st.plotly_chart(fig_xz, use_container_width=True)

            # Data table
            with st.expander("📋 Tabel Data CBA", expanded=False):
                st.dataframe(
                    df_grav.rename(columns={"X": "Easting (m)", "Y": "Northing (m)",
                                            "Z": "Elevasi (m)", "CBA": "CBA (mGal)"}),
                    use_container_width=True, hide_index=True,
                )
                st.download_button("⬇️ Download CSV",
                                   df_grav.to_csv(index=False).encode("utf-8"),
                                   file_name="gravitasi_CBA.csv", mime="text/csv")

        else:
            # ── FALLBACK: Synthetic Gravity ───────────
            st.markdown("""
            <div class="info-box">
              ⬆️ Upload file <code>DATA_GRAV_FIKS_CIHUY.xlsx</code> di atas untuk
              menampilkan data CBA lapangan. Saat ini menampilkan contoh sintetis.
            </div>""", unsafe_allow_html=True)

            col_p, col_c = st.columns([1, 2])
            with col_p:
                depth_g = st.slider("Kedalaman Benda (m)", 10, 500, 80)
                density_contrast = st.slider("Kontras Densitas (g/cm³)", 0.1, 1.5, 0.5)
                radius_g = st.slider("Radius Benda (m)", 5, 200, 40)
                x_center = st.slider("Posisi Horizontal (m)", 0, 500, 250)
            with col_c:
                x_g = np.linspace(0, 500, 300)
                G = 6.674e-11
                mass = density_contrast * 1000 * (4/3) * np.pi * radius_g**3
                r = np.sqrt((x_g - x_center)**2 + depth_g**2)
                gz = G * mass * depth_g / r**3 * 1e8
                regional = 0.005 * x_g - 1.2
                gz_total = gz + regional + np.random.normal(0, 0.02, len(x_g))
                gz_bouguer = gz_total - regional
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_g, y=gz_total, mode="lines", name="Anomali Total",
                                         line=dict(color="#e6edf3", width=1.5, dash="dash")))
                fig.add_trace(go.Scatter(x=x_g, y=gz_bouguer, mode="lines", name="Anomali Bouguer",
                                         line=dict(color="#00d4aa", width=2.5)))
                fig.add_trace(go.Scatter(x=x_g, y=regional, mode="lines", name="Tren Regional",
                                         line=dict(color="#f0a500", width=1.5, dash="dot")))
                apply_theme(fig, "Profil Anomali Gravitasi (Sintetis)")
                fig.update_layout(height=320, xaxis_title="Jarak (m)", yaxis_title="Anomali (mGal)")
                st.plotly_chart(fig, use_container_width=True)

    # ═══════════ MAGNETIC TAB ═══════════
    with tab2:
        # ── FILE IMPORT ──────────────────────────────
        uploaded_mag = upload_box(
            "Import Data Magnetik",
            "MAGNET_FIKS_BANGET.xlsx",
            key="mag_upload",
            types=["xlsx", "csv"],
        )

        mag_profile = None
        mag_slicing = None
        if uploaded_mag is not None:
            try:
                mag_profile, mag_slicing = parse_magnetik(uploaded_mag.read())
                st.success(
                    f"✅ Data magnetik berhasil dimuat — "
                    f"{len(mag_profile)} titik profil, {len(mag_slicing)} titik gridding"
                )
                st.session_state.mag_profile, st.session_state.mag_slicing = mag_profile, mag_slicing
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")

        if mag_profile is not None and not mag_profile.empty:
            # ── KPI ──────────────────────────────────
            kpi_m = [
                ("Stasiun Terukur", f"{mag_profile['Stasiun'].count()}", "titik"),
                ("Anomali Min", f"{mag_profile['Anomali'].min():.2f}", "nT"),
                ("Anomali Max", f"{mag_profile['Anomali'].max():.2f}", "nT"),
                ("Anomali Rata-rata", f"{mag_profile['Anomali'].mean():.2f}", "nT"),
                ("Std Deviasi", f"{mag_profile['Anomali'].std():.2f}", "nT"),
            ]
            cols_m = st.columns(5)
            colors_m = ["#7c5ef5", "#fa7970", "#3fb950", "#00d4aa", "#f0a500"]
            for col, (lbl, val, unit), color in zip(cols_m, kpi_m, colors_m):
                with col:
                    col.markdown(f"""
                    <div class="geo-card">
                      <div class="label">{lbl}</div>
                      <div class="value" style="color:{color};font-size:1.4rem;">{val}
                        <span class="unit">{unit}</span></div>
                    </div>""", unsafe_allow_html=True)

            # ── ANOMALY PROFILE ───────────────────────
            st.markdown('<div class="section-head">Profil Anomali Magnetik Total Field — Data Lapangan</div>', unsafe_allow_html=True)

            # Use UTM X as profile position (sorted)
            df_prof = mag_profile.dropna(subset=["UTM_X", "Anomali"]).sort_values("UTM_X").reset_index(drop=True)
            # Create sequential distance
            if len(df_prof) > 1:
                dx = np.diff(df_prof["UTM_X"].values)
                dy = np.diff(df_prof["UTM_Y"].fillna(0).values)
                dist = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])
            else:
                dist = np.array([0])

            fig_mag = go.Figure()
            fig_mag.add_trace(go.Scatter(
                x=dist, y=df_prof["Anomali"].values,
                mode="lines+markers",
                name="Anomali ΔT",
                line=dict(color="#7c5ef5", width=2.5),
                marker=dict(size=7, color="#7c5ef5", line=dict(width=1, color="#0d1117")),
                fill="tozeroy",
                fillcolor="rgba(124,94,245,0.12)",
                text=[f"Sta: {s}<br>Jarak: {d:.0f}m<br>ΔT: {a:.2f} nT"
                      for s, d, a in zip(df_prof["Stasiun"], dist, df_prof["Anomali"])],
                hovertemplate="%{text}<extra></extra>",
            ))
            fig_mag.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1)
            apply_theme(fig_mag, "Profil Anomali Magnetik Total Field (ΔT) — Koreksi IGRF + Harian")
            fig_mag.update_layout(height=340, xaxis_title="Jarak (m)", yaxis_title="ΔT (nT)")
            st.plotly_chart(fig_mag, use_container_width=True)

            # ── 2D MAP from Slicing data ──────────────
            if mag_slicing is not None and not mag_slicing.empty:
                col_map2, col_side = st.columns([3, 1])
                with col_map2:
                    st.markdown('<div class="section-head">Peta Anomali Magnetik 2D (GRTP — Total Field)</div>', unsafe_allow_html=True)
                    fig_mmap = go.Figure(go.Scatter(
                        x=mag_slicing["X"], y=mag_slicing["Y"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=mag_slicing["GRTP"],
                            colorscale=[[0, "#1e3a8a"], [0.25, "#3b82f6"], [0.5, "#e6edf3"],
                                        [0.75, "#dc2626"], [1, "#7f1d1d"]],
                            colorbar=dict(
                                title=dict(text="GRTP (nT)", font=dict(color="#e6edf3")),
                                tickfont=dict(color="#e6edf3"),
                            ),
                            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                        ),
                        text=[f"X: {x:.0f}<br>Y: {y:.0f}<br>GRTP: {g:.2f} nT<br>RFHD: {r:.4f}"
                              for x, y, g, r in zip(mag_slicing["X"], mag_slicing["Y"],
                                                      mag_slicing["GRTP"], mag_slicing["RFHD"])],
                        hovertemplate="%{text}<extra></extra>",
                    ))
                    apply_theme(fig_mmap, "Peta Distribusi Total Field Magnetik (GRTP)")
                    fig_mmap.update_layout(
                        height=400, xaxis_title="UTM Easting (m)", yaxis_title="UTM Northing (m)",
                        xaxis=dict(scaleanchor="y", scaleratio=1, gridcolor="#30363d"),
                    )
                    st.plotly_chart(fig_mmap, use_container_width=True)

                with col_side:
                    st.markdown('<div class="section-head">RFHD (Horizontal Derivative)</div>', unsafe_allow_html=True)
                    fig_rfhd = go.Figure(go.Scatter(
                        x=mag_slicing["Jarak"], y=mag_slicing["RFHD"],
                        mode="lines",
                        line=dict(color="#f0a500", width=2),
                        fill="tozeroy", fillcolor="rgba(240,165,0,0.12)",
                    ))
                    apply_theme(fig_rfhd, "Profil RFHD")
                    fig_rfhd.update_layout(height=400, xaxis_title="Jarak (m)", yaxis_title="RFHD")
                    st.plotly_chart(fig_rfhd, use_container_width=True)

            # Data table
            with st.expander("📋 Tabel Data Anomali Magnetik", expanded=False):
                show_cols = ["Stasiun", "Longitude", "Latitude", "Elevasi", "Anomali", "UTM_X", "UTM_Y"]
                display_df = mag_profile[[c for c in show_cols if c in mag_profile.columns]]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download CSV",
                                   display_df.to_csv(index=False).encode("utf-8"),
                                   file_name="magnetik_anomali.csv", mime="text/csv")

        else:
            # ── FALLBACK: Synthetic Magnetic ──────────
            st.markdown("""
            <div class="info-box" style="border-color:rgba(124,94,245,.3);background:rgba(124,94,245,.07);">
              ⬆️ Upload file <code>MAGNET_FIKS_BANGET.xlsx</code> di atas untuk
              menampilkan data anomali magnetik lapangan. Saat ini menampilkan contoh sintetis.
            </div>""", unsafe_allow_html=True)

            col_p2, col_c2 = st.columns([1, 2])
            with col_p2:
                depth_m = st.slider("Kedalaman Benda (m)", 10, 500, 60, key="dm")
                sus = st.slider("Suseptibilitas (× 10⁻³ SI)", 1, 100, 30)
                inc = st.slider("Inklinasi (°)", -90, 90, -30)
                x_cm = st.slider("Posisi (m)", 0, 500, 250, key="xcm")
            with col_c2:
                x_m = np.linspace(0, 500, 300)
                k = sus * 1e-3
                B0 = 45000
                inc_r = np.radians(inc)
                r_m = np.sqrt((x_m - x_cm)**2 + depth_m**2)
                theta = np.arctan2(depth_m, x_m - x_cm)
                T = k * B0 * (
                    2 * (np.sin(inc_r))**2 * np.cos(2 * (theta - np.pi/2))
                    - np.cos(inc_r)**2 * np.sin(2 * theta)
                ) / (r_m / depth_m)**3
                T_noise = T + np.random.normal(0, 5, len(x_m))
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=x_m, y=T_noise, mode="lines", name="Total Field",
                                          line=dict(color="#7c5ef5", width=2)))
                fig5.add_trace(go.Scatter(x=x_m, y=T, mode="lines", name="Model",
                                          line=dict(color="#f0a500", width=1.5, dash="dash")))
                apply_theme(fig5, "Profil Anomali Magnetik (Sintetis)")
                fig5.update_layout(height=320, xaxis_title="Jarak (m)", yaxis_title="ΔT (nT)")
                st.plotly_chart(fig5, use_container_width=True)

            # Synthetic 2D map
            st.markdown('<div class="section-head">Peta Anomali Magnetik 2D (Sintetis)</div>', unsafe_allow_html=True)
            nm = 80
            xmg = np.linspace(0, 500, nm)
            ymg = np.linspace(0, 500, nm)
            XM, YM = np.meshgrid(xmg, ymg)
            T_map = np.zeros((nm, nm))
            for cx3, cy3, dep3, sus3 in [(150, 150, 50, 40), (350, 250, 80, -25), (200, 380, 40, 60)]:
                r3 = np.sqrt((XM-cx3)**2 + (YM-cy3)**2 + dep3**2)
                T_map += sus3 * B0 * np.sin(inc_r) * dep3 / r3**3 * dep3**2
            T_map += np.random.normal(0, 10, T_map.shape)
            fig6 = go.Figure(go.Heatmap(
                x=xmg, y=ymg, z=T_map,
                colorscale=[[0, "#1e3a8a"], [0.25, "#3b82f6"], [0.5, "#e6edf3"],
                            [0.75, "#dc2626"], [1, "#7f1d1d"]],
                colorbar=dict(title=dict(text="nT", font=dict(color="#e6edf3")),
                              tickfont=dict(color="#e6edf3")),
                zsmooth="best",
            ))
            apply_theme(fig6, "Peta Anomali Total Magnetik (Sintetis)")
            fig6.update_layout(height=380, xaxis_title="Easting (m)", yaxis_title="Northing (m)")
            st.plotly_chart(fig6, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: PENAMPANG TERPADU
# ═══════════════════════════════════════════════════════════
elif page == "🪨  Penampang Terpadu":
    st.markdown('<div class="page-title">Penampang Bawah Permukaan Terpadu</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Korelasi multi-metode: Geolistrik 2D · Gravitasi (CBA) · Magnetik (ΔT) '
        '— interpretasi lapisan geologi dalam satu penampang terintegrasi.</div>',
        unsafe_allow_html=True)

    page_penampang(
        geo_data    = st.session_state.geo_data,
        df_grav     = st.session_state.df_grav,
        mag_profile = st.session_state.mag_profile,
        mag_slicing = st.session_state.mag_slicing,
    )

    geo_data    = st.session_state.geo_data
    df_grav     = st.session_state.df_grav
    mag_profile = st.session_state.mag_profile

    # Status data
    has_real_geo  = geo_data is not None
    has_real_grav = df_grav is not None
    has_real_mag  = mag_profile is not None and not mag_profile.empty

    badges_html = ""
    for label, real, color in [("Geolistrik",has_real_geo,"#00d4aa"),("Gravitasi",has_real_grav,"#f0a500"),("Magnetik",has_real_mag,"#7c5ef5")]:
        status = "✅ Data lapangan" if real else "⚠ Data sintetis — upload di halaman masing-masing"
        badges_html += (f"<span style='background:rgba(255,255,255,.05);border:1px solid {color}44;border-radius:8px;"
                        f"padding:6px 14px;font-size:.82rem;color:#e6edf3;'><b style='color:{color};'>{label}</b> · {status}</span>")
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;'>{badges_html}</div>", unsafe_allow_html=True)

    # ── Kontrol
    with st.expander("⚙️ Pengaturan Penampang", expanded=True):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            v_exag = st.slider("Vertical Exaggeration", 1.0, 5.0, 2.0, 0.5, key="penampang_v_exag")
            max_depth = st.number_input("Kedalaman Maksimum (m)", 50, 500, 120, 10)
        with col_s2:
            show_geo_layer   = st.checkbox("Profil Resistivitas ρₐ", True)
            show_grav_layer  = st.checkbox("Profil CBA Gravitasi", True)
            show_mag_layer   = st.checkbox("Profil ΔT Magnetik", True)
        with col_s3:
            show_boreholes   = st.checkbox("Posisi Bor (Sintetis)", True)
            show_litho_labels= st.checkbox("Label Litologi", True)
            section_lintasan = st.selectbox("Lintasan Referensi",
                list(geo_data.keys()) if geo_data else ["Lintasan Sintetis"])

    # ── Siapkan data X-profile
    if geo_data and section_lintasan in geo_data:
        df_geo_sec  = geo_data[section_lintasan]
        x_positions = df_geo_sec["Datum_Point"].values.astype(float)
        rho_values  = df_geo_sec["Rho"].values.astype(float)
    else:
        x_positions = np.linspace(0, 144, 60)
        rng = np.random.default_rng(42)
        rho_values = np.clip(20 + 15*np.sin(x_positions/30) + rng.normal(0,4,60)
                             - 8*((x_positions>60)&(x_positions<100)).astype(float), 5, 200)

    x_min, x_max_val = float(x_positions.min()), float(x_positions.max())
    profile_length   = x_max_val - x_min

    # Gravitasi profil
    if has_real_grav:
        grav_x_raw = np.sqrt((df_grav["X"]-df_grav["X"].iloc[0])**2+(df_grav["Y"]-df_grav["Y"].iloc[0])**2).values
        cba_interp = np.interp(np.linspace(0, grav_x_raw.max(), len(x_positions)),
                               np.sort(grav_x_raw), df_grav["CBA"].values[np.argsort(grav_x_raw)])
    else:
        rng2 = np.random.default_rng(7)
        cba_interp = 7.8 + 4*np.sin(x_positions/40) - 2*np.cos(x_positions/15) + rng2.normal(0,0.4,len(x_positions))

    # Magnetik profil
    if has_real_mag:
        df_mp = mag_profile.dropna(subset=["UTM_X","Anomali"]).sort_values("UTM_X").reset_index(drop=True)
        if len(df_mp) > 1:
            dx_m = np.diff(df_mp["UTM_X"].values); dy_m = np.diff(df_mp["UTM_Y"].fillna(0).values)
            dist_m = np.concatenate([[0], np.cumsum(np.sqrt(dx_m**2+dy_m**2))])
        else:
            dist_m = np.array([0.0])
        mag_interp = np.interp(np.linspace(0, dist_m.max() if dist_m.max()>0 else 1, len(x_positions)),
                               dist_m, df_mp["Anomali"].values)
    else:
        rng3 = np.random.default_rng(13)
        mag_interp = 20*np.sin(x_positions/25) - 15*np.cos(x_positions/18+1) + rng3.normal(0,4,len(x_positions))

    # ── Model lapisan dari resistivitas
    rho_smooth = uniform_filter1d(rho_values, size=max(3, len(rho_values)//10))
    rho_norm   = (rho_smooth - rho_smooth.min()) / (np.ptp(rho_smooth) + 1e-6)

    d1 = 10 + 6*np.sin(x_positions/25)   + 3*rho_norm*8
    d2 = d1 + 18 + 10*np.cos(x_positions/35) + 5*rho_norm*4
    d3 = d2 + 20 + 8*np.sin(x_positions/45+0.5) + 4*rho_norm*6
    d4 = np.clip(d3 + 15 + 5*np.cos(x_positions/30), 0, max_depth*0.85)

    LAYER_DEFS = [
        {"name":"Lapisan 1 — Lapukan / Tanah",       "color":"#d4a96a", "bot":d1},
        {"name":"Lapisan 2 — Sedimen Basah / Akuifer","color":"#7c9e72", "bot":d2},
        {"name":"Lapisan 3 — Batupasir / Lanau",      "color":"#8b7355", "bot":d3},
        {"name":"Lapisan 4 — Batuan Lapuk",           "color":"#5a6e80", "bot":d4},
        {"name":"Lapisan 5 — Batuan Dasar",           "color":"#3d4f5c", "bot":np.full_like(x_positions, float(max_depth))},
    ]
    LITHO_RHO = {
        "Lapisan 1 — Lapukan / Tanah":        "5–40 Ω·m",
        "Lapisan 2 — Sedimen Basah / Akuifer":"5–25 Ω·m",
        "Lapisan 3 — Batupasir / Lanau":       "40–200 Ω·m",
        "Lapisan 4 — Batuan Lapuk":            "200–500 Ω·m",
        "Lapisan 5 — Batuan Dasar":            ">500 Ω·m",
    }
    LITHO_INTERP = [
        "Zona lapukan, material tidak terkonsolidasi, potensi kontaminasi tinggi",
        "Kemungkinan zona akuifer / air tanah, target sumur bor prioritas",
        "Sedimen terkonsolidasi, batupasir, kondisi teknik relatif stabil",
        "Batuan yang telah mengalami pelapukan, zona transisi",
        "Batuan dasar (basement), kecepatan seismik tinggi (>2000 m/s)",
    ]

    # ── Subplot utama
    n_rows      = 1 + int(show_geo_layer) + int(show_grav_layer) + int(show_mag_layer)
    row_heights = [4] + [1]*int(show_geo_layer) + [1]*int(show_grav_layer) + [1]*int(show_mag_layer)
    sp_titles   = ["Penampang Geologi Bawah Permukaan"]
    if show_geo_layer:  sp_titles.append("Resistivitas ρₐ (Ω·m)")
    if show_grav_layer: sp_titles.append("CBA Gravitasi (mGal)")
    if show_mag_layer:  sp_titles.append("ΔT Magnetik (nT)")

    fig = make_subplots(rows=n_rows, cols=1, row_heights=row_heights,
                        shared_xaxes=True, vertical_spacing=0.015, subplot_titles=sp_titles)

    # Lapisan berwarna
    prev_tops = np.zeros(len(x_positions))
    for layer in LAYER_DEFS:
        bot  = layer["bot"] * v_exag
        top  = prev_tops
        x_fill = np.concatenate([x_positions, x_positions[::-1]])
        y_fill = np.concatenate([-bot, -top[::-1]])
        fig.add_trace(go.Scatter(x=x_fill, y=y_fill, fill="toself", fillcolor= hex_to_rgba(layer["color"], 0.8),
            line=dict(width=0), name=layer["name"], hoverinfo="skip", showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_positions, y=-bot, mode="lines",
            line=dict(color="rgba(255,255,255,0.2)",width=1), showlegend=False,
            hovertemplate=f"<b>{layer['name']}</b><br>Posisi:%{{x:.0f}}m<br>Batas:%{{customdata:.1f}}m<br>{LITHO_RHO[layer['name']]}<extra></extra>",
            customdata=layer["bot"]), row=1, col=1)
        prev_tops = bot

    # Label litologi
    if show_litho_labels:
        label_fracs = [0.12, 0.38, 0.62, 0.5, 0.82]
        for li, (frac, layer) in enumerate(zip(label_fracs, LAYER_DEFS)):
            xi   = int(frac*(len(x_positions)-1))
            topD = LAYER_DEFS[li-1]["bot"][xi]*v_exag if li > 0 else 0
            botD = layer["bot"][xi]*v_exag
            midD = (topD + botD)/2
            short = layer["name"].split("—")[1].strip() if "—" in layer["name"] else layer["name"]
            fig.add_annotation(x=x_positions[xi], y=-midD, text=short, showarrow=False,
                font=dict(size=10, color="rgba(255,255,255,0.75)", family="DM Sans"), row=1, col=1)

    # Bor sintetis
    if show_boreholes:
        for i, frac in enumerate([0.2, 0.5, 0.78]):
            bx = x_min + frac*profile_length
            fig.add_vline(x=bx, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1.5,
                annotation_text=f"BH-{i+1:02d}", annotation_position="top",
                annotation_font=dict(size=10, color="#e6edf3"), row=1, col=1)

    row_idx = 2
    # Profil resistivitas
    if show_geo_layer:
        fig.add_trace(go.Scatter(x=x_positions, y=rho_values, mode="lines+markers",
            name="ρₐ" + ("" if has_real_geo else " ⚠sintetis"),
            line=dict(color="#4fa3e0",width=1.8), marker=dict(size=4,color="#4fa3e0"),
            fill="tozeroy", fillcolor="rgba(79,163,224,0.12)",
            hovertemplate="Pos:%{x:.0f}m ρₐ:%{y:.1f}Ω·m<extra></extra>"), row=row_idx, col=1)
        row_idx += 1

    # Profil CBA
    if show_grav_layer:
        fig.add_trace(go.Scatter(x=x_positions, y=cba_interp, mode="lines",
            name="CBA" + ("" if has_real_grav else " ⚠sintetis"),
            line=dict(color="#e8923a",width=1.8), fill="tozeroy", fillcolor="rgba(232,146,58,0.10)",
            hovertemplate="Pos:%{x:.0f}m CBA:%{y:.3f}mGal<extra></extra>"), row=row_idx, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1, row=row_idx, col=1)
        row_idx += 1

    # Profil ΔT
    if show_mag_layer:
        fig.add_trace(go.Scatter(x=x_positions, y=mag_interp, mode="lines",
            name="ΔT" + ("" if has_real_mag else " ⚠sintetis"),
            line=dict(color="#b87dd4",width=1.8), fill="tozeroy", fillcolor="rgba(184,125,212,0.10)",
            hovertemplate="Pos:%{x:.0f}m ΔT:%{y:.1f}nT<extra></extra>"), row=row_idx, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#8b949e", line_width=1, row=row_idx, col=1)

    fig.update_layout(**PLOT_LAYOUT,
        height=160*n_rows + 100,
        title=dict(text=f"Penampang Terpadu — {section_lintasan} | V.Exag {v_exag:.1f}× | Panjang {profile_length:.0f} m",
                   font=dict(size=14, family="Syne", color="#e6edf3")),
        legend=dict(bgcolor="rgba(22,27,34,0.85)", bordercolor="#30363d", borderwidth=1,
                    font=dict(size=11), orientation="v", x=1.01, y=1),
        hovermode="x unified")
    fig.update_xaxes(title_text="Jarak / Posisi (m)", gridcolor="#30363d", linecolor="#30363d", row=n_rows, col=1)
    fig.update_yaxes(title_text=f"Kedalaman (m × {v_exag})", gridcolor="#30363d", linecolor="#30363d", row=1, col=1)
    for ann in fig.layout.annotations:
        if ann.text in sp_titles:
            ann.font = dict(size=12, family="Syne", color="#e6edf3")
    st.plotly_chart(fig, use_container_width=True)

    # ── Tabel interpretasi lapisan
    st.markdown('<div class="section-head">Tabel Interpretasi Lapisan</div>', unsafe_allow_html=True)
    interp_rows = []
    prev_d = 0.0
    for i, layer in enumerate(LAYER_DEFS):
        mean_d = float(layer["bot"].mean())
        interp_rows.append({
            "Lapisan": f"L-{i+1}",
            "Deskripsi": layer["name"].split("—")[1].strip() if "—" in layer["name"] else layer["name"],
            "Top rata-rata (m)": f"{prev_d:.1f}",
            "Dasar rata-rata (m)": f"{mean_d:.1f}",
            "Rentang Resistivitas": LITHO_RHO[layer["name"]],
            "Interpretasi": LITHO_INTERP[i],
        })
        prev_d = mean_d
    st.dataframe(pd.DataFrame(interp_rows), use_container_width=True, hide_index=True)

    # ── Korelasi antar metode
    st.markdown('<div class="section-head">Korelasi Anomali vs Resistivitas</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_c1 = go.Figure(go.Scatter(x=rho_values, y=cba_interp, mode="markers",
            marker=dict(size=7, color=x_positions,
                colorscale=[[0,"#4fa3e0"],[0.5,"#00d4aa"],[1,"#e8923a"]],
                colorbar=dict(title=dict(text="Posisi (m)",font=dict(color="#e6edf3")),tickfont=dict(color="#e6edf3"),len=0.8),
                opacity=0.75),
            hovertemplate="ρₐ:%{x:.1f}Ω·m CBA:%{y:.2f}mGal<extra></extra>"))
        apply_theme(fig_c1, "Resistivitas vs CBA Gravitasi")
        fig_c1.update_layout(height=300, xaxis_title="ρₐ (Ω·m)", yaxis_title="CBA (mGal)")
        st.plotly_chart(fig_c1, use_container_width=True)
    with col_c2:
        fig_c2 = go.Figure(go.Scatter(x=rho_values, y=mag_interp, mode="markers",
            marker=dict(size=7, color=x_positions,
                colorscale=[[0,"#4fa3e0"],[0.5,"#7c5ef5"],[1,"#b87dd4"]],
                colorbar=dict(title=dict(text="Posisi (m)",font=dict(color="#e6edf3")),tickfont=dict(color="#e6edf3"),len=0.8),
                opacity=0.75),
            hovertemplate="ρₐ:%{x:.1f}Ω·m ΔT:%{y:.1f}nT<extra></extra>"))
        apply_theme(fig_c2, "Resistivitas vs ΔT Magnetik")
        fig_c2.update_layout(height=300, xaxis_title="ρₐ (Ω·m)", yaxis_title="ΔT (nT)")
        st.plotly_chart(fig_c2, use_container_width=True)

    # ── Ekspor
    st.markdown('<div class="section-head">Ekspor Data Penampang</div>', unsafe_allow_html=True)
    export_df = pd.DataFrame({
        "Posisi (m)": np.round(x_positions,1),
        "ρₐ (Ω·m)": np.round(rho_values,2),
        "CBA (mGal)": np.round(cba_interp,3),
        "ΔT (nT)": np.round(mag_interp,1),
        "Batas L1 (m)": np.round(LAYER_DEFS[0]["bot"],1),
        "Batas L2 (m)": np.round(LAYER_DEFS[1]["bot"],1),
        "Batas L3 (m)": np.round(LAYER_DEFS[2]["bot"],1),
        "Batas L4 (m)": np.round(LAYER_DEFS[3]["bot"],1),
    })
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button("⬇️ Download CSV Penampang", export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"penampang_{section_lintasan.replace(' ','_')}.csv", mime="text/csv", use_container_width=True)
    with col_e2:
        with st.expander("📋 Pratinjau Data"):
            st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# PAGE: PETA SPASIAL
# ═══════════════════════════════════════════════════════════
elif page == "🗺️  Peta Spasial":
    st.markdown('<div class="page-title">Peta Spasial Geofisika</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Visualisasi distribusi spasial titik pengukuran dan anomali geofisika.</div>', unsafe_allow_html=True)

    try:
        import folium
        from streamlit_folium import st_folium

        np.random.seed(42)
        n_pts = 60
        base_lat, base_lon = -6.9, 107.6
        lats = base_lat + np.random.uniform(-0.5, 0.5, n_pts)
        lons = base_lon + np.random.uniform(-0.5, 0.5, n_pts)
        methods = np.random.choice(["Seismik", "Geolistrik", "Gravitasi", "Magnetik"], n_pts)
        values = {"Seismik": np.random.uniform(300, 3500, n_pts), "Geolistrik": np.random.uniform(5, 1000, n_pts),
                  "Gravitasi": np.random.uniform(-10, 10, n_pts), "Magnetik": np.random.uniform(-200, 200, n_pts)}
        point_values = np.array([values[m][i] for i, m in enumerate(methods)])

        col_ctrl, col_map = st.columns([1, 3])
        with col_ctrl:
            selected_method = st.selectbox("Filter Metode", ["Semua", "Seismik", "Geolistrik", "Gravitasi", "Magnetik"])
            map_style = st.selectbox("Gaya Peta", ["CartoDB dark_matter", "OpenStreetMap", "CartoDB positron"])
            show_heatmap = st.checkbox("Tampilkan Heatmap", True)

        method_colors = {"Seismik": "green", "Geolistrik": "purple", "Gravitasi": "orange", "Magnetik": "blue"}
        m = folium.Map(location=[base_lat, base_lon], zoom_start=10, tiles=map_style)
        if show_heatmap:
            from folium.plugins import HeatMap
            HeatMap([[lat, lon, abs(val)] for lat, lon, val in zip(lats, lons, point_values)],
                    radius=20, blur=15, gradient={0.4: "blue", 0.65: "lime", 1: "red"}).add_to(m)
        for i, (lat, lon, method, val) in enumerate(zip(lats, lons, methods, point_values)):
            if selected_method != "Semua" and method != selected_method:
                continue
            folium.CircleMarker(
                location=[lat, lon], radius=7,
                color=method_colors[method], fill=True, fill_opacity=0.8,
                popup=folium.Popup(f"<b>#{i+1}</b><br>{method}<br>{val:.1f}", max_width=200),
                tooltip=f"{method} | {val:.1f}",
            ).add_to(m)
        with col_map:
            st_folium(m, width=None, height=500, returned_objects=[])

    except ImportError:
        np.random.seed(42)
        n_pts = 60
        lats = -6.9 + np.random.uniform(-0.5, 0.5, n_pts)
        lons = 107.6 + np.random.uniform(-0.5, 0.5, n_pts)
        methods = np.random.choice(["Seismik", "Geolistrik", "Gravitasi", "Magnetik"], n_pts)
        values = np.random.uniform(10, 1000, n_pts)
        color_map = {"Seismik": "#00d4aa", "Geolistrik": "#7c5ef5", "Gravitasi": "#f0a500", "Magnetik": "#3fb950"}
        fig = go.Figure()
        for method, color in color_map.items():
            mask = methods == method
            fig.add_trace(go.Scattermap(
                lat=lats[mask], lon=lons[mask], mode="markers",
                marker=dict(size=12, color=color, opacity=0.8),
                name=method,
                text=[f"{method}<br>{v:.1f}" for v in values[mask]],
            ))
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter", center=dict(lat=-6.9, lon=107.6), zoom=9),
            **PLOT_LAYOUT, height=500, title="Peta Distribusi Titik Pengukuran Geofisika",
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: LAPORAN
# ═══════════════════════════════════════════════════════════
elif page == "📋  Laporan":
    st.markdown('<div class="page-title">Laporan & Ekspor Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Ringkasan hasil interpretasi dan ekspor data untuk pelaporan.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        project_name = st.text_input("Nama Proyek", "Survei Geofisika Terpadu 2024")
        location = st.text_input("Lokasi", "Soreang, Jawa Barat")
        operator = st.text_input("Operator / Tim", "Tim Geofisika UNPAD")
    with col2:
        survey_date = st.date_input("Tanggal Survei")
        methods_used = st.multiselect("Metode yang Digunakan",
            ["Geolistrik Wenner 2D", "Gravitasi (CBA)", "Magnetik Total Field", "Seismik Refraksi"],
            default=["Geolistrik Wenner 2D", "Gravitasi (CBA)", "Magnetik Total Field"])
        purpose = st.selectbox("Tujuan Survei",
            ["Eksplorasi Air Tanah", "Eksplorasi Mineral", "Geoteknik", "Lingkungan", "Mitigasi Bencana"])

    st.markdown('<div class="section-head">Ringkasan Hasil Interpretasi</div>', unsafe_allow_html=True)

    summary_data = {
        "Metode": ["Geolistrik Wenner 2D", "Geolistrik Wenner 2D", "Gravitasi", "Gravitasi", "Magnetik", "Magnetik"],
        "Parameter": ["Rentang ρₐ", "Konfigurasi", "CBA Rata-rata", "Rentang CBA", "Anomali Rata-rata", "Rentang ΔT"],
        "Nilai": ["15.2 – 84.0 Ω·m", "Wenner, a=8–40m, L=144m", "7.77 mGal", "-15.2 – 20.0 mGal", "Bervariasi tiap lintasan", "~–14 s.d. +70 nT"],
        "Interpretasi": [
            "Variasi sedimen basah hingga batuan lapuk",
            "Panjang lintasan 28E: 144m, spasi elektroda 8m",
            "Anomali positif dominan — densitas relatif tinggi",
            "Anomali negatif lokal kemungkinan rongga/material ringan",
            "Anomali negatif dominan — rendahnya suseptibilitas",
            "Variasi horizontal mengindikasikan perubahan litologi",
        ],
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-head">Rekomendasi</div>', unsafe_allow_html=True)
    recos = [
        ("💧 Air Tanah", "Resistivitas rendah (15–30 Ω·m) di beberapa titik datum point geolistrik mengindikasikan zona jenuh air. Disarankan pengeboran eksplorasi pada titik anomali rendah.", "#00d4aa"),
        ("🌋 Struktur Geologi", "Variasi CBA negatif lokal pada data gravitasi menunjukkan kemungkinan zona lemah atau material densitas rendah. Perlu korelasi dengan data geologi permukaan.", "#f0a500"),
        ("🧲 Mineral Magnetik", "Anomali total field magnetik menunjukkan perubahan suseptibilitas lateral. Area dengan anomali positif kuat patut menjadi target eksplorasi mineral besi.", "#7c5ef5"),
    ]
    for title, text, color in recos:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.04);border:1px solid {color}44;
                    border-left:4px solid {color};border-radius:10px;
                    padding:14px 18px;margin-bottom:10px;'>
          <div style='font-family:Syne;font-weight:700;color:{color};margin-bottom:4px;'>{title}</div>
          <div style='font-size:.9rem;color:#c9d1d9;'>{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Ekspor Data</div>', unsafe_allow_html=True)
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.download_button("⬇️ Unduh CSV", df_summary.to_csv(index=False).encode("utf-8"),
                           file_name=f"{project_name.replace(' ','_')}_hasil.csv",
                           mime="text/csv", use_container_width=True)
    with col_e2:
        json_export = {"proyek": project_name, "lokasi": location, "operator": operator,
                       "tanggal": str(survey_date), "metode": methods_used, "tujuan": purpose,
                       "hasil": summary_data}
        st.download_button("⬇️ Unduh JSON",
                           json.dumps(json_export, indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name=f"{project_name.replace(' ','_')}_laporan.json",
                           mime="application/json", use_container_width=True)
    with col_e3:
        md_report = f"""# Laporan {project_name}
**Lokasi:** {location} | **Operator:** {operator} | **Tanggal:** {survey_date}

## Metode
{', '.join(methods_used)}

## Hasil Interpretasi
{df_summary.to_markdown(index=False)}

## Rekomendasi
{chr(10).join([f'- **{t}**: {tx}' for t, tx, _ in recos])}
"""
        st.download_button("⬇️ Unduh Markdown", md_report.encode("utf-8"),
                           file_name=f"{project_name.replace(' ','_')}_laporan.md",
                           mime="text/markdown", use_container_width=True)

    st.markdown("""
    <div style='margin-top:20px;padding:16px 20px;background:rgba(0,212,170,.06);
                border:1px solid rgba(0,212,170,.2);border-radius:12px;font-size:.85rem;color:#8b949e;'>
      <b style='color:#e6edf3;'>ℹ️ Catatan:</b> Data dalam laporan ini merupakan hasil interpretasi geofisika.
      Keakuratan bergantung pada kualitas data lapangan dan validasi silang antar metode.
      Disarankan konfirmasi dengan data pengeboran (borehole log).
    </div>""", unsafe_allow_html=True)