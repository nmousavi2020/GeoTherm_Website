
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import geopandas as gpd

from scipy.interpolate import griddata
from shapely.geometry import Point, box
from shapely.prepared import prep

DEPTHS = [100, 200, 300, 400, 500, 600]
TEMP_COLS = [f"T{d}" for d in DEPTHS]
TEMP_MAX = 150

st.set_page_config(
    page_title="GeoTherm | Thermal Explorer",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:#f4f7fb;color:#172b4d}
[data-testid="stSidebar"] {background:#e9eff7}
.block-container {padding-top:1.5rem}
h1,h2,h3 {color:#163b65!important}
p,label {color:#263b53}
div[data-testid="stMetric"] {
 background:white;border:1px solid #dce5ef;
 padding:15px;border-radius:12px;
 box-shadow:0 2px 8px #17365d0a
}
div[data-testid="stMetricLabel"] {color:#52677f!important}
div[data-testid="stMetricValue"] {color:#075985!important}
.stDownloadButton button {
 background:#075985;color:white;border:0;border-radius:8px
}
.stDownloadButton button:hover {background:#0e7490;color:white}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(raw):
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    df.columns = [
        str(c).replace("\ufeff", "").strip().lstrip("#").strip()
        for c in df.columns
    ]

    required = ["lon", "lat", "TG", "ST"] + TEMP_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["lon", "lat"]).query(
        "lon >= -180 and lon <= 180 and lat >= -90 and lat <= 90"
    )


@st.cache_data
def load_land():
    # Natural Earth land polygons. Internet is required on first load.
    url = (
        "https://naturalearth.s3.amazonaws.com/"
        "110m_physical/ne_110m_land.zip"
    )
    land = gpd.read_file(url)
    return land.geometry.union_all()


def make_grid(data, column, bounds, method, resolution, land_union):
    xmin, xmax, ymin, ymax = bounds

    points_df = data.dropna(subset=["lon", "lat", column])
    points_df = points_df.groupby(
        ["lon", "lat"], as_index=False
    )[column].mean()

    if len(points_df) < 3:
        raise ValueError("At least 3 unique data points are required.")

    coords = points_df[["lon", "lat"]].to_numpy()
    values = points_df[column].to_numpy()

    # Interpolation is restricted to the data convex hull.
    hull = Point(coords[0]).buffer(0)
    from shapely.geometry import MultiPoint
    hull = MultiPoint(coords).convex_hull

    gx = np.linspace(xmin, xmax, resolution)
    gy = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(gx, gy)

    zz = griddata(coords, values, (xx, yy), method=method)

    # Mask ocean and all areas outside the measurement-data hull.
    land_prepared = prep(land_union)
    hull_prepared = prep(hull)
    mask = np.zeros(xx.shape, dtype=bool)

    for j in range(xx.shape[0]):
        for i in range(xx.shape[1]):
            pt = Point(float(xx[j, i]), float(yy[j, i]))
            mask[j, i] = (
                land_prepared.covers(pt)
                and hull_prepared.covers(pt)
            )

    zz[~mask] = np.nan
    return xx, yy, zz


def map_height(bounds, width=1100):
    xmin, xmax, ymin, ymax = bounds
    lat_mid = (ymin + ymax) / 2
    lon_width = max((xmax - xmin) * np.cos(np.radians(lat_mid)), 0.1)
    lat_height = max(ymax - ymin, 0.1)
    return int(np.clip(width * lat_height / lon_width, 350, 850))


def map_png(data, column, bounds, interpolate, method, resolution, land):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f7fb")

    if interpolate:
        xx, yy, zz = make_grid(
            data, column, bounds, method, resolution, land
        )
        mesh = ax.pcolormesh(
            xx, yy, zz, cmap="turbo", shading="auto",
            vmin=0, vmax=TEMP_MAX
        )
    else:
        mesh = ax.scatter(
            data.lon, data.lat, c=data[column],
            cmap="turbo", vmin=0, vmax=TEMP_MAX,
            s=36, edgecolors="white", linewidths=.35
        )

    ax.scatter(
        data.lon, data.lat, s=4,
        c="#17365d", alpha=.5, linewidths=0
    )
    fig.colorbar(mesh, ax=ax, label="Temperature (°C)")
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"GeoTherm | {column} | Subsurface temperature")
    ax.grid(alpha=.2)
    fig.tight_layout()

    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return output.getvalue()


st.title("🌍 GeoTherm")
st.markdown(
    "### Global Subsurface Thermal Explorer\n"
    "Geothermal engineering · Scientific data visualization"
)

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload GeoTherm CSV", type=["csv"])
    st.caption("Required: lon, lat, TG, ST, T100 … T600")

    st.divider()
    st.header("Geographic filters")
    lon_min = st.number_input(
        "Longitude minimum", min_value=-180.0,
        max_value=180.0, value=-180.0
    )
    lon_max = st.number_input(
        "Longitude maximum", min_value=-180.0,
        max_value=180.0, value=180.0
    )
    lat_min = st.number_input(
        "Latitude minimum", min_value=-90.0,
        max_value=90.0, value=-90.0
    )
    lat_max = st.number_input(
        "Latitude maximum", min_value=-90.0,
        max_value=90.0, value=90.0
    )
    depth = st.select_slider("Depth (m)", options=DEPTHS, value=300)

    st.divider()
    st.header("Map settings")
    use_interp = st.toggle("Enable land-only interpolation", value=False)
    method = st.selectbox(
        "Interpolation method", ["linear", "cubic", "nearest"],
        disabled=not use_interp
    )
    resolution = st.slider(
        "Grid resolution", 40, 180, 100, 10,
        disabled=not use_interp
    )

try:
    if uploaded is not None:
        df = load_data(uploaded.getvalue())
    else:
        with open("IHCF_subsurface_temperature.csv", "rb") as f:
            df = load_data(f.read())
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.info("Upload IHCF_subsurface_temperature.csv in the sidebar.")
    st.stop()

if lon_min >= lon_max or lat_min >= lat_max:
    st.error("Minimum coordinates must be smaller than maximum coordinates.")
    st.stop()

column = f"T{depth}"
bounds = (lon_min, lon_max, lat_min, lat_max)

filtered = df[
    df.lon.between(lon_min, lon_max)
    & df.lat.between(lat_min, lat_max)
].copy()

if filtered.empty:
    st.warning("No measurements found in this geographic region.")
    st.stop()

land_union = None
if use_interp:
    try:
        with st.spinner("Loading land boundaries..."):
            land_union = load_land()
    except Exception as exc:
        st.error(f"Could not load land boundaries: {exc}")
        st.info("Check your internet connection and try again.")
        st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Measurements", f"{len(filtered):,}")
m2.metric("Mean temperature", f"{filtered[column].mean():.2f} °C")
m3.metric("Minimum", f"{filtered[column].min():.2f} °C")
m4.metric("Maximum", f"{filtered[column].max():.2f} °C")

map_tab, profile_tab, data_tab = st.tabs([
    "🌍 Temperature Map",
    "📉 Vertical Profile",
    "📁 Data & Downloads"
])

with map_tab:
    st.subheader(f"Subsurface temperature at {depth} m")

    fig = go.Figure()

    if use_interp:
        try:
            xx, yy, zz = make_grid(
                filtered, column, bounds, method, resolution, land_union
            )
            fig.add_trace(go.Heatmap(
                x=xx[0], y=yy[:, 0], z=zz,
                colorscale="Turbo",
                zmin=0, zmax=TEMP_MAX,
                colorbar=dict(title="Temperature (°C)"),
                hovertemplate=(
                    "Lon: %{x:.3f}°<br>Lat: %{y:.3f}°"
                    "<br>Temperature: %{z:.2f} °C<extra></extra>"
                ),
                connectgaps=False,
                name="Interpolated estimate"
            ))
        except Exception as exc:
            st.warning(f"Interpolation unavailable: {exc}")

    fig.add_trace(go.Scattergl(
        x=filtered.lon, y=filtered.lat,
        mode="markers", name="Measured points",
        marker=dict(
            size=12,
            color=filtered[column],
            colorscale="Turbo",
            cmin=0, cmax=TEMP_MAX,
            showscale=not use_interp,
            colorbar=dict(title="Temperature (°C)"),
            line=dict(color="white", width=.7),
            opacity=.95
        ),
        customdata=filtered[["lat", column, "TG", "ST"]].to_numpy(),
        hovertemplate=(
            "Lon: %{x:.4f}°<br>Lat: %{customdata[0]:.4f}°"
            "<br>Temperature: %{customdata[1]:.2f} °C"
            "<br>Gradient: %{customdata[2]:.2f} °C/km"
            "<br>Surface: %{customdata[3]:.2f} °C<extra></extra>"
        )
    ))

    midlat = (lat_min + lat_max) / 2
    height = map_height(bounds)
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=dict(color="#263b53", size=12),
        margin=dict(l=15, r=15, t=20, b=20),
        xaxis=dict(
            title="Longitude (°)",
            range=[lon_min, lon_max],
            gridcolor="#dce5ef",
        ),
        yaxis=dict(
            title="Latitude (°)",
            range=[lat_min, lat_max],
            gridcolor="#dce5ef",
            scaleanchor="x",
            scaleratio=max(np.cos(np.radians(midlat)), 0.05),
        ),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    try:
        png = map_png(
            filtered, column, bounds, use_interp,
            method, resolution, land_union
        )
        st.download_button(
            "⬇️ Download map PNG", png,
            f"geotherm_{depth}m.png", "image/png"
        )
    except Exception as exc:
        st.warning(f"PNG export unavailable: {exc}")

with profile_tab:
    st.subheader("Temperature versus depth")
    c1, c2 = st.columns(2)
    qlon = c1.number_input(
        "Target longitude", min_value=-180.0,
        max_value=180.0, value=0.0, format="%.5f"
    )
    qlat = c2.number_input(
        "Target latitude", min_value=-90.0,
        max_value=90.0, value=0.0, format="%.5f"
    )

    valid = df.dropna(subset=["lon", "lat"] + TEMP_COLS).copy()
    distances = (
        ((valid.lon - qlon) * np.cos(np.radians(qlat))) ** 2
        + (valid.lat - qlat) ** 2
    )
    nearest = valid.loc[distances.idxmin()]
    temps = nearest[TEMP_COLS].to_numpy(dtype=float)

    st.caption(
        f"Nearest measured point: {nearest.lon:.4f}°, "
        f"{nearest.lat:.4f}°"
    )

    profile = go.Figure(go.Scatter(
        x=temps, y=DEPTHS,
        mode="lines+markers",
        line=dict(color="#087ea4", width=3),
        marker=dict(
            size=10, color="#0e7490",
            line=dict(color="white", width=1)
        ),
        hovertemplate=(
            "Depth: %{y} m<br>"
            "Temperature: %{x:.2f} °C<extra></extra>"
        )
    ))
    profile.update_layout(
        template="plotly_white",
        height=480,
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=dict(color="#263b53", size=12),
        margin=dict(l=60, r=25, t=25, b=55),
        xaxis=dict(
            title="Temperature (°C)",
            range=[0, TEMP_MAX],
            gridcolor="#dce5ef"
        ),
        yaxis=dict(
            title="Depth (m)",
            autorange="reversed",
            gridcolor="#dce5ef"
        ),
    )
    st.plotly_chart(profile, use_container_width=True)

    profile_df = pd.DataFrame({
        "depth_m": DEPTHS,
        "temperature_C": temps
    })
    p1, p2 = st.columns(2)
    p1.download_button(
        "⬇️ Download profile CSV",
        profile_df.to_csv(index=False).encode("utf-8"),
        "geotherm_profile.csv", "text/csv",
        use_container_width=True
    )

    profile_png = io.BytesIO()
    profile.write_image if False else None
    profile_fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    profile_fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.plot(temps, DEPTHS, color="#087ea4", marker="o", linewidth=2.5)
    ax.set_ylim(600, 0)
    ax.set_xlim(0, TEMP_MAX)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("GeoTherm | Vertical temperature profile")
    ax.grid(alpha=.2)
    profile_fig.tight_layout()
    profile_fig.savefig(profile_png, format="png", dpi=180)
    plt.close(profile_fig)
    p2.download_button(
        "⬇️ Download profile PNG",
        profile_png.getvalue(),
        "geotherm_profile.png", "image/png",
        use_container_width=True
    )

with data_tab:
    st.subheader("Filtered measurements")
    st.dataframe(
        filtered[["lon", "lat", "TG", "ST"] + TEMP_COLS],
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "⬇️ Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "geotherm_filtered.csv", "text/csv"
    )

    if use_interp:
        try:
            xx, yy, zz = make_grid(
                filtered, column, bounds, method, resolution, land_union
            )
            grid_df = pd.DataFrame({
                "lon": xx.ravel(),
                "lat": yy.ravel(),
                "temperature_C": zz.ravel()
            }).dropna()
            st.download_button(
                "⬇️ Download land-only interpolated CSV",
                grid_df.to_csv(index=False).encode("utf-8"),
                f"geotherm_interpolated_{depth}m.csv", "text/csv"
            )
        except Exception as exc:
            st.warning(f"Grid export unavailable: {exc}")

st.divider()
st.caption("GeoTherm · Copyright © 2026 N. Mousavi. All rights reserved.")
