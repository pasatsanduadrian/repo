import os
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, send_file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import osmnx as ox
import contextily as ctx
from tqdm import tqdm
import base64
from datetime import datetime
from pyngrok import ngrok
import threading

load_dotenv()

app = Flask(__name__)

# =======================
# Utilitare model dispersie
# =======================

def calc_sigmas(CATEGORY, x1):
    x = np.abs(x1)
    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))
    c = np.zeros(np.shape(x))
    d = np.zeros(np.shape(x))
    # ... (același cod ca în exemplul tău, vezi mai sus pentru toate categoriile) ...
    # (codul pentru fiecare categorie, vezi mesajul anterior pentru detalii)
    # [cod complet identic cu versiunea precedentă]
    if CATEGORY == 1:
        ind = np.where((x < 100.) & (x > 0.)); a[ind] = 122.800; b[ind] = 0.94470
        ind = np.where((x >= 100.) & (x < 150.)); a[ind] = 158.080; b[ind] = 1.05420
        ind = np.where((x >= 150.) & (x < 200.)); a[ind] = 170.220; b[ind] = 1.09320
        ind = np.where((x >= 200.) & (x < 250.)); a[ind] = 179.520; b[ind] = 1.12620
        ind = np.where((x >= 250.) & (x < 300.)); a[ind] = 217.410; b[ind] = 1.26440
        ind = np.where((x >= 300.) & (x < 400.)); a[ind] = 258.89; b[ind] = 1.40940
        ind = np.where((x >= 400.) & (x < 500.)); a[ind] = 346.75; b[ind] = 1.7283
        ind = np.where((x >= 500.) & (x < 3110.)); a[ind] = 453.85; b[ind] = 2.1166
        ind = np.where(x >= 3110.); a[ind] = 453.85; b[ind] = 2.1166
        c[:] = 24.1670; d[:] = 2.5334
    # (continuă cu categoriile 2-6, exact ca în codul tău)
    # ... codul complet calc_sigmas (sau cere-l explicit dacă vrei tot)
    elif CATEGORY == 2:
        ind = np.where((x < 200.) & (x > 0.)); a[ind] = 90.673; b[ind] = 0.93198
        ind = np.where((x >= 200.) & (x < 400.)); a[ind] = 98.483; b[ind] = 0.98332
        ind = np.where(x >= 400.); a[ind] = 109.3; b[ind] = 1.09710
        c[:] = 18.3330; d[:] = 1.8096
    elif CATEGORY == 3:
        a[:] = 61.141; b[:] = 0.91465; c[:] = 12.5; d[:] = 1.0857
    elif CATEGORY == 4:
        ind = np.where((x < 300.) & (x > 0.)); a[ind] = 34.459; b[ind] = 0.86974
        ind = np.where((x >= 300.) & (x < 1000.)); a[ind] = 32.093; b[ind] = 0.81066
        ind = np.where((x >= 1000.) & (x < 3000.)); a[ind] = 32.093; b[ind] = 0.64403
        ind = np.where((x >= 3000.) & (x < 10000.)); a[ind] = 33.504; b[ind] = 0.60486
        ind = np.where((x >= 10000.) & (x < 30000.)); a[ind] = 36.650; b[ind] = 0.56589
        ind = np.where(x >= 30000.); a[ind] = 44.053; b[ind] = 0.51179
        c[:] = 8.3330; d[:] = 0.72382
    elif CATEGORY == 5:
        ind = np.where((x < 100.) & (x > 0.)); a[ind] = 24.26; b[ind] = 0.83660
        ind = np.where((x >= 100.) & (x < 300.)); a[ind] = 23.331; b[ind] = 0.81956
        ind = np.where((x >= 300.) & (x < 1000.)); a[ind] = 21.628; b[ind] = 0.75660
        ind = np.where((x >= 1000.) & (x < 2000.)); a[ind] = 21.628; b[ind] = 0.63077
        ind = np.where((x >= 2000.) & (x < 4000.)); a[ind] = 22.534; b[ind] = 0.57154
        ind = np.where((x >= 4000.) & (x < 10000.)); a[ind] = 24.703; b[ind] = 0.50527
        ind = np.where((x >= 10000.) & (x < 20000.)); a[ind] = 26.970; b[ind] = 0.46713
        ind = np.where((x >= 20000.) & (x < 40000.)); a[ind] = 35.420; b[ind] = 0.37615
        ind = np.where(x >= 40000.); a[ind] = 47.618; b[ind] = 0.29592
        c[:] = 6.25; d[:] = 0.54287
    elif CATEGORY == 6:
        ind = np.where((x < 200.) & (x > 0.)); a[ind] = 15.209; b[ind] = 0.81558
        ind = np.where((x >= 200.) & (x < 700.)); a[ind] = 14.457; b[ind] = 0.78407
        ind = np.where((x >= 700.) & (x < 1000.)); a[ind] = 13.953; b[ind] = 0.68465
        ind = np.where((x >= 1000.) & (x < 2000.)); a[ind] = 13.953; b[ind] = 0.63227
        ind = np.where((x >= 2000.) & (x < 3000.)); a[ind] = 14.823; b[ind] = 0.54503
        ind = np.where((x >= 3000.) & (x < 7000.)); a[ind] = 16.187; b[ind] = 0.46490
        ind = np.where((x >= 7000.) & (x < 15000.)); a[ind] = 17.836; b[ind] = 0.41507
        ind = np.where((x >= 15000.) & (x < 30000.)); a[ind] = 22.651; b[ind] = 0.32681
        ind = np.where((x >= 30000.) & (x < 60000.)); a[ind] = 27.074; b[ind] = 0.27436
        ind = np.where(x >= 60000.); a[ind] = 34.219; b[ind] = 0.21716
        c[:] = 4.1667; d[:] = 0.36191
    else:
        raise ValueError("Unknown stability category.")
    sig_z = a * (x / 1000.)**b
    sig_z[x > 0][sig_z[x > 0] > 5000.] = 5000.
    theta = 0.017453293 * (c - d * np.log(np.abs(x + 1e-15) / 1000.))
    sig_y = 465.11628 * x / 1000. * np.tan(theta)
    return sig_y, sig_z

def gauss_func(Q, u, dir1, x, y, z, xs, ys, H_eff, Dy, Dz, STABILITY):
    u1 = u
    x1 = x - xs
    y1 = y - ys
    wx = u1 * np.sin((dir1 - 180.) * np.pi / 180.)
    wy = u1 * np.cos((dir1 - 180.) * np.pi / 180.)
    dot_product = wx * x1 + wy * y1
    magnitudes = u1 * np.sqrt(x1**2 + y1**2)
    subtended = np.arccos(dot_product / (magnitudes + 1e-15))
    downwind = np.cos(subtended) * magnitudes
    crosswind = np.sin(subtended) * magnitudes
    sig_y, sig_z = calc_sigmas(STABILITY, downwind)
    C = np.zeros(x.shape)
    ind = downwind > 0
    exp_part = np.exp(-crosswind[ind]**2 / (2 * sig_y[ind]**2 + 1e-15))
    exp_z = (np.exp(- (z[ind] - H_eff)**2 / (2 * sig_z[ind]**2 + 1e-15)) +
             np.exp(- (z[ind] + H_eff)**2 / (2 * sig_z[ind]**2 + 1e-15)))
    C[ind] = Q / (2 * np.pi * u1 * sig_y[ind] * sig_z[ind] + 1e-15) * exp_part * exp_z
    return C

def calculate_plume_rise(T_stack, T_ambient, wind_speed, flow_rate, stack_diameter=0.5):
    T_stack_K = T_stack + 273.15
    T_ambient_K = T_ambient + 273.15
    delta_T = T_stack_K - T_ambient_K
    delta_H = stack_diameter * (flow_rate / (wind_speed + 1e-15))**0.25 * (1 + (delta_T / (T_stack_K + 1e-15)))
    return delta_H

def estimate_stability(row):
    if row['WS'] > 4.5: return 6
    elif row['WS'] > 3.5: return 5
    elif row['RAD'] > 200 and row['TC'] > 20: return 4
    elif row['RAD'] > 120 and row['TC'] > 15: return 3
    elif row['RAD'] > 50: return 2
    elif row['RH'] > 75: return 1
    else: return 1

def overlay_on_map(local_x, local_y, C1, image_data, extent_range):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    local_extent = [-extent_range, extent_range, -extent_range, extent_range]
    ax.imshow(image_data, extent=local_extent, origin='lower', alpha=1.0)
    C1_max = np.max(C1, axis=2) * 1e6
    C1_max = np.nan_to_num(C1_max)
    max_val = np.max(C1_max)
    min_val = np.min(C1_max[C1_max > 0]) if np.any(C1_max > 0) else 0
    if max_val == min_val:
        thresholds = np.linspace(min_val, max_val + 1e-9, 7)
    else:
        thresholds = np.linspace(min_val, max_val, 7)
    thresholds = np.unique(thresholds)
    if len(thresholds) < 2:
        thresholds = np.array([min_val, max_val + 1e-9])
    cmap = mcolors.ListedColormap(['#edf8fb', '#b2e2e2', '#8cd2c3', '#66c2a4', '#238b45', '#005824'])
    if len(thresholds) -1 < len(cmap.colors):
        cmap = mcolors.ListedColormap(cmap.colors[:len(thresholds)-1])
    cs = ax.contourf(local_x, local_y, C1_max, levels=thresholds, cmap=cmap, alpha=0.5, extend='neither')
    cbar = fig.colorbar(cs, ticks=thresholds)
    cbar.set_label(r'$\mu g / m^{3}$')
    ax.scatter(0, 0, color='red', s=80, edgecolor='k', zorder=5, label='Emission Source')
    ax.set_xlim(local_extent[0], local_extent[1])
    ax.set_ylim(local_extent[2], local_extent[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m, local)')
    ax.set_ylabel('y (m, local)')
    ax.set_title('Dispersion Plume Overlaid on Local Street Map')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig, C1_max, min_val, max_val

def run_dispersion_model_local(center_lon, center_lat, street_network_image,
                               emission_source_x, emission_source_y, flow_rate, H, T_stack, crs_proj, extent_range, size_in_pixels):
    x_range = np.linspace(emission_source_x - extent_range, emission_source_x + extent_range, size_in_pixels)
    y_range = np.linspace(emission_source_y - extent_range, emission_source_y + extent_range, size_in_pixels)
    x, y = np.meshgrid(x_range, y_range)
    local_x = x - emission_source_x
    local_y = y - emission_source_y
    data = {
        "Time": pd.date_range("2023-07-17 00:00", periods=24, freq='H'),
        "RH": [65.68, 70, 72.73, 75.07, 69.47, 58.89, 51.1, 46.44, 41.02, 35.76, 33.26, 32.76, 32.17, 31.73, 32.69, 34.06, 35.27, 39.56, 45.95, 46.25, 52.12, 54.68, 57.32, 58.74],
        "TC": [23.01, 21.97, 21.2, 21.03, 23.26, 26.52, 29.82, 32.33, 34.69, 37.16, 38.34, 38.45, 38.61, 38.16, 36.48, 35.81, 34.8, 32.25, 30.27, 30.84, 28.74, 27.62, 26.49, 25.79],
        "WD": [70.91, 70.91, 70.9, 70.91, 70.83, 70.83, 50.1, 242.71, 137.74, 157.93, 80.85, 146.77, 175.06, 309.03, 8.67, 6.98, 223.43, 274.78, 280.65, 243.65, 219.07, 322.56, 316.16, 316.01],
        "WS": [0.11, 0.11, 0.11, 0.11, 3.11, 0.12, 0.13, 0.19, 0.19, 0.24, 0.25, 0.27, 0.29, 0.16, 0.13, 0.12, 0.13, 0.13, 0.12, 0.17, 0.12, 0.12, 0.12, 0.12],
        "S1": [17.06, 15.19, 34.14, 43.12, 30.21, 36.39, 41.02, 33.42, 29.36, 36.16, 128.46, 42.12, 50.53, 58.94, 44.62, 26.17, 90.71, 256.10, 144.29, 20.24, 62.01, 41.82, 26.24, 45.18],
        "RAD": [150] * 24
    }
    df = pd.DataFrame(data)
    df['Stability'] = df.apply(estimate_stability, axis=1)
    df['Q'] = df['S1'] / 1e6
    y_dim, x_dim = x.shape
    C1 = np.zeros((y_dim, x_dim, 24))
    for i in tqdm(range(24), desc="Simulating dispersion"):
        wind_dir = df.loc[i, 'WD']
        wind_speed = df.loc[i, 'WS']
        T_ambient = df.loc[i, 'TC']
        stability = df.loc[i, 'Stability']
        H_eff = H + calculate_plume_rise(T_stack, T_ambient, wind_speed, flow_rate)
        C = gauss_func(df.loc[i, 'Q'], wind_speed, wind_dir, x, y, np.zeros((y_dim, x_dim)),
                       emission_source_x, emission_source_y, H_eff, 10, 10, stability)
        mask = (
            (x < emission_source_x - extent_range) | (x > emission_source_x + extent_range) |
            (y < emission_source_y - extent_range) | (y > emission_source_y + extent_range)
        )
        C[mask] = np.nan
        C1[:, :, i] = C
    fig_overlay, C1_max, min_conc, max_conc = overlay_on_map(local_x, local_y, C1, street_network_image, extent_range)
    return fig_overlay, min_conc, max_conc

# ======================
# Funcții de generare hartă (ai deja)
# ======================

# def generate_network_basemap(center_lat, center_lon, dist=2500, dxy=10, dpi=100):
#     G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
#     node_id = ox.nearest_nodes(G, center_lon, center_lat)
#     G_proj = ox.project_graph(G)
#     crs_proj = G_proj.graph['crs']
#     emission_source_x = G_proj.nodes[node_id]['x']
#     emission_source_y = G_proj.nodes[node_id]['y']
#     extent_range = dist
#     size_in_pixels = int((2 * extent_range) / dxy)
#     figsize = (size_in_pixels / dpi, size_in_pixels / dpi)
#     fig, ax = ox.plot_graph(
#         G_proj, bgcolor='none', edge_color='#003366', node_size=0,
#         edge_linewidth=0.5, show=False, close=False, figsize=figsize
#     )
#     ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
#     ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])
#     try:
#         ctx.add_basemap(ax, crs=crs_proj, source=ctx.providers.Stamen.Terrain)
#     except Exception:
#         pass
#     fig.tight_layout(pad=0)
#     fig.canvas.draw()
#     image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
#     path = "/tmp/network_basemap.png"
#     fig.savefig(path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels, path

# def generate_network_only(center_lat, center_lon, dist=2500, dxy=10, dpi=100):
#     G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
#     node_id = ox.nearest_nodes(G, center_lon, center_lat)
#     G_proj = ox.project_graph(G)
#     crs_proj = G_proj.graph['crs']
#     emission_source_x = G_proj.nodes[node_id]['x']
#     emission_source_y = G_proj.nodes[node_id]['y']
#     extent_range = dist
#     size_in_pixels = int((2 * extent_range) / dxy)
#     figsize = (size_in_pixels / dpi, size_in_pixels / dpi)
#     fig, ax = ox.plot_graph(
#         G_proj, bgcolor='white', edge_color='#003366', node_size=0,
#         edge_linewidth=0.5, show=False, close=False, figsize=figsize
#     )
#     ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
#     ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])
#     fig.tight_layout(pad=0)
#     fig.canvas.draw()
#     image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
#     plt.close(fig)
#     return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels

def generate_network_basemap(center_lat, center_lon, dist=2500, dxy=10, dpi=100):
    import pyproj

    # Proiecție EPSG:3857
    crs_3857 = "EPSG:3857"

    # Creează graful și proiectează în EPSG:3857
    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
    G_proj = ox.project_graph(G, to_crs=crs_3857)
    crs_proj = G_proj.graph['crs']

    # Coordonate sursă în EPSG:4326 -> EPSG:3857
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_3857, always_xy=True)
    emission_source_x, emission_source_y = transformer.transform(center_lon, center_lat)

    extent_range = dist
    size_in_pixels = int((2 * extent_range) / dxy)
    figsize = (size_in_pixels / dpi, size_in_pixels / dpi)

    fig, ax = ox.plot_graph(
        G_proj, bgcolor='none', edge_color='#003366', node_size=0,
        edge_linewidth=0.5, show=False, close=False, figsize=figsize
    )
    # Limitele axei în EPSG:3857
    ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
    ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])

    # Adaugă basemap color, neapărat cu reset_extent=True!
    try:
        ctx.add_basemap(
            ax, crs=crs_proj, source=ctx.providers.Stamen.Terrain,
            zoom=15, reset_extent=False  # Păstrează exact axa setată mai sus!
        )
    except Exception as e:
        print(f"Eroare la contextily: {e}")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
    path = "/tmp/network_basemap.png"
    fig.savefig(path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels, path


def generate_network_only(center_lat, center_lon, dist=2500, dxy=10, dpi=100):
    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
    node_id = ox.nearest_nodes(G, center_lon, center_lat)
    G_proj = ox.project_graph(G, to_crs="EPSG:3857")
    crs_proj = G_proj.graph['crs']
    emission_source_x = G_proj.nodes[node_id]['x']
    emission_source_y = G_proj.nodes[node_id]['y']
    extent_range = dist
    size_in_pixels = int((2 * extent_range) / dxy)
    figsize = (size_in_pixels / dpi, size_in_pixels / dpi)
    fig, ax = ox.plot_graph(
        G_proj, bgcolor='white', edge_color='#003366', node_size=0,
        edge_linewidth=0.5, show=False, close=False, figsize=figsize
    )
    ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
    ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
    plt.close(fig)
    return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels


# =====================
# Flask routes & Template
# =====================

form_template = """
<!DOCTYPE html>
<html lang="ro">
<head>
  <meta charset="UTF-8">
  <title>Simulare dispersie poluanți — Vizualizare pe hartă</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f6f8fb; margin:0; padding:0; }
    .container { max-width: 960px; margin:40px auto 40px auto; background:#fff; border-radius:18px; box-shadow:0 2px 24px #00244422; padding: 36px 28px; }
    h1, h2, h3 { color: #003366; }
    label { font-weight:bold; margin-top: 8px;}
    input[type="text"] { padding:6px; border-radius:6px; border:1px solid #ddd; width:120px; margin-bottom: 12px;}
    .btn { background: #0066cc; color: #fff; border:none; border-radius:6px; padding: 8px 20px; cursor:pointer; font-size:16px; }
    .cards { display: flex; gap:32px; flex-wrap:wrap; margin-top:24px; }
    .card { background:#f7faff; border-radius:14px; box-shadow:0 1px 8px #0050a111; padding:18px; flex:1 1 320px; min-width:280px; }
    img { width:100%; max-width: 480px; border-radius:8px; border:1.5px solid #e0e0e0; box-shadow: 0 1px 6px #00447722;}
    table { width:100%; margin-top: 14px;}
    th, td { text-align:left; padding:6px 12px;}
    th { background:#eaf4fc;}
    tr:nth-child(even){background-color: #f2f9fe;}
    .backlink { display:block; margin-top:18px; font-size:15px; color: #0366d6;}
    @media (max-width:900px) { .cards{flex-direction:column; gap:18px;} }
    .osm-note { margin:10px 0 18px 0; color:#333;}
  </style>
</head>
<body>
  <div class="container">
    <h1>🛰️ Simulare dispersie poluanți</h1>
    <p>Generează rapid o simulare și vizualizează rezultatul pe hartă cu străzi și suprapunerea plumei de poluare.</p>
    <div class="osm-note">
      <b>Consultă harta:</b>
      <a href="https://www.openstreetmap.org/search?query=strada%20mosoaia%2037%2C%20bucuresti#map=19/44.379146/26.130614"
         target="_blank" style="color:#0366d6; text-decoration:underline;">
         Deschide OpenStreetMap (pentru identificare coordonate)
      </a>
      <br/>Copiezi coordonatele în câmpurile de mai jos.
    </div>
    <form method="post" action="/simulate">
      <label>Latitudine centru:</label><br>
      <input type="text" name="latitude" value="44.4399"><br>
      <label>Longitudine centru:</label><br>
      <input type="text" name="longitude" value="26.0550"><br>
      <label>Rază analiză (m):</label><br>
      <input type="text" name="radius" value="2500"><br>
      <label>Debit (flow rate, m³/s):</label><br>
      <input type="text" name="flow_rate" value="5"><br>
      <label>Înălțime sursă (H, m):</label><br>
      <input type="text" name="H" value="25"><br>
      <label>Temperatură gaze (°C):</label><br>
      <input type="text" name="T_stack" value="150"><br><br>
      <input class="btn" type="submit" value="Rulează simularea">
    </form>
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(form_template)

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        latitude = float(request.form.get("latitude", 44.4399))
        longitude = float(request.form.get("longitude", 26.0550))
        radius = float(request.form.get("radius", 2500))
        flow_rate = float(request.form.get("flow_rate", 5))
        H = float(request.form.get("H", 25))
        T_stack = float(request.form.get("T_stack", 150))
        dxy = 10
        dpi = 100

        # Card 1: hartă rețea stradală + basemap color
        network_img, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels, network_path = generate_network_basemap(
            latitude, longitude, dist=radius, dxy=dxy, dpi=dpi
        )
        # Card 2: hartă doar cu străzi pentru pluma
        street_network_image, _, _, _, _, _ = generate_network_only(
            latitude, longitude, dist=radius, dxy=dxy, dpi=dpi
        )

        # Rulează modelul și plotează pluma DOAR pe străzi (fără basemap color)
        fig_overlay, min_conc, max_conc = run_dispersion_model_local(
            longitude, latitude, street_network_image,
            emission_source_x, emission_source_y, flow_rate, H, T_stack, crs_proj, extent_range, size_in_pixels
        )
        overlay_path = "/tmp/plume.png"
        fig_overlay.savefig(overlay_path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_overlay)

        # Imagini base64 pentru embed + url local pentru click/zoom
        with open(network_path, "rb") as imgfile:
            network_bytes = imgfile.read()
        with open(overlay_path, "rb") as imgfile:
            overlay_bytes = imgfile.read()

        network_b64 = base64.b64encode(network_bytes).decode()
        overlay_b64 = base64.b64encode(overlay_bytes).decode()

        # Metadate tabelar
        metadate = [
            ("Data generării", datetime.now().strftime("%d/%m/%Y %H:%M")),
            ("Coordonate centru", f"{latitude}, {longitude}"),
            ("Rază analiză", f"{radius} m"),
            ("Flow rate", f"{flow_rate} m³/s"),
            ("Înălțime sursă", f"{H} m"),
            ("Temperatură gaze", f"{T_stack} °C"),
            ("Grid step", f"{dxy} m"),
            ("Emission source (x, y)", f"{emission_source_x:.2f}, {emission_source_y:.2f}"),
            ("Max conc", f"{max_conc:.2f} μg/m³"),
            ("Min conc (>0)", f"{min_conc:.2f} μg/m³")
        ]

        html = f"""
        <div class="container">
          <h1>🛰️ Rezultat simulare dispersie</h1>
          <div class="cards">
            <div class="card">
              <h2>Hartă rețea stradală cu basemap</h2>
              <a href="/image/network" target="_blank" title="Deschide imagine completă">
                <img src='data:image/png;base64,{network_b64}' alt="Basemap"/>
              </a>
            </div>
            <div class="card">
              <h2>Plume suprapus pe rețea stradală</h2>
              <a href="/image/overlay" target="_blank" title="Deschide imagine completă">
                <img src='data:image/png;base64,{overlay_b64}' alt="Overlay"/>
              </a>
            </div>
          </div>
          <div class="card" style="margin-top:32px;">
            <h3>Metadate simulare</h3>
            <table>
              <tbody>
                {''.join([f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in metadate])}
              </tbody>
            </table>
          </div>
          <a class="backlink" href='/'>⟵ Înapoi la simulare</a>
        </div>
        <style>
            {form_template.split("<style>")[1].split("</style>")[0]}
        </style>
        """
        # Salvează și pentru acces la imagine full
        with open("/tmp/last_network.png", "wb") as f:
            f.write(network_bytes)
        with open("/tmp/last_overlay.png", "wb") as f:
            f.write(overlay_bytes)

        return render_template_string(html)
    except Exception as e:
        html = f"""
        <div class="container">
            <h2 style="color:#a00;">Eroare la simulare</h2>
            <div class="card" style="background:#ffebee;color:#a00;">
              <b>{str(e)}</b>
            </div>
            <a class="backlink" href='/'>⟵ Înapoi la simulare</a>
        </div>
        <style>
            {form_template.split("<style>")[1].split("</style>")[0]}
        </style>
        """
        return render_template_string(html)

@app.route("/image/network")
def img_network():
    return send_file("/tmp/last_network.png", mimetype="image/png")

@app.route("/image/overlay")
def img_overlay():
    return send_file("/tmp/last_overlay.png", mimetype="image/png")

# ====================
# Pornire Flask + ngrok
# ====================
def run_with_ngrok(app, port=5008):
    ngrok_token = os.getenv("NGROK_TOKEN")
    ngrok_hostname = os.getenv("NGROK_HOSTNAME")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        try:
            if ngrok_hostname:
                public_url = ngrok.connect(port, hostname=ngrok_hostname)
            else:
                public_url = ngrok.connect(port)
        except Exception as e:
            print("Eroare la conectarea cu hostname static:", e)
            public_url = ngrok.connect(port)
        print(f"\n*** LINK STATIC NGROK: {public_url.public_url}\n")
    else:
        print("Warning: nu ai setat token ngrok. Aplicația va fi disponibilă doar local.")

    def run_flask():
        app.run(host='0.0.0.0', port=port, debug=False)

    thread = threading.Thread(target=run_flask)
    thread.daemon = True
    thread.start()
    thread.join()

if __name__ == "__main__":
    run_with_ngrok(app, port=5009)
