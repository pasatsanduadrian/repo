import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib

from flask import Flask, request, render_template_string, send_file
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke
from matplotlib.patches import FancyArrowPatch
import osmnx as ox
import contextily as ctx
from tqdm import tqdm
import pyproj
from shapely.ops import transform
import shapely.geometry

# Configure Flask
app = Flask(__name__)

##############################
# Utility Functions
##############################

def calc_sigmas(CATEGORY, x1):
    x = np.abs(x1)
    a = np.zeros(np.shape(x))
    b = np.zeros(np.shape(x))
    c = np.zeros(np.shape(x))
    d = np.zeros(np.shape(x))
    if CATEGORY == 1:  # very unstable
        ind = np.where((x < 100.) & (x > 0.))
        a[ind] = 122.800; b[ind] = 0.94470
        ind = np.where((x >= 100.) & (x < 150.))
        a[ind] = 158.080; b[ind] = 1.05420
        ind = np.where((x >= 150.) & (x < 200.))
        a[ind] = 170.220; b[ind] = 1.09320
        ind = np.where((x >= 200.) & (x < 250.))
        a[ind] = 179.520; b[ind] = 1.12620
        ind = np.where((x >= 250.) & (x < 300.))
        a[ind] = 217.410; b[ind] = 1.26440
        ind = np.where((x >= 300.) & (x < 400.))
        a[ind] = 258.89; b[ind] = 1.40940
        ind = np.where((x >= 400.) & (x < 500.))
        a[ind] = 346.75; b[ind] = 1.7283
        ind = np.where((x >= 500.) & (x < 3110.))
        a[ind] = 453.85; b[ind] = 2.1166
        ind = np.where(x >= 3110.)
        a[ind] = 453.85; b[ind] = 2.1166
        c[:] = 24.1670
        d[:] = 2.5334
    elif CATEGORY == 2:  # moderately unstable
        ind = np.where((x < 200.) & (x > 0.))
        a[ind] = 90.673; b[ind] = 0.93198
        ind = np.where((x >= 200.) & (x < 400.))
        a[ind] = 98.483; b[ind] = 0.98332
        ind = np.where(x >= 400.)
        a[ind] = 109.3; b[ind] = 1.09710
        c[:] = 18.3330
        d[:] = 1.8096
    elif CATEGORY == 3:  # slightly unstable
        a[:] = 61.141
        b[:] = 0.91465
        c[:] = 12.5
        d[:] = 1.0857
    elif CATEGORY == 4:  # neutral
        ind = np.where((x < 300.) & (x > 0.))
        a[ind] = 34.459; b[ind] = 0.86974
        ind = np.where((x >= 300.) & (x < 1000.))
        a[ind] = 32.093; b[ind] = 0.81066
        ind = np.where((x >= 1000.) & (x < 3000.))
        a[ind] = 32.093; b[ind] = 0.64403
        ind = np.where((x >= 3000.) & (x < 10000.))
        a[ind] = 33.504; b[ind] = 0.60486
        ind = np.where((x >= 10000.) & (x < 30000.))
        a[ind] = 36.650; b[ind] = 0.56589
        ind = np.where(x >= 30000.)
        a[ind] = 44.053; b[ind] = 0.51179
        c[:] = 8.3330
        d[:] = 0.72382
    elif CATEGORY == 5:  # moderately stable
        ind = np.where((x < 100.) & (x > 0.))
        a[ind] = 24.26; b[ind] = 0.83660
        ind = np.where((x >= 100.) & (x < 300.))
        a[ind] = 23.331; b[ind] = 0.81956
        ind = np.where((x >= 300.) & (x < 1000.))
        a[ind] = 21.628; b[ind] = 0.75660
        ind = np.where((x >= 1000.) & (x < 2000.))
        a[ind] = 21.628; b[ind] = 0.63077
        ind = np.where((x >= 2000.) & (x < 4000.))
        a[ind] = 22.534; b[ind] = 0.57154
        ind = np.where((x >= 4000.) & (x < 10000.))
        a[ind] = 24.703; b[ind] = 0.50527
        ind = np.where((x >= 10000.) & (x < 20000.))
        a[ind] = 26.970; b[ind] = 0.46713
        ind = np.where((x >= 20000.) & (x < 40000.))
        a[ind] = 35.420; b[ind] = 0.37615
        ind = np.where(x >= 40000.)
        a[ind] = 47.618; b[ind] = 0.29592
        c[:] = 6.25
        d[:] = 0.54287
    elif CATEGORY == 6:  # very stable
        ind = np.where((x < 200.) & (x > 0.))
        a[ind] = 15.209; b[ind] = 0.81558
        ind = np.where((x >= 200.) & (x < 700.))
        a[ind] = 14.457; b[ind] = 0.78407
        ind = np.where((x >= 700.) & (x < 1000.))
        a[ind] = 13.953; b[ind] = 0.68465
        ind = np.where((x >= 1000.) & (x < 2000.))
        a[ind] = 13.953; b[ind] = 0.63227
        ind = np.where((x >= 2000.) & (x < 3000.))
        a[ind] = 14.823; b[ind] = 0.54503
        ind = np.where((x >= 3000.) & (x < 7000.))
        a[ind] = 16.187; b[ind] = 0.46490
        ind = np.where((x >= 7000.) & (x < 15000.))
        a[ind] = 17.836; b[ind] = 0.41507
        ind = np.where((x >= 15000.) & (x < 30000.))
        a[ind] = 22.651; b[ind] = 0.32681
        ind = np.where((x >= 30000.) & (x < 60000.))
        a[ind] = 27.074; b[ind] = 0.27436
        ind = np.where(x >= 60000.)
        a[ind] = 34.219; b[ind] = 0.21716
        c[:] = 4.1667
        d[:] = 0.36191
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
    exp_part = np.exp(-crosswind[ind]**2 / (2 * sig_y[ind]**2))
    exp_z = (np.exp(- (z[ind] - H_eff)**2 / (2 * sig_z[ind]**2)) +
             np.exp(- (z[ind] + H_eff)**2 / (2 * sig_z[ind]**2)))
    C[ind] = Q / (2 * np.pi * u1 * sig_y[ind] * sig_z[ind]) * exp_part * exp_z
    return C

def calculate_plume_rise(T_stack, T_ambient, wind_speed, flow_rate, stack_diameter=0.5):
    T_stack_K = T_stack + 273.15
    T_ambient_K = T_ambient + 273.15
    delta_T = T_stack_K - T_ambient_K
    delta_H = stack_diameter * (flow_rate / wind_speed)**0.25 * (1 + (delta_T / T_stack_K))
    return delta_H

def estimate_stability(row):
    if row['WS'] > 4.5:
        return 6
    elif row['WS'] > 3.5:
        return 5
    elif row['RAD'] > 200 and row['TC'] > 20:
        return 4
    elif row['RAD'] > 120 and row['TC'] > 15:
        return 3
    elif row['RAD'] > 50:
        return 2
    elif row['RH'] > 75:
        return 1
    else:
        return 1

def overlay_on_map(local_x, local_y, C1, image_data, extent_range):
    """
    Plot the street map and the dispersion plume using *local* coordinates
    where the emission source is at (0,0) and the extent is [-extent_range, extent_range].
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # Define the local extent
    local_extent = [-extent_range, extent_range, -extent_range, extent_range]

    # Plot the base map image using the local extent
    ax.imshow(image_data, extent=local_extent, origin='lower', alpha=1.0)

    # Plot the dispersion concentration
    C1_max = np.max(C1, axis=2) * 1e6
    # Ensure C1_max has finite values where calculations are valid, replace NaNs if necessary for contouring
    C1_max = np.nan_to_num(C1_max) 
    max_val = np.max(C1_max)
    min_val = np.min(C1_max[C1_max > 0]) if np.any(C1_max > 0) else 0
    if max_val == min_val:
        # Handle case with uniform concentration or no plume
        thresholds = np.linspace(min_val, max_val + 1e-9, 7) # Add small epsilon to avoid single level
    else:
        # Define thresholds from min positive value to max value
        thresholds = np.linspace(min_val, max_val, 7) # Use 7 levels for 6 intervals

    # Ensure thresholds are unique and sorted
    thresholds = np.unique(thresholds)
    if len(thresholds) < 2: # Need at least 2 levels for contourf
        thresholds = np.array([min_val, max_val + 1e-9])

    cmap = mcolors.ListedColormap(['#edf8fb', '#b2e2e2', '#8cd2c3', '#66c2a4', '#238b45', '#005824'])
    # Adjust colormap length if number of levels is less than expected
    if len(thresholds) -1 < len(cmap.colors):
        cmap = mcolors.ListedColormap(cmap.colors[:len(thresholds)-1])

    cs = ax.contourf(local_x, local_y, C1_max, levels=thresholds, cmap=cmap, alpha=0.5, extend='neither') # Use extend='neither'
    cbar = fig.colorbar(cs, ticks=thresholds)
    cbar.set_label(r'$\mu g / m^{3}$')

    # Clip contours to the axes rectangle patch
    patch = plt.Rectangle((local_extent[0], local_extent[2]), local_extent[1]-local_extent[0], local_extent[3]-local_extent[2], transform=ax.transData)
    for collection in cs.collections:
        collection.set_clip_path(patch)

    # Mark the emission source at (0,0)
    ax.scatter(0, 0, color='red', s=80, edgecolor='k', zorder=5, label='Emission Source')

    # Set axes limits to match the basemap extent exactly
    ax.set_xlim(local_extent[0], local_extent[1])
    ax.set_ylim(local_extent[2], local_extent[3])
    ax.set_aspect('equal')  # Ensure square pixels

    ax.set_xlabel('x (m, local)')
    ax.set_ylabel('y (m, local)')
    ax.set_title('Dispersion Plume Overlaid on Local Street Map')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def generate_street_network_image_local(center_lat, center_lon, dist=2500, dxy=10, dpi=100):
    """
    Generate the street network image and return the image, emission source coordinates, crs, extent, and image shape.
    The image and the meshgrid for the plume will use the same extent and resolution.
    """
    # Step 1: Get the unprojected graph in lat/lon
    G = ox.graph_from_point((center_lat, center_lon), dist=dist, network_type='drive')
    # Step 2: Find nearest node in unprojected coordinates using lat/lon
    node_id = ox.nearest_nodes(G, center_lon, center_lat)
    # Step 3: Project the graph to EPSG:3857
    G_proj = ox.project_graph(G)
    crs_proj = G_proj.graph['crs']
    # Step 4: Get the emission source coordinates (projected)
    emission_source_x = G_proj.nodes[node_id]['x']
    emission_source_y = G_proj.nodes[node_id]['y']
    print("Emission source (projected):", emission_source_x, emission_source_y)

    # Use the passed-in dist as the display extent
    extent_range = dist
    # Calculate the number of pixels based on dist and dxy
    size_in_pixels = int((2 * extent_range) / dxy)
    figsize = (size_in_pixels / dpi, size_in_pixels / dpi)
    fig, ax = ox.plot_graph(
        G_proj, bgcolor='none', edge_color='blue', node_size=0,
        edge_linewidth=0.5, show=False, close=False, figsize=figsize
    )
    ax.set_xlim([emission_source_x - extent_range, emission_source_x + extent_range])
    ax.set_ylim([emission_source_y - extent_range, emission_source_y + extent_range])
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_array = np.array(fig.canvas.renderer._renderer)[..., :3]
    plt.close(fig)
    return image_array, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels

def run_dispersion_model_local(center_lon, center_lat, street_network_image,
                               emission_source_x, emission_source_y, flow_rate, H, T_stack, crs_proj, extent_range, size_in_pixels):
    """
    Run the plume dispersion model over a meshgrid defined over the same extent and shape as the map image.
    """
    # Create meshgrid in projected coordinates matching the basemap image
    x_range = np.linspace(emission_source_x - extent_range, emission_source_x + extent_range, size_in_pixels)
    y_range = np.linspace(emission_source_y - extent_range, emission_source_y + extent_range, size_in_pixels)
    x, y = np.meshgrid(x_range, y_range)
    local_x = x - emission_source_x
    local_y = y - emission_source_y

    # Test DataFrame for 24 hours of meteorological data
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

    fig_overlay = overlay_on_map(local_x, local_y, C1, street_network_image, extent_range)
    return fig_overlay

##################################
# Flask Routes & Interface
##################################

form_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Pollutant Dispersion Simulation</title>
</head>
<body>
  <h2>Enter Input Parameters</h2>
  <form method="post" action="/simulate">
    <label>Center Latitude:</label><br>
    <input type="text" name="latitude" value="44.4399"><br><br>
    <label>Center Longitude:</label><br>
    <input type="text" name="longitude" value="26.0550"><br><br>
    <label>Interest Radius (m):</label><br>
    <input type="text" name="radius" value="2500"><br><br>
    <h3>Emission Parameters</h3>
    <label>Flow Rate (m³/s):</label><br>
    <input type="text" name="flow_rate" value="5"><br><br>
    <label>Source Height (H in m):</label><br>
    <input type="text" name="H" value="25"><br><br>
    <label>Stack Gas Temperature (T_stack in °C):</label><br>
    <input type="text" name="T_stack" value="150"><br><br>
    <input type="submit" value="Run Simulation">
  </form>
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
        dxy = 10  # Use the same grid step as in the notebook
        
        # Generate the street network image and determine emission source in projected coordinates
        street_network_image, emission_source_x, emission_source_y, crs_proj, extent_range, size_in_pixels = generate_street_network_image_local(latitude, longitude, dist=radius, dxy=dxy)
        
        # Run the dispersion model and shift coordinates to a local coordinate system (emission source at (0, 0))
        fig_overlay = run_dispersion_model_local(longitude, latitude, street_network_image,
                                                   emission_source_x, emission_source_y, flow_rate, H, T_stack, crs_proj, extent_range, size_in_pixels)
        
        buf = io.BytesIO()
        fig_overlay.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig_overlay)
        return send_file(buf, mimetype='image/png', as_attachment=False, download_name='ModelDispersie.png')
    except Exception as e:
        return f"Error during simulation: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
