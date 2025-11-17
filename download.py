import os
import requests
from tqdm import tqdm
import subprocess
import rasterio
import numpy as np
import trimesh

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
OUTPUT_DIR = "usgs_lidar"
MERGED_TIF = "merged_dem.tif"
OUTPUT_STL = "terrain_model.stl"
RESOLUTION = 2.0   # meters per pixel
VERTICAL_EXAGGERATION = 1.5
BASE_THICKNESS = 20  # mm

# -----------------------------------------------------------------------------
# STEP 1. READ LIST OF FILES AND DOWNLOAD
# -----------------------------------------------------------------------------
with open("0_file_download_links.txt", "r") as f:
    urls = [line.strip() for line in f if line.strip().endswith(".laz")]

os.makedirs(OUTPUT_DIR, exist_ok=True)

for url in tqdm(urls, desc="Downloading LAZ files"):
    filename = os.path.join(OUTPUT_DIR, os.path.basename(url))
    if not os.path.exists(filename):
        r = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# -----------------------------------------------------------------------------
# STEP 2. CREATE PDAL PIPELINE TO MERGE INTO DEM (TIF)
# -----------------------------------------------------------------------------
# Generate PDAL pipeline JSON dynamically
pipeline = {
    "pipeline": [
        *[{"type": "readers.las", "filename": os.path.join(OUTPUT_DIR, os.path.basename(url))} for url in urls],
        {"type": "filters.range", "limits": "Classification[2:2]"},  # ground points only
        {
            "type": "writers.gdal",
            "filename": MERGED_TIF,
            "gdaldriver": "GTiff",
            "resolution": RESOLUTION,
            "output_type": "mean",
            "nodata": -9999
        }
    ]
}

import json
pipeline_path = "pipeline.json"
with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=2)

print("\nRunning PDAL to create DEM...")
subprocess.run(["pdal", "pipeline", pipeline_path], check=True)

# -----------------------------------------------------------------------------
# STEP 3. LOAD DEM AND GENERATE MESH
# -----------------------------------------------------------------------------
print("\nLoading DEM and generating mesh...")
with rasterio.open(MERGED_TIF) as src:
    elevation = src.read(1)
    elevation = np.where(elevation == src.nodata, np.nan, elevation)
    transform = src.transform

    # Downsample if very large
    max_dim = 1000
    if elevation.shape[0] > max_dim or elevation.shape[1] > max_dim:
        scale = min(max_dim / elevation.shape[0], max_dim / elevation.shape[1])
        elevation = elevation[::int(1/scale), ::int(1/scale)]

# Convert to 3D points
rows, cols = elevation.shape
x = np.arange(cols) * RESOLUTION
y = np.arange(rows) * RESOLUTION
X, Y = np.meshgrid(x, y)
Z = elevation * VERTICAL_EXAGGERATION

# Replace NaNs with minimum elevation (for continuity)
Z = np.nan_to_num(Z, nan=np.nanmin(Z))

# -----------------------------------------------------------------------------
# STEP 4. CREATE TRIANGULATED MESH WITH BASE
# -----------------------------------------------------------------------------
print("\nCreating STL mesh...")
vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        faces.append([idx, idx + 1, idx + cols])
        faces.append([idx + 1, idx + cols + 1, idx + cols])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Add a flat base (for 3D printing stability)
min_z = np.min(Z)
base = mesh.copy()
base.apply_translation([0, 0, -min_z + BASE_THICKNESS])
mesh = trimesh.util.concatenate([mesh, base.convex_hull])

# -----------------------------------------------------------------------------
# STEP 5. SAVE STL
# -----------------------------------------------------------------------------
mesh.export(OUTPUT_STL)
print(f"\n✅ STL saved as {OUTPUT_STL}")
