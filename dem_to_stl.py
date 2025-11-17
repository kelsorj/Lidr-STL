import os
import requests
from tqdm import tqdm
import rasterio
from rasterio.merge import merge
import numpy as np
import trimesh

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
URL_LIST_FILE = "0_file_download_links.txt"
OUTPUT_DIR = "dem_tiles"
MERGED_TIF = "merged_dem.tif"
OUTPUT_STL = "terrain_model.stl"

VERTICAL_EXAGGERATION = 1.8   # exaggerate terrain height
BASE_THICKNESS = 20.0         # mm or same unit as Z scale
MAX_DIM = 1200                # limit grid size for memory efficiency
# -----------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# STEP 1. READ LINKS AND DOWNLOAD DEM FILES
# -----------------------------------------------------------
with open(URL_LIST_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip().endswith(".tif")]

for url in tqdm(urls, desc="Downloading DEMs"):
    fname = os.path.basename(url)
    outpath = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(outpath):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(outpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# -----------------------------------------------------------
# STEP 2. MERGE THE DEM FILES INTO A SINGLE MOSAIC
# -----------------------------------------------------------
srcs = [rasterio.open(os.path.join(OUTPUT_DIR, os.path.basename(url))) for url in urls]
mosaic, transform = merge(srcs)
meta = srcs[0].meta.copy()
meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform,
    "count": 1
})
with rasterio.open(MERGED_TIF, "w", **meta) as dst:
    dst.write(mosaic[0], 1)

for src in srcs:
    src.close()

# -----------------------------------------------------------
# STEP 3. LOAD THE MERGED DEM AND CREATE A MESH
# -----------------------------------------------------------
with rasterio.open(MERGED_TIF) as src:
    elev = src.read(1)
    elev = np.where(elev == src.nodata, np.nan, elev)
    transform = src.transform

# Downsample if very large
rows, cols = elev.shape
if max(rows, cols) > MAX_DIM:
    scale = int(np.ceil(max(rows, cols) / MAX_DIM))
    elev = elev[::scale, ::scale]
    rows, cols = elev.shape

# Coordinate grids
x = np.arange(cols) * transform.a + transform.c
y = np.arange(rows) * transform.e + transform.f
X, Y = np.meshgrid(x, y)

# Elevation (apply vertical exaggeration)
Z = np.nan_to_num(elev, nan=np.nanmin(elev)) * VERTICAL_EXAGGERATION

# -----------------------------------------------------------
# STEP 4. TRIANGULATE AND ADD BASE
# -----------------------------------------------------------
verts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        faces.append([idx, idx + 1, idx + cols])
        faces.append([idx + 1, idx + cols + 1, idx + cols])

mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Add a flat base
min_z = np.min(Z)
base = mesh.copy()
base.apply_translation([0, 0, -min_z - BASE_THICKNESS])
mesh = trimesh.util.concatenate([mesh, base.convex_hull])

# -----------------------------------------------------------
# STEP 5. EXPORT TO STL
# -----------------------------------------------------------
mesh.export(OUTPUT_STL)
print(f"✅ STL saved as {OUTPUT_STL}")
