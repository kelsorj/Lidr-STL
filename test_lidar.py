#!/usr/bin/env python3
"""
Quick test: Process 4 adjacent LAZ files to preview LiDAR quality.
Much faster than processing all 653 files!
"""

import os
import sys
import json
import subprocess
import numpy as np
import rasterio
import trimesh

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
LAZ_DIR = "lidar_laz"
TEST_DEM = "test_lidar.tif"
TEST_STL = "test_lidar.stl"

# Pick 4 adjacent LAZ files (based on coordinate grid in filename)
# Format: USGS_LPC_CA_SanFrancisco_B23_XXXXYYYY.laz
# Where XXXX and YYYY are grid coordinates

TEST_FILES = [
    # Downtown SF area - pick 4 adjacent tiles
    "USGS_LPC_CA_SanFrancisco_B23_04800250.laz",
    "USGS_LPC_CA_SanFrancisco_B23_04800255.laz",
    "USGS_LPC_CA_SanFrancisco_B23_04850250.laz",
    "USGS_LPC_CA_SanFrancisco_B23_04850255.laz"
]

# Processing settings
DEM_RESOLUTION = 0.5          # meters per pixel
VERTICAL_EXAGGERATION = 1.0   # 1:1 scale for buildings
BASE_THICKNESS = 5.0          # mm
MAX_PRINT_SIZE = 300.0        # mm (fit on printer bed)

print("=" * 60)
print("🧪 LiDAR Test - 4 Tile Preview")
print("=" * 60)
print(f"Testing with 4 adjacent LAZ files to preview quality\n")

# -----------------------------------------------------------
# STEP 1. CHECK FILES EXIST
# -----------------------------------------------------------
print("📂 Checking for LAZ files...")
test_paths = []
for filename in TEST_FILES:
    filepath = os.path.join(LAZ_DIR, filename)
    if os.path.exists(filepath):
        test_paths.append(filepath)
        print(f"  ✓ {filename}")
    else:
        print(f"  ✗ {filename} - NOT FOUND")

if len(test_paths) == 0:
    print(f"\n❌ No test files found in {LAZ_DIR}/")
    print("Make sure LAZ files are downloaded first")
    sys.exit(1)

print(f"\n✅ Found {len(test_paths)}/{len(TEST_FILES)} test files\n")

# -----------------------------------------------------------
# STEP 2. CREATE PDAL PIPELINE
# -----------------------------------------------------------
print("🔨 Creating PDAL pipeline...")

pipeline_stages = []

# Add test LAZ files
for filepath in test_paths:
    pipeline_stages.append({
        "type": "readers.las",
        "filename": filepath
    })

# Keep all points except noise
pipeline_stages.append({
    "type": "filters.range",
    "limits": "Classification![7:7]"  # Exclude noise class
})

# Write to GeoTIFF
pipeline_stages.append({
    "type": "writers.gdal",
    "filename": TEST_DEM,
    "gdaldriver": "GTiff",
    "output_type": "max",  # Max elevation to capture building tops
    "resolution": DEM_RESOLUTION,
    "nodata": -9999
})

pipeline = {"pipeline": pipeline_stages}
pipeline_path = "pipeline_test.json"

with open(pipeline_path, "w") as f:
    json.dump(pipeline, f, indent=2)

print(f"Running PDAL on {len(test_paths)} files...")
print("(This should take ~30-60 seconds)\n")

try:
    result = subprocess.run(["pdal", "pipeline", pipeline_path], 
                          check=True, capture_output=True, text=True)
    print("✅ PDAL processing complete!\n")
except subprocess.CalledProcessError as e:
    print(f"❌ PDAL failed: {e}")
    print(f"stderr: {e.stderr}")
    sys.exit(1)
except FileNotFoundError:
    print("❌ PDAL not found! Install with: brew install pdal")
    sys.exit(1)

# -----------------------------------------------------------
# STEP 3. LOAD DEM AND CREATE MESH
# -----------------------------------------------------------
print("📊 Loading DEM...")
with rasterio.open(TEST_DEM) as src:
    elev = src.read(1)
    transform = src.transform
    nodata = src.nodata
    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)
    print(f"DEM size: {elev.shape[1]} x {elev.shape[0]} pixels")

rows, cols = elev.shape

# Calculate extent
x_min = transform.c
x_max = transform.c + cols * transform.a
y_min = transform.f + rows * transform.e
y_max = transform.f

extent_x = abs(x_max - x_min)
extent_y = abs(y_max - y_min)
print(f"Area: {extent_x:.1f}m × {extent_y:.1f}m")

# Elevation stats
z_min = np.nanmin(elev) * VERTICAL_EXAGGERATION
z_max = np.nanmax(elev) * VERTICAL_EXAGGERATION
z_range = z_max - z_min
print(f"Elevation range: {z_range:.1f}m")

# Calculate scale to fit on printer
scale_xy = MAX_PRINT_SIZE / max(extent_x, extent_y)
scale_z = MAX_PRINT_SIZE / z_range if z_range > 0 else 1.0
scale = min(scale_xy, scale_z)

final_x = extent_x * scale
final_y = extent_y * scale
final_z = z_range * scale

print(f"\nScaled model: {final_x:.1f}mm × {final_y:.1f}mm × {final_z:.1f}mm")
print(f"Scale: {scale:.6f} (meters → mm)\n")

# -----------------------------------------------------------
# STEP 4. CREATE 3D MESH
# -----------------------------------------------------------
print("🔨 Creating 3D mesh...")

# Create coordinate grids
x = np.arange(cols) * transform.a + transform.c
y = np.arange(rows) * transform.e + transform.f
X, Y = np.meshgrid(x, y)

# Apply vertical exaggeration
Z = np.nan_to_num(elev, nan=z_min / VERTICAL_EXAGGERATION) * VERTICAL_EXAGGERATION

# Downsample if too large
MAX_DIM = 2000
if max(rows, cols) > MAX_DIM:
    downsample = int(np.ceil(max(rows, cols) / MAX_DIM))
    X = X[::downsample, ::downsample]
    Y = Y[::downsample, ::downsample]
    Z = Z[::downsample, ::downsample]
    rows, cols = Z.shape
    print(f"Downsampled to {rows} × {cols} for performance")

# IMPORTANT: Use GLOBAL minimum elevation as the base
# This ensures all tiles sit at the same "sea level" on a flat board
# Higher areas = taller pieces, lower areas = shorter pieces
GLOBAL_BASE_Z = z_min * scale  # Use the minimum elevation as global base

print(f"Global base elevation: {z_min:.1f}m (scaled: {GLOBAL_BASE_Z:.1f}mm)")

# Scale vertices - use GLOBAL base, not local minimum
verts = np.column_stack((
    (X.flatten() - x_min) * scale,
    (Y.flatten() - y_min) * scale,
    (Z.flatten() * scale) - GLOBAL_BASE_Z  # Shift so lowest point touches z=0
))

# Create triangulated mesh
print("Triangulating...")
faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        faces.append([idx, idx + 1, idx + cols])
        faces.append([idx + 1, idx + cols + 1, idx + cols])

mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# Create a SOLID model by adding perimeter walls and bottom
print("Creating solid model with walls...")

# Get perimeter vertices (edges of the grid)
perimeter_top_indices = []

# Top edge (row 0)
for j in range(cols):
    perimeter_top_indices.append(j)

# Right edge (last column)
for i in range(1, rows):
    perimeter_top_indices.append(i * cols + (cols - 1))

# Bottom edge (last row, right to left)
for j in range(cols - 2, -1, -1):
    perimeter_top_indices.append((rows - 1) * cols + j)

# Left edge (bottom to top)
for i in range(rows - 2, 0, -1):
    perimeter_top_indices.append(i * cols)

# Create new vertices: original terrain + perimeter at base level
all_verts = list(verts)
base_offset = len(all_verts)

# Add base-level versions of perimeter vertices
for idx in perimeter_top_indices:
    v = verts[idx].copy()
    v[2] = -BASE_THICKNESS  # Set to base level
    all_verts.append(v)

# Create wall faces connecting perimeter top to perimeter base
wall_faces = list(faces)
num_perim = len(perimeter_top_indices)

for i in range(num_perim):
    top1 = perimeter_top_indices[i]
    top2 = perimeter_top_indices[(i + 1) % num_perim]
    base1 = base_offset + i
    base2 = base_offset + ((i + 1) % num_perim)
    
    # Two triangles for this wall segment
    wall_faces.append([top1, base1, top2])
    wall_faces.append([base1, base2, top2])

# Create bottom face (triangulate the base perimeter)
base_indices = list(range(base_offset, base_offset + num_perim))
# Simple fan triangulation from first vertex
for i in range(1, num_perim - 1):
    wall_faces.append([base_indices[0], base_indices[i+1], base_indices[i]])

all_verts = np.array(all_verts)
solid = trimesh.Trimesh(vertices=all_verts, faces=wall_faces, process=False)

# Report actual print height
actual_max_z = np.max(verts[:, 2])
print(f"Print height: {actual_max_z + BASE_THICKNESS:.1f}mm (from board surface)")
print(f"Model is now completely solid (no hollow spaces)")

# -----------------------------------------------------------
# STEP 5. EXPORT STL
# -----------------------------------------------------------
print(f"💾 Exporting STL...")
solid.export(TEST_STL)

print(f"\n🎉 Success! Test STL created:")
print(f"   📦 File: {TEST_STL}")
print(f"   📏 Size: {final_x:.1f}mm × {final_y:.1f}mm × {final_z:.1f}mm")
print(f"   🏙️  Covers: {extent_x:.0f}m × {extent_y:.0f}m of terrain")
print(f"\n💡 Open in your slicer to preview!")
print(f"   If quality looks good, run: python lidar_to_stl.py")
print(f"   (That will process all 653 files for the full 6ft×6ft model)")

