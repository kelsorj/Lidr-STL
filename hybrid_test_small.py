#!/usr/bin/env python3
"""
HYBRID TEST - Small Area Only
==============================
Tests terrain + buildings on a small 500m × 500m area
"""

import os
import numpy as np
import trimesh
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.affinity import translate, scale as shapely_scale
import laspy
import glob
from scipy.interpolate import griddata
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Test area: Small neighborhood near city center with hills
TEST_CENTER_LAT = 37.7749
TEST_CENTER_LON = -122.4194
TEST_SIZE_DEGREES = 0.005  # ~500m × 500m

# Target dimensions
TARGET_TILE_SIZE = 304.8     # 1 foot in mm
BASE_THICKNESS = 5.0
MIN_BUILDING_HEIGHT = 3.0
VERTICAL_SCALE = 1.0
TERRAIN_RESOLUTION = 2.0

# Output
OUTPUT_DIR = "hybrid_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("🧪 HYBRID TEST - Small Area")
print("=" * 60)
print(f"📍 Test area: {TEST_SIZE_DEGREES*111:.0f}m × {TEST_SIZE_DEGREES*111:.0f}m")
print(f"🔲 Tile size: {TARGET_TILE_SIZE:.1f}mm\n")

# Download OSM buildings for test area
print("📥 Downloading OSM buildings for test area...")
bbox = (
    TEST_CENTER_LON - TEST_SIZE_DEGREES,
    TEST_CENTER_LAT - TEST_SIZE_DEGREES,
    TEST_CENTER_LON + TEST_SIZE_DEGREES,
    TEST_CENTER_LAT + TEST_SIZE_DEGREES
)
buildings_gdf = ox.features_from_bbox(bbox=bbox, tags={'building': True})
buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
print(f"   ✅ Found {len(buildings_gdf)} buildings")

# Load LAZ files
print("\n📂 Loading LAZ files...")
laz_files = sorted(glob.glob("lidar_laz/*.laz"))
print(f"   ✅ Found {len(laz_files)} LAZ files")

# Determine LiDAR CRS and convert buildings
print("\n🗺️  Converting to LiDAR coordinate system...")
LIDAR_CRS = 'EPSG:7131'
buildings_gdf = buildings_gdf.to_crs(LIDAR_CRS)
print(f"   ✅ Buildings in {LIDAR_CRS}")

# Get test area bounds in LiDAR coordinates
bounds = buildings_gdf.total_bounds  # [minx, miny, maxx, maxy]
print(f"   Bounds: X={bounds[0]:.0f}-{bounds[2]:.0f}, Y={bounds[1]:.0f}-{bounds[3]:.0f}")

# Find relevant LAZ files
print("\n🔍 Finding relevant LAZ files...")
relevant_laz = []
for laz_file in laz_files:
    with laspy.open(laz_file) as laz:
        h = laz.header
        if (h.x_max >= bounds[0] and h.x_min <= bounds[2] and
            h.y_max >= bounds[1] and h.y_min <= bounds[3]):
            relevant_laz.append(laz_file)
            print(f"   ✅ {os.path.basename(laz_file)}")

if len(relevant_laz) == 0:
    print("\n❌ No LAZ files cover this area! Try a different location.")
    exit(1)

print(f"\n✅ Using {len(relevant_laz)} LAZ files")

# Extract building heights from LiDAR
print("\n🏗️  Extracting building heights from LiDAR...")
buildings_with_heights = []

for idx, building in tqdm(buildings_gdf.iterrows(), total=len(buildings_gdf), desc="Processing buildings"):
    geom = building.geometry
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    
    # Get points within building footprint
    all_z = []
    for laz_file in relevant_laz:
        try:
            with laspy.open(laz_file) as laz:
                las = laz.read()
                x, y, z = las.x, las.y, las.z
                
                b = geom.bounds
                mask = (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])
                if not mask.any():
                    continue
                
                points_in_bbox = np.column_stack((x[mask], y[mask]))
                z_in_bbox = z[mask]
                
                inside = np.array([geom.contains(Point(px, py)) for px, py in points_in_bbox])
                if inside.any():
                    all_z.extend(z_in_bbox[inside])
        except:
            continue
    
    if len(all_z) >= 3:
        z_arr = np.array(all_z)
        ground_z = np.percentile(z_arr, 5)
        roof_z = np.percentile(z_arr, 95)
        height = roof_z - ground_z
        
        if height > 1.0:
            building_copy = building.copy()
            building_copy['ground_elev_m'] = ground_z
            building_copy['roof_elev_m'] = roof_z
            building_copy['height_m'] = height
            buildings_with_heights.append(building_copy)

buildings_gdf = gpd.GeoDataFrame(buildings_with_heights)
print(f"   ✅ Extracted heights for {len(buildings_gdf)} buildings")

if len(buildings_gdf) == 0:
    print("   ❌ No buildings with heights found!")
    exit(1)

# Calculate scaling
print("\n📐 Calculating scaling...")
world_width_m = bounds[2] - bounds[0]
world_height_m = bounds[3] - bounds[1]
scale = TARGET_TILE_SIZE / max(world_width_m, world_height_m)
print(f"   Real size: {world_width_m:.1f}m × {world_height_m:.1f}m")
print(f"   Scale: {scale:.4f} mm/m")
print(f"   Model size: {world_width_m*scale:.1f}mm × {world_height_m*scale:.1f}mm")

# Get global base
GLOBAL_BASE_Z = buildings_gdf['ground_elev_m'].min() * scale * VERTICAL_SCALE
print(f"   Global base: {buildings_gdf['ground_elev_m'].min():.1f}m")

# Extract TERRAIN
print("\n🏔️  Extracting terrain from LiDAR...")
all_ground_x, all_ground_y, all_ground_z = [], [], []

for laz_file in tqdm(relevant_laz, desc="Reading LAZ files"):
    with laspy.open(laz_file) as laz:
        las = laz.read()
        
        classification = las.classification
        x, y, z = las.x, las.y, las.z
        
        mask = (
            (classification == 2) &  # Ground points
            (x >= bounds[0]) & (x <= bounds[2]) &
            (y >= bounds[1]) & (y <= bounds[3])
        )
        
        if mask.any():
            all_ground_x.extend(x[mask])
            all_ground_y.extend(y[mask])
            all_ground_z.extend(z[mask])

print(f"   ✅ Found {len(all_ground_x)} ground points")

# Create terrain grid
nx = int((bounds[2] - bounds[0]) / TERRAIN_RESOLUTION) + 1
ny = int((bounds[3] - bounds[1]) / TERRAIN_RESOLUTION) + 1
print(f"   Creating {nx}×{ny} terrain grid...")

grid_x = np.linspace(bounds[0], bounds[2], nx)
grid_y = np.linspace(bounds[1], bounds[3], ny)
X, Y = np.meshgrid(grid_x, grid_y)

points = np.column_stack((all_ground_x, all_ground_y))
values = np.array(all_ground_z)
Z = griddata(points, values, (X, Y), method='linear', fill_value=np.median(values))

rows, cols = Z.shape
print(f"   ✅ Terrain grid: {cols}×{rows}")
print(f"   Elevation range: {Z.min():.1f}m to {Z.max():.1f}m (Δ={Z.max()-Z.min():.1f}m)")

# Convert terrain to model coordinates
terrain_verts = []
for i in range(rows):
    for j in range(cols):
        x_local = (X[i, j] - bounds[0]) * scale
        y_local = (Y[i, j] - bounds[1]) * scale
        z_local = (Z[i, j] * scale * VERTICAL_SCALE) - GLOBAL_BASE_Z
        terrain_verts.append([x_local, y_local, z_local])

terrain_verts = np.array(terrain_verts)

# Create terrain faces
terrain_faces = []
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        terrain_faces.append([idx, idx + 1, idx + cols])
        terrain_faces.append([idx + 1, idx + cols + 1, idx + cols])

# Add base and walls
base_z = terrain_verts[:, 2].min() - BASE_THICKNESS

# Perimeter
perimeter_indices = []
perimeter_indices.extend([j for j in range(cols)])  # Bottom edge
perimeter_indices.extend([(i+1) * cols - 1 for i in range(rows)])  # Right edge
perimeter_indices.extend([((rows-1) * cols + j) for j in range(cols-1, -1, -1)])  # Top edge (reverse)
perimeter_indices.extend([i * cols for i in range(rows-1, -1, -1)])  # Left edge (reverse)

seen = set()
perimeter_indices = [x for x in perimeter_indices if not (x in seen or seen.add(x))]

all_verts = list(terrain_verts)
base_offset = len(all_verts)
for idx in perimeter_indices:
    v = terrain_verts[idx].copy()
    v[2] = base_z
    all_verts.append(v)

all_faces = terrain_faces.copy()
num_perim = len(perimeter_indices)
for i in range(num_perim):
    top1 = perimeter_indices[i]
    top2 = perimeter_indices[(i + 1) % num_perim]
    base1 = base_offset + i
    base2 = base_offset + ((i + 1) % num_perim)
    all_faces.append([top1, base1, top2])
    all_faces.append([base1, base2, top2])

base_indices = list(range(base_offset, base_offset + num_perim))
for i in range(1, num_perim - 1):
    all_faces.append([base_indices[0], base_indices[i+1], base_indices[i]])

terrain_mesh = trimesh.Trimesh(vertices=np.array(all_verts), faces=all_faces, process=False)
print(f"   ✅ Terrain mesh created")

# Create building meshes
print(f"\n🏢 Creating {len(buildings_gdf)} building meshes...")
building_meshes = []

for idx, building in tqdm(buildings_gdf.iterrows(), total=len(buildings_gdf), desc="Creating meshes"):
    geom = building.geometry
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    
    coords = list(geom.exterior.coords)[:-1]
    if len(coords) < 3:
        continue
    
    # Transform to local coordinates
    local_coords = [
        ((x - bounds[0]) * scale, (y - bounds[1]) * scale)
        for x, y in coords
    ]
    
    ground_z = (building['ground_elev_m'] * scale * VERTICAL_SCALE) - GLOBAL_BASE_Z
    roof_z = (building['roof_elev_m'] * scale * VERTICAL_SCALE) - GLOBAL_BASE_Z
    
    if roof_z - ground_z < MIN_BUILDING_HEIGHT:
        roof_z = ground_z + MIN_BUILDING_HEIGHT
    
    bottom_verts = [(x, y, ground_z) for x, y in local_coords]
    top_verts = [(x, y, roof_z) for x, y in local_coords]
    
    verts = bottom_verts + top_verts
    n = len(bottom_verts)
    
    faces = []
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
    for i in range(n):
        next_i = (i + 1) % n
        faces.append([i, next_i, n + i])
        faces.append([next_i, n + next_i, n + i])
    
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        building_meshes.append(mesh)
    except:
        continue

print(f"   ✅ Created {len(building_meshes)} building meshes")

# Combine everything
print("\n🔨 Combining terrain + buildings...")
final_mesh = trimesh.util.concatenate([terrain_mesh] + building_meshes)

# Export
output_file = os.path.join(OUTPUT_DIR, "hybrid_test.stl")
final_mesh.export(output_file)

print(f"\n🎉 Complete!")
print(f"✅ Saved: {output_file}")
print(f"📊 Stats:")
print(f"   • Terrain: {len(terrain_verts)} vertices")
print(f"   • Buildings: {len(building_meshes)}")
print(f"   • Total vertices: {len(final_mesh.vertices)}")
print(f"   • Total faces: {len(final_mesh.faces)}")
print(f"   • Elevation change: {Z.max()-Z.min():.1f}m ({(Z.max()-Z.min())*scale:.1f}mm in model)")
print(f"\n🏔️  San Francisco's hills are captured in the terrain!")


