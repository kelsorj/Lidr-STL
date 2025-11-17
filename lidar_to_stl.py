#!/usr/bin/env python3
"""
Download LiDAR LAZ files and convert to tiled 3D-printable STLs.
Includes buildings, trees, and all above-ground features.
Final model: 6ft × 6ft max, tiled into 1ft (304.8mm) pieces for Bambu H2D.
"""

import os
import sys
import json
import subprocess
import numpy as np
import rasterio
import trimesh
from tqdm import tqdm
import requests

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
LAZ_LIST_FILE = "0_file_download_lidr.txt"  # Note: you have a typo in filename
LAZ_DIR = "lidar_laz"
MERGED_DEM = "lidar_merged.tif"
OUTPUT_DIR = "stl_tiles_lidar"

# Printer settings (Bambu H2D)
TARGET_TILE_SIZE = 304.8      # mm (1 foot per tile)
MAX_HEIGHT = 300.0            # mm (max Z height)

# Final assembled model size constraint
MAX_ASSEMBLED_SIZE = 1828.8   # mm (6 feet - total final model limit)

# Terrain settings
VERTICAL_EXAGGERATION = 1.0   # No exaggeration for buildings (1:1 scale)
BASE_THICKNESS = 5.0          # mm base thickness
TILE_OVERLAP = 0.0            # mm overlap between tiles
MAX_DIM = 1500                # maximum rows/cols for meshing (per tile)

# LiDAR processing
DEM_RESOLUTION = 0.5          # meters per pixel (0.5m = higher detail for buildings)
USE_ALL_POINTS = True         # Include buildings, trees, etc. (not just ground)
BATCH_SIZE = 50               # Process LAZ files in batches (lower if running out of memory)

# Alignment features
ADD_ALIGNMENT_PINS = False  # Set to True to add pins/holes between tiles
PIN_DIAMETER = 4.0
PIN_HEIGHT = 8.0
PIN_CLEARANCE = 0.2
# -----------------------------------------------------------

def create_alignment_pin(diameter, height):
    """Create a cylindrical alignment pin."""
    pin = trimesh.creation.cylinder(radius=diameter/2, height=height, sections=16)
    return pin

def create_alignment_hole(diameter, height, clearance=0.2):
    """Create a hole for alignment pin (slightly larger)."""
    hole = trimesh.creation.cylinder(radius=(diameter+clearance)/2, height=height*1.5, sections=16)
    return hole

def add_alignment_features(mesh, tile_x, tile_y, num_tiles_x, num_tiles_y, tile_size, base_z):
    """Add alignment pins on right/bottom edges, holes on left/top edges."""
    features = []
    pin_positions = [0.25, 0.75]
    
    # Right edge pins (if not last column)
    if tile_x < num_tiles_x - 1:
        for pos in pin_positions:
            pin = create_alignment_pin(PIN_DIAMETER, PIN_HEIGHT)
            x = tile_size
            y = tile_size * pos
            z = base_z + BASE_THICKNESS
            pin.apply_translation([x, y, z])
            features.append(pin)
    
    # Bottom edge pins (if not last row)
    if tile_y < num_tiles_y - 1:
        for pos in pin_positions:
            pin = create_alignment_pin(PIN_DIAMETER, PIN_HEIGHT)
            x = tile_size * pos
            y = tile_size
            z = base_z + BASE_THICKNESS
            pin.apply_translation([x, y, z])
            features.append(pin)
    
    # Left edge holes (if not first column)
    if tile_x > 0:
        for pos in pin_positions:
            hole = create_alignment_hole(PIN_DIAMETER, BASE_THICKNESS, PIN_CLEARANCE)
            x = 0
            y = tile_size * pos
            z = base_z + BASE_THICKNESS/2
            hole.apply_translation([x, y, z])
            features.append(hole)
    
    # Top edge holes (if not first row)
    if tile_y > 0:
        for pos in pin_positions:
            hole = create_alignment_hole(PIN_DIAMETER, BASE_THICKNESS, PIN_CLEARANCE)
            x = tile_size * pos
            y = 0
            z = base_z + BASE_THICKNESS/2
            hole.apply_translation([x, y, z])
            features.append(hole)
    
    if features:
        mesh = trimesh.util.concatenate([mesh] + features)
    
    return mesh

# -----------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------
os.makedirs(LAZ_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("🏙️  LiDAR to STL Converter")
print("=" * 60)
print(f"📂 LAZ file list: {LAZ_LIST_FILE}")
print(f"📦 Output directory: {OUTPUT_DIR}")
print(f"🎯 Goal: Entire model ≤ {MAX_ASSEMBLED_SIZE/304.8:.0f}ft × {MAX_ASSEMBLED_SIZE/304.8:.0f}ft")
print(f"🖨️  Tiled into {TARGET_TILE_SIZE/304.8:.0f}ft pieces for Bambu H2D\n")

# -----------------------------------------------------------
# STEP 1. READ LAZ FILE URLS
# -----------------------------------------------------------
SKIP_DOWNLOAD = True  # Set to False to re-download files

if not SKIP_DOWNLOAD:
    print("📋 Reading LAZ file list...")
    with open(LAZ_LIST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip().endswith(".laz")]

    print(f"Found {len(urls)} LAZ files to download\n")

    # -----------------------------------------------------------
    # STEP 2. DOWNLOAD LAZ FILES
    # -----------------------------------------------------------
    print("📥 Downloading LAZ files...")
    downloaded_files = []
    for url in tqdm(urls, desc="Downloading"):
        filename = os.path.basename(url)
        filepath = os.path.join(LAZ_DIR, filename)
        downloaded_files.append(filepath)
        
        if os.path.exists(filepath):
            continue  # Skip if already downloaded
        
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"\n⚠️  Failed to download {filename}: {e}")

    print(f"✅ Downloaded/verified {len(downloaded_files)} LAZ files\n")
else:
    # Use already downloaded files
    print("📂 Using already downloaded LAZ files...")
    downloaded_files = []
    if os.path.exists(LAZ_DIR):
        for filename in sorted(os.listdir(LAZ_DIR)):
            if filename.endswith(".laz"):
                downloaded_files.append(os.path.join(LAZ_DIR, filename))
    
    if len(downloaded_files) == 0:
        print(f"❌ No LAZ files found in {LAZ_DIR}/")
        print("Set SKIP_DOWNLOAD = False to download files")
        sys.exit(1)
    
    print(f"✅ Found {len(downloaded_files)} LAZ files\n")
# -----------------------------------------------------------
# STEP 3. CREATE PDAL PIPELINE TO MERGE LAZ -> DEM (IN BATCHES)
# -----------------------------------------------------------
print("🔧 Processing LAZ files in batches (to avoid resource limits)...")

# Process in batches to avoid overwhelming system resources
batch_dems = []

num_batches = int(np.ceil(len(downloaded_files) / BATCH_SIZE))
print(f"📦 Processing {len(downloaded_files)} files in {num_batches} batches\n")

for batch_idx in range(num_batches):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min((batch_idx + 1) * BATCH_SIZE, len(downloaded_files))
    batch_files = downloaded_files[start_idx:end_idx]
    
    batch_dem = f"lidar_batch_{batch_idx}.tif"
    batch_dems.append(batch_dem)
    
    if os.path.exists(batch_dem):
        print(f"[Batch {batch_idx+1}/{num_batches}] ✓ Already processed: {batch_dem}")
        continue
    
    print(f"[Batch {batch_idx+1}/{num_batches}] Processing {len(batch_files)} files...")
    
    # Build PDAL pipeline for this batch
    pipeline_stages = []
    
    # Add LAZ files for this batch
    for filepath in batch_files:
        pipeline_stages.append({
            "type": "readers.las",
            "filename": filepath
        })
    
    # Filter points
    if not USE_ALL_POINTS:
        pipeline_stages.append({
            "type": "filters.range",
            "limits": "Classification[2:2]"
        })
    else:
        pipeline_stages.append({
            "type": "filters.range",
            "limits": "Classification![7:7]"
        })
    
    # Write to GeoTIFF
    pipeline_stages.append({
        "type": "writers.gdal",
        "filename": batch_dem,
        "gdaldriver": "GTiff",
        "output_type": "max",
        "resolution": DEM_RESOLUTION,
        "nodata": -9999
    })
    
    pipeline = {"pipeline": pipeline_stages}
    pipeline_path = f"pipeline_batch_{batch_idx}.json"
    
    with open(pipeline_path, "w") as f:
        json.dump(pipeline, f, indent=2)
    
    try:
        subprocess.run(["pdal", "pipeline", pipeline_path], 
                      check=True, capture_output=True, text=True)
        print(f"  ✅ Created: {batch_dem}")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Batch {batch_idx+1} failed: {e}")
        print(f"  stderr: {e.stderr}")
        print("  Continuing with other batches...")
        continue
    except FileNotFoundError:
        print("\n❌ PDAL not found!")
        print("Install with: brew install pdal")
        sys.exit(1)

# -----------------------------------------------------------
# STEP 4. MERGE BATCH DEMs INTO SINGLE DEM
# -----------------------------------------------------------
print(f"\n🔗 Merging {len(batch_dems)} batch DEMs into final DEM...")

try:
    from rasterio.merge import merge as rasterio_merge
    
    # Open all batch DEMs
    src_files = []
    for dem_path in batch_dems:
        if os.path.exists(dem_path):
            src_files.append(rasterio.open(dem_path))
    
    if len(src_files) == 0:
        print("❌ No batch DEMs were created successfully!")
        sys.exit(1)
    
    print(f"Merging {len(src_files)} batch DEMs...")
    mosaic, out_transform = rasterio_merge(src_files)
    
    # Write merged DEM
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })
    
    with rasterio.open(MERGED_DEM, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close all source files
    for src in src_files:
        src.close()
    
    print(f"✅ Merged DEM saved: {MERGED_DEM}\n")
    
except Exception as e:
    print(f"❌ Merge failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------
# STEP 5. LOAD DEM AND CALCULATE SCALING
# -----------------------------------------------------------
print("📊 Loading DEM and calculating scale...")
with rasterio.open(MERGED_DEM) as src:
    elev = src.read(1)
    transform = src.transform
    nodata = src.nodata
    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)
    print(f"Loaded DEM: {elev.shape[1]} x {elev.shape[0]} pixels")
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)

rows, cols = elev.shape

# Calculate real-world extent
x_min = transform.c
x_max = transform.c + cols * transform.a
y_min = transform.f + rows * transform.e
y_max = transform.f

extent_x = abs(x_max - x_min)
extent_y = abs(y_max - y_min)
print(f"Real-world extent: {extent_x:.1f}m × {extent_y:.1f}m")

# Calculate elevation range
if rows * cols > 100_000_000:
    print("⚡ Large dataset detected, sampling elevation range...")
    step = max(1, int(np.sqrt(rows * cols / 10_000_000)))
    elev_sample = elev[::step, ::step]
    z_min = np.nanmin(elev_sample) * VERTICAL_EXAGGERATION
    z_max = np.nanmax(elev_sample) * VERTICAL_EXAGGERATION
else:
    z_min = np.nanmin(elev) * VERTICAL_EXAGGERATION
    z_max = np.nanmax(elev) * VERTICAL_EXAGGERATION

z_range = z_max - z_min
print(f"Elevation range: {z_range:.1f}m")

# Calculate scale to fit ENTIRE model within MAX_ASSEMBLED_SIZE
scale_xy = MAX_ASSEMBLED_SIZE / max(extent_x, extent_y)
scale_z = MAX_HEIGHT / z_range if z_range > 0 else 1.0
scale = min(scale_xy, scale_z)

# Calculate final assembled dimensions
final_width = extent_x * scale
final_height = extent_y * scale
final_z = z_range * scale

print(f"\n📏 Final assembled model size: {final_width:.1f}mm × {final_height:.1f}mm × {final_z:.1f}mm")
print(f"   ({final_width/304.8:.2f}ft × {final_height/304.8:.2f}ft × {final_z/25.4:.2f}in)")

# Calculate tile grid
effective_tile_size = TARGET_TILE_SIZE - TILE_OVERLAP
num_tiles_x = int(np.ceil(final_width / effective_tile_size))
num_tiles_y = int(np.ceil(final_height / effective_tile_size))

print(f"📐 Scale factor: {scale:.6f} (real-world meters → mm)")
print(f"🔲 Tile grid: {num_tiles_x} × {num_tiles_y} = {num_tiles_x * num_tiles_y} total tiles")
print(f"🖨️  Each tile: {TARGET_TILE_SIZE:.1f}mm × {TARGET_TILE_SIZE:.1f}mm (1 foot)")

# CRITICAL: Store global base elevation so all tiles sit at same height on board
GLOBAL_BASE_Z = z_min * scale
print(f"🏔️  Global base elevation: {z_min:.1f}m (scaled: {GLOBAL_BASE_Z:.1f}mm)")
print(f"   All tiles will use this as 'sea level' - higher terrain = taller pieces\n")

# -----------------------------------------------------------
# STEP 6. GENERATE EACH TILE
# -----------------------------------------------------------
tile_count = 0
for tile_y in range(num_tiles_y):
    for tile_x in range(num_tiles_x):
        tile_count += 1
        print(f"[{tile_count}/{num_tiles_x * num_tiles_y}] Generating tile ({tile_x}, {tile_y})...")
        
        # Calculate tile bounds in real-world coordinates
        tile_x_start_world = x_min + tile_x * effective_tile_size / scale
        tile_x_end_world = tile_x_start_world + (TARGET_TILE_SIZE / scale)
        tile_y_start_world = y_max - (tile_y + 1) * TARGET_TILE_SIZE / scale
        tile_y_end_world = tile_y_start_world + (TARGET_TILE_SIZE / scale)
        
        # Convert to pixel indices
        col_start = int((tile_x_start_world - x_min) / abs(transform.a))
        col_end = int((tile_x_end_world - x_min) / abs(transform.a))
        row_start = int((y_max - tile_y_end_world) / abs(transform.e))
        row_end = int((y_max - tile_y_start_world) / abs(transform.e))
        
        # Clamp to array bounds
        col_start = max(0, min(cols, col_start))
        col_end = max(0, min(cols, col_end))
        row_start = max(0, min(rows, row_start))
        row_end = max(0, min(rows, row_end))
        
        if col_end <= col_start or row_end <= row_start:
            print(f"  ⚠️  Skipping empty tile")
            continue
        
        # Extract tile elevation data
        tile_elev = elev[row_start:row_end, col_start:col_end].copy()
        
        # Create coordinate grids for this tile
        tile_cols = col_end - col_start
        tile_rows = row_end - row_start
        tile_x_coords = x_min + (np.arange(col_start, col_end) * transform.a)
        tile_y_coords = y_max + (np.arange(row_start, row_end) * transform.e)
        tile_X, tile_Y = np.meshgrid(tile_x_coords, tile_y_coords)
        
        # Apply vertical exaggeration
        tile_Z = np.nan_to_num(tile_elev, nan=z_min / VERTICAL_EXAGGERATION) * VERTICAL_EXAGGERATION
        
        tile_rows, tile_cols = tile_Z.shape
        
        # Downsample if tile is too large
        if max(tile_rows, tile_cols) > MAX_DIM:
            downsample = int(np.ceil(max(tile_rows, tile_cols) / MAX_DIM))
            tile_X = tile_X[::downsample, ::downsample]
            tile_Y = tile_Y[::downsample, ::downsample]
            tile_Z = tile_Z[::downsample, ::downsample]
            tile_rows, tile_cols = tile_Z.shape
            print(f"  📉 Downsampled to {tile_rows} × {tile_cols}")
        
        # Scale and center the tile - USE GLOBAL BASE for consistent heights!
        verts = np.column_stack((
            (tile_X.flatten() - tile_x_start_world) * scale,
            (tile_Y.flatten() - tile_y_start_world) * scale,
            (tile_Z.flatten() * scale) - GLOBAL_BASE_Z  # Subtract global base so z=0 is "sea level"
        ))
        
        # Create triangulated mesh
        faces = []
        for i in range(tile_rows - 1):
            for j in range(tile_cols - 1):
                idx = i * tile_cols + j
                faces.append([idx, idx + 1, idx + tile_cols])
                faces.append([idx + 1, idx + tile_cols + 1, idx + tile_cols])
        
        if len(faces) == 0:
            print(f"  ⚠️  No faces generated, skipping")
            continue
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        
        # Create SOLID model by adding perimeter walls and bottom
        # Get perimeter vertices (edges of the grid)
        perimeter_top_indices = []
        
        # Top edge (row 0)
        for j in range(tile_cols):
            perimeter_top_indices.append(j)
        
        # Right edge (last column)
        for i in range(1, tile_rows):
            perimeter_top_indices.append(i * tile_cols + (tile_cols - 1))
        
        # Bottom edge (last row, right to left)
        for j in range(tile_cols - 2, -1, -1):
            perimeter_top_indices.append((tile_rows - 1) * tile_cols + j)
        
        # Left edge (bottom to top)
        for i in range(tile_rows - 2, 0, -1):
            perimeter_top_indices.append(i * tile_cols)
        
        # Create vertices: original terrain + perimeter at base level
        all_verts = list(verts)
        base_offset = len(all_verts)
        
        for idx in perimeter_top_indices:
            v = verts[idx].copy()
            v[2] = -BASE_THICKNESS  # Set to base level
            all_verts.append(v)
        
        # Create wall faces connecting perimeter top to base
        wall_faces = list(faces)
        num_perim = len(perimeter_top_indices)
        
        for i in range(num_perim):
            top1 = perimeter_top_indices[i]
            top2 = perimeter_top_indices[(i + 1) % num_perim]
            base1 = base_offset + i
            base2 = base_offset + ((i + 1) % num_perim)
            
            wall_faces.append([top1, base1, top2])
            wall_faces.append([base1, base2, top2])
        
        # Create bottom face
        base_indices = list(range(base_offset, base_offset + num_perim))
        for i in range(1, num_perim - 1):
            wall_faces.append([base_indices[0], base_indices[i+1], base_indices[i]])
        
        all_verts = np.array(all_verts)
        solid = trimesh.Trimesh(vertices=all_verts, faces=wall_faces, process=False)
        
        # Report this tile's height from board surface
        tile_max_z = np.max(verts[:, 2])
        tile_height = tile_max_z + BASE_THICKNESS
        print(f"  📏 Height: {tile_height:.1f}mm from board (solid)")
        
        # Add alignment features
        if ADD_ALIGNMENT_PINS:
            solid = add_alignment_features(
                solid, tile_x, tile_y, num_tiles_x, num_tiles_y,
                TARGET_TILE_SIZE, -BASE_THICKNESS
            )
        
        # Export
        output_file = os.path.join(OUTPUT_DIR, f"lidar_tile_{tile_x}_{tile_y}.stl")
        solid.export(output_file)
        print(f"  ✅ Saved: {output_file}")

print(f"\n🎉 Complete! Generated {tile_count} tiles in '{OUTPUT_DIR}/'")
print(f"💡 Assembly instructions:")
print(f"   • Print all {tile_count} tiles")
print(f"   • Arrange in a {num_tiles_x}×{num_tiles_y} grid (left-to-right, top-to-bottom)")
print(f"   • Final model: {final_width:.1f}mm × {final_height:.1f}mm ({final_width/304.8:.1f}ft × {final_height/304.8:.1f}ft)")
print(f"   • Each tile is exactly 1 foot (304.8mm) square")
print(f"   • Buildings and structures are included at 1:1 scale! 🏙️")

