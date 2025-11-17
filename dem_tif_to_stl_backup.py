#!/usr/bin/env python3
"""
Download DEM TIFF files and convert to tiled 3D-printable STLs.
Processes 1m resolution elevation data from USGS.
Final model: 6ft × 6ft max, tiled into 1ft (304.8mm) pieces for Bambu H2D.
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.merge import merge as rasterio_merge
import trimesh
from tqdm import tqdm
import requests

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
TIF_LIST_FILE = "dem_tif_urls.txt"  # DEM TIF download list
TIF_DIR = "dem_tif_downloads"        # Folder for downloaded TIFs
MERGED_DEM = "dem_tif_merged.tif"
OUTPUT_DIR = "stl_tiles_dem"

# Printer settings (Bambu H2D)
TARGET_TILE_SIZE = 304.8      # mm (1 foot per tile)
MAX_HEIGHT = 300.0            # mm (max Z height)

# FORCE specific grid size
FORCE_TILES_X = 2   # Number of tiles horizontally
FORCE_TILES_Y = 1   # Number of tiles vertically
TARGET_MODEL_WIDTH = FORCE_TILES_X * TARGET_TILE_SIZE   # 609.6mm
TARGET_MODEL_HEIGHT = FORCE_TILES_Y * TARGET_TILE_SIZE  # 304.8mm

# Terrain settings
VERTICAL_EXAGGERATION = 3.0   # Increase for more dramatic terrain (was 1.5)
BASE_THICKNESS = 5.0          # mm base thickness
TILE_OVERLAP = 0.0            # mm overlap between tiles
MAX_DIM = 2000                # maximum rows/cols for meshing (per tile)

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
os.makedirs(TIF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("🗻 DEM TIFF to STL Converter")
print("=" * 60)
print(f"📂 TIF file list: {TIF_LIST_FILE}")
print(f"📦 Output directory: {OUTPUT_DIR}")
print(f"🎯 Target: {FORCE_TILES_X}×{FORCE_TILES_Y} grid = {TARGET_MODEL_WIDTH:.1f}mm × {TARGET_MODEL_HEIGHT:.1f}mm")
print(f"🖨️  Tiled into {TARGET_TILE_SIZE/304.8:.0f}ft pieces for Bambu H2D\n")

# -----------------------------------------------------------
# STEP 1. READ TIF FILE URLS
# -----------------------------------------------------------
SKIP_DOWNLOAD = False  # Set to True to skip download if files already exist

if not SKIP_DOWNLOAD:
    print("📋 Reading TIF file list...")
    with open(TIF_LIST_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip() and line.strip().startswith("http")]

    print(f"Found {len(urls)} TIF files to download\n")

    # -----------------------------------------------------------
    # STEP 2. DOWNLOAD TIF FILES
    # -----------------------------------------------------------
    print("📥 Downloading TIF files...")
    downloaded_files = []
    for url in tqdm(urls, desc="Downloading"):
        filename = os.path.basename(url)
        filepath = os.path.join(TIF_DIR, filename)
        downloaded_files.append(filepath)
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"  ✓ Already exists: {filename} ({file_size/1024/1024:.1f} MB)")
            continue
        
        try:
            print(f"\n  Downloading: {filename}")
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            block_size = 8192
            
            with open(filepath, "wb") as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"    {filename}") as pbar:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            file_size = os.path.getsize(filepath)
            print(f"  ✅ Downloaded: {filename} ({file_size/1024/1024:.1f} MB)")
            
        except Exception as e:
            print(f"\n  ⚠️  Failed to download {filename}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up partial download
                downloaded_files.remove(filepath)

    print(f"\n✅ Downloaded/verified {len(downloaded_files)} TIF files\n")
else:
    # Use already downloaded files
    print("📂 Using already downloaded TIF files...")
    downloaded_files = []
    if os.path.exists(TIF_DIR):
        for filename in sorted(os.listdir(TIF_DIR)):
            if filename.endswith((".tif", ".tiff")):
                downloaded_files.append(os.path.join(TIF_DIR, filename))
    
    if len(downloaded_files) == 0:
        print(f"❌ No TIF files found in {TIF_DIR}/")
        print("Set SKIP_DOWNLOAD = False to download files")
        sys.exit(1)
    
    print(f"✅ Found {len(downloaded_files)} TIF files\n")

# -----------------------------------------------------------
# STEP 3. PROCESS EACH TIF FILE SEPARATELY
# -----------------------------------------------------------
print(f"📄 Processing {len(downloaded_files)} TIF files separately...\n")

if len(downloaded_files) == 0:
    print("❌ No DEM files to process!")
    sys.exit(1)

dem_files_to_process = downloaded_files  # Process all files individually

# -----------------------------------------------------------
# STEP 4. PROCESS EACH DEM FILE
# -----------------------------------------------------------
for file_idx, dem_file in enumerate(dem_files_to_process):
    print("=" * 60)
    print(f"Processing file {file_idx + 1}/{len(dem_files_to_process)}: {os.path.basename(dem_file)}")
    print("=" * 60)
    
    # Create separate output directory for this file
    file_basename = os.path.splitext(os.path.basename(dem_file))[0]
    file_output_dir = os.path.join(OUTPUT_DIR, file_basename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    print("📊 Loading DEM and calculating scale...")
    with rasterio.open(dem_file) as src:
        elev = src.read(1)
        transform = src.transform
        nodata = src.nodata
        if nodata is not None:
            elev = np.where(elev == nodata, np.nan, elev)
        print(f"Loaded DEM: {elev.shape[1]} x {elev.shape[0]} pixels")
        print(f"Resolution: {abs(transform.a):.2f}m × {abs(transform.e):.2f}m per pixel")

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
        z_min = np.nanmin(elev_sample)
        z_max = np.nanmax(elev_sample)
    else:
        z_min = np.nanmin(elev)
        z_max = np.nanmax(elev)

    z_range_raw = z_max - z_min
    z_range = z_range_raw * VERTICAL_EXAGGERATION
    print(f"Elevation range: {z_min:.1f}m to {z_max:.1f}m (raw range: {z_range_raw:.1f}m)")
    print(f"With {VERTICAL_EXAGGERATION}x exaggeration: {z_range:.1f}m")

    # Check if terrain is very flat
    if z_range_raw < 10:
        print(f"⚠️  WARNING: Terrain has very little elevation variation ({z_range_raw:.1f}m)")
        print(f"   Consider increasing VERTICAL_EXAGGERATION (currently {VERTICAL_EXAGGERATION}x)")
    print()

    # FORCE scale to fit exactly into target grid size
    print(f"🎯 Forcing terrain to fit {FORCE_TILES_X}×{FORCE_TILES_Y} tile grid")
    print(f"   Target size: {TARGET_MODEL_WIDTH:.1f}mm × {TARGET_MODEL_HEIGHT:.1f}mm")
    print(f"   Real terrain: {extent_x:.1f}m × {extent_y:.1f}m\n")

    # Calculate XY scale to fit terrain into target dimensions
    scale_x = TARGET_MODEL_WIDTH / extent_x
    scale_y = TARGET_MODEL_HEIGHT / extent_y
    scale_xy = min(scale_x, scale_y)  # Use smaller scale to fit within both dimensions

    # Calculate Z scale INDEPENDENTLY - don't let XY scale crush the elevation!
    # We want vertical exaggeration relative to the XY scale
    scale_z_max = MAX_HEIGHT / z_range if z_range > 0 else 1.0
    scale_z = min(scale_z_max, scale_xy * 100)  # Allow Z to be up to 100x the XY scale

    print(f"XY scale: {scale_xy:.6f} ({1/scale_xy:.1f}:1 ratio)")
    print(f"Z scale:  {scale_z:.6f} (allows up to {MAX_HEIGHT:.0f}mm height)")
    print(f"Final model: {extent_x * scale_xy:.1f}mm × {extent_y * scale_xy:.1f}mm × {z_range * scale_z:.1f}mm")

    # Calculate final assembled dimensions using separate XY and Z scales
    final_width = extent_x * scale_xy
    final_height = extent_y * scale_xy
    final_z = z_range * scale_z

    print(f"📏 Final assembled model size: {final_width:.1f}mm × {final_height:.1f}mm × {final_z:.1f}mm")
    print(f"   ({final_width/304.8:.2f}ft × {final_height/304.8:.2f}ft × {final_z/25.4:.2f}in)")

    # FORCE tile grid to exact dimensions
    num_tiles_x = FORCE_TILES_X
    num_tiles_y = FORCE_TILES_Y
    effective_tile_size = TARGET_TILE_SIZE - TILE_OVERLAP

    print(f"📐 XY Scale: {scale_xy:.6f}, Z Scale: {scale_z:.6f}")
    print(f"🔲 Tile grid: {num_tiles_x} × {num_tiles_y} = {num_tiles_x * num_tiles_y} total tiles")
    print(f"🖨️  Each tile: {TARGET_TILE_SIZE:.1f}mm × {TARGET_TILE_SIZE:.1f}mm (1 foot)")

    # CRITICAL: Store global base elevation so all tiles sit at same height on board
    GLOBAL_BASE_Z = z_min * VERTICAL_EXAGGERATION * scale_z
    print(f"🏔️  Global base elevation: {z_min:.1f}m (scaled: {GLOBAL_BASE_Z:.1f}mm)")
    print(f"   All tiles will use this as 'sea level' - higher terrain = taller pieces\n")

    # -----------------------------------------------------------
    # STEP 5. GENERATE EACH TILE
    # -----------------------------------------------------------
    tile_count = 0
    for tile_y in range(num_tiles_y):
        for tile_x in range(num_tiles_x):
            tile_count += 1
            print(f"[{tile_count}/{num_tiles_x * num_tiles_y}] Generating tile ({tile_x}, {tile_y})...")
            
            # Calculate tile bounds in real-world coordinates (using XY scale)
            tile_x_start_world = x_min + tile_x * effective_tile_size / scale_xy
            tile_x_end_world = tile_x_start_world + (TARGET_TILE_SIZE / scale_xy)
            tile_y_start_world = y_max - (tile_y + 1) * TARGET_TILE_SIZE / scale_xy
            tile_y_end_world = tile_y_start_world + (TARGET_TILE_SIZE / scale_xy)
        
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
        
            # Apply vertical exaggeration and handle nodata
            tile_Z = np.nan_to_num(tile_elev, nan=z_min) * VERTICAL_EXAGGERATION
        
            tile_rows, tile_cols = tile_Z.shape
        
            # Downsample if tile is too large
            if max(tile_rows, tile_cols) > MAX_DIM:
            downsample = int(np.ceil(max(tile_rows, tile_cols) / MAX_DIM))
            tile_X = tile_X[::downsample, ::downsample]
            tile_Y = tile_Y[::downsample, ::downsample]
            tile_Z = tile_Z[::downsample, ::downsample]
            tile_rows, tile_cols = tile_Z.shape
            print(f"  📉 Downsampled to {tile_rows} × {tile_cols}")
        
            # Scale and center the tile - USE SEPARATE XY and Z scales!
            verts = np.column_stack((
            (tile_X.flatten() - tile_x_start_world) * scale_xy,
            (tile_Y.flatten() - tile_y_start_world) * scale_xy,
            (tile_Z.flatten() * scale_z) - GLOBAL_BASE_Z  # Use Z scale for elevation
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
            tile_min_z = np.min(verts[:, 2])
            tile_max_z = np.max(verts[:, 2])
            tile_height = tile_max_z + BASE_THICKNESS
            tile_relief = tile_max_z - tile_min_z
            print(f"  📏 Height: {tile_height:.1f}mm from board (relief: {tile_relief:.1f}mm)")
        
            # Add alignment features
            if ADD_ALIGNMENT_PINS:
            solid = add_alignment_features(
                solid, tile_x, tile_y, num_tiles_x, num_tiles_y,
                TARGET_TILE_SIZE, -BASE_THICKNESS
            )
        
            # Export
            output_file = os.path.join(file_output_dir, f"tile_{tile_x}_{tile_y}.stl")
            solid.export(output_file)
            print(f"  ✅ Saved: {output_file}")

    print(f"\n✅ File complete! Generated {tile_count} tiles in '{file_output_dir}/'\n")

print("=" * 60)
print("🎉 ALL FILES COMPLETE!")
print("=" * 60)
print(f"💡 Each TIF file generated {FORCE_TILES_X}×{FORCE_TILES_Y} tiles in separate folders:")
for dem_file in dem_files_to_process:
    file_basename = os.path.splitext(os.path.basename(dem_file))[0]
    file_output_dir = os.path.join(OUTPUT_DIR, file_basename)
    print(f"   📂 {file_output_dir}/")
print(f"\n   • Each tile is exactly 1 foot (304.8mm) square")
print(f"   • Terrain shown at {VERTICAL_EXAGGERATION}x vertical exaggeration 🗻")

