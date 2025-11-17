#!/usr/bin/env python3
"""
Convert a DEM GeoTIFF into multiple tiled 3D-printable STLs for large format printers.
Splits terrain into tiles that fit your printer bed, with alignment pins for assembly.
Optimized for Bambu H2D (320x320x320mm build volume).
"""

import os
import sys
import numpy as np
import rasterio
import trimesh
from tqdm import tqdm

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
INPUT_TIF = "reprojected.tif"   # Path to your DEM
OUTPUT_DIR = "stl_tiles"        # Directory for output STL files

# Printer settings (Bambu H2D)
TARGET_TILE_SIZE = 304.8      # mm (1 foot per tile - fits on 320mm bed with margin)
MAX_HEIGHT = 300.0            # mm (max Z height, leaving margin)

# Final assembled model size constraint
MAX_ASSEMBLED_SIZE = 1828.8   # mm (6 feet - total final model limit)

# Terrain settings
VERTICAL_EXAGGERATION = 1.8   # exaggerate terrain relief
BASE_THICKNESS = 5.0          # mm base thickness
TILE_OVERLAP = 0.0            # mm overlap between tiles (0 for exact 1ft grid)
MAX_DIM = 1500                # maximum rows/cols for meshing (per tile)

# Alignment features (optional - helps with assembly)
ADD_ALIGNMENT_PINS = False    # Set to True to add pins/holes for tile registration
PIN_DIAMETER = 4.0            # mm
PIN_HEIGHT = 8.0              # mm (extends into adjacent tile base)
PIN_CLEARANCE = 0.2           # mm (holes slightly larger than pins)
# -----------------------------------------------------------

if len(sys.argv) > 1:
    INPUT_TIF = sys.argv[1]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📂 Input DEM: {INPUT_TIF}")
print(f"📦 Output directory: {OUTPUT_DIR}")
print(f"🎯 Goal: Entire terrain ≤ {MAX_ASSEMBLED_SIZE/304.8:.0f}ft × {MAX_ASSEMBLED_SIZE/304.8:.0f}ft, tiled into {TARGET_TILE_SIZE/304.8:.0f}ft pieces")

# -----------------------------------------------------------
# HELPER FUNCTIONS
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
    
    # Pin positions: 1/4 and 3/4 along edges
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
    
    # Add pins to mesh
    if features:
        mesh = trimesh.util.concatenate([mesh] + features)
    
    return mesh

# -----------------------------------------------------------
# STEP 1. LOAD THE DEM
# -----------------------------------------------------------
with rasterio.open(INPUT_TIF) as src:
    elev = src.read(1)
    transform = src.transform
    nodata = src.nodata
    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)
    print(f"Loaded DEM: {elev.shape[1]} x {elev.shape[0]} pixels")
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)

# -----------------------------------------------------------
# STEP 2. CALCULATE GLOBAL SCALE AND TILING
# -----------------------------------------------------------
rows, cols = elev.shape

# Calculate real-world extent WITHOUT creating huge meshgrids
x_min = transform.c
x_max = transform.c + cols * transform.a
y_min = transform.f + rows * transform.e
y_max = transform.f

extent_x = abs(x_max - x_min)
extent_y = abs(y_max - y_min)
print(f"Real-world extent: {extent_x:.1f}m × {extent_y:.1f}m")

# Calculate elevation range (sample if too large to avoid memory issues)
if rows * cols > 100_000_000:  # If > 100M pixels, sample for z-range
    print("⚡ Large dataset detected, sampling elevation range...")
    step = max(1, int(np.sqrt(rows * cols / 10_000_000)))  # Sample ~10M points
    elev_sample = elev[::step, ::step]
    z_min = np.nanmin(elev_sample) * VERTICAL_EXAGGERATION
    z_max = np.nanmax(elev_sample) * VERTICAL_EXAGGERATION
else:
    z_min = np.nanmin(elev) * VERTICAL_EXAGGERATION
    z_max = np.nanmax(elev) * VERTICAL_EXAGGERATION

z_range = z_max - z_min
print(f"Elevation range (after {VERTICAL_EXAGGERATION}× exaggeration): {z_range:.1f}m")

# Calculate scale to fit ENTIRE model within MAX_ASSEMBLED_SIZE (6ft × 6ft)
# Scale based on longest horizontal dimension
scale_xy = MAX_ASSEMBLED_SIZE / max(extent_x, extent_y)

# Also ensure height fits within MAX_HEIGHT
scale_z = MAX_HEIGHT / z_range if z_range > 0 else 1.0

# Use the most restrictive scale (smallest)
scale = min(scale_xy, scale_z)

# Calculate final assembled dimensions
final_width = extent_x * scale
final_height = extent_y * scale
final_z = z_range * scale

print(f"\n📏 Final assembled model size: {final_width:.1f}mm × {final_height:.1f}mm × {final_z:.1f}mm")
print(f"   ({final_width/304.8:.2f}ft × {final_height/304.8:.2f}ft × {final_z/25.4:.2f}in)")

# Calculate how many tiles we need (each tile is TARGET_TILE_SIZE)
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
# STEP 3. GENERATE EACH TILE
# -----------------------------------------------------------
tile_count = 0
for tile_y in range(num_tiles_y):
    for tile_x in range(num_tiles_x):
        tile_count += 1
        print(f"[{tile_count}/{num_tiles_x * num_tiles_y}] Generating tile ({tile_x}, {tile_y})...")
        
        # Calculate tile bounds in real-world coordinates
        tile_x_start_world = x_min + tile_x * effective_tile_size / scale
        tile_x_end_world = tile_x_start_world + (TARGET_TILE_SIZE / scale)
        tile_y_start_world = y_max - (tile_y + 1) * TARGET_TILE_SIZE / scale  # Y decreases
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
        
        # Extract ONLY this tile's elevation data (memory efficient!)
        tile_elev = elev[row_start:row_end, col_start:col_end].copy()
        
        # Create coordinate grids for THIS TILE ONLY
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
        output_file = os.path.join(OUTPUT_DIR, f"tile_{tile_x}_{tile_y}.stl")
        solid.export(output_file)
        print(f"  ✅ Saved: {output_file}")

print(f"\n🎉 Complete! Generated {tile_count} tiles in '{OUTPUT_DIR}/'")
print(f"💡 Assembly instructions:")
print(f"   • Print all {tile_count} tiles")
print(f"   • Arrange in a {num_tiles_x}×{num_tiles_y} grid (left-to-right, top-to-bottom)")
print(f"   • Final model: {final_width:.1f}mm × {final_height:.1f}mm ({final_width/304.8:.1f}ft × {final_height/304.8:.1f}ft)")
print(f"   • Each tile is exactly 1 foot (304.8mm) square")
