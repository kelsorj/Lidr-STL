#!/usr/bin/env python3
"""
Download OpenStreetMap building data for San Francisco and convert to tiled STLs.
Creates clean 3D building models with proper heights.
Final model: 6ft × 6ft max, tiled into 1ft (304.8mm) pieces for Bambu H2D.
"""

import os
import sys
import numpy as np
import trimesh
from tqdm import tqdm

try:
    import osmnx as ox
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from pyproj import Transformer
except ImportError as e:
    print("❌ Missing required packages!")
    print("\nInstall with:")
    print("  pip install osmnx geopandas shapely pyproj")
    print(f"\nError: {e}")
    sys.exit(1)

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
OUTPUT_DIR = "stl_tiles_osm"
CACHE_FILE = "sf_buildings.geojson"

# San Francisco bounding box (approx)
# You can adjust these to focus on specific areas
SF_NORTH = 37.8324  # North boundary
SF_SOUTH = 37.7079  # South boundary  
SF_EAST = -122.3482 # East boundary
SF_WEST = -122.5155 # West boundary

# Printer settings (Bambu H2D)
TARGET_TILE_SIZE = 304.8      # mm (1 foot per tile)
MAX_HEIGHT = 300.0            # mm (max Z height)

# Final assembled model size constraint
MAX_ASSEMBLED_SIZE = 1828.8   # mm (6 feet - total final model limit)

# Building settings
DEFAULT_BUILDING_HEIGHT = 10.0    # meters (if height not specified)
GROUND_ELEVATION = 0.0            # meters (base ground level)
BASE_THICKNESS = 5.0              # mm base thickness
VERTICAL_EXAGGERATION = 1.0       # 1:1 scale for buildings

# Simplification
SIMPLIFY_TOLERANCE = 0.5          # meters (simplify building footprints)
MAX_BUILDINGS_PER_TILE = 10000    # Limit for performance

print("=" * 60)
print("🏙️  OpenStreetMap to STL Converter")
print("=" * 60)
print(f"📦 Output directory: {OUTPUT_DIR}")
print(f"🎯 Goal: Entire model ≤ {MAX_ASSEMBLED_SIZE/304.8:.0f}ft × {MAX_ASSEMBLED_SIZE/304.8:.0f}ft")
print(f"🖨️  Tiled into {TARGET_TILE_SIZE/304.8:.0f}ft pieces\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# STEP 1. DOWNLOAD OSM BUILDING DATA
# -----------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"📂 Loading cached building data from {CACHE_FILE}...")
    buildings = gpd.read_file(CACHE_FILE)
    print(f"✅ Loaded {len(buildings)} buildings from cache\n")
else:
    print("📥 Downloading OSM building data for San Francisco...")
    print("(This may take 2-5 minutes...)\n")
    
    try:
        # Download building footprints from OSM using place name
        # This is more reliable across OSMnx versions
        print("Fetching San Francisco boundary...")
        
        # Method 1: Try using place name (most compatible)
        try:
            buildings = ox.features_from_place(
                "San Francisco, California, USA",
                tags={'building': True}
            )
        except:
            # Method 2: Fall back to bbox as tuple
            print("Trying bbox method...")
            bbox = (SF_NORTH, SF_SOUTH, SF_EAST, SF_WEST)
            buildings = ox.features_from_bbox(
                bbox,
                tags={'building': True}
            )
        
        # Keep only polygon geometries
        buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        print(f"✅ Downloaded {len(buildings)} buildings\n")
        
        # Save to cache
        print(f"💾 Saving to cache: {CACHE_FILE}...")
        buildings.to_file(CACHE_FILE, driver='GeoJSON')
        
    except Exception as e:
        print(f"❌ Failed to download OSM data: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try a smaller bounding box")
        print("3. Check if OSM servers are available")
        sys.exit(1)

# -----------------------------------------------------------
# STEP 2. EXTRACT BUILDING HEIGHTS
# -----------------------------------------------------------
print("📊 Processing building heights...")

def get_building_height(row):
    """Extract building height from OSM tags."""
    # Try various height fields
    if 'height' in row and row['height']:
        try:
            h = float(str(row['height']).replace('m', '').strip())
            return h
        except:
            pass
    
    if 'building:levels' in row and row['building:levels']:
        try:
            levels = float(row['building:levels'])
            return levels * 3.5  # ~3.5m per floor
        except:
            pass
    
    # Default height based on building type
    btype = row.get('building', 'yes')
    if btype in ['house', 'residential', 'detached']:
        return 7.0
    elif btype in ['commercial', 'retail']:
        return 12.0
    elif btype in ['apartments', 'office']:
        return 25.0
    elif btype == 'yes':
        return DEFAULT_BUILDING_HEIGHT
    else:
        return DEFAULT_BUILDING_HEIGHT

buildings['height_m'] = buildings.apply(get_building_height, axis=1)

print(f"Height stats:")
print(f"  Min: {buildings['height_m'].min():.1f}m")
print(f"  Max: {buildings['height_m'].max():.1f}m")
print(f"  Mean: {buildings['height_m'].mean():.1f}m\n")

# -----------------------------------------------------------
# STEP 3. CONVERT TO METRIC COORDINATES
# -----------------------------------------------------------
print("🗺️  Converting coordinates...")

# Reproject to UTM (meters) for easier processing
# UTM Zone 10N for San Francisco
buildings_utm = buildings.to_crs(epsg=32610)

# Get bounding box in UTM
bounds = buildings_utm.total_bounds  # [minx, miny, maxx, maxy]
extent_x = bounds[2] - bounds[0]
extent_y = bounds[3] - bounds[1]

print(f"Area coverage: {extent_x:.0f}m × {extent_y:.0f}m\n")

# -----------------------------------------------------------
# STEP 4. CALCULATE SCALING AND TILING
# -----------------------------------------------------------
print("📐 Calculating scale and tile grid...")

# Get elevation range
z_min = GROUND_ELEVATION
z_max = buildings['height_m'].max()
z_range = z_max - z_min

# Calculate scale to fit within MAX_ASSEMBLED_SIZE
scale_xy = MAX_ASSEMBLED_SIZE / max(extent_x, extent_y)
scale_z = MAX_HEIGHT / z_range if z_range > 0 else 1.0
scale = min(scale_xy, scale_z)

# Final assembled dimensions
final_width = extent_x * scale
final_height = extent_y * scale
final_z = z_range * scale

print(f"📏 Final assembled model: {final_width:.1f}mm × {final_height:.1f}mm × {final_z:.1f}mm")
print(f"   ({final_width/304.8:.2f}ft × {final_height/304.8:.2f}ft × {final_z/25.4:.2f}in)")

# Calculate tile grid
num_tiles_x = int(np.ceil(final_width / TARGET_TILE_SIZE))
num_tiles_y = int(np.ceil(final_height / TARGET_TILE_SIZE))

print(f"📐 Scale factor: {scale:.6f} (meters → mm)")
print(f"🔲 Tile grid: {num_tiles_x} × {num_tiles_y} = {num_tiles_x * num_tiles_y} total tiles")

# Global base elevation
GLOBAL_BASE_Z = z_min * scale
print(f"🏔️  Global base elevation: {z_min:.1f}m (scaled: {GLOBAL_BASE_Z:.1f}mm)\n")

# -----------------------------------------------------------
# STEP 5. GENERATE TILES
# -----------------------------------------------------------
print("🏗️  Generating building tiles...\n")

def create_building_mesh(footprint, height_m, scale, base_z):
    """Create a 3D mesh for a single building."""
    if footprint.is_empty or height_m <= 0:
        return None
    
    try:
        # Get exterior coordinates
        if isinstance(footprint, MultiPolygon):
            # Use largest polygon if multipolygon
            footprint = max(footprint.geoms, key=lambda p: p.area)
        
        coords = list(footprint.exterior.coords)
        if len(coords) < 3:
            return None
        
        # Create bottom vertices (at ground level, z=0)
        bottom_verts = [(x, y, 0) for x, y in coords[:-1]]
        
        # Create top vertices (at building height)
        scaled_height = height_m * scale - GLOBAL_BASE_Z
        top_verts = [(x, y, scaled_height) for x, y in coords[:-1]]
        
        all_verts = bottom_verts + top_verts
        n = len(bottom_verts)
        
        faces = []
        
        # Bottom face (reversed for correct normal)
        for i in range(1, n - 1):
            faces.append([0, i + 1, i])
        
        # Top face
        for i in range(1, n - 1):
            faces.append([n, n + i, n + i + 1])
        
        # Side walls
        for i in range(n):
            next_i = (i + 1) % n
            faces.append([i, next_i, n + i])
            faces.append([next_i, n + next_i, n + i])
        
        return trimesh.Trimesh(vertices=all_verts, faces=faces, process=False)
    
    except Exception as e:
        return None

# Generate each tile
tile_count = 0
for tile_y in range(num_tiles_y):
    for tile_x in range(num_tiles_x):
        tile_count += 1
        print(f"[{tile_count}/{num_tiles_x * num_tiles_y}] Tile ({tile_x}, {tile_y})...")
        
        # Calculate tile bounds in real-world coordinates (UTM meters)
        tile_x_min = bounds[0] + (tile_x * TARGET_TILE_SIZE / scale)
        tile_x_max = tile_x_min + (TARGET_TILE_SIZE / scale)
        tile_y_min = bounds[1] + (tile_y * TARGET_TILE_SIZE / scale)
        tile_y_max = tile_y_min + (TARGET_TILE_SIZE / scale)
        
        # Filter buildings in this tile
        tile_buildings = buildings_utm.cx[tile_x_min:tile_x_max, tile_y_min:tile_y_max]
        
        if len(tile_buildings) == 0:
            print(f"  ⚠️  No buildings in this tile, skipping")
            continue
        
        print(f"  🏢 {len(tile_buildings)} buildings")
        
        # Create meshes for all buildings in tile
        meshes = []
        for idx, building in tile_buildings.iterrows():
            # Transform building to tile-local coordinates
            geom = building.geometry
            
            # Translate to tile origin
            from shapely.affinity import translate
            local_geom = translate(geom, xoff=-tile_x_min, yoff=-tile_y_min)
            
            # Scale to mm
            from shapely.affinity import scale as shapely_scale
            scaled_geom = shapely_scale(local_geom, xfact=scale, yfact=scale, origin=(0, 0))
            
            # Simplify if needed
            if SIMPLIFY_TOLERANCE > 0:
                scaled_geom = scaled_geom.simplify(SIMPLIFY_TOLERANCE * scale, preserve_topology=True)
            
            # Create 3D mesh
            mesh = create_building_mesh(
                scaled_geom,
                building['height_m'],
                scale,
                GLOBAL_BASE_Z
            )
            
            if mesh:
                meshes.append(mesh)
        
        # Combine all building meshes (or empty if no buildings)
        if len(meshes) > 0:
            print(f"  🔨 Combining {len(meshes)} buildings...")
            tile_mesh = trimesh.util.concatenate(meshes)
        else:
            print(f"  ⚠️  No buildings in this tile - creating ground only")
            tile_mesh = None
        
        # Create SOLID ground (surface + perimeter walls + base)
        # Ground surface at z=0 (buildings sit on this)
        ground_top = np.array([
            [0, 0, 0],
            [TARGET_TILE_SIZE, 0, 0],
            [TARGET_TILE_SIZE, TARGET_TILE_SIZE, 0],
            [0, TARGET_TILE_SIZE, 0]
        ])
        ground_top_faces = [[0, 1, 2], [0, 2, 3]]
        
        # Ground bottom at -BASE_THICKNESS
        ground_bottom = np.array([
            [0, 0, -BASE_THICKNESS],
            [TARGET_TILE_SIZE, 0, -BASE_THICKNESS],
            [TARGET_TILE_SIZE, TARGET_TILE_SIZE, -BASE_THICKNESS],
            [0, TARGET_TILE_SIZE, -BASE_THICKNESS]
        ])
        ground_bottom_faces = [[0, 2, 1], [0, 3, 2]]  # Reversed for downward normal
        
        # Perimeter walls connecting top to bottom
        perimeter_faces = [
            # Front wall (y=0)
            [0, 1, 5], [0, 5, 4],
            # Right wall (x=max)
            [1, 2, 6], [1, 6, 5],
            # Back wall (y=max)
            [2, 3, 7], [2, 7, 6],
            # Left wall (x=0)
            [3, 0, 4], [3, 4, 7]
        ]
        
        # Combine ground vertices and faces
        ground_verts = np.vstack([ground_top, ground_bottom])
        ground_faces = ground_top_faces + ground_bottom_faces + perimeter_faces
        ground_mesh = trimesh.Trimesh(vertices=ground_verts, faces=ground_faces, process=False)
        
        # Combine buildings with ground (or just ground if no buildings)
        if tile_mesh is not None:
            tile_mesh = trimesh.util.concatenate([tile_mesh, ground_mesh])
        else:
            tile_mesh = ground_mesh
        
        # Export
        output_file = os.path.join(OUTPUT_DIR, f"osm_tile_{tile_x}_{tile_y}.stl")
        tile_mesh.export(output_file)
        print(f"  ✅ Saved: {output_file}")

print(f"\n🎉 Complete! Generated {tile_count} tiles in '{OUTPUT_DIR}/'")
print(f"💡 Assembly instructions:")
print(f"   • Print all {tile_count} tiles")
print(f"   • Arrange in a {num_tiles_x}×{num_tiles_y} grid")
print(f"   • Final model: {final_width:.1f}mm × {final_height:.1f}mm")
print(f"   • Clean building geometry from OpenStreetMap! 🏙️")

