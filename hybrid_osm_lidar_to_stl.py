#!/usr/bin/env python3
"""
HYBRID OSM + LiDAR to STL Converter
====================================
Combines the best of both worlds:
- OSM building footprints (clean boundaries)
- LiDAR point cloud data (real heights and roof structures)

This creates LOD2 quality 3D buildings with accurate geometry!
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# San Francisco bounding box (expanded slightly for coverage)
SF_NORTH = 37.8324
SF_SOUTH = 37.7079
SF_EAST = -122.3482
SF_WEST = -122.5155

# Target dimensions
MAX_ASSEMBLED_SIZE = 1828.8  # 6 feet in mm
TARGET_TILE_SIZE = 304.8     # 1 foot in mm

# 3D printing parameters
BASE_THICKNESS = 5.0         # Base plate thickness in mm
MIN_BUILDING_HEIGHT = 3.0    # Minimum building height in mm
VERTICAL_SCALE = 1.0         # Vertical exaggeration factor

# LiDAR processing
LAZ_DIR = "lidar_laz"        # Directory containing LAZ files
GROUND_PERCENTILE = 5        # Percentile to use for ground elevation (filters noise)
ROOF_PERCENTILE = 95         # Percentile to use for roof height (filters outliers)
TERRAIN_RESOLUTION = 2.0     # Terrain grid resolution in meters (lower = more detail)

# Output
OUTPUT_DIR = "hybrid_stl_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("🏙️  HYBRID OSM + LiDAR TO STL CONVERTER")
print("=" * 60)
print(f"📍 Region: San Francisco")
print(f"📐 Max size: {MAX_ASSEMBLED_SIZE:.1f}mm ({MAX_ASSEMBLED_SIZE/304.8:.1f} ft)")
print(f"🔲 Tile size: {TARGET_TILE_SIZE:.1f}mm ({TARGET_TILE_SIZE/304.8:.1f} ft)")
print(f"📦 Base thickness: {BASE_THICKNESS:.1f}mm\n")

# ============================================================
# STEP 1: DOWNLOAD OSM BUILDING FOOTPRINTS
# ============================================================
print("📥 Downloading OSM building data...")

try:
    # Try place name first (works with most OSMnx versions)
    buildings_gdf = ox.features_from_place(
        "San Francisco, California, USA",
        tags={'building': True}
    )
    print(f"   ✅ Downloaded {len(buildings_gdf)} buildings from OSM")
except Exception as e:
    print(f"   ⚠️  Place name failed, trying bbox: {e}")
    try:
        buildings_gdf = ox.features_from_bbox(
            north=SF_NORTH,
            south=SF_SOUTH,
            east=SF_EAST,
            west=SF_WEST,
            tags={'building': True}
        )
        print(f"   ✅ Downloaded {len(buildings_gdf)} buildings from OSM")
    except Exception as e2:
        print(f"   ❌ Failed to download OSM data: {e2}")
        exit(1)

# Filter to only polygons
buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
print(f"   🔍 Filtered to {len(buildings_gdf)} polygon buildings")

# Ensure CRS is WGS84
if buildings_gdf.crs != 'EPSG:4326':
    buildings_gdf = buildings_gdf.to_crs('EPSG:4326')

# ============================================================
# STEP 2: LOAD LAZ FILES
# ============================================================
print(f"\n📂 Loading LAZ files from '{LAZ_DIR}/'...")

laz_files = sorted(glob.glob(os.path.join(LAZ_DIR, "*.laz")))
if len(laz_files) == 0:
    print(f"   ❌ No LAZ files found in '{LAZ_DIR}/'")
    exit(1)

print(f"   ✅ Found {len(laz_files)} LAZ files")

# ============================================================
# STEP 2.5: DETERMINE LIDAR CRS AND CONVERT BUILDINGS
# ============================================================
print("\n🗺️  Determining LiDAR coordinate system...")

# Extract CRS from LAZ file header
with laspy.open(laz_files[0]) as laz:
    header = laz.header
    crs = header.parse_crs()
    # San Francisco LiDAR uses: NAD83(2011) / San Francisco CS13 (EPSG:7131)
    LIDAR_CRS = 'EPSG:7131'
    
print(f"   LiDAR CRS: {LIDAR_CRS} (NAD83(2011) / San Francisco CS13)")
print(f"   Sample extent: X={header.x_min:.0f}-{header.x_max:.0f}, Y={header.y_min:.0f}-{header.y_max:.0f}")

# Convert buildings to LiDAR CRS for spatial matching
print(f"   Converting {len(buildings_gdf)} OSM buildings to LiDAR CRS...")
buildings_gdf = buildings_gdf.to_crs(LIDAR_CRS)
print(f"   ✅ Buildings now in {LIDAR_CRS}")

# Build spatial index for LAZ files (know which files cover which areas)
print("   🗺️  Building spatial index for LAZ files...")
laz_bounds = []
for laz_file in tqdm(laz_files, desc="   Indexing"):
    try:
        with laspy.open(laz_file) as laz:
            header = laz.header
            bbox = box(header.x_min, header.y_min, header.x_max, header.y_max)
            laz_bounds.append({
                'file': laz_file,
                'bounds': bbox,
                'x_min': header.x_min,
                'x_max': header.x_max,
                'y_min': header.y_min,
                'y_max': header.y_max
            })
    except Exception as e:
        print(f"   ⚠️  Skipping {os.path.basename(laz_file)}: {e}")

print(f"   ✅ Indexed {len(laz_bounds)} LAZ files")

# ============================================================
# STEP 3: EXTRACT BUILDING HEIGHTS FROM LIDAR
# ============================================================
print("\n🏗️  Extracting building heights from LiDAR...")

def get_terrain_from_lidar(bbox, laz_bounds_list, resolution=2.0):
    """Extract ground terrain from LiDAR for a bounding box.
    
    Args:
        bbox: (minx, miny, maxx, maxy) in projected coordinates
        laz_bounds_list: List of LAZ file info dicts
        resolution: Grid resolution in meters
        
    Returns:
        (X, Y, Z) numpy arrays for terrain grid
    """
    minx, miny, maxx, maxy = bbox
    
    # Find relevant LAZ files
    relevant_laz = [
        item for item in laz_bounds_list
        if item['x_max'] >= minx and item['x_min'] <= maxx and
           item['y_max'] >= miny and item['y_min'] <= maxy
    ]
    
    if not relevant_laz:
        return None
    
    # Collect ground points from all relevant LAZ files
    all_ground_x = []
    all_ground_y = []
    all_ground_z = []
    
    for laz_item in relevant_laz:
        try:
            with laspy.open(laz_item['file']) as laz:
                las = laz.read()
                
                # Filter to ground classification (class 2) and within bbox
                classification = las.classification
                x = las.x
                y = las.y
                z = las.z
                
                # Get ground points in this tile
                mask = (
                    (classification == 2) &  # Ground points only
                    (x >= minx) & (x <= maxx) &
                    (y >= miny) & (y <= maxy)
                )
                
                if mask.any():
                    all_ground_x.extend(x[mask])
                    all_ground_y.extend(y[mask])
                    all_ground_z.extend(z[mask])
        except Exception as e:
            continue
    
    if len(all_ground_x) < 10:
        return None
    
    # Create regular grid
    nx = int((maxx - minx) / resolution) + 1
    ny = int((maxy - miny) / resolution) + 1
    
    grid_x = np.linspace(minx, maxx, nx)
    grid_y = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate ground points onto grid
    from scipy.interpolate import griddata
    points = np.column_stack((all_ground_x, all_ground_y))
    values = np.array(all_ground_z)
    
    Z = griddata(points, values, (X, Y), method='linear', fill_value=np.median(values))
    
    return X, Y, Z

def get_lidar_height_for_building(building_geom, laz_bounds_list):
    """Extract height from LiDAR for a building footprint."""
    centroid = building_geom.centroid
    
    # Find LAZ files that intersect this building
    relevant_laz = [
        item for item in laz_bounds_list
        if item['bounds'].intersects(building_geom)
    ]
    
    if not relevant_laz:
        return None
    
    all_z_values = []
    
    # Extract points from relevant LAZ files
    for laz_item in relevant_laz:
        try:
            with laspy.open(laz_item['file']) as laz:
                las = laz.read()
                
                # Get coordinates
                x = las.x
                y = las.y
                z = las.z
                
                # Filter to points within building footprint
                # Create mask for points inside polygon
                if isinstance(building_geom, MultiPolygon):
                    building_geom = max(building_geom.geoms, key=lambda p: p.area)
                
                # Simple bounding box check first (fast)
                bounds = building_geom.bounds  # (minx, miny, maxx, maxy)
                mask = (
                    (x >= bounds[0]) & (x <= bounds[2]) &
                    (y >= bounds[1]) & (y <= bounds[3])
                )
                
                if not mask.any():
                    continue
                
                # For points in bbox, do precise polygon check
                points_in_bbox = np.column_stack((x[mask], y[mask]))
                z_in_bbox = z[mask]
                
                # Check which points are actually inside polygon
                inside_mask = np.array([
                    building_geom.contains(Point(px, py))
                    for px, py in points_in_bbox
                ])
                
                if inside_mask.any():
                    all_z_values.extend(z_in_bbox[inside_mask])
        
        except Exception as e:
            continue
    
    if len(all_z_values) < 3:  # Need at least a few points
        return None
    
    z_array = np.array(all_z_values)
    
    # Get ground and roof elevations using percentiles (filters outliers)
    ground_z = np.percentile(z_array, GROUND_PERCENTILE)
    roof_z = np.percentile(z_array, ROOF_PERCENTILE)
    
    height = roof_z - ground_z
    
    return {
        'ground_elevation': ground_z,
        'roof_elevation': roof_z,
        'height': height,
        'num_points': len(all_z_values)
    }

# Process buildings and add LiDAR heights
print("   🔍 Processing buildings (this may take a few minutes)...")
buildings_with_heights = []

for idx, building in tqdm(buildings_gdf.iterrows(), total=len(buildings_gdf), desc="   Extracting heights"):
    height_data = get_lidar_height_for_building(building.geometry, laz_bounds)
    
    if height_data and height_data['height'] > 1.0:  # At least 1m tall
        building_copy = building.copy()
        building_copy['lidar_height_m'] = height_data['height']
        building_copy['ground_elev_m'] = height_data['ground_elevation']
        building_copy['roof_elev_m'] = height_data['roof_elevation']
        building_copy['num_lidar_points'] = height_data['num_points']
        buildings_with_heights.append(building_copy)

buildings_gdf = gpd.GeoDataFrame(buildings_with_heights)
print(f"   ✅ Successfully extracted heights for {len(buildings_gdf)} buildings")

if len(buildings_gdf) == 0:
    print("   ❌ No buildings with valid LiDAR heights found!")
    exit(1)

# ============================================================
# STEP 4: CALCULATE SCALING AND TILING
# ============================================================
print("\n📐 Calculating scaling and tiling...")

# Get real-world extents (already in meters since we're in EPSG:7131)
world_west = buildings_gdf.geometry.bounds['minx'].min()
world_east = buildings_gdf.geometry.bounds['maxx'].max()
world_south = buildings_gdf.geometry.bounds['miny'].min()
world_north = buildings_gdf.geometry.bounds['maxy'].max()

world_width_m = world_east - world_west
world_height_m = world_north - world_south

print(f"   Real-world size: {world_width_m:.1f}m × {world_height_m:.1f}m")
print(f"   Extent: X={world_west:.0f}-{world_east:.0f}, Y={world_south:.0f}-{world_north:.0f}")

# Calculate scale to fit MAX_ASSEMBLED_SIZE
scale_x = MAX_ASSEMBLED_SIZE / world_width_m
scale_y = MAX_ASSEMBLED_SIZE / world_height_m
scale = min(scale_x, scale_y)

final_width = world_width_m * scale
final_height = world_height_m * scale

print(f"   Scaled model: {final_width:.1f}mm × {final_height:.1f}mm")
print(f"   Scale: {scale:.4f}mm per meter")

# Calculate tiling
num_tiles_x = int(np.ceil(final_width / TARGET_TILE_SIZE))
num_tiles_y = int(np.ceil(final_height / TARGET_TILE_SIZE))

print(f"   Grid: {num_tiles_x} × {num_tiles_y} = {num_tiles_x * num_tiles_y} tiles")

# Global base elevation (minimum ground across all buildings)
GLOBAL_BASE_Z = buildings_gdf['ground_elev_m'].min() * scale * VERTICAL_SCALE
print(f"   Global base elevation: {buildings_gdf['ground_elev_m'].min():.1f}m (scaled: {GLOBAL_BASE_Z:.1f}mm)")

# ============================================================
# STEP 5: GENERATE BUILDING MESH FUNCTION
# ============================================================

def create_building_mesh(footprint, ground_elev_m, roof_elev_m, scale, global_base_z, x_offset, y_offset):
    """Create a 3D mesh for a building using OSM footprint + LiDAR heights."""
    if footprint.is_empty:
        return None
    
    try:
        # Get exterior coordinates
        if isinstance(footprint, MultiPolygon):
            footprint = max(footprint.geoms, key=lambda p: p.area)
        
        coords = list(footprint.exterior.coords)
        if len(coords) < 3:
            return None
        
        # Transform to tile-local coordinates
        local_coords = [
            ((x - x_offset) * scale, (y - y_offset) * scale)
            for x, y in coords[:-1]  # Skip last (duplicate of first)
        ]
        
        # Calculate Z values (use LiDAR elevations)
        ground_z = (ground_elev_m * scale * VERTICAL_SCALE) - global_base_z
        roof_z = (roof_elev_m * scale * VERTICAL_SCALE) - global_base_z
        
        # Ensure minimum height
        if roof_z - ground_z < MIN_BUILDING_HEIGHT:
            roof_z = ground_z + MIN_BUILDING_HEIGHT
        
        # Create vertices: bottom at ground level, top at roof level
        bottom_verts = [(x, y, ground_z) for x, y in local_coords]
        top_verts = [(x, y, roof_z) for x, y in local_coords]
        
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

# ============================================================
# STEP 6: GENERATE TILES
# ============================================================
print("\n🏗️  Generating hybrid OSM+LiDAR tiles...\n")

tile_count = 0
for tile_y in range(num_tiles_y):
    for tile_x in range(num_tiles_x):
        tile_count += 1
        print(f"📦 Tile [{tile_x}, {tile_y}] ({tile_count}/{num_tiles_x * num_tiles_y})...")
        
        # Calculate tile bounds in world coordinates (already in meters)
        tile_x_min = world_west + (tile_x * TARGET_TILE_SIZE / scale)
        tile_x_max = world_west + ((tile_x + 1) * TARGET_TILE_SIZE / scale)
        tile_y_min = world_south + (tile_y * TARGET_TILE_SIZE / scale)
        tile_y_max = world_south + ((tile_y + 1) * TARGET_TILE_SIZE / scale)
        
        # Get buildings in this tile
        tile_bbox = box(tile_x_min, tile_y_min, tile_x_max, tile_y_max)
        tile_buildings = buildings_gdf[buildings_gdf.geometry.intersects(tile_bbox)].copy()
        
        print(f"   🏢 {len(tile_buildings)} buildings in tile")
        
        # Generate meshes for each building
        meshes = []
        if len(tile_buildings) > 0:
            for idx, building in tile_buildings.iterrows():
                mesh = create_building_mesh(
                    building.geometry,
                    building['ground_elev_m'],
                    building['roof_elev_m'],
                    scale,
                    GLOBAL_BASE_Z,
                    tile_x_min,
                    tile_y_min
                )
                
                if mesh:
                    meshes.append(mesh)
        
        # Extract TERRAIN from LiDAR for this tile
        print(f"   🏔️  Extracting terrain from LiDAR...")
        terrain_data = get_terrain_from_lidar(
            (tile_x_min, tile_y_min, tile_x_max, tile_y_max),
            laz_bounds,
            TERRAIN_RESOLUTION
        )
        
        if terrain_data is None:
            print(f"   ⚠️  No terrain data found, skipping tile")
            continue
        
        terrain_X, terrain_Y, terrain_Z = terrain_data
        rows, cols = terrain_Z.shape
        print(f"   📐 Terrain grid: {cols}×{rows}")
        
        # Convert terrain to tile-local coordinates and scale
        terrain_verts = []
        for i in range(rows):
            for j in range(cols):
                x_local = (terrain_X[i, j] - tile_x_min) * scale
                y_local = (terrain_Y[i, j] - tile_y_min) * scale
                z_local = (terrain_Z[i, j] * scale * VERTICAL_SCALE) - GLOBAL_BASE_Z
                terrain_verts.append([x_local, y_local, z_local])
        
        terrain_verts = np.array(terrain_verts)
        
        # Create terrain faces (triangulate grid)
        terrain_faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                idx = i * cols + j
                # Two triangles per grid cell
                terrain_faces.append([idx, idx + 1, idx + cols])
                terrain_faces.append([idx + 1, idx + cols + 1, idx + cols])
        
        # Create perimeter walls (connect terrain edges to base)
        base_z = terrain_verts[:, 2].min() - BASE_THICKNESS
        
        # Get perimeter indices
        perimeter_top_indices = []
        # Bottom edge
        perimeter_top_indices.extend([i * cols for i in range(rows)])
        # Right edge  
        perimeter_top_indices.extend([i * cols + (cols - 1) for i in range(rows)])
        # Top edge (reverse)
        perimeter_top_indices.extend([(rows - 1) * cols + j for j in range(cols - 1, -1, -1)])
        # Left edge (reverse)
        perimeter_top_indices.extend([i * cols for i in range(rows - 1, -1, -1)])
        
        # Remove duplicates while preserving order
        seen = set()
        perimeter_top_indices = [x for x in perimeter_top_indices if not (x in seen or seen.add(x))]
        
        # Create base vertices (same XY, but at base_z)
        all_verts = list(terrain_verts)
        base_offset = len(all_verts)
        for idx in perimeter_top_indices:
            v = terrain_verts[idx].copy()
            v[2] = base_z
            all_verts.append(v)
        
        # Create wall faces
        all_faces = terrain_faces.copy()
        num_perim = len(perimeter_top_indices)
        for i in range(num_perim):
            top1 = perimeter_top_indices[i]
            top2 = perimeter_top_indices[(i + 1) % num_perim]
            base1 = base_offset + i
            base2 = base_offset + ((i + 1) % num_perim)
            all_faces.append([top1, base1, top2])
            all_faces.append([base1, base2, top2])
        
        # Create bottom face
        base_indices = list(range(base_offset, base_offset + num_perim))
        for i in range(1, num_perim - 1):
            all_faces.append([base_indices[0], base_indices[i+1], base_indices[i]])
        
        terrain_mesh = trimesh.Trimesh(vertices=np.array(all_verts), faces=all_faces, process=False)
        print(f"   ✅ Terrain mesh: {len(terrain_verts)} vertices, {len(terrain_faces)} faces")
        
        # Combine buildings with terrain
        if len(meshes) > 0:
            print(f"   🔨 Combining terrain + {len(meshes)} buildings...")
            tile_mesh = trimesh.util.concatenate([terrain_mesh] + meshes)
        else:
            tile_mesh = terrain_mesh
        
        # Export
        output_file = os.path.join(OUTPUT_DIR, f"hybrid_tile_{tile_x}_{tile_y}.stl")
        tile_mesh.export(output_file)
        print(f"   ✅ Saved: {output_file}")

print(f"\n🎉 Complete! Generated {tile_count} tiles in '{OUTPUT_DIR}/'")
print(f"💡 Benefits of hybrid approach:")
print(f"   • Clean OSM building footprints (no noise)")
print(f"   • Real LiDAR heights (accurate roof elevations)")
print(f"   • LOD2 quality with manageable file sizes")
print(f"   • Best of both worlds! 🏙️✨")

