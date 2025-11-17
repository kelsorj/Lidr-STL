#!/usr/bin/env python3
"""
SIMPLIFIED HYBRID OSM + LiDAR Test
===================================
Tests coordinate system matching on a small area
"""

import numpy as np
import osmnx as ox
import geopandas as gpd
import laspy
import glob

# Download small area of OSM buildings (centered on LiDAR coverage)
print("📥 Downloading OSM buildings for small test area...")
test_lat, test_lon = 37.74803, -122.50571  # Center of LiDAR coverage
buildings_gdf = ox.features_from_bbox(
    bbox=(test_lon - 0.002, test_lat - 0.002, test_lon + 0.002, test_lat + 0.002),  # (west, south, east, north)
    tags={'building': True}
)
buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
print(f"   Found {len(buildings_gdf)} buildings")

# Convert to LiDAR CRS (extracted from LAZ file header!)
LIDAR_CRS = 'EPSG:7131'  # NAD83(2011) / San Francisco CS13
buildings_proj = buildings_gdf.to_crs(LIDAR_CRS)
print(f"   Converted to {LIDAR_CRS}")
print(f"   Building extent: X={buildings_proj.geometry.bounds['minx'].min():.0f}-{buildings_proj.geometry.bounds['maxx'].max():.0f}")
print(f"   Building extent: Y={buildings_proj.geometry.bounds['miny'].min():.0f}-{buildings_proj.geometry.bounds['maxy'].max():.0f}")

# Load LAZ files
print("\n📂 Loading LAZ files...")
laz_files = sorted(glob.glob("lidar_laz/*.laz"))
print(f"   Found {len(laz_files)} LAZ files")

# Find LAZ files that overlap with buildings
print("\n🔍 Finding overlapping LAZ files...")
overlapping = []
for laz_file in laz_files[:20]:  # Check first 20 files
    with laspy.open(laz_file) as laz:
        h = laz.header
        # Check if this LAZ file overlaps with our buildings
        buildings_bounds = buildings_proj.total_bounds  # [minx, miny, maxx, maxy]
        if (h.x_max >= buildings_bounds[0] and h.x_min <= buildings_bounds[2] and
            h.y_max >= buildings_bounds[1] and h.y_min <= buildings_bounds[3]):
            overlapping.append(laz_file)
            print(f"   ✅ {laz_file.split('/')[-1]}: X={h.x_min:.0f}-{h.x_max:.0f}, Y={h.y_min:.0f}-{h.y_max:.0f}")

if len(overlapping) == 0:
    print("\n❌ No LAZ files overlap with test area!")
    print("   This means the CRS might be wrong, or we need a different test location")
else:
    print(f"\n✅ Found {len(overlapping)} overlapping LAZ files!")
    print("   Coordinate systems are correctly aligned!")

