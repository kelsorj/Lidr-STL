# New Map Section - LiDAR to STL

This is a separate workflow for processing a new map section (91 LAZ files from `downloadlist.txt`) into 304.8 x 304.8mm STL tiles.

## Key Differences from Main Project

### Separate Folders & Files
To keep everything organized and prevent mixing with your existing work:

| Component | Original | New Map |
|-----------|----------|---------|
| **Download list** | `0_file_download_lidr.txt` | `downloadlist.txt` |
| **LAZ storage** | `lidar_laz/` | `newmap_lidar_laz/` |
| **Batch DEMs** | `lidar_batch_*.tif` | `newmap_batch_*.tif` |
| **Merged DEM** | `lidar_merged.tif` | `newmap_lidar_merged.tif` |
| **Pipeline files** | `pipeline_batch_*.json` | `newmap_pipeline_batch_*.json` |
| **Output tiles** | `stl_tiles_lidar/` | `stl_tiles_newmap/` |
| **Tile names** | `lidar_tile_X_Y.stl` | `newmap_tile_X_Y.stl` |

## Quick Start

### 1. Run the Script

```bash
python lidar_to_stl_newmap.py
```

The script will:
1. **Download** 91 LAZ files from `downloadlist.txt` → `newmap_lidar_laz/`
2. **Process in batches** (50 files each) → `newmap_batch_0.tif`, `newmap_batch_1.tif`
3. **Merge overlapping data** → `newmap_lidar_merged.tif` (PDAL automatically handles overlap!)
4. **Generate STL tiles** → `stl_tiles_newmap/newmap_tile_0_0.stl`, etc.

### 2. Print the Tiles

Each tile is exactly **304.8mm × 304.8mm (1 foot)** and fits on the Bambu H2D (320mm bed) with margin.

### 3. Assemble

Arrange tiles in a grid pattern (left-to-right, top-to-bottom). The script will tell you:
- Grid dimensions (e.g., 3×4 = 12 tiles)
- Final assembled size

## How Overlapping Data is Combined

**PDAL automatically handles overlaps!** When multiple LAZ files cover the same area:

1. **Batch Processing**: Each batch is processed with `output_type: "max"` which takes the highest point
2. **Merging**: `rasterio.merge()` combines batch DEMs, automatically handling overlaps
3. **Result**: You get a seamless single elevation model

The `output_type: "max"` setting means:
- Buildings and structures are preserved (highest points win)
- No gaps or seams between overlapping tiles
- Natural terrain is properly represented

## Configuration

Edit `lidar_to_stl_newmap.py` to adjust:

```python
# Size settings
TARGET_TILE_SIZE = 304.8      # mm per tile (1 foot)
MAX_ASSEMBLED_SIZE = 1828.8   # mm (6 feet max)

# Terrain
VERTICAL_EXAGGERATION = 1.0   # 1.0 = true scale, 2.0 = 2x height
BASE_THICKNESS = 5.0          # mm

# Processing
DEM_RESOLUTION = 0.5          # meters per pixel (lower = more detail)
BATCH_SIZE = 50               # files per batch
USE_ALL_POINTS = True         # True = buildings, False = ground only

# Features
ADD_ALIGNMENT_PINS = False    # True to add pins/holes for assembly
```

## Skip Download on Subsequent Runs

If you need to regenerate tiles after files are downloaded:

```python
SKIP_DOWNLOAD = True  # Line ~52 in lidar_to_stl_newmap.py
```

## Troubleshooting

### Memory Issues
Reduce `BATCH_SIZE` from 50 to 25 or lower:
```python
BATCH_SIZE = 25  # Process fewer files at once
```

### Missing PDAL
```bash
brew install pdal  # macOS
```

### Want More Detail
Lower the resolution (more pixels = more detail):
```python
DEM_RESOLUTION = 0.25  # meters per pixel (default 0.5)
```

### Tiles Too Tall
If tiles exceed printer height (300mm):
```python
VERTICAL_EXAGGERATION = 0.5  # Reduce terrain height
```

## Files Generated

```
newmap_lidar_laz/          # Downloaded LAZ files (91 files)
newmap_batch_0.tif         # Batch DEM #1
newmap_batch_1.tif         # Batch DEM #2
newmap_lidar_merged.tif    # Final combined DEM
newmap_pipeline_batch_*.json  # PDAL pipeline configs
stl_tiles_newmap/          # Final STL tiles
  ├── newmap_tile_0_0.stl
  ├── newmap_tile_0_1.stl
  └── ...
```

## What Makes This Different?

1. **All new folder names** - won't interfere with your existing project
2. **Reads from `downloadlist.txt`** - your 91 new URLs
3. **Separate output** - tiles go to `stl_tiles_newmap/`
4. **Same proven workflow** - identical processing logic to your working script

## Data Sources

Your new map includes LAZ files from:
- ARRA_CA_GOLDENGATE_2010 (14 files)
- ARRA_CA_SANFRANCOAST_2010 (7 files)
- CA_NoCAL_3DEP_Supp_Funding_2018_D18 (34 files)
- CA_SANFRANBAYFEMA_2004 (6 files)
- CA_SanFrancisco_B23 (15 files)
- CA_West_Coast_LiDAR_2016_B16 (11 files)

Total: **91 files** covering your new map section.



