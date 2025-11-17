# DEM TIFF to STL Converter

Convert USGS 1-meter resolution DEM TIFF files directly to 304.8mm × 304.8mm STL tiles.

## Quick Start

### 1. Run the Script

```bash
python dem_tif_to_stl.py
```

The script will:
1. **Download** 2 TIF files from `dem_tif_urls.txt` → `dem_tif_downloads/`
2. **Merge** overlapping TIFs → `dem_tif_merged.tif`
3. **Generate STL tiles** → `stl_tiles_dem/dem_tile_0_0.stl`, etc.

### 2. Your Files

The URLs in `dem_tif_urls.txt` are:
- `USGS_1m_x54y419_CA_NoCAL_Wildfires_B5b_2018.tif`
- `USGS_1m_x54y419_CA_NoCal_Wildfires_B5b_QL1_2018.tif`

These are **1-meter resolution** Digital Elevation Models (DEMs) from the Northern California Wildfires project.

## What's Different: DEM vs. LiDAR

| Feature | DEM TIFF (This Script) | LiDAR LAZ |
|---------|------------------------|-----------|
| **Data Type** | Raster (grid of elevations) | Point cloud (3D points) |
| **Resolution** | 1m per pixel | Variable, often sub-meter |
| **Buildings** | ❌ Smoothed out | ✅ Included (if using all points) |
| **File Size** | Smaller (~100-500 MB) | Larger (1-2 GB per file) |
| **Processing** | Faster, simpler | Slower, requires PDAL |
| **Best For** | Terrain, topography | Buildings, structures, trees |

## Configuration

Edit `dem_tif_to_stl.py` to adjust:

```python
# Size settings
TARGET_TILE_SIZE = 304.8      # mm per tile (1 foot)
MAX_ASSEMBLED_SIZE = 1828.8   # mm (6 feet max)

# Terrain
VERTICAL_EXAGGERATION = 1.5   # 1.5x height for better visibility
BASE_THICKNESS = 5.0          # mm

# Processing
MAX_DIM = 2000                # max pixels per tile dimension
```

## Add More TIF Files

To add more DEM TIFs, just add their URLs to `dem_tif_urls.txt`:

```
https://example.com/DEM_file1.tif
https://example.com/DEM_file2.tif
https://example.com/DEM_file3.tif
```

The script will automatically:
- Download all files
- Merge overlapping regions
- Generate a single seamless terrain model

## Skip Download on Re-runs

If you've already downloaded the files and want to regenerate tiles:

```python
SKIP_DOWNLOAD = True  # Line ~61 in dem_tif_to_stl.py
```

## Adjust Vertical Exaggeration

The default is 1.5x to make terrain features more visible:

```python
VERTICAL_EXAGGERATION = 1.0   # True scale (flat)
VERTICAL_EXAGGERATION = 1.5   # Default (recommended)
VERTICAL_EXAGGERATION = 2.0   # More dramatic
VERTICAL_EXAGGERATION = 3.0   # Very dramatic
```

## Files Generated

```
dem_tif_downloads/                      # Downloaded TIF files
  ├── USGS_1m_x54y419_CA_NoCAL_Wildfires_B5b_2018.tif
  └── USGS_1m_x54y419_CA_NoCal_Wildfires_B5b_QL1_2018.tif
dem_tif_merged.tif                      # Merged DEM
stl_tiles_dem/                          # Final STL tiles
  ├── dem_tile_0_0.stl
  ├── dem_tile_0_1.stl
  └── ...
```

## Comparison: All Three Workflows

### 1. **LiDAR LAZ** (`lidar_to_stl.py`)
- ✅ Includes buildings, trees, structures
- ✅ Highest detail
- ❌ Slower processing
- ❌ Requires PDAL
- **Best for:** Urban areas, architectural detail

### 2. **New Map LAZ** (`lidar_to_stl_newmap.py`)
- Same as above, but separate folder structure
- Uses `downloadlist.txt` (91 LAZ files)

### 3. **DEM TIFF** (`dem_tif_to_stl.py`) ⭐ **This One**
- ✅ Fast processing
- ✅ Simple workflow
- ✅ Good for terrain
- ❌ No buildings/structures (smoothed out)
- **Best for:** Natural terrain, mountains, valleys

## Troubleshooting

### Download Fails
Files are large (~200-400 MB each). If download fails:
1. Check internet connection
2. Try downloading manually from URLs
3. Place files in `dem_tif_downloads/` folder
4. Set `SKIP_DOWNLOAD = True`

### Memory Issues
Reduce `MAX_DIM`:
```python
MAX_DIM = 1000  # Lower = less memory, coarser detail
```

### Tiles Too Flat
Increase exaggeration:
```python
VERTICAL_EXAGGERATION = 2.0  # or higher
```

### Tiles Too Tall (>300mm)
Reduce exaggeration:
```python
VERTICAL_EXAGGERATION = 1.0  # or lower
```

## Assembly

1. **Print all tiles** at 100% scale, no supports needed for terrain top
2. **Arrange in grid** - script tells you dimensions (e.g., 2×3)
3. **Align edges** - tiles fit together at exactly 304.8mm (1 foot) each
4. **Optional:** Glue edges with CA glue or use alignment pins

## What Are These Specific Files?

Your two TIF files are from the **Northern California Wildfires 2018** USGS survey:

- **x54y419** = Grid coordinates (54th column, 419th row in the tiling system)
- **B5b** = Batch/Project identifier
- **QL1** = Quality Level 1 (highest quality)
- **1m** = 1-meter ground sample distance

These cover the same geographic area but may have been processed differently (standard vs. high quality). The script will merge them and use the best data from each.

## Converting to LiDAR

If you want buildings/structures from this same area, look for the corresponding LAZ files:
- Search USGS for "CA_NoCAL_Wildfires_B5b_2018" LAZ files
- They'll have similar names but with `.laz` extension
- Add URLs to `downloadlist.txt` and use `lidar_to_stl_newmap.py`



