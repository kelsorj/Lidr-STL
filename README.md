# San Francisco 3D Terrain Models

Generate 3D-printable terrain models of San Francisco for the Bambu H2D printer (320×320mm bed).

## Scripts

### 1. `fix.py` - DEM to STL (Terrain Only)
Converts Digital Elevation Model (DEM) data to STL. **Does NOT include buildings** - only bare ground terrain.

```bash
python fix.py reprojected.tif
```

Output: `stl_tiles/tile_X_Y.stl`

### 2. `lidar_to_stl.py` - LiDAR to STL (With Buildings!) 🏙️
Downloads and processes LiDAR point cloud data. **Includes buildings, trees, and structures**.

```bash
python lidar_to_stl.py
```

Output: `stl_tiles_lidar/lidar_tile_X_Y.stl`

## Features

- **6ft × 6ft maximum model size** - entire terrain scaled to fit
- **1ft (304.8mm) tiles** - each tile fits on 320mm bed with margin
- **Alignment pins** - automatic pins/holes for easy assembly
- **Memory efficient** - processes tiles one at a time
- **No overlap** - tiles form exact 1ft grid

## Configuration

Both scripts have adjustable parameters at the top:

```python
TARGET_TILE_SIZE = 304.8      # mm per tile (1 foot)
MAX_ASSEMBLED_SIZE = 1828.8   # mm (6 feet total)
VERTICAL_EXAGGERATION = 1.8   # terrain height multiplier
BASE_THICKNESS = 5.0          # mm
```

## Requirements

### Python packages:
```bash
pip install -r requirements.txt
```

### PDAL (for LiDAR processing):
```bash
brew install pdal  # macOS
```

## Data Sources

- **DEM**: Pre-processed `reprojected.tif` file
- **LiDAR**: URLs in `0_file_download_lidr.txt` (653 LAZ files)

## Output

Each script generates a grid of STL files:
- `tile_0_0.stl`, `tile_0_1.stl`, ... (DEM)
- `lidar_tile_0_0.stl`, `lidar_tile_0_1.stl`, ... (LiDAR)

Assemble left-to-right, top-to-bottom to recreate the full terrain.

## Tips

- **LiDAR is slower** but includes buildings - expect ~30min+ for 653 files
- **Adjust resolution**: In `lidar_to_stl.py`, change `DEM_RESOLUTION` (default 0.5m)
- **More exaggeration**: Increase `VERTICAL_EXAGGERATION` for dramatic terrain
- **Disable pins**: Set `ADD_ALIGNMENT_PINS = False` if not needed


