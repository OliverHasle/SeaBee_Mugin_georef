# Georef Pipeline

A Python package for processing geospatial reference data.

## Installation

### Automated Installation (Windows)
1. Clone the repository
2. Run `setup_windows.bat`

#### Linux/Mac
1. Clone the repository
2. Make the script executable: `chmod +x setup_linux.sh`
3. Run `./setup_environment.sh`

### Manual Installation using Conda (Recommended)
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate georef_env

# Install the package
pip install .
```

### How to get a DEM
1. Find relevant elevation models (DEMs / DTMs / DOMs) on "geonorge.no" and download them (*.geotiff)
2. Load into these geotiff files into QGIS
3. Merge all elevation models into one raster

  In QGIS:
  - Raster -> Miscallenious
  - Merge (Select all rasters to be merged) -> Run
4. Clip raster to the area needed

  In QGIS:
  - Raster 
  - Extract
  - Clip raster by Extent
  - Select layer (assign a no-data value (e.g. -9999.0), save to file, clipping extent)
  - Run
5. Export raster to file as *.geotiff
