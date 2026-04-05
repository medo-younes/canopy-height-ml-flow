# Methodology



## 1. Dataset Construction

N = 5,000

### 1-A. Tile Preparation
- Intersecting lidar tile polygons with AOI Boundary 
- Download Forest and Water Bodies Data
- Compute % Forest cover in Each Tile, filter out tiles with <25% forest cover
- Clip tile geometries with water bodies to avoid sampling water
- Construct 10 x 10 km blocks with for spatial-k-folds
- Stratify tiles by forest cover and K-fold ID, sample 10 tiles from each strata
- Compute n samples per tile based on total samples required

### 1-B Structurally Guided Sampling
For each tile:
- Download LiDAR Point Cloud from AWS S3
- Compute Canopy Height Model (CHM) as 95th percentile in height, export as geotiff
- Stratify CHM layer
- Randomly sample height, take n samples per strata

### 1-C Extract GEE Emmbeddings
- Attain target year from tile index
- Aquire Google Earth Engine Satellite Embeddings (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) image collection for the target year
- 