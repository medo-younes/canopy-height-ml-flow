# CanopyFlow: ML-driven Tree Canopy Height Estimation Pipeline


TODO:
- Running with Docker
- Hosting can elevation parquet files on github or aws
- wood buffalo case study
- Instructions for setting up config.yaml for target aoi
- passing AOI path as argument in data download flow

End-to-end metaflow pipeline for estimating tree canopy height across large areas [Google AlphaEarth Satellite Embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) using airborne LiDAR Point Clouds as ground-truth.

This project consists of three metaflow pipelines for preparing 10m tree canopy height maps:
1. [**Data Download**](flows/download_data.py) - downloading LiDAR point clouds from AWS S3 ([learn more](#lidar-data)), compute CHM, structurally guided sampling of tree canopy height. Sampling GEE satellite embeddings 
2. [**Training**](flows/train.py) - regression model training (Elastic Net, Random Forest and XGBoost) with Spatial K-fold Cross Validation. Automated model selection.
3. [**Inference**](flows/inference.py) - predict tree canopy height across your AOI using best model checkpoint. 


## Setting Up GEE API Access

1. Export a GEE Service Account Key JSON file by [following these instructions.](https://gee-documentation.readthedocs.io/en/latest/authentication/service-account-auth.html).
2. Create an .env file in the project root directory and include your GEE service account email and the path to your GEE Service Account Key (JSON)

```bash
GEE_SERVICE_ACCOUNT_EMAIL=<YOUR-GEE-SERVICE-ACCOUNT-EMAIL>
GEE_SERVICE_ACCOUNT_KEY_PATH=<PATH-TO-YOUR-GEE-SERVICE-ACCOUNT-KEY>
```


## Run with Conda Environemnt
```bash
conda env create -f environment.yaml
conda activate canopy-flow
```

## Run with Docker

```bash
# Export AWS Credentials to environment
eval "$(aws configure export-credentials --profile default --format env)"

# Run with docker image (myounes88/canopy-flow:test)
docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
           -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
           -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
            myounes88/canopy-flow:test python data.py run --max-workers 3 --max-num-splits 4000
```

## Running Flows

```bash
## Dataset Preparation
python flows/download_data.py run --max-workers 3 --max-num-splits 4000 

## Model Training - use metaflow Run ID from dataset preparation flow
python flows/train.py run --max-workers 3 --max-num-splits 4000 

## Inference - use path to best model checkpoint 
python flows/inference.py run --max-workers 8 --max-num-splits 8000 --model-checkpoint <MODEL-CHECKPOINT-PATH>
```

## LiDAR Data

The current workflow is currently integrated with the CanElevation Series Dataset.

To run this pipeline on your own data you will need:
- **S3-hosted LiDAR Point Clouds** - your own [COPC](https://copc.io/) files hosted on AWS S3
- **Tile Index** - indicating the spatial bounds of each point cloud, including its S3 URI or URL
- **Google Earth Engine Account** with a valid service account key for programmatic access via Python API
- AOI Polygon Boundary** AOI in GeoParquet format
