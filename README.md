# ML Tree Canopy Height Estimation Pipeline

End-to-end metaflow pipeline for estimating tree canopy height across large areas [Google AlphaEarth Satellite Embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) using airborne LiDAR Point Clouds as ground-truth.


The current workflow is currently integrated with the CanElevation Series Dataset.

To run this pipeline on your own data you will need:
- LiDAR Point Clouds in [COPC](https://copc.io/) format hosted on an AWS S3 Bucket   
- A Tile Index indicating the spatial bounds of each point cloud, including its S3 URI or URL
- A Google Earth Engine (GEE) Account with a valid API Key for programmatic access via Python API

## Setting Up GEE API Access

For GEE access you need to create a .env file in the root directory and add the following environmental variables:

```bash
GEE_SERVICE_ACCOUNT_EMAIL=<YOUR-GEE-SERVICE-ACCOUNT-EMAIL>
GEE_SERVICE_ACCOUNT_KEY_PATH=<YOUR-GEE-SERVICE-ACCOUNT-KEY-PATH>
```
If you haven't configured your GEE API Credentials follow these instructions.

## Installation

```bash
conda env create -f envrionment.yaml
conda activate pc-flow
```


## Run with Docker

```bash
docker pull <image-name>
docker run ....
```


## Dataset Preparation Flow

```bash
python flows/data_flow.py run --max-workers 3 --max-num-splits 4000 --test true
```


## Training  Flow

```bash
 python flows/train_flow.py run --max-workers 3 --run-id <YOUR-RUN-ID> --test true
```