# ML Tree Canopy Height Estimation Pipeline

End-to-end metaflow pipeline for upscaling airborne LiDAR tree height data with Google AlphaEarth Satellite Embeddings.


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