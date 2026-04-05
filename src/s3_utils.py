
# Create an LAZ Dataset Pyramid from LiDAR Point Clouds - CanElevation Series


'''
Refer to the Government of Canada Website for more Information: https://open.canada.ca/data/en/dataset/7069387e-9986-4297-9f55-0288e9676947

The LiDAR Point Clouds is a product that is part of the CanElevation Series created to support the National Elevation Data Strategy implemented by NRCan.

This product contains point clouds from various airborne LiDAR acquisition projects conducted in Canada. These airborne LiDAR acquisition projects may have been conducted by NRCan or by various partners. The LiDAR point cloud data is licensed under an open government license and has been incorporated into the National Elevation Data Strategy.

Point cloud files are distributed by LiDAR acquisition project without integration between projects.

The point cloud files are distributed using the compressed .LAZ / Cloud Optimized Point Cloud (COPC) format. The COPC open format is an octree reorganization of the data inside a .LAZ 1.4 file. It allows efficient use and visualization rendering via HTTP calls (e.g. via the web), while offering the capabilities specific to the compressed .LAZ format which is already well established in the industry. Point cloud files are therefore both downloadable for local use and viewable via URL links from a cloud computing environment.

The reference system used for all point clouds in the product is NAD83(CSRS), epoch 2010. The projection used is the UTM projection with the corresponding zone. Elevations are orthometric and expressed in reference to the Canadian Geodetic Vertical Datum of 2013 (CGVD2013).

'''

## Import Libraries
import geopandas as gpd
import os
from pathlib import Path
import boto3
import re
from datetime import datetime
# import pdal 
# import json
import urllib.request
import zipfile
import os

from urllib.parse import urlparse


def get_s3_objects_from_url(s3_url):
    s3_url = urlparse(s3_url)
    bucket_name = s3_url.netloc.lstrip('/').split('.')[0]
    object_name = s3_url.path.lstrip('/')
    return bucket_name, object_name

## FUNCTIONS
import time
import socket
import urllib3.exceptions

def retry_on_connection(fn, max_retries=5, initial_wait=2, backoff=2):
    wait = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (socket.gaierror, urllib3.exceptions.NewConnectionError, urllib3.exceptions.MaxRetryError) as e:
            # transient network/DNS failure
            if attempt == max_retries:
                raise
            print(f"[retry {attempt}/{max_retries}] network error: {e}; sleeping {wait}s")
            time.sleep(wait)
            wait *= backoff
        except Exception:
            raise
        
def download_s3(s3_url, out_dir, max_retries = 4, intial_wait = 2):
    # Boto3 AWS S3 Client
    s3 = boto3.client('s3') # Define Boto3 client

    bucket_name, object_name = get_s3_objects_from_url(s3_url)

    file_name = object_name.split('/')[-1]
    out_path = os.path.join(out_dir, file_name)

    if os.path.exists(out_path):
        print(f'-ALREADY EXISTS: {file_name}')
    else:
        def download():
            s3.download_file(bucket_name, object_name, out_path)

        retry_on_connection(
            fn = download,
            max_retries=max_retries,
            initial_wait=intial_wait
        )
        print(f'-DOWNLOADED: {file_name}')

    return out_path


## Download LAZ Files 
def download_laz_from_s3(s3_urls, laz_dir):
    ## Download Matching LAZ Files from NRCAN S3 Bucket
    for s3_url in s3_urls:
        download_s3(
            s3_url = s3_url,
            out_dir = laz_dir
        )


def retrieve_tile_index(out_dir):
    if os.path.exists(out_dir) == False:
            os.mkdir(out_dir)
    url = 'https://canelevation-lidar-point-clouds.s3-ca-central-1.amazonaws.com/pointclouds_nuagespoints/Index_LiDARtiles_tuileslidar.zip'
    filename = os.path.join(out_dir, 'tile_index.zip')


    urllib.request.urlretrieve(url, filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

    os.remove(filename)


def upload_folder_to_s3(bucket_name, folder_path, s3_prefix=""):
    s3_client = boto3.client('s3')

    # Walk through all directories and files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_path = os.path.join(root, file)

            # Create the relative S3 key
            relative_path = os.path.relpath(local_path, folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Ensure forward slashes

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_path, bucket_name, s3_key)


def upload_files_to_s3(bucket_name, file_paths, s3_prefix = ''):
    s3_client = boto3.client('s3')

    # Walk through all directories and files
    for file_path in file_paths:

        # Create the relative S3 key
        relative_path = os.path.relpath(file_path)
        s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # Ensure forward slashes

        print(f"Uploading {file_path} to s3://{bucket_name}/{s3_prefix}")
        s3_client.upload_file(file_path, bucket_name, s3_key)


def get_latest_experiment_id(bucket_name, file_key):
    # Initialize S3 client
    s3 = boto3.client('s3')
    # Get the object from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    # Read and decode the content
    return response['Body'].read().decode('utf-8')