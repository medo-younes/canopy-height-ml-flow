import ee
import os
from dotenv import load_dotenv



def get_gee_credentials(env_path = ".env"):
    load_dotenv(env_path)
    return ee.ServiceAccountCredentials(
        os.getenv("GEE_SERVICE_ACCOUNT_EMAIL"),
        os.getenv("GEE_SERVICE_ACCOUNT_KEY_PATH")
    )


def auth_gee_from_env(env_path = ".env"):
    credentials = get_gee_credentials(env_path)
    ee.Initialize(credentials)
    print("GEE Authenticated and Initialized")


def gdf_points_to_ee(gdf, id_col ="id"):
        gdf =  gdf.to_crs(4326)
        features = [ ee.Feature(
                        ee.Geometry.Point(row.geometry.x, row.geometry.y),
                        {"id": row[id_col]})
                        for idx, row in gdf.iterrows()
                        ]

        return  ee.FeatureCollection(features)


def get_embeddings_image(bbox, year, out_epsg = None):
    bbox_ee =  ee.Geometry.Rectangle(*bbox)
    year2 = f'{int(year) + 1}-01-01'
    year1 = f'{year}-01-01'
    
    ## Get Satellite Embeddings Image
    embeddings_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    embeddings_image = embeddings_col.filterDate(year1, year2).filterBounds(bbox_ee).mosaic().clip(bbox_ee)
    
    # Check if image exists
    if embeddings_image is None:
        raise ValueError(f"No embeddings image found for year {year} in bbox {bbox}")
    
    # Only reproject if out_epsg is provided and valid
    if out_epsg is not None:
        embeddings_image = embeddings_image.reproject(
            crs=out_epsg,
            scale=10
        )
    
    return embeddings_image


import geemap
def download_ee_image_from_bbox(image, bbox, out_path, scale, epsg):
    bbox_ee  = ee.Geometry.Rectangle(*bbox)
    crs = f"EPSG:{epsg}"
    geemap.download_ee_image(
                            image=image, 
                            filename=out_path, 
                            scale = scale,
                            crs = crs,
                            region = bbox_ee,
                        )
