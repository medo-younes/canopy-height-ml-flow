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