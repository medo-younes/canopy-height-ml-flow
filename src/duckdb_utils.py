import os
import duckdb
import boto3
import requests
import geopandas as gpd


def load_duckdb_extension(con, extension, from_community=False):
    try:
        con.execute(f"LOAD {extension};")
    except duckdb.IOException:
        install_cmd = f"INSTALL {extension}"
        if from_community:
            install_cmd += " FROM community"
        try:
            con.execute(f"{install_cmd}; LOAD {extension};")
        except duckdb.IOException as exc:
            raise RuntimeError(
                f"Failed to install/load DuckDB extension '{extension}'. "
                "This likely means the Docker container has no internet access to download DuckDB extensions. "
                "Pre-install the extension in the image or enable network access."
            ) from exc


def init_db():
    con = duckdb.connect()
    load_duckdb_extension(con, "httpfs")
    load_duckdb_extension(con, "spatial")
    load_duckdb_extension(con, "pdal", from_community=True)
    return con

def configure_aws_credentials(con, region):
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials:
        con.execute(f"SET s3_access_key_id = '{credentials.access_key}';")
        con.execute(f"SET s3_secret_access_key = '{credentials.secret_key}';")
        if credentials.token:
            con.execute(f"SET s3_session_token = '{credentials.token}';")
    
    con.execute(f"SET s3_region = '{region}';")
    return con



def get_latest_overture_release() -> str:
    """Fetch the latest Overture Maps release tag from their STAC catalog."""
    env_release = os.environ.get("OVERTURE_MAPS_RELEASE")
    if env_release:
        return env_release

    try:
        response = requests.get("https://stac.overturemaps.org/catalog.json", timeout=10)
        response.raise_for_status()
        catalog = response.json()
        return catalog["latest"]
    except requests.RequestException as exc:
        raise RuntimeError(
            "Unable to fetch latest Overture Maps release from stac.overturemaps.org. "
            "If the container cannot access the internet, set OVERTURE_MAPS_RELEASE to a known release tag."
        ) from exc


def download_overture_water_bodies(
    bbox: list,
    out_path: str | None = None,
    release: str  = "2026-03-18.0",
) -> duckdb.DuckDBPyRelation:
    """
    Query Overture Maps water body polygons from S3 as a DuckDB relation.
    """
    
    con = duckdb.connect()
    load_duckdb_extension(con, "httpfs")
    load_duckdb_extension(con, "spatial")
    con.execute("SET s3_region = 'us-west-2';")
    con.execute("SET s3_use_ssl = true;")

    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=base/type=water/*"

    x1, y1, x2, y2 = bbox
    query = f"""
        SELECT
            id,
            ST_AsText(geometry) AS wkt,
            names,
            subtype,
            class
        FROM read_parquet('{s3_path}', hive_partitioning=true)
        WHERE bbox.xmin BETWEEN {x1} AND {x2}
        AND bbox.ymin BETWEEN {y1} AND {y2}
        AND ST_GeometryType(geometry) != 'LINESTRING'
        AND ST_GeometryType(geometry) != 'POINT'
    """

    df = con.sql(query).to_df()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['wkt']), crs=4326)
    if out_path is not None:
        gdf.to_parquet(out_path)
    return gdf




def merge_parquets_to_gdf(in_dir, crs = 4326, out_path = None):
    con = init_db()
    con.sql(f"CREATE TABLE geodataframe AS SELECT *,ST_AsText(geometry) AS wkt FROM read_parquet('{in_dir}/*.parquet')")
    df = con.sql("SELECT * FROM geodataframe").to_df()
    # Create a GeoDataFrame from the DataFrame, converting the WKT column to a geometry column
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['wkt']), crs=crs).drop(columns = 'wkt')

    if out_path is not None:
        gdf.to_parquet(out_path)

    return gdf