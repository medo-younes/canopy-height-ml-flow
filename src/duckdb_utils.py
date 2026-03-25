import duckdb 
import boto3



import duckdb
import requests
import geopandas as gpd

def init_db():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL pdal FROM community; LOAD pdal")
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
    catalog = requests.get("https://stac.overturemaps.org/catalog.json").json()
    return catalog["latest"]

def download_overture_water_bodies(
    bbox: list,
    out_path: str | None = None,
    release: str | None = None,
) -> duckdb.DuckDBPyRelation:
    """
    Query Overture Maps water body polygons from S3 as a DuckDB relation.
    """
    if release is None:
        release = get_latest_overture_release()
        print(f"Using Overture release: {release}")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
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