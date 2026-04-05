
from shapely.geometry import box
import geopandas as gpd
import numpy as np 
import rasterio as rio
import pyarrow.parquet as pq
import json

def meshgrid_from_coords(bbox, x_step, y_step):
        x1, y1, x2, y2 = bbox
        xs1, ys1 = np.arange(x1, x2, x_step), np.arange(y2, y1, -y_step)
        xs2, ys2 = np.arange(x1 + x_step, x2 + x_step, x_step), np.arange(y2 - y_step, y1 - y_step, -y_step)
        xs1, ys1 = np.meshgrid(xs1,ys1)
        xs2, ys2 = np.meshgrid(xs2, ys2)
        return xs1, ys1, xs2, ys2


def construct_mesh_grid_gdf(bbox, size_factor = 10):

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    x_step = width / size_factor
    y_step = height/  size_factor
    
    xs1, ys1, xs2, ys2 = meshgrid_from_coords(bbox, x_step, y_step)
    tiles = [box(*[xc1, yc1, xc2, yc2]) for xc1, yc1, xc2, yc2 in zip(xs1.flatten(), ys1.flatten(), xs2.flatten(), ys2.flatten()) ]
    return gpd.GeoDataFrame(geometry=tiles)



def get_pixel_bounds_from_points(point_coords, bbox, res):
    x1, y1, x2, y2 = bbox

    transform = rio.transform.from_origin(x1,y2, res,res)

    ## Get Pixel Index
    xs,ys =point_coords
    rows, cols = rio.transform.rowcol(transform, xs, ys, op = np.ceil)
    rows, cols = rows.astype(int), cols.astype(int)
    # print(rows,cols)

    ## Get Pixel Bounds
    xs1, ys2 = rio.transform.xy(transform, rows,cols, offset = "ul")
    xs2, ys1 = rio.transform.xy(transform, rows,cols, offset = "lr")

    return xs1, ys1, xs2, ys2


def stratify_raster(raster_path, out_raster_path, bins, band = 1):
    with rio.open(raster_path) as src:

        data = src.read(band)
        mask = (data != src.nodata)
        data_masked = data[mask]

        
        stratified = np.digitize(data_masked,bins, right=True)
        out_arr = np.full(data.shape, 0, dtype=rio.int8)

        out_arr = out_arr.flatten()
        out_arr[np.where(mask.flatten())[0]] = stratified
        out_arr = out_arr.reshape(data.shape)

    
        profile = src.profile
        profile.update({
            "dtype": rio.int8,
            "nodata": 0
        })

        with rio.open(out_raster_path, "w", 
                    **profile
                    ) as out:
            out.write(out_arr, 1)





def duckdb_get_intersection(gdf, in_epsg, intersect_vector_path):
    import duckdb

    if gdf is None or len(gdf) == 0:
        return None

    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial")

    in_epsg = f"'EPSG:{in_epsg}'"
    out_epsg = f"'EPSG:{gdf.crs.to_epsg()}'"
    union_geom = gdf.unary_union
    if union_geom.is_empty:
        return None
    wkt = union_geom.wkt

    df = con.sql(f''' 
            SELECT *,
            ST_Transform(geometry,{in_epsg},{out_epsg}) AS geom_reproj, 
            ST_AsText(geom_reproj) as wkt
            FROM read_parquet('{intersect_vector_path}')
            WHERE ST_Intersects(geom_reproj, ST_GeomFromText('{wkt}'));
            ''').to_df()

    if len(df) > 0:
        return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['wkt']), crs=gdf.crs)   
    else: 
        return None




def read_canopy_height_data(vector_path, min_height, max_height):
    # Read the parquet file directly with geopandas
    gdf = gpd.read_parquet(vector_path)
    
    # Filter by height values
    gdf = gdf[(gdf['height'] > min_height) & (gdf['height'] <= max_height)]
    
    # Remove rows with null geometries
    return gdf.dropna()



def sample_percent_cover_from_mask(gdf, raster_path, band):
    with rio.open(raster_path) as src:
        gdf = gdf.to_crs(src.crs)
        transform = src.transform
        coords = gdf.geometry.bounds.values
        windows = [rio.windows.from_bounds(x1, y1,x2,y2, transform) for x1,y1,x2,y2 in coords]
        return [src.read(band, window = window).mean() for window in windows]
    


def get_crs_epsg_from_parquet(file_path):
    # Read only the metadata (no data loaded into memory)
    metadata = pq.read_metadata(file_path)
    
    # Decode the 'geo' metadata (GeoParquet spec stores CRS here)
    geo_metadata = json.loads(metadata.metadata[b'geo'].decode('utf-8'))
    
    # Extract the EPSG code from the geometry column's CRS
    crs_info = geo_metadata['columns']['geometry']['crs']
    epsg_code = crs_info['id']['code']
    return epsg_code


def preprocess_tiles(gdf, forest_raster_path, water_vector_path, min_forest_cover= 0.25):

    ## Compute Forest Percentage Cover
    gdf['forest_cover'] = sample_percent_cover_from_mask(
        raster_path = forest_raster_path,
        band = 1,
        gdf = gdf,
    )
    ## Remove Tiles with low forst cover
    gdf = gdf[gdf.forest_cover >= min_forest_cover]

    ## Mask out water bocies
    water_gdf = gpd.read_parquet(water_vector_path).to_crs(gdf.crs)
    gdf = gdf.overlay(water_gdf[['geometry']], how="difference")

    ## Remove Small tiles based on # of Stds from mean area
    outlier_area = gdf.area.mean() - gdf.area.std() * 3.0
    gdf = gdf[gdf.area > outlier_area]
    return gdf



from osgeo import gdal


    
def clip_raster_with_vector(raster_path, vector_path, output_path, nodata_value=-9999.0):
    """
    Clips a raster file using a vector polygon mask.

    :param input_raster: Path to the input raster file (e.g., 'input.tif').
    :param input_vector: Path to the clipping vector file (e.g., 'clip.shp').
    :param output_raster: Path for the output clipped raster file (e.g., 'output.tif').
    :param nodata_value: The value to use for pixels outside the clip area.
    """
    vector_epsg = get_crs_epsg_from_parquet(vector_path)
    try:

        # Use gdal.Warp to perform the clipping operation
        ds = gdal.Warp(
            output_path,
            raster_path,
            format='GTiff',  # Output format (GeoTIFF is common)
            cutlineDSName=vector_path,  # The vector dataset to use as a cutline
            cropToCutline=True,  # Crops the output extent to the cutline's bounding box
            dstNodata=nodata_value,  # Set the NoData value for areas outside the mask
            creationOptions=['COMPRESS=LZW'], # Optional: Adds LZW compression
            cutlineSRS = f"EPSG:{vector_epsg}"
        )
        # Close the dataset
        ds = None
        print(f"Successfully clipped raster saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")