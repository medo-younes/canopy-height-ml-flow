
from shapely.geometry import box
import geopandas as gpd
import numpy as np 
import rasterio as rio


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




def read_canopy_height_data(vector_path, min_height, max_height, out_epsg):
    import duckdb

    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial")

    con.sql(f"CREATE TABLE sample_points AS SELECT *,ST_AsText(geometry) AS wkt FROM read_parquet('{vector_path}')")
    df = con.sql(f"SELECT * FROM sample_points WHERE height > {min_height} AND height <= {max_height}").to_df()

    # Create a GeoDataFrame from the DataFrame, converting the WKT column to a geometry column
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['wkt']), crs=out_epsg).drop(columns = 'wkt')
