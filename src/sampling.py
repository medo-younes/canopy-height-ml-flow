import rasterio as rio
import numpy as np
from rasterio.features import shapes
import geopandas as gpd

def vectorize_strata(raster_path, strata_col = "strata", band = 1, dissolve = True, labels = None):
    with rio.open(raster_path) as src:
        data= src.read(band)
        mask = data != src.nodata
        out_shapes = list(shapes(data, mask = mask, transform=src.transform))
        crs = src.crs
        features = [{"geometry": geom, "properties": {strata_col: value}} for geom,value in out_shapes]
        if len(features) > 0: 
            strata_gdf = gpd.GeoDataFrame.from_features(features, src.crs).astype({"strata": int})
            strata_gdf = strata_gdf.set_crs(crs)
            
            if labels is not None:
                strata_gdf["label"] = strata_gdf.strata.map(labels)
            return strata_gdf.dissolve(strata_col).reset_index() if dissolve else strata_gdf
        else:
            return None

def generate_stratified_random_points(strata_gdf, n_per_strata, strata_col ="strata",  explode = True, labels = None, out_vector_path = None):
    strata_gdf = strata_gdf.dissolve(strata_col).reset_index()
    point_samples = np.concat([strata_gdf[strata_gdf.strata == strata[strata_col]].sample_points(n_per_strata[strata[strata_col]]) for idx, strata in strata_gdf.iterrows()])
    point_samples_gdf = gpd.GeoDataFrame(dict(strata = [strata.strata for idx, strata in strata_gdf.iterrows()]), geometry=point_samples, crs = strata_gdf.crs)
    if labels is not None:
            point_samples_gdf["label"] = point_samples_gdf.strata.map(labels)

    point_samples_gdf = point_samples_gdf.explode() if explode else point_samples_gdf

    if out_vector_path is not None:
        point_samples_gdf.to_parquet(out_vector_path)

    return point_samples_gdf


def sample_raster_points(vector_path, raster_path):
    
    gdf = gpd.read_parquet(vector_path)
    if gdf.geometry.type.iloc[0] != "Point":
        gdf = gdf.set_geometry(gdf.centroid)
    xy = gdf.get_coordinates().values
    with rio.open(raster_path) as src:
        samples = src.sample(xy)
        return [float(sample.squeeze()) for sample in samples]

    