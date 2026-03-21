import pandas as pd 
import geopandas as gpd



def get_aoi(sites_path, name_col, aoi_name):
    sites_gdf = gpd.read_file(sites_path)
    return sites_gdf[sites_gdf[name_col] == aoi_name].reset_index(drop=True)


def get_stats(data_df, col, out_path):
    from tabulate import tabulate
    stats = data_df[col].describe()
    stats.reset_index().to_csv(out_path)
    

