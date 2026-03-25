import pandas as pd 
import geopandas as gpd
import os


def get_aoi(sites_path, name_col, aoi_name):
    sites_gdf = gpd.read_parquet(sites_path)
    return sites_gdf[sites_gdf[name_col] == aoi_name].reset_index(drop=True)


def get_stats(data_df, col, out_path):
    stats = data_df[col].describe()
    stats.reset_index().to_csv(out_path)
    


def create_project_dirs(config):
    """
    Create all required directories for the project paths defined in config.yaml.
    
    Args:
        config: OmegaConf config object with resolved paths
    """
    dirs_to_create = [
        config.paths.project.root,
        config.paths.raw.root,
        config.paths.raw.laz,
        config.paths.raw.embeddings,
        config.paths.prc.root,
        config.paths.prc.chm,
        config.paths.prc.stratified,
        config.paths.training.root,
        config.paths.training.data,
        config.paths.training.outputs,
        config.paths.training.ch_samples,
        config.paths.training.embedding_samples,
        config.paths.models.root,
    ]
    
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)
    
    print(f"Created project directories under {config.paths.project.root}")