import sys
from pathlib import Path

# Make imports robust regardless of run path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metaflow import FlowSpec, step, Config, catch, Parameter, current, project
import os
from src import *
from src.pdal_ops import *
from src.s3_utils import download_s3
from src.parser import omega_parse
import geopandas as gpd
from src.data import get_aoi
from hydra.utils import instantiate, call
from spatialkfold.blocks import spatial_blocks
import logging
import uuid


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

@project(name = "construct_canopy_height_dataset")
class DataFlow(FlowSpec):
    '''
    Canopy Height Estimation with Google Earth Embeddings (GEE) Dataset Preparation Flow
    1. Retrieves the area of interest (AOI) from config
    2. Get intersection between AOI and Point Cloud Tile Index
    3. Generate random n points within Intersection geometry and run spatial K-fold on sample points
    4. Extract GEE Satellite Embeddings from Sample Points and Export to CSV
    5. Download LAZ files from AWS S3, compute Height Above Ground (HAG) with PDAL
    6. Compute Canopy Height Model using 95th Percentile of HAG
    7. Extract Tree Height Data using 10 x 10 grid cell with UTM CRS transform
    
    Example:
    python flows/construct_dataset.py run --max-workers 3 --max-num-splits 4000 --test true
    '''
    
    config_path = Parameter(
        "config",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = "config.yaml"
    )
    s3_bucket = Parameter(
        "s3-bucket",
        help = "S3 URI to root directory hosting lidar metadata (tiles and site boundaries)",
        default = "canopy-flow-data",
        required = False
    )
    tile_index_path = Parameter(
        "tile-index-path",
        default = "s3://canopy-flow-data/canelevation/tile_index.parquet",
        required = False
    )
    sites_path = Parameter(
        "sites-path",
        default = "s3://canopy-flow-data/canelevation/sites.parquet",
        required = False
    )
    datasource_key = Parameter(
        "datasource-key",
        help = "Data source key path within s3 bucket. Directory should contain tile_index.parquet and sites.parquet",
        default = "canelevation",
        required = False
    )
    output_dir = Parameter(
        "output_dir",
        help = "Name of ouput directory on S3 bucket",
        default = "projects",
        required = False
    )
    cache_experiment_id = Parameter(
        "cache-run-id",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = None,
        required = False
    )
    test = Parameter(
        "test",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = False,
        required = False
    )
   
   
    
    
    config = Config("config", default = "config.yaml", parser = omega_parse)

    @step
    def start(self):
        from src.data import create_project_dirs
        
        ## Setup output Directory
        self.experiment_id = str(uuid.uuid4())

        logger.info(f"Project Root Directory: {self.config.paths.project.root}")
        logger.info(f"Experiment ID: {self.experiment_id}")

        # Create all project directories
        create_project_dirs(self.config)

        self.experiment_dir = os.path.join(self.config.paths.project.root, 'experiments', self.experiment_id)
        self.training_dataset_path = os.path.join(self.experiment_dir, self.config.paths.experiments.dataset)
        self.embedding_samples_path =  os.path.join(self.experiment_dir, 'embeddings')
        self.training_prc_dir = os.path.join(self.config.paths.project.root, 'data/training', self.experiment_id)

      
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.training_prc_dir, exist_ok=True)
        os.makedirs(self.embedding_samples_path, exist_ok=True)

        self.next(self.get_aoi_and_tiles)


    @step
    def get_aoi_and_tiles(self):

        from src.ee_utils import get_ee_forest_mask, auth_gee_from_env
        from src.geo_utils import preprocess_tiles
        from src.duckdb_utils import download_overture_water_bodies
        from glob import glob
        from src.sampling import construct_stratified_sample

        os.makedirs('data', exist_ok=True)
        paths = self.config.paths
        if self.cache_experiment_id is not None:
            self.aoi_gdf = gpd.read_parquet(self.aoi_path)
            self.tiles_aoi_bounds_gdf = gpd.read_parquet(self.tiles_aoi_bounds_path)
            tiles_aoi_gdf = gpd.read_parquet(self.config.paths.project.tiles_aoi)
            ch_ids = [path.split('.')[0] for path in os.listdir(self.ch_dir)]
            self.tile_ids = tiles_aoi_gdf[~tiles_aoi_gdf.Tile_name.isin(ch_ids)].tile_id.to_list()
            if self.skip_lidar:
                self.tile_ids = [self.tile_ids[0]]
            logger.info(f"Loading Data from Cache..")
        else:

            sites_path = download_s3(self.sites_path, 'data')

            ## Read AOI GDF
            aoi_config = self.config.data.get_aoi
            aoi_gdf = get_aoi(
                            # sites_path=self.config.datasources[self.config.project.datasource].sites,
                            sites_path=sites_path,
                            name_col=aoi_config.name_col,
                            aoi_name=self.config.project.aoi_name
                            )
            self.utm_crs = aoi_gdf.estimate_utm_crs()
            aoi_gdf.to_crs(self.utm_crs)
            
            ## Read Tile Index
            tile_index_path = download_s3(self.tile_index_path, 'data')
            tile_index_gdf = gpd.read_parquet(tile_index_path).to_crs(self.utm_crs)

            ## Spatial Join Tile Index and AOI
            tiles_aoi_gdf = gpd.sjoin(tile_index_gdf,aoi_gdf[['geometry']].to_crs(tile_index_gdf.crs), predicate='intersects', how='inner')
            tiles_aoi_gdf = tiles_aoi_gdf[tiles_aoi_gdf.geometry.is_valid]
            tiles_aoi_gdf['tile_id'] = list(range(len(tiles_aoi_gdf)))

            ## Get Single geometry of tile boundaries
            tiles_aoi_geom = tiles_aoi_gdf.to_crs(self.utm_crs).buffer(2).reset_index().dissolve().buffer(-2).rename('geometry').reset_index()
            tiles_aoi_bounds_gdf = tiles_aoi_gdf.dissolve().set_geometry(tiles_aoi_geom.geometry.values)
            

            ## Download Forest Mask
            bbox_4326 = aoi_gdf.to_crs(4326).total_bounds
            
            if not os.path.exists(self.config.paths.project.forest_mask):
                auth_gee_from_env()
                get_ee_forest_mask(
                    bbox_4326,
                    out_path =self.config.paths.project.forest_mask, 
                    out_epsg = self.utm_crs.to_epsg()
                    )
            
            ## Download Water Bodies Vector Layer
            if not os.path.exists(self.config.paths.project.water_bodies):
                download_overture_water_bodies(
                    bbox =bbox_4326,
                    out_path = self.config.paths.project.water_bodies
                )
            ## Preprocess Tiles - filter out tiles without forest cover and intersecting with water
            org_length = len(tiles_aoi_gdf)
            tiles_aoi_gdf = preprocess_tiles(
                gdf = tiles_aoi_gdf,
                forest_raster_path= self.config.paths.project.forest_mask,
                min_forest_cover = self.config.data.filter.min_forest_cover,
                water_vector_path = self.config.paths.project.water_bodies
            ).drop(columns = ['index_right'])


            ## Run Get Spatial Blocks for Spatial Cross validation
            blocks_gdf = spatial_blocks(
                gdf = tiles_aoi_gdf, 
                nfolds = self.config.data.sampling.k,
                width = self.config.data.sampling.width, 
                height = self.config.data.sampling.height, 
                method = self.config.data.sampling.method, 
                orientation = self.config.data.sampling.orientation, 
                grid_type = self.config.data.sampling.grid_type,  
                random_state = self.config.data.sampling.random_state
            )
            blocks_gdf = blocks_gdf.set_crs(self.utm_crs, allow_override = True)
            

            
            ## Pass SpatialFolds to tiles
            tiles_aoi_gdf = gpd.overlay(blocks_gdf, tiles_aoi_gdf)
            tiles_aoi_gdf['forest_cover_strata'] = pd.cut(tiles_aoi_gdf.forest_cover, [0.0, 0.3, 0.5, 0.8, 1.0], labels=['low', 'medium','high', 'full'])
            

            ## Stratified sampling of tiles based on forest cover and spatial k-fold ID
            tiles_aoi_gdf, n_samples_per_tile = construct_stratified_sample(tiles_aoi_gdf,
                                        strata_cols=['forest_cover_strata', 'folds'],
                                        min_n = self.config.data.sampling.min_n_per_tile,
                                        n_samples = self.config.data.sampling.n
                                    )
            
            tiles_aoi_gdf = tiles_aoi_gdf.reset_index(drop=True).groupby('folds').apply(lambda group: group.sample(1)) if self.test else tiles_aoi_gdf
            logger.info(f"Tile Preprocessing: {org_length - len(tiles_aoi_gdf)} Tiles Filtered out")
            logger.info(f"Samples per Tile {n_samples_per_tile}")

            ## Export geometries
            aoi_gdf.to_parquet(paths.project.aoi)
            tiles_aoi_bounds_gdf.to_parquet(paths.project.tiles_aoi_bounds)
            tiles_aoi_gdf.to_parquet(paths.project.tiles_aoi)
            blocks_gdf.to_parquet(paths.project.blocks)

            ## Pass artifacts
            self.aoi_gdf = aoi_gdf
            self.tiles_aoi_bounds_gdf = tiles_aoi_bounds_gdf
            # self.tiles_aoi_gdf = tiles_aoi_gdf
            self.n_samples = n_samples_per_tile
            self.blocks_gdf = blocks_gdf

            ## Filter out existing samples
            ch_files = glob(os.path.join(self.config.paths.training.ch_samples, "*.parquet"))
            ch_exists = tiles_aoi_gdf.Tile_name.isin([os.path.basename(file).split('.')[0] for file in ch_files])
            self.tile_ids = tiles_aoi_gdf[~ch_exists].tile_id.to_list()
      

            if len(self.tile_ids) == 0:
                self.tile_ids = [tiles_aoi_gdf.iloc[0].tile_id]

        logger.info(f"Processing {len(self.tile_ids)} Point Cloud Tiles")
        self.next(self.get_lidar_data, foreach='tile_ids')

    @catch(var = "get_lidar_data_failed")
    @step
    def get_lidar_data(self):
        from src.laz_utils import is_copc_vlr_present
        from src.chm import compute_chm
        from src.geo_utils import stratify_raster
        from src.sampling import generate_stratified_random_points, vectorize_strata, sample_raster_points
        import numpy as np
        tile_id = self.input
        tiles_aoi_gdf = gpd.read_parquet(self.config.paths.project.tiles_aoi)
        tile_gdf = tiles_aoi_gdf[tiles_aoi_gdf.tile_id == tile_id]

        ## Setup Paths
        paths = self.config.paths
        s3_url = tile_gdf.URL.item()
        laz_basename = os.path.basename(s3_url)
        tif_basename = laz_basename.split(".")[0] + ".tif"
        logger.info(f"Downloading COPC: {laz_basename}")
        chm_path = os.path.join(paths.prc.chm, tif_basename)
        stratified_path = os.path.join(paths.prc.stratified, tif_basename)
        samples_path = os.path.join(paths.training.ch_samples, laz_basename.split('.')[0] + ".parquet")

        ## Download LAZ COPC from AWS S3
        laz_local_path = download_s3(s3_url , paths.raw.laz)
        # laz_epsg = get_epsg_authority_from_laz(laz_local_path)

        ## Configure Paths
        is_copc = is_copc_vlr_present(laz_local_path)
        
        logger.info(f"Is COPC VLR {is_copc}")

        c1 = is_copc and os.path.exists(chm_path) == False
        if c1:
            logger.info(f"Computing Canopy Height Model: {laz_basename}")
            ## Compute Canopy Height Model
            compute_chm(
                laz_path=laz_local_path,
                out_raster_path=chm_path,
                res_m=self.config.data.res_m,
                statistic=self.config.data.chm.statistic,
                compute_hag=True,
                # polygon = tile_gdf.to_crs(laz_epsg).geometry.item().wkt,
                max_height=self.config.data.filter.max_height
            )

        ## Compute Strata
        c2 = not os.path.exists(stratified_path) and os.path.exists(chm_path)
        if c2:
            logger.info(f"Stratifying CHM: {laz_basename}")
            
            stratify_raster(
                raster_path= chm_path,
                out_raster_path=stratified_path,
                bins = self.config.data.sampling.height_strata,
                band = 1
            )

        ## Structurally Guide Sampling of Strata
        c3 = not os.path.exists(samples_path) and os.path.exists(stratified_path)
        if c3:
            logger.info(f"Running Structurally Guided Sampling: {laz_basename}")
            labels = dict(zip( list(range(len(self.config.data.sampling.height_strata))), self.config.data.sampling.strata_names ))
            bin_ids = list(range(1 , len(self.config.data.sampling.height_strata)+1))
            weights = dict(zip(bin_ids,bin_ids))
            n_per_strata = [int(np.ceil(x / sum(weights) * self.n_samples)) for x in weights]   

            ## Vectorize CHM strata raster
            strata_gdf = vectorize_strata(
                raster_path=stratified_path,
                dissolve=False,
                labels = labels
            )


            if strata_gdf is not None:
                ## Filter Strata vector by area (reduce noise)
                strata_gdf = strata_gdf[strata_gdf.area > self.config.data.sampling.min_area]
                # Generate Strateified Random Points, weighted by Strata
                if len(strata_gdf) > 0:
                    point_samples_gdf = generate_stratified_random_points(
                        strata_gdf=strata_gdf,
                        strata_col="strata",
                        n_per_strata=n_per_strata,
                        explode=True,
                        labels=labels,
                        out_vector_path = samples_path
                    )
                    point_samples_gdf["height"] = sample_raster_points(samples_path, chm_path)
                    point_samples_gdf = point_samples_gdf[(point_samples_gdf.height > self.config.data.filter.min_height) & (point_samples_gdf.height < self.config.data.filter.max_height)]
                    point_samples_gdf = point_samples_gdf.to_crs(self.utm_crs)
                    if len(point_samples_gdf) > 0:
                        point_samples_gdf["tile_id"] = laz_basename.split('.')[0]
                        point_samples_gdf[self.config.data.sampling.id] = [uuid.uuid4().__str__() for i in range(len(point_samples_gdf))]
                        point_samples_gdf.to_parquet(samples_path)
                        self.samples_path = samples_path

    
        self.next(self.join)

    @step
    def join(self, inputs):
        from src.duckdb_utils import merge_parquets_to_gdf
        ## Merge Artifacts
        self.merge_artifacts(inputs, exclude= ["samples_path",
                                                "get_lidar_data_failed",
                                                "_catch_exception",
                                            
                                                ])
        
        paths = self.config.paths
        ## Merge Training Data GeoParquets into single GeoDataFrame and write to file
        self.training_data = merge_parquets_to_gdf(
            in_dir = paths.training.ch_samples, 
            crs = self.utm_crs,
            out_path = self.training_dataset_path
        )
  
        
        self.next(self.construct_spatial_kfold)

    
    @step
    def construct_spatial_kfold(self):

        self.training_data = gpd.overlay(self.training_data, self.blocks_gdf.to_crs(self.training_data.crs))
        self.training_data.to_parquet(self.training_dataset_path)
    
        self.next(self.init_gee)

    @step
    def init_gee(self):
        import ee
        from src.ee_utils import auth_gee_from_env, get_embeddings_image
       
        tiles_aoi_gdf = gpd.read_parquet(self.config.paths.project.tiles_aoi)
        ## Authenticate GEE
        auth_gee_from_env()
    
        ## Get Satelite Embeddings Image
        embeddings_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        ## Configure Query Geometry
        bbox_coords = self.tiles_aoi_bounds_gdf.to_crs(4326).total_bounds
        ## Setup Year Variables
        year1 = tiles_aoi_gdf.year.max().item()
 
        ## Pass Embeddings Images to Metaflow object
        self.embeddings_image = get_embeddings_image(bbox_coords, year1)
        self.folds = self.blocks_gdf.folds.unique()
        
        self.next(self.extract_gee_embeddings, foreach = "folds")

    @step
    def extract_gee_embeddings(self):
        from src.ee_utils import gdf_points_to_ee, auth_gee_from_env
        import geemap
        import pandas as pd

        paths = self.config.paths
        embeddings_csv_path = os.path.join(self.embedding_samples_path, f"embeddings_k{self.input}.csv")
        embeddings_parquet_path = embeddings_csv_path.replace('.csv','.parquet')

        ## Authenticate GEE
        auth_gee_from_env()
        

        k_point_samples = self.training_data[self.training_data.folds == self.input].to_crs(4326)

        if len(k_point_samples)> 0:
            ## Upload point samples to GEE
            logger.info("Uploading sample points to GEE")
            point_samples_ee = gdf_points_to_ee(
                gdf= k_point_samples,
                id_col = self.config.data.sampling.id
            )

            ## Sample Emebdddings from points
            logger.info(f"Sampling GEE Satellite Embeddings with {len(k_point_samples)} Points")
            sampled_data =  self.embeddings_image.sampleRegions(
                collection = point_samples_ee,
                scale = self.config.data.res_m
            )

            logger.info(f"Exporting Sampled Embeddings")

            ## Export Embeddings to CSV File
            geemap.ee_to_csv(sampled_data, embeddings_csv_path)

            try:
                embeddings_df = pd.read_csv(embeddings_csv_path)
                embeddings_df.to_parquet(embeddings_parquet_path)
                os.remove(embeddings_csv_path)
            except:
                logger.warning(f"Failed to read {embeddings_csv_path}")


        self.next(self.merge_training_data)

    @step
    def merge_training_data(self, inputs):
        from src.sampling import remove_height_outliers
        self.merge_artifacts(inputs)
        ## Read Emebeddings CSV File
        import duckdb
        con = duckdb.connect()
        embeddings_df = con.sql(f"SELECT * FROM read_parquet('{self.embedding_samples_path}/*.parquet')").to_df()
        self.training_data = self.training_data.set_index('id').join(
                                                                    embeddings_df.set_index('id'),
                                                                    rsuffix='_embeddings'  # Rename duplicates with suffix
                                                                ).reset_index()

         ## Remove outliers
        n = len(self.training_data )
        self.training_data  = remove_height_outliers(
            df = self.training_data,
            z_threshold=2.5
        )
        logger.info(f"Outliers Removed: {n - len(self.training_data)}")

        logger.info("Exporting Training Data")
        self.training_data.to_parquet(self.training_dataset_path)
    
        self.next(self.upload_dataset_to_s3)

    @step
    def upload_dataset_to_s3(self):

        from src.s3_utils import upload_files_to_s3
        ## Write Latest ID to file
        latest_path = os.path.join(self.config.paths.project.root, "latest.txt")
        open(latest_path, 'w').write(self.experiment_id)

        paths_cfg = self.config.paths.project
        upload_files = [
            paths_cfg.aoi,
            paths_cfg.tiles_aoi,
            paths_cfg.tiles_aoi_bounds,
            paths_cfg.blocks,
            self.training_dataset_path,
            latest_path
        ]

        upload_files_to_s3(
            bucket_name=self.s3_bucket,
            file_paths=upload_files,   
        )
        logger.info(f"Files uploaded to AWS S3 Bucket: {self.s3_bucket}")

        self.next(self.end)
    @step
    def end(self):
        print("ENDING FLOW")




if __name__ == '__main__':
    DataFlow()
