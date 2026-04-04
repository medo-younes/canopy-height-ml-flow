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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

@project(name = "data_flow")
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
    python flows/download_data.py run --max-workers 3 --max-num-splits 4000 --test true
    '''
    
    config_path = Parameter(
        "config",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = "config.yaml"
    )
    cache_run_id = Parameter(
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
    test_size = Parameter(
        "test-size",
        help = "Size of test sample size",
        default = 10,
        required = False
    )
    skip_lidar = Parameter(
        "skip-lidar",
        help = "Size of test sample size",
        default = False,
        required = False
    )
    
    
    config = Config("config", default = "config.yaml", parser = omega_parse)

    @step
    def start(self):
        from src.data import create_project_dirs
        print(self.config.paths.project.root)
        ## Setup output Directory
        project_id = str(self.cache_run_id) if self.cache_run_id is not None else str(current.run_id) 
 
        # Create all project directories
        create_project_dirs(self.config)
      
        self.next(self.get_aoi_and_tiles)


    @step
    def get_aoi_and_tiles(self):

        from src.ee_utils import get_ee_forest_mask, auth_gee_from_env
        from src.geo_utils import preprocess_tiles
        from src.duckdb_utils import download_overture_water_bodies
        from glob import glob

        paths = self.config.paths
        if self.cache_run_id is not None:
            self.aoi_gdf = gpd.read_parquet(self.aoi_path)
            self.tiles_aoi_bounds_gdf = gpd.read_parquet(self.tiles_aoi_bounds_path)
            self.tiles_aoi_gdf = gpd.read_parquet(self.tiles_aoi_path)
            ch_ids = [path.split('.')[0] for path in os.listdir(self.ch_dir)]
            self.tile_ids = self.tiles_aoi_gdf[~self.tiles_aoi_gdf.Tile_name.isin(ch_ids)].tile_id.to_list()
            if self.skip_lidar:
                self.tile_ids = [self.tile_ids[0]]
            logger.info(f"Loading Data from Cache..")
        else:
            ## Read AOI GDF
            aoi_config = self.config.data.get_aoi
            aoi_gdf = get_aoi(
                            sites_path=self.config.datasources[self.config.project.datasource].sites,
                            name_col=aoi_config.name_col,
                            aoi_name=self.config.project.aoi_name
                            )
            self.utm_crs = aoi_gdf.estimate_utm_crs()
            aoi_gdf.to_crs(self.utm_crs)
            
            ## Read Tile Index
            tile_index_gdf = gpd.read_parquet(self.config.datasources[self.config.project.datasource].tile_index).to_crs(self.utm_crs)

            ## Spatial Join Tile Index and AOI
            tiles_aoi_gdf = gpd.sjoin(tile_index_gdf,aoi_gdf[['geometry']].to_crs(tile_index_gdf.crs), predicate='intersects', how='inner')
            tiles_aoi_gdf = tiles_aoi_gdf[tiles_aoi_gdf.geometry.is_valid]
            tiles_aoi_gdf['tile_id'] = list(range(len(tiles_aoi_gdf)))

            ## Get Single geometry of tile boundaries
            tiles_aoi_geom = tiles_aoi_gdf.to_crs(self.utm_crs).buffer(2).reset_index().dissolve().buffer(-2).rename('geometry').reset_index()
            tiles_aoi_bounds_gdf = tiles_aoi_gdf.dissolve().set_geometry(tiles_aoi_geom.geometry.values)
            tiles_aoi_gdf = tiles_aoi_gdf.sample(self.test_size) if self.test else tiles_aoi_gdf

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
            )
            logger.info(f"Tile Preprocessing: {org_length - len(tiles_aoi_gdf)} Tiles Filtered out")

            ## Export geometries
            aoi_gdf.to_parquet(paths.project.aoi)
            tiles_aoi_bounds_gdf.to_parquet(paths.project.tiles_aoi_bounds)
            tiles_aoi_gdf.to_parquet(paths.project.tiles_aoi)

            ## Pass artifacts
            self.aoi_gdf = aoi_gdf
            self.tiles_aoi_bounds_gdf = tiles_aoi_bounds_gdf
            self.tiles_aoi_gdf = tiles_aoi_gdf

            ## Filter out existing samples
            ch_files = glob(os.path.join(self.config.paths.training.ch_samples, "*.parquet"))
            ch_exists = self.tiles_aoi_gdf.Tile_name.isin([os.path.basename(file).split('.')[0] for file in ch_files])
            self.tile_ids = self.tiles_aoi_gdf[~ch_exists].tile_id.to_list()
      

            if len(self.tile_ids) == 0:
                self.tile_ids = [self.tiles_aoi_gdf.iloc[0].tile_id]

        logger.info(f"Processing {len(self.tile_ids)} Point Cloud Tiles")
        self.next(self.get_lidar_data, foreach='tile_ids')

    @catch(var = "get_lidar_data_failed")
    @step
    def get_lidar_data(self):
        from src.laz_utils import is_copc_vlr_present
        from src.chm import compute_chm
        from src.geo_utils import stratify_raster, duckdb_get_intersection
        from src.sampling import generate_stratified_random_points, vectorize_strata, sample_raster_points
        import uuid
        from src.laz_utils import get_epsg_authority_from_laz
        tile_id = self.input
        tile_gdf = self.tiles_aoi_gdf[self.tiles_aoi_gdf.tile_id == tile_id]

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
                        n_per_strata=weights,
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
                                                "_catch_exception"])
        
        paths = self.config.paths
        ## Merge Training Data GeoParquets into single GeoDataFrame and write to file
        self.training_data = merge_parquets_to_gdf(
            in_dir = paths.training.ch_samples, 
            crs = self.utm_crs,
            out_path = paths.training.training_data
        )
  
        
        self.next(self.construct_spatial_kfold)

    
    @step
    def construct_spatial_kfold(self):
        paths = self.config.paths

          
        ## Run Spatial K fold with Kmeans  
        self.blocks_gdf = spatial_blocks(
            gdf = self.training_data, 
            nfolds = self.config.data.sampling.k,
            width = self.config.data.sampling.width, 
            height = self.config.data.sampling.height, 
            method = self.config.data.sampling.method, 
            orientation = self.config.data.sampling.orientation, 
            grid_type = self.config.data.sampling.grid_type,  
            random_state = self.config.data.sampling.random_state
        )
        self.blocks_gdf = self.blocks_gdf.set_crs(self.utm_crs, allow_override = True)
        self.blocks_gdf.to_parquet(paths.training.blocks)


        self.training_data = gpd.overlay(self.training_data, self.blocks_gdf.to_crs(self.training_data.crs))
        self.training_data.to_parquet(self.config.paths.training.training_data)
    
        self.next(self.init_gee)

    @step
    def init_gee(self):
        import ee
        from src.ee_utils import auth_gee_from_env
       
     
        ## Authenticate GEE
        auth_gee_from_env()
    
        ## Get Satelite Embeddings Image
        embeddings_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        ## Configure Query Geometry
        bbox_coords = self.tiles_aoi_bounds_gdf.to_crs(4326).total_bounds
        bbox_ee =  ee.Geometry.Rectangle(*bbox_coords)
        ## Setup Year Variables
        year1 = self.tiles_aoi_gdf.year.max().item()
        year2 = f'{int(year1) + 1}-01-01'
        year1 = f'{year1}-01-01'
        logger.info(f"Configured Year Range: {year1} -> {year2}")
        
        ## Get Emebeddings image from Target Year
        embeddings_image = embeddings_col.filterDate(year1, year2).filterBounds(bbox_ee).first()

        ## Pass Embeddings Images to Metaflow object
        self.embeddings_image = embeddings_image
        self.folds = self.blocks_gdf.folds.unique()
        
        self.next(self.extract_gee_embeddings, foreach = "folds")

    @step
    def extract_gee_embeddings(self):
        from src.ee_utils import gdf_points_to_ee, auth_gee_from_env
        import geemap
        import pandas as pd

        paths = self.config.paths
        embeddings_csv_path = os.path.join(paths.training.embedding_samples, f"embeddings_k{self.input}.csv")
        embeddings_parquet_path = embeddings_csv_path.replace('.csv','.parquet')

        ## Authenticate GEE
        auth_gee_from_env()
        

        k_point_samples = self.training_data[self.training_data.folds == self.input].to_crs(4326)
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

        embeddings_df = pd.read_csv(embeddings_csv_path)
        embeddings_df.to_parquet(embeddings_parquet_path)
        os.remove(embeddings_csv_path)
        self.next(self.merge_training_data)

    @step
    def merge_training_data(self, inputs):
        from src.sampling import remove_height_outliers
        self.merge_artifacts(inputs)
        ## Read Emebeddings CSV File
        import duckdb
        con = duckdb.connect()
        embeddings_df = con.sql(f"SELECT * FROM read_parquet('{self.config.paths.training.embedding_samples}/*.parquet')").to_df()
        self.training_data = self.training_data.set_index('id').join(embeddings_df.set_index('id')).reset_index()

         ## Remove outliers
        n = len(self.training_data )
        self.training_data  = remove_height_outliers(
            df = self.training_data,
            z_threshold=2.5
        )
        logger.info(f"Outliers Removed: {n - len(self.training_data )}")

        logger.info("Exporting Training Data")
        self.training_data.to_parquet(self.config.paths.training.training_data)
        self.next(self.end)


    @step
    def end(self):
        print("ENDING FLOW")




if __name__ == '__main__':
    DataFlow()
