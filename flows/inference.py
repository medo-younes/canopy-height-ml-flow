import sys
from pathlib import Path

# Make imports robust regardless of run path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from metaflow import FlowSpec, step, Config, resources, Parameter, current
import os
from src.parser import omega_parse
import logging
import optuna

from src.ee_utils import auth_gee_from_env
## Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce other logs


class CanopyHeightInferenceFlow(FlowSpec):
    '''
    
    Example:
    python flows/inference.py run --max-workers 8 --max-num-splits 4000 --test true --model-checkpoint ElasticNet_RMSE_3.94_1774094444055350.pkl
    '''
    
    config_path = Parameter(
        "config",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = "config.yaml"
    )
    model_checkpoint = Parameter(
        "model-checkpoint",
        help = "Path to pretrained regression model checkpoint",
        default = "ElasticNet_RMSE_3.94_1774094444055350.pkl",
        required = True
    )
    test = Parameter(
        "test",
        help = "If to run a test",
        default = False,
        required = False
    )
    cache_run_id = Parameter(
        "cache-run-id",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = None,
        required = False
    )
    
    
    config = Config("config", default = "config.yaml", parser = omega_parse)

    @step
    def start(self):
      
        ## Setup output Directory
        aoi_dir_name =  self.config.project.aoi_name
        self.embeddings_dir = self.config.paths.raw.embeddings
        self.model_checkpoint_path = os.path.join(self.config.paths.models.root, self.model_checkpoint) ## TODO: Get Model from MLFLow registry
        self.canopy_height_preds_dir = os.path.join(self.config.paths.predictions.root, str(current.run_id) if self.cache_run_id is None else self.cache_run_id)
        self.grid_path = os.path.join(self.canopy_height_preds_dir, 'grid.parquet')
        self.embeddings_vrt_path = os.path.join(self.embeddings_dir, f'embeddings_{aoi_dir_name}.vrt')
        self.canopy_heights_preds_vrt_path = os.path.join(self.canopy_height_preds_dir, f'canopy_height_{aoi_dir_name}.vrt')
        ## Create Dirs
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.canopy_height_preds_dir, exist_ok=True)

        self.next(self.read_data)


    @step
    def read_data(self):
        from src.data import get_aoi
        from spatialkfold.blocks import spatial_blocks
        import geopandas as gpd
        logger.info(f"Fetching AOI for {self.config.project.aoi_name}")
        self.aoi_gdf = get_aoi(
                    sites_path=self.config.data.get_aoi.sites_path,
                    name_col=self.config.data.get_aoi.name_col,
                    aoi_name=self.config.project.aoi_name
                    )
        
        self.epsg = self.aoi_gdf.estimate_utm_crs().to_epsg()
        self.aoi_gdf = self.aoi_gdf.to_crs(self.epsg)


        if self.test:
            from shapely.geometry import Polygon
            polygon = self.aoi_gdf.geometry.item()
            width = 10e3
            clipped_polygon = polygon.intersection(Polygon([(polygon.centroid.x - width, polygon.centroid.y - width), (polygon.centroid.x + width, polygon.centroid.y - width), (polygon.centroid.x + width, polygon.centroid.y + width), (polygon.centroid.x - width, polygon.centroid.y + width)]))
            self.aoi_gdf = self.aoi_gdf.set_geometry([clipped_polygon])
        
        logger.info(f"Constructing Mesh Grid for {self.config.project.aoi_name}")

        if os.path.exists(self.grid_path):
            from glob import glob
            grid_gdf = gpd.read_parquet(self.grid_path)
            paths = glob(os.path.join(self.canopy_height_preds_dir, "*.tif"))
            pred_ids = [int(os.path.basename(p).split('.')[0]) for p in paths]
            self.grid_gdf = grid_gdf[~grid_gdf.id.isin(pred_ids)]
        else:
            self.grid_gdf = spatial_blocks(
                gdf = self.aoi_gdf,
                width = self.config.predict.tile_size,
                height = self.config.predict.tile_size,
                nfolds=1,
            ).reset_index().rename(columns = {'index': 'id'}).sort_values('id')
            self.grid_gdf.to_parquet(self.grid_path)

        self.grid_ids = self.grid_gdf.id.to_list()
        logger.info(f"Yielded {len(self.grid_ids)} Grid Cells Accross {self.config.project.aoi_name}")
        
        self.next(self.load_model)

    
    @step
    def load_model(self):
        import pickle
        logger.info(f"Loading Regression Model from: {self.model_checkpoint_path}")

        ## Load Model using Pickle 
        # TODO: Load model from MLFlow Model Registry
        with open(self.model_checkpoint_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

        logger.info("Model Successfully Loaded")
        

        self.next(self.get_ee_image)


    @step
    def get_ee_image(self):
        
        import ee
        from glob import glob
        from src.ee_utils import get_embeddings_image
        auth_gee_from_env()

        ## Configure Query Geometry
        bbox_coords = self.aoi_gdf.to_crs(4326).total_bounds
        year = 2019 ## TODO: get year programmatically
        self.embeddings_image = get_embeddings_image(bbox = bbox_coords, year =year)    

        self.next(self.predict_canopy_height, foreach = 'grid_ids')

    @step
    def predict_canopy_height(self):
        import geemap
        import ee
        from src.ee_utils import download_ee_image_from_bbox
        from src.models import predict_canopy_height_from_embeddings
        auth_gee_from_env()

        grid_id = self.input
        bbox = self.grid_gdf[self.grid_gdf.id == grid_id].to_crs(4326).total_bounds
        embeddings_path = os.path.join(self.embeddings_dir, f"{grid_id}.tif")
        canopy_height_path = os.path.join(self.canopy_height_preds_dir, f'{grid_id}.tif')

        ## download Embeddings as GeoTiff
        
        
        logger.info(f"Downloading GEE Embeddings Images: Grid ID {grid_id} | BBOX {bbox}")
        if not os.path.exists(embeddings_path):
            download_ee_image_from_bbox(
                image = self.embeddings_image, 
                bbox = bbox, 
                out_path = embeddings_path, 
                scale = 10, 
                epsg=self.epsg
                
                )
            
        ##PRedict Tree Canopy Height with pretrained Model
        if os.path.exists(canopy_height_path) == False:
            logger.info(f"Predicting Canopy Height: Grid ID {grid_id} | BBOX {bbox}")
            predict_canopy_height_from_embeddings(
                model = self.model,
                embeddings_path = embeddings_path,
                out_path = canopy_height_path
            )   

        if os.path.exists(canopy_height_path):
            self.canopy_height_path = canopy_height_path
            self.embeddings_path = embeddings_path

        self.next(self.join)


    @step
    def join(self, inputs):
        from osgeo import gdal
        from glob import glob
        from src.geo_utils import clip_raster_with_vector

        logger.info("Merging Artifacts")
        self.merge_artifacts(inputs,
                             exclude = ["grid_ids", 
                                        "embeddings_image", 
                                        "embeddings_path", 
                                        "canopy_height_path"])
        


        ## Write VRT for Emebddings and Canopy Height Predictions
        ## Get VRT Tile Paths
        logger.info(f"Building VRT")
        ch_pred_paths = glob(os.path.join(self.canopy_height_preds_dir, "*.tif"))
        vrt_ch = gdal.BuildVRT(
            destName = self.canopy_heights_preds_vrt_path, 
            srcDSOrSrcDSTab=ch_pred_paths
        )

        # vrt_embeddings = None
        vrt_ch = None

        ## Clip VRT To AOI Polygon
        logger.info(f"Clipping VRT to AOI Extent")
        clip_raster_with_vector(
            raster_path=self.canopy_heights_preds_vrt_path, 
            vector_path=self.config.paths.project.aoi, 
            output_path=self.canopy_heights_preds_vrt_path
            )

        self.next(self.end)
    @step
    def end(self):
        logger.info("INFERENCE COMPLETE")



if __name__ == '__main__':
    CanopyHeightInferenceFlow()



        