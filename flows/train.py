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


## Configure Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce other logs


class TrainFlow(FlowSpec):
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
    python flows/data_flow.py run --max-workers 3 --max-num-splits 4000 --test true
    '''
    
    config_path = Parameter(
        "config",
        help = "Path to config YAML for configuring metaflow pipeline.",
        default = "config.yaml"
    )
    dataset_id = Parameter(
        "dataset-id",
        help = "Required run id for training dataset",
        default = "test",
        required = True
    )
    test = Parameter(
        "test",
        help = "If to run a test",
        default = False,
        required = False
    )
    
    
    config = Config("config", default = "config.yaml", parser = omega_parse)

    @step
    def start(self):
      
        ## Setup output Directory
        project_id = str(current.run_id)
        dataset_id = str(self.dataset_id)
        self.training_outputs_path = os.path.join(self.config.output.training_outputs_dir,project_id)
        self.out_dir = os.path.join(self.config.output.dir, dataset_id)
        self.embeddings_dir = os.path.join(self.out_dir, "embeddings")
        self.ch_dir = os.path.join(self.out_dir, "canopy_heights")
        self.aoi_path = os.path.join(self.out_dir, "aoi.parquet")
        self.tiles_aoi_path = os.path.join(self.out_dir, "tiles_aoi.parquet")
        self.tiles_aoi_bounds_path = os.path.join(self.out_dir, "tiles_aoi_bounds.parquet")
        self.embeddings_csv_path = os.path.join(self.out_dir,'embeddings.csv')
        self.training_data_path = os.path.join(self.out_dir,'training.parquet')
        self.sample_points_path = os.path.join(self.out_dir,'sample_points.parquet')
        self.blocks_path = os.path.join(self.out_dir,'blocks.parquet')
        
        ## Create Dirs
        os.makedirs(self.training_outputs_path, exist_ok=True)

        ## Setup 
        self.n_jobs = os.cpu_count() - 4
        self.X_cols = [f"A{str(i).zfill(2)}" for i in range(64)]
        self.y_col = 'height'
        self.run_models = self.config.project.run_models
        self.next(self.read_data)


    @step
    def read_data(self):
        from src.geo_utils import read_canopy_height_data
        from src.data import get_stats
        import numpy as np
        import geopandas as gpd
        import pandas as pd

        ## Read Training Data GeoParquet using DuckDB
        training_gdf: gpd.GeoDataFrame = read_canopy_height_data(
            vector_path=self.training_data_path,
            min_height = self.config.data.filter.min_height ,
            max_height= self.config.data.filter.max_height,
            out_epsg = self.config.project.epsg
        )

        if self.test:
            training_gdf = pd.concat([fold_data.sample(55) for idx, fold_data in training_gdf.groupby(['folds','strata'])]).reset_index()
        # ## Prepare Training Data
        self.X  = training_gdf.loc[:, self.X_cols].values ## Independent Variables: Satelite Embeddings 
        self.y = training_gdf.loc[:,self.y_col].values ## Dependent Variable: Tree Canopy Height
        self.folds, self.n_folds = training_gdf.folds.values, np.unique(training_gdf.folds.values)

        # Example usage
        get_stats(training_gdf, 'height', os.path.join(self.training_outputs_path, "height_stats.csv"))     
        self.training_gdf = training_gdf
        self.next(self.dataset_eda)

    @step
    def dataset_eda(self):
        logger.info("Running EDA...")


        self.next(self.spatial_kfold_cv, foreach = "run_models")

    @step 
    def spatial_kfold_cv(self):
        from hydra.utils import instantiate, call
        import optuna
        from optuna.integration import OptunaSearchCV
        from sklearn.model_selection import PredefinedSplit
        from src.models import init_model, get_param_distributions
        from src.model_evaluation import compute_cv_scores
        import pandas as pd

        model_name = self.input
        model_outputs_path = os.path.join(self.training_outputs_path, f"{model_name}_results.parquet")

        ## Initialize Model Class
        model = init_model(model_name, random_state=self.config.project.seed)

        print(f"Training Model {model_name} \n {model}")
        param_distributions = get_param_distributions(model_name)

        ## Optuna Cross-validation
        self.optuna_cv = OptunaSearchCV(
            estimator = model,
            param_distributions = param_distributions,
            cv = PredefinedSplit(test_fold=self.folds),
            scoring = self.config.training.scoring,
            n_trials = self.config.training.n_trials,
            n_jobs = self.n_jobs,
            verbose = self.config.training.verbose,
            refit = True,
            random_state = self.config.project.seed,
            
        )

        ## Run Optuna Cross-validation
        self.optuna_cv.fit(self.X, self.y)


        ## Compute Cross-Validation Scores
        logger.info(f"Computing {model_name} CV Scores")
        self.results_df = compute_cv_scores(
            estimator=self.optuna_cv.best_estimator_,
            X = self.X,
            y = self.y,
            folds = self.folds,
            n_jobs= self.n_jobs
        )

        self.results_df.to_parquet(model_outputs_path)
        
        self.next(self.combine_cv_results)


    @step
    def combine_cv_results(self, inputs):
        import pandas as pd
        ## Get Best Estimators from each Cross Validation Object
        self.results_df = pd.concat([model_results.results_df for model_results in inputs])
        self.optuna_cv_list = [model_results.optuna_cv for model_results in inputs]

        
        ## Merge Artifacts
        self.merge_artifacts(inputs, 
                             exclude = ['optuna_cv', 'results_df', 'optuna_cv_list'])
        
        self.results_df.to_parquet(os.path.join(self.training_outputs_path, "results.parquet"))
        self.next(self.create_plots)

    @step
    def create_plots(self):
        from src.plots import plot_model_comparison_boxplots, regression_scatter_plot
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import ElasticNet
        from xgboost import XGBRegressor
        ## KFold Box Plot
        plot_model_comparison_boxplots(
            self.results_df, 
             score_names = ["R2", "RMSE", "MAE"],
            models = ['EN','RF','XG'],
            out_path = os.path.join(self.training_outputs_path, "boxplot_comparison.png"))
        
        ## Regresiion Scatter Plot
        model_names = ['ElasticNet', 'RandomForest', 'XGBoost']
        model_params = [cv_obj.best_params_ for cv_obj in self.optuna_cv_list]
        model_map = dict(zip(['ElasticNet', 'RandomForest', 'XGBoost'], [ElasticNet, RandomForestRegressor, XGBRegressor]))

        regression_scatter_plot(
            model_map =model_map,
            model_names=model_names,
            data_df=self.training_gdf,
            X_cols=self.X_cols,
            y_col=self.y_col,
            folds=self.folds,
            model_params=model_params,
            figsize=(15, 5),
            lim=(0, self.training_gdf.height.max()+2),
            xlabel="LiDAR Canopy Height (m)",
            ylabel="Predicted Canopy Height (m)",
            out_path = os.path.join(self.training_outputs_path,"regression_plot.png")
        )

        self.next(self.select_best_model)

    @step
    def select_best_model(self):
        from src.model_evaluation import get_best_model
        import pickle
        best_model_name, mean_score, std_score = get_best_model(self.results_df, self.config.project.criterion)
        model_idx = self.run_models.index(best_model_name)
        logger.info(f"Best model: {best_model_name} with mean {self.config.project.criterion}: {mean_score:.4f} +- {std_score:.4f}")

        ## Refit on the Full Dataset
        optuna_cv = self.optuna_cv_list[model_idx]
        model = optuna_cv.best_estimator_
        logger.info(f"Reftting {best_model_name} on Entire Dataset")
        model.fit(self.X, self.y)

        ## Export Pretrained Model
        # Save the model to a file
        best_model_path = os.path.join(self.config.output.models_dir, f'{best_model_name}_{self.config.project.criterion}_{mean_score:.2f}_{current.run_id}.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.next(self.end)
    @step
    def end(self):
        logger.info("Complete")



if __name__ == "__main__":
    TrainFlow()