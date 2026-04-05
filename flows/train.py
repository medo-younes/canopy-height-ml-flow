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
    
    Example:
    python flows/train.py run --max-workers 3 --max-num-splits 4000 --test true

    docker run -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
           -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
           -e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \e
            myounes88/canopy-flow:test python train.py run --max-workers 3 --max-num-splits 4000 --experiment-id a19fc055-f5d-4941-a29c-c76e68ba9238
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
    experiment_id = Parameter(
        "experiment-id",
        help = "Required experiment id for training dataset",
        required = False
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
        from src.s3_utils import get_latest_experiment_id
        ## Setup output Directory
        run_id = str(current.run_id)
        
        
        
        # Get the object from S3
        if self.experiment_id is None:
            experiment_id = get_latest_experiment_id(
                    bucket_name = 'canopy-flow-data',
                    file_key = os.path.join('projects', self.config.project.aoi_name, 'latest.txt')
                    )
        else:
            experiment_id = self.experiment_id
        
        logger.info(f"Experiment ID: {experiment_id}")
        self.project_dir = os.path.join('projects', self.config.project.aoi_name)
        self.experiment_dir = os.path.join(self.project_dir, 'experiments', experiment_id)
        self.dataset_path = "s3://" + os.path.join(self.s3_bucket, self.experiment_dir, self.config.paths.experiments.dataset)
        self.results_path = os.path.join(self.experiment_dir, "results.parquet")
        self.box_plot_path =  os.path.join(self.experiment_dir, "boxplot_comparison.png")
        self.regression_plot_path =  os.path.join(self.experiment_dir, "regression_plot.png")
        self.upload_files = [self.results_path, self.box_plot_path, self.regression_plot_path]

        ## Create Dirs
        os.makedirs(self.experiment_dir, exist_ok=True)


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
        from src.sampling import remove_height_outliers
        import numpy as np
        import geopandas as gpd
        import pandas as pd
        from src.s3_utils import download_s3


        logger.info(f"Dataset Path: {self.dataset_path}")
        local_dataset_path= download_s3(self.dataset_path, self.experiment_dir)
        ## Read Training Data GeoParquet using DuckDB
        training_gdf = read_canopy_height_data(
            vector_path = local_dataset_path,
            min_height = self.config.data.filter.min_height ,
            max_height= self.config.data.filter.max_height,
        )


        ## Subsample 50 points from each strata / fold for testing
        if self.test:
            training_gdf = pd.concat([fold_data.sample(50, replace= True) for idx, fold_data in training_gdf.groupby(['folds','strata'])]).reset_index()

        logger.info(f"Dataset size: {len(training_gdf)}")
       

        # ## Prepare Training Data
        self.X  = training_gdf.loc[:, self.X_cols].values ## Independent Variables: Satelite Embeddings 
        self.y = training_gdf.loc[:,self.y_col].values ## Dependent Variable: Tree Canopy Height
        self.folds, self.n_folds = training_gdf.folds.values, np.unique(training_gdf.folds.values)

        # Example usage
        get_stats(training_gdf, 'height', os.path.join(self.experiment_dir, "height_stats.csv"))     
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
        model_outputs_path = os.path.join(self.experiment_dir, f"{model_name}_results.parquet")

        ## Initialize Model Class
        model = init_model(model_name, random_state=self.config.project.seed)

        print(f"Training Model {model_name} \n {model}")
        param_distributions = get_param_distributions(model_name)

        ## Optuna Cross-validation
        self.optuna_cv = OptunaSearchCV(
            estimator = model,
            param_distributions = param_distributions,
            cv = PredefinedSplit(test_fold=self.folds),
            scoring = self.config.experiment.scoring,
            n_trials = self.config.experiment.n_trials,
            n_jobs = self.n_jobs,
            verbose = self.config.experiment.verbose,
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
        
        
        self.results_df.to_parquet(self.results_path)
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
            out_path = self.box_plot_path
            
            )
        
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
            out_path = self.regression_plot_path
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
        self.best_model_path = os.path.join(self.experiment_dir, f'{best_model_name}_{self.config.project.criterion}_{mean_score:.2f}.pkl')
        with open(self.best_model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.upload_files.append(self.best_model_path)
        self.next(self.upload_results_to_s3)

    @step
    def upload_results_to_s3(self):

        from src.s3_utils import upload_files_to_s3

        ## Write Latest ID to file
        model_path = os.path.join(self.config.paths.project.root, "model.txt")
        open(model_path, 'w').write(self.best_model_path)

        ## Upload files to S3 bucket
        upload_files_to_s3(
            bucket_name=self.s3_bucket,
            file_paths=self.upload_files,   
        )
        logger.info(f"Files uploaded to AWS S3 Bucket: {self.s3_bucket}")

        
        self.next(self.end)


    @step
    def end(self):
        logger.info("Complete")



if __name__ == "__main__":
    TrainFlow()