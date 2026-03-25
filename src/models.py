from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna
import os




def init_model(model_name, random_state):
    """
    Initialize the correct model based on model name.

    Args:
        model_name (str): Name of the model ('ElasticNet', 'RandomForestRegressor', 'XGBoostRegressor')

    Returns:
        sklearn estimator: The initialized model
    """
    if model_name == "ElasticNet":
        return ElasticNet(max_iter=10000, random_state=random_state)
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(random_state=random_state)
    elif model_name == "XGBRegressor":
        return XGBRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_param_distributions(model_name):
    """
    Get the parameter distributions for Optuna hyperparameter search based on model name.

    Args:
        model_name (str): Name of the model

    Returns:
        dict: Parameter distributions for Optuna
    """
    if model_name == "ElasticNet":
        return {
            "alpha": optuna.distributions.FloatDistribution(1e-5, 10.0, log=True),
            "l1_ratio": optuna.distributions.FloatDistribution(0.0, 1.0)
        }
    elif model_name == "RandomForestRegressor":
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 500, log=True),
            "max_depth": optuna.distributions.IntDistribution(5, 50, log=True),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
            "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", 0.5]),
        }
    elif model_name == "XGBRegressor":
        return {
            'n_estimators': optuna.distributions.IntDistribution(10, 500, log=True),
            'learning_rate': optuna.distributions.FloatDistribution(1e-4, 0.3, log=True),
            'max_depth': optuna.distributions.IntDistribution(3, 50, log=True),
            'subsample': optuna.distributions.FloatDistribution(0.5, 1.0),
            'colsample_bytree': optuna.distributions.FloatDistribution(0.5, 1.0),
            'gamma': optuna.distributions.FloatDistribution(0.0, 5.0),
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    


import rasterio as rio
import numpy as np


def preprocess_embeddings(data, embedding_dims= 64):
    data = data.transpose(1,2,0).reshape(-1, embedding_dims)
    nan_mask = (np.isinf(data) | np.isnan(data)).any(axis = 1)
    masked_data = data[~nan_mask]
    return masked_data, nan_mask


def postprocess_predictions(preds, data, nan_mask):
    height, width = data.shape[1:]
    size = width * height
    output = np.full(shape=(size,), fill_value=np.nan, dtype=np.float32)
    output[~nan_mask] = preds
    output = np.maximum(output, 0, where=~np.isnan(output), out=output)
    return output.reshape(height, width)

def predict_canopy_height_from_embeddings(model, embeddings_path, out_path, embedding_dims= 64, out_dtype = rio.float32):
    with rio.open(embeddings_path) as src:
        data = src.read()
        masked_data, nan_mask = preprocess_embeddings(data, embedding_dims)  ## Preprocess Embeddings - mask out nan / inf values

        size = masked_data.shape[0]
        if size > 0:
            preds = model.predict(masked_data) ## Make canopy height predictions with pretrainedm odel
            preds_image = postprocess_predictions(preds, data, nan_mask) # Postprocess canopy height predictions - fill nan values with zeroes and add predicitons to image

            ## Export PRedicted Canopy Height Model
            profile = src.profile
            profile.update({
                "count": 1,
                "dtype": out_dtype
            })
            with rio.open(out_path, "w", **profile) as out:
                out.write(preds_image, 1)
            return preds_image
        else:
            return None
