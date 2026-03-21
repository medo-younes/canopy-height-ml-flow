from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna


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
    elif model_name == "XGBoostRegressor":
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
    elif model_name == "XGBoostRegressor":
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