
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import PredefinedSplit
import numpy as np
import pandas as pd

def compute_cv_scores(estimator, X, y, folds, n_jobs = -1):


    cv = PredefinedSplit(test_fold=folds)
    r2_scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2", n_jobs=n_jobs)
    rmse_scores = cross_val_score(estimator, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=n_jobs) * -1
    mae_scores = cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=n_jobs) * -1

    scores = np.array( [r2_scores, rmse_scores, mae_scores]).T
    results_df = pd.DataFrame(scores, columns = ["R2", 'RMSE','MAE'])
    results_df['folds'] = list(range(1, len(results_df)+ 1))
    results_df['model'] = estimator.__class__.__name__

    return results_df


def get_fold_data(data_df, X_cols, y_col, fold):
    X_train = data_df[data_df.folds != fold][X_cols].values
    X_val = data_df[data_df.folds == fold][X_cols].values
    y_train = data_df[data_df.folds != fold][y_col].values
    y_val = data_df[data_df.folds == fold][y_col].values
    return X_train, y_train, X_val, y_val



def get_cv_predictions(data_df, model_map, model_params, n_folds, X_cols, y_col):

    model_names = model_map.keys()   
    val_data = []
    outputs = [[] for _ in model_names]
    folds_ = []
    for fold in n_folds:
        X_train, y_train, X_val, y_val = get_fold_data(data_df,X_cols, y_col, fold)
        models = [model_map[model_name](**params) for model_name, params in zip(model_names, model_params)]
        y_pred = [model.fit(X_train, y_train).predict(X_val) for model in models]
        for i in range(len(y_pred)):
            outputs[i].extend(y_pred[i])

        val_data.extend(y_val)
        folds_.extend([fold] * len(y_val))
    outputs.append(val_data)
    outputs.append(folds_)

    columns = list(model_names)
    columns.extend(['y_true','folds'])

    return pd.DataFrame(np.array(outputs).T, columns= columns).astype({"folds": int})
