
import seaborn as sns
import matplotlib.pyplot as plt


def plot_height_distribution(data, xlabel,title, bins = 50, kde=True, color ='green', out_path = None, figsize = (10,6), ax = None):
    # Simple - seaborn does it all

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    sns.histplot(data, bins=bins, kde=kde, color=color, ax = ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path)


    plt.show()


def plot_embeddings_scatter(data_df, emb_cols, height_col, color ='black', figsize= (20,20), out_path = None):
    fig,axes = plt.subplots(8,8, figsize = figsize)
    for i,ax in enumerate(axes.flatten()):
        data_df.plot.scatter(x = emb_cols[i], y = height_col, ax = ax,  s= 1, c=color)
        corr= data_df[[height_col, emb_cols[i]]].corr()[emb_cols[i]].iloc[0].item()
        corr = round(corr,2)
        ax.set_title(f"{emb_cols[i]} | {corr}")
        
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path)



def plot_model_comparison_boxplots(results_df, score_names, models,figsize=(10, 4), out_path= None):
    """
    Create beautiful box plots comparing model performance across different metrics.
    
    Parameters:
    - all_results: List of DataFrames containing model results
    - score_names: List of score names (e.g., ["R2", "RMSE", "MAE"])
    - models: List of model names (e.g., ['EN','RF','XG'])
    - ranges: List of [min, max] ranges for each score
    - figsize: Tuple for figure size
    """
  
    fig, ax = plt.subplots(1, len(score_names), figsize=figsize)
    ranges = [[0, 1], [0, results_df.RMSE.max() + 1], [0, results_df.MAE.max() + 1]]
    results_grouped = results_df.groupby('model')
    # Define colors for different models
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightcyan']

    j = 0
    for i, score_name in enumerate(score_names):
        for model_name, model_data in results_grouped:
            bp = ax[i].boxplot(model_data[score_name], positions=[j], 
                              patch_artist=True, 
                              boxprops=dict(facecolor=colors[j % len(colors)], color='black'),
                              medianprops=dict(color='black', linewidth=2),
                              whiskerprops=dict(color='black'),
                              capprops=dict(color='black'),
                              flierprops=dict(marker='o', markerfacecolor='red', markersize=5, linestyle='none'),
                              widths=0.4)
            j += 1

            ax[i].set_ylabel(score_name)

        ax[i].set_xticklabels(models)
        ax[i].set_ylim(ranges[i][0], ranges[i][1])
        ax[i].set_title(score_names[i])
    
    plt.tight_layout()
    
    if out_path is not None:
        plt.savefig(out_path)





import numpy as np
from src.model_evaluation import get_cv_predictions
def regression_scatter_plot(model_names, data_df, X_cols, y_col, folds, model_map,
                                  model_params, figsize=(20, 5), lim=(0, 40),
                                  xlabel="LiDAR Canopy Height (m)", ylabel="Predicted Canopy Height (m)",
                                  scatter_color='black', scatter_size=2, diagonal_color='red', out_path = None,
                                  text_bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", boxstyle="round")):
    """
    Create scatter plots comparing predicted vs actual values for multiple models.

    Parameters:
    - model_names: List of model name strings (e.g., ['ElasticNet', 'RandomForest', 'XGBoost'])
    - data_df: DataFrame containing the data with folds and target column
    - embedding_columns: List of column names for features
    - height_col: String name of target column
    - folds: Array of fold assignments
    - cv_objects: List of fitted cross-validation objects with best_params_
    - figsize: Tuple for figure size
    - xlim, ylim: Tuples for axis limits
    - xlabel, ylabel: Strings for axis labels
    - scatter_color: Color for scatter points
    - scatter_size: Size of scatter points
    - diagonal_color: Color for 1:1 diagonal line
    - text_bbox: Dict for text box styling
    """
    from sklearn.metrics import r2_score, root_mean_squared_error
    from src.models import init_model

    n_models = len(model_names)
    fig, ax = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        ax = [ax]


    n_folds = np.unique(folds)

    cv_outputs = get_cv_predictions(data_df, model_map, model_params, n_folds, X_cols, y_col)

    vals = cv_outputs.y_true
    for i, model_name in enumerate(model_names):

        preds = cv_outputs[model_name]
        ## Plot Prediction Scatter Points
        ax[i].scatter(x = vals, y = preds, s = scatter_size, c = scatter_color)

        # Add diagonal line
        # ax[i].plot([lim[0], lim[1]], [lim[0], lim[1]], color='red')
        ax[i].set_aspect('equal', adjustable='box')
        ax[i].grid(True, linestyle=':', alpha=0.5)
        ax[i].plot(lim, lim, color=diagonal_color, linestyle='--', linewidth=1.5)

        # Scatter plot
        # Set labels and title
        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlabel(xlabel)
        ax[i].set_title(model_name)

        ## Add R2 Labels
        r2 = r2_score(vals, preds)
        rmse = root_mean_squared_error(vals, preds)

        ## Add Line of Best Fit
        a, b = np.polyfit(vals, preds, 1)
        ax[i].plot(vals, a*vals+b, color = 'green', linewidth = 1.5)

        ## Add Text for results
        ax[i].text(
            0.02, 0.98,
            f"R²={r2:.3f}\nRMSE={rmse:.3f}\ny = {a.round(2)}x + {b.round(2)}",
            transform=ax[i].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=text_bbox
        )
        

        
        

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
