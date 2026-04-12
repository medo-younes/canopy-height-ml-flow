# Case Study: Wood Buffalo National Park of Canada

Forest canopy height is a key variable in estimating aboveground biomass and carbon sequestration. Capturing accurate tree height data can be achieved with airborne or UAV-based LiDAR, however cost-constraints make it challenging to attain wall-to-wall canopy height models (CHMs) of large areas using this technology. Scaling up forest canopy height modelling can be treated as a regression problem by combining LiDAR-derived CHMs as the target variable, while using globally available and continuous satellite data as predictors. Current approaches construct predictor variables using optical and radar imagery, which can require a lot of preprocessing to feed into a model. Geospatial foundational models can produce powerful embeddings, helping scientists bypass tedious feature engineering and to focus on model development. By leveraging Google Earth Engine's (GEE) satellite embeddings dataset, you can easily construct a stack of 64 feature-rich predictor variables for forest canopy height estimation with strong results.

I developed a repeatable, end-to-end ML pipeline for automatically constructing forest canopy height + embeddings training datasets, model training and canopy height prediction on an area of interest. The pipeline uses spatial k-fold cross validation to account for spatial autocorrelation. Three regression models (Elastic Net, Random Forest and XGBoost) are fine-tuned with bayesian hyperparameter optimization across all folds. Ultimately the best model (lowest mean RMSE) is selected for inference. 

To test it, I ran the pipeline on Wood Buffalo National Park — Canada's largest national park at ~4.48M hectares — using airborne LiDAR from Natural Resources Canada.  Elastic Net came out on top: R² = 0.841 ± 0.035 | RMSE = 3.851 ± 0.44m. In another post, I'll share more about the tools I used to develop this pipeline.

Github link in the comments, would appreciate any feedback/thoughts from anyone working in the remote sensing space!

GitHub: https://github.com/medo-younes/canopy-height-ml-flow.git
This work is inspired by: https://medium.com/google-earth/improved-forest-carbon-estimation-with-alphaearth-foundations-and-airborne-lidar-data-af2d93e94c55


![Alt text description](predicted_chm.png)

## Sampling Points
![plot](sample_points.png)

## Model Performance
![plot](regression_plot.png)

![plot](boxplot_comparison.png)