# Case Study: Wood Buffalo National Park of Canada

## Introduction



<img src="wood_buffalo.jpg" align = "right" width ="500px" padding ="50px"/>

Forest canopy height is a key variable in estimating aboveground biomass and carbon sequestration. Capturing accurate tree height data can be achieved with airborne or UAV-based LiDAR, however cost-constraints make it challenging to attain wall-to-wall canopy height models (CHMs) of large areas using this technology. Scaling up forest canopy height modelling can be treated as a regression problem by combining LiDAR-derived CHMs as the target variable, while using globally available and continuous satellite data as predictors. Current approaches construct predictor variables using optical and radar imagery, which can require a lot of preprocessing to feed into a model. Geospatial foundational models can produce powerful embeddings, helping scientists bypass tedious feature engineering and to focus on model development. By leveraging Google Earth Engine's (GEE) satellite embeddings dataset, you can easily construct a stack of 64 feature-rich predictor variables for forest canopy height estimation with strong results.

To exhibit the pipeline's scalability, a case study was implemented on Canada's largest national park; [Wood Buffalo National Park](https://en.wikipedia.org/wiki/Wood_Buffalo_National_Park). Situated in northeastern Alberta, Wood Buffalo National Park covers 44,741 km2, an area larger than Switzerland, making it the second-largest national park in the world.  



## Extracting Canopy Heights and Satellite Embeddings

<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: 2;">

  

  </div>
  <div style="flex: 1;">
    <img src="sample_points.png"/>

  </div>
</div>



## Multi-Model Opimization

<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: ;">
      — using airborne LiDAR from Natural Resources Canada.  Elastic Net came out on top: R² = 0.841 ± 0.035 | RMSE = 3.851 ± 0.44m. In another post, I'll share more about the tools I used to develop this pipeline.
  

  </div>
  <div style="flex: 1;">
    <img src="regression_plot.png"/>
    <img src="boxplot_comparison.png"/>

  </div>
</div>





![Alt text description](predicted_chm.png)

