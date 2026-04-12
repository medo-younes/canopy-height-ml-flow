# Case Study: Wood Buffalo National Park of Canada

<img src="wood_buffalo.jpg" align = "left" width ="500px" padding ="50px"/>

Forest canopy height is a key variable in estimating aboveground biomass and carbon sequestration. Capturing accurate tree height data can be achieved with airborne or UAV-based LiDAR, however cost-constraints make it challenging to attain wall-to-wall canopy height models (CHMs) of large areas using this technology. Scaling up forest canopy height modelling can be treated as a regression problem by combining LiDAR-derived CHMs as the target variable, while using globally available and continuous satellite data as predictors. Current approaches construct predictor variables using optical and radar imagery, which can require a lot of preprocessing to feed into a model. Geospatial foundational models can produce powerful embeddings, helping scientists bypass tedious feature engineering and to focus on model development. By leveraging Google Earth Engine's (GEE) satellite embeddings dataset, you can easily construct a stack of 64 feature-rich predictor variables for forest canopy height estimation with strong results.

To exhibit the pipeline's scalability, a case study was implemented on Canada's largest national park; [Wood Buffalo National Park](https://en.wikipedia.org/wiki/Wood_Buffalo_National_Park). Situated in northeastern Alberta, Wood Buffalo National Park covers 44,741 km2, an area larger than Switzerland, making it the second-largest national park in the world. Using publically available data, a canopy height estimation model was developed for the major national park.
<br clear="left"/>

## Extracting Canopy Heights and Satellite Embeddings

<img src="tile_index.png" align = "right" width ="500px" padding ="50px"/>

With the end goal of constructing a training dataset including sampled forest canopy height as the target variable and 64 AlphaEarth Satellite Embeddings as predictor variables, a [metaflow](https://metaflow.org/) was developed.

Airborne LiDAR Point Clouds were utilized as ground-truth canopy height data for the study area. The [CanElevation Series](https://open.canada.ca/data/en/dataset/7069387e-9986-4297-9f55-0288e9676947) is a publically available LiDAR Point Cloud dataset produced by Natural Resources Canada (NRCan). There is wide coverage across major Canadian cities and natural sites, making it a exteremley valuable elevation dataset for the country. Made available as Cloud Optimized Point Clouds (COPC) on AWS S3, CanElevation Series enables fast spatial querying of point cloud data. It comes with a 1 x 1 km tile index, including the spatial boundaries of each COPC file, project metadata (including aquisition year) and its S3 URL. 

The first step is to retrieve the tiles overlapping the area of interest (AOI) using as simple spatial join. The intersection amounted to 3,536 tiles. Next, the interesecting tiles were filtered based on forest cover (>=25%) using a [global raster of natural and planted forest extent on GEE](https://gee-community-catalog.org/projects/global_ftype/) (Xiao, Y., 2024). Water bodies (oceans and lakes) were clipped out from tile geometries with the help of [Overture's globally available water features dataset](https://docs.overturemaps.org/schema/reference/base/water/). As a result, the post processed tiles ensure that only forested areas are sampled, while avoiding the sampling of water. Following these preprocessing steps, a total of 2,240 tiles covering 1,813 km2 remained; only ~4% of the study area.


<br clear="left"/>
<img src="sample_points.png" align = "left" width ="500px" padding ="50px"/>

<br clear="left"/>


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

## References

1. Xiao, Y. (2024). Global Natural and Planted Forests Mapping at Fine Spatial Resolution of 30 m [Data set].
Zenodo. https://doi.org/10.5281/zenodo.10701417