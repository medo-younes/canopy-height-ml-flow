import pdal
import json
from src.pdal_ops import filter_pdal,reader
import logging


def compute_chm(laz_path, out_raster_path, res_m = 10, statistic = "P95", compute_hag = True, height_dimension = "HeightAboveGround", polygon = None, max_height = None):

    # Always use reader() to ensure consistent dict structure
    kwargs = {"polygon": polygon} if polygon is not None else None
    pipeline_dict = {
        "pipeline": [
            reader('copc', laz_path, kwargs=kwargs)
        ]
    }


    if compute_hag:
        pipeline_dict['pipeline'].extend([filter_pdal("smrf", None),filter_pdal("hag_nn", kwargs = None)])
        height_dimension = "HeightAboveGround"
    
    # Add height cap if specified
    if max_height is not None:
        pipeline_dict['pipeline'].append(filter_pdal("range", kwargs={"limits": f"HeightAboveGround[0:{max_height}]"}))
    pipeline_dict['pipeline'].append({
                "type": "writers.gdal",
                "filename": out_raster_path,
                "resolution": res_m,
                "dimension": height_dimension,
                "output_type": statistic,            # 95th percentile directly
                "data_type": "float32",
                "binmode": True
            })
            
    

    pipeline = pdal.Pipeline(json.dumps(pipeline_dict))

    try:
        n_pts = pipeline.execute()
    except RuntimeError as err:
        logging.error("CHM PDAL execution failed for %s", out_raster_path)
        logging.error("Pipeline JSON: %s", json.dumps(pipeline_dict, indent=2))
        logging.error("PDAL log output: %s", pipeline.log)
        raise

    if n_pts == 0:
        logging.warning("No points in polygon for %s, skipping CHM", laz_path)
        return




