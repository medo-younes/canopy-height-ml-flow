import pdal
import json
from src.pdal_ops import filter_pdal

def compute_chm(laz_path, out_raster_path, res_m = 10, statistic = "P95", compute_hag = True, height_dimension = "HeightAboveGround"):

    pipeline = {
        "pipeline": [
            laz_path,
        ]
    }

    if compute_hag:
        pipeline['pipeline'].extend([filter_pdal("smrf", None),filter_pdal("hag_nn", kwargs = None)])
        height_dimension = "HeightAboveGround"
    

    # pipeline['pipeline'].append({
    #             "type":"filters.outlier",
    #             "method":"radius",
    #             "radius":1.0,
    #             "min_k": 4
    # })
    pipeline['pipeline'].append({
                "type": "writers.gdal",
                "filename": out_raster_path,
                "resolution": res_m,
                "dimension": height_dimension,
                "output_type": statistic,            # 95th percentile directly
                "data_type": "float32",
                "binmode": True
            })
            

    pipeline = pdal.Pipeline(json.dumps(pipeline))
    pipeline.execute()



