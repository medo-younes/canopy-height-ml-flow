import json
import pdal 
import pandas as pd

def reader(reader_type, filename, kwargs= None):

    stages = {
            "type" : f"readers.{reader_type}", 
            "filename" : filename,   
        }
    if kwargs: stages.update(kwargs)
    return stages

def writer(writer_type, filename):
    return {
        "type" : f"writers.{writer_type}", 
        "filename" : filename,
        }
 

def filter_expression(expression):
    return {
        "type":"filters.expression",
        "expression": expression
    }

def filter_pdal(filter_type, kwargs = None):
    stages= {"type":f"filters.{filter_type}"}
    if kwargs: stages.update(kwargs)
    return stages

def merge():
    return {"type": "filters.merge"}

def build_pipeline(stages):
    pipeline_dict = dict(pipeline=stages)
    pipeline_json = json.dumps(pipeline_dict)
    return pdal.Pipeline(pipeline_json)
     
def filter_ground():
    return [
        filter("csf"),
        filter_expression(expression="Classification == 2")
    ]




def query_point_cloud(point_cloud_path, bounds, reader_type ="copc", return_df = True):
    stages = [reader(reader_type, point_cloud_path, kwargs = {"bounds" : bounds})]
    pipeline = build_pipeline(stages)
    pipeline.execute()
    if len(pipeline.arrays) == 0:
        # No points in bounds; return empty table/dataframe
        empty = pd.DataFrame()
        return empty if return_df else []
    if return_df:
        return pd.DataFrame(pipeline.arrays[0])
    else:
        return pipeline.arrays[0]