from metaflow import FlowSpec, step, Config, resources, batch
import os
from db import init_db, configure_aws_credentials
from tile_index import get_tiles
from pdal_ops import *
from s3_utils import download_s3

bbox = [-79.542732,43.701579,-79.371758,43.781425]
year = 2022

class PointCloudFlow(FlowSpec):
    config = Config("config", default = "config.yaml", parser = "yaml.full_load")

    @step
    def start(self):
        

        self.tile_index_path = os.path.join(self.config.data.root , self.config.data.tile_index_file)
        self.raw_path = os.path.join(self.config.data.root, 'raw')

        if not os.path.exists(self.raw_path):
            os.makedirs(self.raw_path, exist_ok=True)

        print("Creating DuckDB Connection...")
        con = init_db()
        con = configure_aws_credentials(con, region = self.config.aws.region)

        tile_index_df = get_tiles(con, self.tile_index_path, bbox = bbox, year = year)
        tile_index_df = tile_index_df.sample(1)
        self.urls = tile_index_df.URL.to_list()
        
        print(f"Total of {len(self.urls)} COPCs Queried")
        self.next(self.process, foreach = "urls")

    # @resources(cpu = 8, memory=32000)
    @step
    def process(self):
        s3_url = self.input
        download_s3(s3_url = s3_url, out_dir=self.raw_path)

        out_file = os.path.basename(s3_url)
        local_path = os.path.join(self.raw_path, out_file).replace('.copc','')
        prc_path = local_path.replace('.laz', '_filtered.laz')
        
        print("> Filtering Points")
        stages = [
            reader('las', local_path),
            # filter_csf(),
            filter_expression(expression="NumberOfReturns > 1"),
            writer('las', prc_path)
        ]

        print("Running pipeline")
        pipeline = build_pipeline(stages)
        pipeline.execute()
        
        ## Remove raw point clouds
        os.remove(local_path)
        self.pc_basename = os.path.basename(s3_url)
        self.next(self.join)

    @step
    def join(self, inputs):
        # self.basemames = [url for url in inputs]
        self.next(self.end)


    @step
    def end(self):
        print("ENDING FLOW")




if __name__ == '__main__':
    PointCloudFlow()