import duckdb 
import boto3




def init_db():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL pdal FROM community; LOAD pdal")
    return con

def configure_aws_credentials(con, region):
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials:
        con.execute(f"SET s3_access_key_id = '{credentials.access_key}';")
        con.execute(f"SET s3_secret_access_key = '{credentials.secret_key}';")
        if credentials.token:
            con.execute(f"SET s3_session_token = '{credentials.token}';")
    
    con.execute(f"SET s3_region = '{region}';")
    return con