



def get_tiles(con, tile_index_path, bbox, year = None):
    bbox_str = ", ".join([str(x) for x in bbox])
    con.sql(f"""CREATE TABLE query AS
            SELECT *, 
            CAST(RIGHT(project, 4) AS INTEGER) AS year
            FROM '{tile_index_path}' 
            WHERE
            ST_Intersects(
                geometry,
                ST_MakeEnvelope({bbox_str})
                )
            """)

    max_year = con.sql(f"""SELECT MAX(year) FROM query;""").fetchone()[0]
    if not year:
        return con.sql(f"SELECT * FROM QUERY WHERE year = {max_year}").df()
    else:
        query_df = con.sql(f"SELECT * FROM QUERY WHERE year = {year}").df()
        if len(query_df) > 0:
            return query_df
        else:
            return con.sql(f"SELECT * FROM QUERY WHERE year = {max_year}").df()