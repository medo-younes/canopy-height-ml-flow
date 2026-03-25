
import laspy
def is_copc_vlr_present(path):

    try:
        with laspy.open(path) as f:
            vlrs = [vlr.user_id.lower() for vlr in f.header.vlrs]
            evlrs = [evlr.user_id.lower() for evlr in f.header.evlrs]
            return "copc" in vlrs or "copc" in evlrs
    except Exception as e:
        return False
    


def get_epsg_authority_from_laz(laz_path):
    las = laspy.read(laz_path)
    crs = las.header.parse_crs()
    try: 
        return ":".join(crs.to_authority())
    except:
        return ":".join(crs.sub_crs_list[0].to_authority())