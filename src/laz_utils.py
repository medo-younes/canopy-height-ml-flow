def is_copc_vlr_present(path):
    import laspy
    try:
        with laspy.open(path) as f:
            vlrs = [vlr.user_id.lower() for vlr in f.header.vlrs]
            evlrs = [evlr.user_id.lower() for evlr in f.header.evlrs]
            return "copc" in vlrs or "copc" in evlrs
    except Exception as e:
        return False