from gnss_lib_py.utils.time_conversions import datetime_to_tow, tow_to_datetime
from gnss_lib_py.utils.sim_gnss import find_sat
import numpy as np
from pathlib import Path
from datetime import datetime
import georinex as gr
import process_ephemerides as eph

def test_compare_rnx3_sat_pos_with_magnitude():
    """"Loads a RNX3 file, compute a position for different satellites and time, and compare to MAGNITUDE results
    Test will be a success if the difference in position is lower than threshold_pos_error = 0.01
    """
    threshold_pos_error = 0.01

    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001",
                                                                            "BRDC00IGS_R_20220010000_01D_GN.rnx")
    # load RNX3 NAV file
    nav_ds = eph.load_rnx3_nav_ds(path_to_rnx3_nav_file)

    # select sv and time
    sv = np.array('G01', dtype='<U3')
    gps_week = 2190
    gps_tow = 523800
    sv_pos_magnitude = np.array([13053451.235, -12567273.060, 19015357.126])

    # select right ephemeris
    date = np.datetime64(tow_to_datetime(gps_week, gps_tow))
    nav_df = eph.select_nav_ephemeris(nav_ds, sv, date)

    # call findsat from gnss_lib_py
    sv_posvel_rnx3_df = find_sat(nav_df, gps_tow, gps_week)
    sv_pos_rnx3 = np.array([sv_posvel_rnx3_df["x"].values[0],
                            sv_posvel_rnx3_df["y"].values[0],
                            sv_posvel_rnx3_df["z"].values[0]])

    assert (np.linalg.norm(sv_pos_rnx3 - sv_pos_magnitude) < threshold_pos_error)

if __name__ == "__main__":
    test_compare_rnx3_sat_pos_with_magnitude()