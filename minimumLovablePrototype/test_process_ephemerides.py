from gnss_lib_py.utils.time_conversions import datetime_to_tow, tow_to_datetime, get_leap_seconds
from gnss_lib_py.utils.sim_gnss import find_sat
import numpy as np
from pathlib import Path
from datetime import datetime
import georinex as gr
import process_ephemerides as eph
import zipfile

def test_compare_rnx3_sat_pos_with_magnitude():
    """Loads a RNX3 file, compute a position for different satellites and time, and compare to MAGNITUDE results
    Test will be a success if the difference in position is lower than threshold_pos_error_m = 0.01
    """
    threshold_pos_error_m = 0.01

    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001",
                                                                            "BRDC00IGS_R_20220010000_01D_GN.rnx")

    # check existence of file, and if not, try to find and unzip a compressed version
    if not path_to_rnx3_nav_file.exists():
        # check existence of zipped file
        path_to_zip_file = path_to_rnx3_nav_file.with_suffix(".zip")
        if path_to_zip_file.exists():
            print("Unzipping RNX3 NAV file...")
            # unzip file
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(path_to_rnx3_nav_file.resolve().parent)
        else:
            print(f"File {path_to_rnx3_nav_file} (or zipped version) does not exist")

    # select sv and time
    sv = np.array('G01', dtype='<U3')
    gps_week = 2190
    gps_tow = 523800

    # MAGNITUDE position
    sv_pos_magnitude = np.array([13053451.235, -12567273.060, 19015357.126])

    # Compute RNX3 satellite position
        # load RNX3 NAV file
    nav_ds = eph.convert_rnx3_nav_file_to_dataset(path_to_rnx3_nav_file)

        # select right ephemeris
    date = np.datetime64(tow_to_datetime(gps_week, gps_tow))
    nav_df = eph.select_nav_ephemeris(nav_ds, sv, date)

        # call findsat from gnss_lib_py
    sv_posvel_rnx3_df = find_sat(nav_df, gps_tow, gps_week)
    sv_pos_rnx3 = np.array([sv_posvel_rnx3_df["x"].values[0],
                            sv_posvel_rnx3_df["y"].values[0],
                            sv_posvel_rnx3_df["z"].values[0]])

    assert (np.linalg.norm(sv_pos_rnx3 - sv_pos_magnitude) < threshold_pos_error_m)


'''
def test_compare_rnx3_sat_pos_with_sp3():
    """Loads a RNX3 file, compute a position for different satellites and time, and compare to MAGNITUDE results
    Test will be a success if the difference in position is lower than threshold_pos_error_m = 0.01
    """
    threshold_pos_error_m = 0.01

    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001",
                                                                            "BRDC00IGS_R_20220010000_01D_GN.rnx")
    # filepath towards SP3 file
    path_to_sp3_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001", "igs21906.sp3")

    # select sv and time
    sv = np.array('G01', dtype='<U3')
    gps_week = 2190
    gps_tow = 523800

    # compute RNX3 satellite position
        # load RNX3 NAV file
    nav_ds = eph.load_rnx3_nav_ds(path_to_rnx3_nav_file)

        # select right ephemeris
    date = np.datetime64(tow_to_datetime(gps_week, gps_tow))
    nav_df = eph.select_nav_ephemeris(nav_ds, sv, date)

        # call findsat from gnss_lib_py
    sv_posvel_rnx3_df = find_sat(nav_df, gps_tow, gps_week)
    sv_pos_rnx3 = np.array([sv_posvel_rnx3_df["x"].values[0],
                            sv_posvel_rnx3_df["y"].values[0],
                            sv_posvel_rnx3_df["z"].values[0]])

    # compute SP3 satellite position
    # TODO: compute sp3 satellite position using https://gnss-lib-py.readthedocs.io/en/latest/tutorials/tutorials_parsers_notebook.html#Precise-Ephemerides-Processing
    # The code uses the NavData structure from a Google dataset import
    # Some modifications are required in order to be able to compute a SP3 position without NavData
    sv_pos_sp3 = np.zeros(3)

    """
    # Early attempt to compute an SP3 position, but the error is too large, probably due to interpolation error
        # load sp3
    sp3_ds = gr.load(path_to_sp3_file)

        # convert sp3 time from utc to gps_time
    ls = get_leap_seconds(date.astype(datetime))
    sp3_sv = sp3_ds.sel(sv=sv)
    sv_pos_sp3 = sp3_sv.interp(time=date + np.timedelta64(ls, 's'))['position'].values * 1000
    """

    assert (np.linalg.norm(sv_pos_rnx3 - sv_pos_sp3) < threshold_pos_error_m)
'''