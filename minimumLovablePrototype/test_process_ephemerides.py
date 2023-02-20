import math

import pandas as pd
from gnss_lib_py.utils.time_conversions import (
    datetime_to_tow,
    tow_to_datetime,
    get_leap_seconds,
)
from gnss_lib_py.utils.sim_gnss import find_sat
import numpy as np
from pathlib import Path
from datetime import datetime
import georinex as gr
import process_ephemerides as eph
import zipfile

import helpers
import converters
import parse_rinex
import constants


def test_compare_rnx3_sat_pos_with_magnitude():
    """Loads a RNX3 file, compute a position for different satellites and time, and compare to MAGNITUDE results
    Test will be a success if the difference in position is lower than threshold_pos_error_m = 0.01
    """
    threshold_pos_error_m = 0.01

    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = (
        Path(__file__)
        .resolve()
        .parent.parent.joinpath(
            "datasets", "TLSE_2022001", "BRDC00IGS_R_20220010000_01D_GN.rnx"
        )
    )
    # select sv and time
    sv = np.array("G01", dtype="<U3")
    gps_week = 2190
    gps_tow = 523800

    # MAGNITUDE position
    sv_pos_magnitude = np.array([13053451.235, -12567273.060, 19015357.126])

    # check existence of RNX3 file, and if not, try to find and unzip a compressed version
    if not path_to_rnx3_nav_file.exists():
        # check existence of zipped file
        path_to_zip_file = path_to_rnx3_nav_file.with_suffix(".zip")
        if path_to_zip_file.exists():
            print("Unzipping RNX3 NAV file...")
            # unzip file
            with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
                zip_ref.extractall(path_to_rnx3_nav_file.resolve().parent)
        else:
            print(f"File {path_to_rnx3_nav_file} (or zipped version) does not exist")

    # Compute RNX3 satellite position
    # load RNX3 NAV file
    nav_ds = eph.convert_rnx3_nav_file_to_dataset(path_to_rnx3_nav_file)

    # select right ephemeris
    date = np.datetime64(tow_to_datetime(gps_week, gps_tow))
    nav_df = eph.select_nav_ephemeris(nav_ds, sv, date)

    # call findsat from gnss_lib_py
    sv_posvel_rnx3_df = find_sat(nav_df, gps_tow, gps_week)
    sv_pos_rnx3 = np.array(
        [
            sv_posvel_rnx3_df["x"].values[0],
            sv_posvel_rnx3_df["y"].values[0],
            sv_posvel_rnx3_df["z"].values[0],
        ]
    )

    assert np.linalg.norm(sv_pos_rnx3 - sv_pos_magnitude) < threshold_pos_error_m


def test_compute_satellite_clock_offset():
    # When computing the satellite clock offset of GPS-001 for January 1st 2022 at 1am GPST
    # We expect the clock offset to be computed from the following RINEX 3 ephemeris
    """
G01 2022 01 01 00 00 00 4.691267386079e-04-1.000444171950e-11 0.000000000000e+00
     3.900000000000e+01-1.411250000000e+02 3.988380417768e-09-6.242942382352e-01
    -7.363036274910e-06 1.121813920327e-02 4.695728421211e-06 5.153674995422e+03
     5.184000000000e+05-3.166496753693e-08-1.036611240093e+00 1.955777406693e-07
     9.864187694897e-01 2.997500000000e+02 8.840876015687e-01-8.133553080847e-09
    -3.778728827795e-10 1.000000000000e+00 2.190000000000e+03 0.000000000000e+00
     2.000000000000e+00 0.000000000000e+00 5.122274160385e-09 3.900000000000e+01
     5.171890000000e+05 4.000000000000e+00 0.000000000000e+00 0.000000000000e+00
     """
    # copied from the following file
    rinex_3_navigation_file = helpers.prx_root().joinpath(
        f"datasets/TLSE_2022001/BRDC00IGS_R_20220010000_01D_GN.rnx"
    )
    computed_offset_s, computed_offset_rate_sps = eph.compute_satellite_clock_offset_and_clock_offset_rate(
        eph.convert_rnx3_nav_file_to_dataset(rinex_3_navigation_file),
        "G01",
        pd.Timestamp(np.datetime64("2022-01-01T01:00:00.000000000")),
    )
    # We expect the following clock offset and clock offset rate computed by hand from the parameters above.
    delta_t_s = constants.cSecondsPerHour
    expected_offset = (
        4.691267386079e-04
        + (-1.000444171950e-11 * delta_t_s)
        + math.pow(0.000000000000e00, 2)
    )
    expected_offset_rate = -1.000444171950e-11
    assert (
        constants.cGpsIcdSpeedOfLight_mps * (expected_offset - computed_offset_s) < 1e-6
    )
