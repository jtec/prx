import math

import georinex
import georinex as gr
import xarray
from gnss_lib_py.utils.constants import WEEKSEC
from gnss_lib_py.utils.time_conversions import (
    datetime_to_tow,
    get_leap_seconds,
    tow_to_datetime,
)
from gnss_lib_py.utils.sim_gnss import find_sat
from gnss_lib_py.parsers.precise_ephemerides import (
    parse_sp3,
    multi_gnss_from_precise_eph,
    extract_sp3,
)
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import parse_rinex
import constants


def convert_rnx3_nav_file_to_dataframe(path):
    # parse RNX3 NAV file using georinex module
    nav_ds = parse_rinex.load(path, use_caching=True)
    nav_df = convert_nav_dataset_to_dataframe(nav_ds)
    return nav_df


constellation_2_system_time_scale = {
    "G": "GPST",
    "S": "SBAST",
    "E": "GST",
    "C": "BDT",
    "R": "GLONASST",
    "J": "QZSST",
    "I": "IRNWT",
}


def satellite_id_2_system_time_scale(satellite_id):
    return constellation_2_system_time_scale[satellite_id[0]]


def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py"""
    nav_df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    nav_df.dropna(how="all", inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df["source"] = nav_ds.filename

    nav_df["time_scale"] = nav_df.apply(
        lambda row: satellite_id_2_system_time_scale(row["sv"]), axis=1
    )

    # TODO Can we be sure that this is always GPST?
    gpst_s = (
        pd.to_numeric(nav_df["time"] - constants.cGpstEpoch)
        / constants.cNanoSecondsPerSecond
    )
    # Week second:
    nav_df["t_oc"] = gpst_s - constants.cSecondsPerWeek * np.floor(
        gpst_s / constants.cSecondsPerWeek
    )

    nav_df.rename(
        columns={
            "M0": "M_0",
            "Eccentricity": "e",
            "Toe": "t_oe",
            "DeltaN": "deltaN",
            "Cuc": "C_uc",
            "Cus": "C_us",
            "Cic": "C_ic",
            "Crc": "C_rc",
            "Cis": "C_is",
            "Crs": "C_rs",
            "Io": "i_0",
            "Omega0": "Omega_0",
        },
        inplace=True,
    )
    return nav_df


def convert_single_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py"""
    nav_df = nav_ds.to_dataframe()
    time = pd.to_datetime(nav_ds["time"].values)
    sv = nav_ds["sv"].values
    SVclockBias = nav_ds["SVclockBias"].values
    SVclockDrift = nav_ds["SVclockDrift"].values
    SVclockDriftRate = nav_ds["SVclockDriftRate"].values
    IODE = nav_ds["IODE"].values
    C_rs = nav_ds["Crs"].values
    deltaN = nav_ds["DeltaN"].values
    M_0 = nav_ds["M0"].values
    C_uc = nav_ds["Cuc"].values
    e = nav_ds["Eccentricity"].values
    C_us = nav_ds["Cus"].values
    sqrtA = nav_ds["sqrtA"].values
    t_oe = nav_ds["Toe"].values
    C_ic = nav_ds["Cic"].values
    Omega_0 = nav_ds["Omega0"].values
    C_is = nav_ds["Cis"].values
    i_0 = nav_ds["Io"].values
    C_rc = nav_ds["Crc"].values
    omega = nav_ds["omega"].values
    OmegaDot = nav_ds["OmegaDot"].values
    IDOT = nav_ds["IDOT"].values
    CodesL2 = nav_ds["CodesL2"].values
    GPSWeek = nav_ds["GPSWeek"].values
    L2Pflag = nav_ds["L2Pflag"].values
    SVacc = nav_ds["SVacc"].values
    health = nav_ds["health"].values
    TGD = nav_ds["TGD"].values
    IODC = nav_ds["IODC"].values
    TransTime = nav_ds["TransTime"].values
    source = nav_ds.filename
    time_gpst_ns = nav_ds["time"].values.astype("datetime64[ms]") - constants.cGpstEpoch
    time_gpst_ns_np = np.timedelta64(time_gpst_ns)
    t_oc = (
        pd.to_numeric(time_gpst_ns_np).astype("float") / constants.cNanoSecondsPerSecond
    )
    t_oc = t_oc - WEEKSEC * np.floor(t_oc / WEEKSEC)

    dataframe_data = {
        "time": time,
        "sv": sv,
        "SVclockBias": SVclockBias,
        "SVclockDrift": SVclockDrift,
        "SVclockDriftRate": SVclockDriftRate,
        "IODE": IODE,
        "C_rs": C_rs,
        "deltaN": deltaN,
        "M_0": M_0,
        "C_uc": C_uc,
        "e": e,
        "C_us": C_us,
        "sqrtA": sqrtA,
        "t_oe": t_oe,
        "C_ic": C_ic,
        "Omega_0": Omega_0,
        "C_is": C_is,
        "i_0": i_0,
        "C_rc": C_rc,
        "omega": omega,
        "OmegaDot": OmegaDot,
        "IDOT": IDOT,
        "CodesL2": CodesL2,
        "GPSWeek": GPSWeek,
        "L2Pflag": L2Pflag,
        "SVacc": SVacc,
        "health": health,
        "TGD": TGD,
        "IODC": IODC,
        "TransTime": TransTime,
        "source": source,
        "t_oc": t_oc,
        # Glonass orbit parameters:
        "X": nav_ds["X"].values,
    }

    nav_df = pd.DataFrame(dataframe_data, index=[0])

    return nav_df


def select_nav_ephemeris(nav_dataframe, satellite_id, gpst_datetime):
    """select an ephemeris from a RNX3 nav dataframe for a particular sv and time, and return the ephemeris.

    Input examples:
    nav_dataset = convert_nav_dataset_to_dataframe(path_to_rnx3_nav_file)
    satellite_id = np.array('G01', dtype='<U3') # satellite ID for a single satellite,
    gpst_datetime = np.datetime64('2022-01-01T00:00:00.000'), np.datetime64(tow_to_datetime(gps_week, gps_tow))

    Output:
    nav_dataframe: a pandas.dataframe containing the selected ephemeris
    """
    ephemerides_of_requested_sat = nav_dataframe.loc[
        (nav_dataframe["sv"] == satellite_id)
    ]
    # Find first ephemeris before time of interest
    ephemerides_of_requested_sat = ephemerides_of_requested_sat.sort_values(by=["time"])
    ephemerides_of_requested_sat_before_requested_time = (
        ephemerides_of_requested_sat.loc[
            ephemerides_of_requested_sat["time"] < gpst_datetime
        ]
    )
    assert (
        ephemerides_of_requested_sat_before_requested_time.shape[0] > 0
    ), f"Did not find ephemeris with timestamp before {gpst_datetime}"
    return ephemerides_of_requested_sat_before_requested_time.iloc[[-1]]


def compute_satellite_clock_offset_and_clock_offset_rate(
    parsed_rinex_3_nav_file: pd.DataFrame,
    satellite: str,
    time_constellation_time_ns: pd.Timestamp,
):
    ephemeris_df = select_nav_ephemeris(
        parsed_rinex_3_nav_file, satellite, time_constellation_time_ns.to_datetime64()
    )
    time_wrt_ephemeris_epoch_s = pd.Timedelta(
        time_constellation_time_ns - ephemeris_df["time"].iloc[0]
    ).total_seconds()
    if satellite[0] != "R":
        offset_at_epoch_s = ephemeris_df["SVclockBias"].iloc[0]
        offset_rate_sps = ephemeris_df["SVclockDrift"].iloc[0]
        offset_acceleration_sps2 = ephemeris_df["SVclockDriftRate"].iloc[0]
    else:
        offset_at_epoch_s = ephemeris_df["SVclockBias"].iloc[0]
        offset_rate_sps = ephemeris_df["SVrelFreqBias"].iloc[0]
        offset_acceleration_sps2 = 0

    offset_s = (
        offset_at_epoch_s
        + offset_rate_sps * time_wrt_ephemeris_epoch_s
        + offset_acceleration_sps2 * math.pow(time_wrt_ephemeris_epoch_s, 2)
    )
    offset_rate_sps = (
        offset_rate_sps + 2 * offset_acceleration_sps2 * time_wrt_ephemeris_epoch_s
    )
    return offset_s, offset_rate_sps
