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

cGpstEpoch = datetime(1980, 1, 6, 0, 0, 0)
cNanoSecondsPerSecond = 1e-9


def convert_rnx3_nav_file_to_dataset(path):
    """load rnx3 nav file specified in path as xarray.dataset

    Input:
    path: pathlib.path object to a RNX3 file

    Load the file using the georinex package.
    return as an xarray.dataset object.
    """

    # parse RNX3 NAV file using georinex module
    nav_ds = gr.load(path)
    return nav_ds


def convert_rnx3_nav_file_to_dataframe(path):
    """load rnx3 nav file specified in path in pandas.dataframe

    Input:
    path can be either the path to a RNX3 file

    Load the file as dataset.
    Convert it to pandas.
    """
    nav_ds = convert_rnx3_nav_file_to_dataset(path)
    nav_df = convert_nav_dataset_to_dataframe(nav_ds)

    return nav_df


def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py"""
    nav_df = nav_ds.to_dataframe()
    nav_df.dropna(how="all", inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df["source"] = nav_ds.filename
    # convert time to number of elapsed seconds since GPST origin
    nav_df["t_oc"] = pd.to_numeric(nav_df["time"] - cGpstEpoch) * cNanoSecondsPerSecond
    # convert time to number of elapsed seconds since beginning of week
    nav_df["t_oc"] = nav_df["t_oc"] - WEEKSEC * np.floor(nav_df["t_oc"] / WEEKSEC)
    nav_df["time"] = nav_df["time"].dt.tz_localize("UTC")

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
    time_gpst_ns = (
        nav_ds["time"].values.astype("datetime64[ms]").astype(datetime) - cGpstEpoch
    )
    time_gpst_ns_np = np.timedelta64(time_gpst_ns)
    t_oc = pd.to_numeric(time_gpst_ns_np).astype("float") * cNanoSecondsPerSecond
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
    }

    nav_df = pd.DataFrame(dataframe_data, index=[0])

    return nav_df


def select_nav_ephemeris(nav_dataset, satellite_id, gpst_datetime):
    """select an ephemeris from a RNX3 nav dataset for a particular sv and time, and return a dataframe

    Input examples:
    nav_dataset = convert_rnx3_nav_file_to_dataset(path_to_rnx3_nav_file)
    satellite_id = np.array('G01', dtype='<U3') # satellite ID for a single satellite,
    gpst_datetime = np.datetime64('2022-01-01T00:00:00.000'), np.datetime64(tow_to_datetime(gps_week, gps_tow))

    Output:
    nav_dataframe: a pandas.dataframe containing the selected ephemeris
    """
    # select ephemeris for right satellite
    nav_dataset_of_requested_satellite_id = nav_dataset.sel(sv=satellite_id)
    # find first ephemeris before date of interest
    ephemeris_index = np.searchsorted(
        nav_dataset_of_requested_satellite_id.time.values, gpst_datetime
    )
    nav_dataset_of_requested_satellite_id_and_time = nav_dataset_of_requested_satellite_id.isel(
        time=ephemeris_index - 1
    )
    # convert to dataframe
    nav_dataframe = convert_single_nav_dataset_to_dataframe(
        nav_dataset_of_requested_satellite_id_and_time
    )

    return nav_dataframe


"""if __name__ == "__main__":"""
