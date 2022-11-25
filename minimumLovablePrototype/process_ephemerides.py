import georinex as gr
import xarray
from gnss_lib_py.utils.constants import WEEKSEC
from gnss_lib_py.utils.time_conversions import datetime_to_tow
from gnss_lib_py.utils.sim_gnss import find_sat
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from datetime import datetime, timedelta


def load_rnx3_nav_ds(path):
    """load rnx3 nav file specified in path

    If zipped, unzip.
    Load the file using the georinex package.
    return as an xarray.dataset object
    """
    # unzip file if necessary
    if not path.exists():
        print("Unzipping RNX3 NAV file...")
        # unzip file
        path_to_zip_file = path.with_suffix(".zip")
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            print(path.resolve().parent)
            zip_ref.extractall(path.resolve().parent)

    # parse RNX3 NAV file using georinex module
    nav_ds = gr.load(path)
    return nav_ds


def load_rnx3_nav_df(path):
    """load rnx3 nav file specified in path

    If zipped, unzip.
    Load the file using the georinex package.
    Convert it to pandas
    """
    load_rnx3_nav_ds(path)

    # convert ephemerides from xarray.Dataset to pandas.DataFrame
    nav_df = convert_nav_ds_to_df(nav_ds)

    return nav_df


def convert_nav_ds_to_df(nav_ds):
    # convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py
    # nav_df = nav_ds.to_dataframe()
    # nav_df.dropna(how='all', inplace=True)
    # nav_df.reset_index(inplace=True)
    # nav_df['source'] = nav_ds.filename
    # nav_df['t_oc'] = pd.to_numeric(nav_df['time'] - datetime(1980, 1, 6, 0, 0, 0))
    # nav_df['t_oc'] = 1e-9 * nav_df['t_oc'] - WEEKSEC * np.floor(1e-9 * nav_df['t_oc'] / WEEKSEC)
    # nav_df['time'] = nav_df['time'].dt.tz_localize('UTC')

    nav_df = nav_ds.to_pandas()
    nav_df['source'] = nav_ds.filename
    nav_df['t_oc'] = pd.to_numeric(nav_ds['time'].values.astype('datetime64[ms]').astype(datetime) - datetime(1980, 1, 6, 0, 0, 0))
    nav_df['t_oc'] = 1e-9 * nav_df['t_oc'] - WEEKSEC * np.floor(1e-9 * nav_df['t_oc'] / WEEKSEC)
    nav_df['time'] = nav_ds['time'].values.dt.tz_localize('UTC')

    nav_df.rename(
        columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                 'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'},
        inplace=True)
    return nav_df


if __name__ == "__main__":
    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001",
                                                                            "BRDC00IGS_R_20220010000_01D_GN.rnx")
    # load RNX3 NAV file
    nav_ds = load_rnx3_nav_ds(path_to_rnx3_nav_file)

    # select sv and time
    sv = np.array('G01', dtype='<U3')
    date = np.datetime64('2022-01-01T00:00:00.000')
    gps_week, gps_tow = datetime_to_tow(date.astype(datetime))

    # select right ephemeris
    nav_prn = nav_ds.sel(sv=sv)
    # find first epehemeris before date of interest
    indexEph = np.searchsorted(nav_prn.time.values,date)
    nav_time = nav_prn.isel(time=0)

    # call findsat from gnss_lib_py
    nav_df = convert_nav_ds_to_df(nav_time)
    sv_posvel = find_sat(nav_df, gps_tow, gps_week)

    print(sv_posvel)
