import georinex as gr
from gnss_lib_py.utils.constants import WEEKSEC
from gnss_lib_py.utils.time_conversions import datetime_to_tow
from gnss_lib_py.utils.sim_gnss import find_sat
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from datetime import datetime


def load_rnx3_obs(path):
    """load rnx3 obs file specified in path

    If zipped, unzip.
    Load the file using the georinex package.
    Convert it to pandas
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

    # convert ephemerides from xarray.Dataset to pandas.DataFrame
    nav_df = nav_ds.to_dataframe()
    nav_df.dropna(how='all', inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df['source'] = path
    nav_df['t_oc'] = pd.to_numeric(nav_df['time'] - datetime(1980, 1, 6, 0, 0, 0))
    nav_df['t_oc'] = 1e-9 * nav_df['t_oc'] - WEEKSEC * np.floor(1e-9 * nav_df['t_oc'] / WEEKSEC)
    nav_df['time'] = nav_df['time'].dt.tz_localize('UTC')
    nav_df.rename(
        columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                 'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'},
        inplace=True)
    return nav_df


if __name__ == "__main__":
    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001",
                                                                            "BRDC00IGS_R_20220010000_01D_MN.rnx")
    # load RNX3 NAV file
    nav_df = load_rnx3_obs(path_to_rnx3_nav_file)

    # select sv and time
    sv = np.array('G01', dtype='<U5')
    date = datetime.fromisoformat('2022-01-01T00:00:00.000')
    gps_week, gps_tow = datetime_to_tow(date)

    # call findsat from gnss_lib_py
    sv_posvel = find_sat(nav_df, gps_tow, gps_week)

    print(sv_posvel)
