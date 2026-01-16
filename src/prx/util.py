import logging
import math
import os
import re
import shutil
import subprocess
from functools import wraps
from pathlib import Path

import georinex
import joblib
import numpy as np
import pandas as pd
import xarray
from imohash import imohash
from astropy.utils import iers
from astropy import time as astrotime
from astropy.coordinates import get_sun, ITRS
import astropy.units

from prx import constants

logger = logging.getLogger(__name__)


def file_exists_and_can_read_first_line(file: Path):
    assert file.exists(), f"Provided file path {file} does not exist"
    try:
        with open(file) as f:
            return f.readline()
    except UnicodeDecodeError:
        return None


def is_rinex_3_obs_file(file: Path):
    if first_line := file_exists_and_can_read_first_line(file):
        return re.search("3.0.*OBS.*RINEX VERSION.*", first_line) is not None
    return False


def is_rinex_3_nav_file(file: Path):
    if first_line := file_exists_and_can_read_first_line(file):
        return re.search("3.0.*NAV.*RINEX VERSION.*", first_line) is not None
    return False


def is_rinex_2_obs_file(file: Path):
    first_line = file_exists_and_can_read_first_line(file)
    if first_line is None:
        return False
    if "RINEX VERSION" not in first_line or "2.0" not in first_line:
        return False
    return True


def is_rinex_2_nav_file(file: Path):
    first_line = file_exists_and_can_read_first_line(file)
    if first_line is None:
        return False
    if "NAV" not in first_line or "2." not in first_line:
        return False
    return True


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = pd.Timestamp.now()
        result = func(*args, **kwargs)
        end_time = pd.Timestamp.now()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {total_time} to run.")
        return result

    return timeit_wrapper


@timeit
def try_repair_with_gfzrnx(file):
    with open(file) as f:
        if "gfzrnx" in f.read():
            logging.warning(f"File {file} already contains 'gfzrnx', skipping repair.")
            return file
    tool_path = shutil.which("gfzrnx")
    if tool_path is None:
        logger.info(
            "gfzrnx binary not found (try adding it to PATH), skipping repair..."
        )
        return file
    command = [
        str(tool_path),
        "-finp",
        str(file),
        "-fout",
        str(file),
        "-chk",
        "-kv",
        "-f",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info(f"Ran gfzrnx file repair on {file}")
        with open(file, "r") as f:
            file_content = f.read()
            file_content = re.sub(
                r"gfzrnx-(.*?)FILE PROCESSING(.*)UTC COMMENT",
                r"gfzrnx-\1FILE PROCESSING (timestamp removed by prx) UTC COMMENT",
                file_content,
                count=1,
            )
        with open(file, "w") as f:
            f.write(file_content)
            logger.info(
                f"Removed repair timestamp from gfzrnx file {file} to avoid content file hash changes."
            )
    else:
        logger.info(f"gfzrnx file repair run failed: {result}")
        assert False
    return file


def parse_boolean_env_variable(env_variable_name: str, value_if_not_set: bool):
    # Based on https://stackoverflow.com/a/65407083/2567449
    var_string = os.environ.get(env_variable_name, None)
    if var_string is None:
        return value_if_not_set
    var_string = var_string.lower().strip()
    assert var_string in ("True", "true", "1", "False", "false", "0")
    return var_string in ("True", "true", "1")


disk_cache = joblib.Memory(Path(__file__).parent.joinpath("diskcache"), verbose=0)


def get_logger(label):
    return logging.getLogger(label)


def prx_repository_root() -> Path:
    return Path(__file__).parents[2]


def hash_of_file_content(file: Path, use_sampling: bool = False):
    assert file.exists(), f"Looks like {file} does not exist."
    sample_threshhold = imohash.SAMPLE_THRESHOLD
    if not use_sampling:
        sample_threshhold = math.inf
    t0 = pd.Timestamp.now()
    hash_string = imohash.hashfile(
        file, hexdigest=True, sample_threshhold=sample_threshhold
    )
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        logger.info(
            f"Hashing file content took {hash_time}, we might want want to think about partially hashing the"
            f" file, e.g. using https://github.com/kalafut/py-imohash"
        )

    return hash_string


def timestamp_2_timedelta(timestamp: pd.Timestamp, time_scale):
    assert isinstance(timestamp, pd.Timestamp), "timestamp must be of type pd.Timestamp"
    # RINEX 3 adds the offset between GPST and GST/QZSST/IRNSST epochs, so we can use the GPST epoch here.
    # The SBAST epoch is the same as the GPST epoch.
    if (
        time_scale == "GPST"
        or time_scale == "SBAST"
        or time_scale == "QZSST"
        or time_scale == "IRNSST"
        or time_scale == "GST"
    ):
        return timestamp - constants.cGpstUtcEpoch
    if time_scale == "BDT":
        return (
            (timestamp - constants.cGpstUtcEpoch)
            - pd.Timedelta(1356 * constants.cSecondsPerWeek, "seconds")
            - pd.Timedelta(14, "seconds")
        )
    # The GLONASS epoch is (probably) the UTC epoch, to keep the Timedelta within the same order of magnitude
    # as for the other constellations, we use an arbitrary epoch here.
    if time_scale == "GLONASST":
        return timestamp - constants.cArbitraryGlonassUtcEpoch
    assert False, f"Time scale {time_scale} not supported."


def timestamp_to_mid_day(ts):
    return (
        pd.Timestamp(year=ts.year, month=1, day=1)
        + pd.Timedelta(days=ts.day_of_year - 1)
        + pd.Timedelta(hours=12)
    )


def timedelta_2_weeks_and_seconds(time_delta: pd.Timedelta | pd.Series):
    if isinstance(time_delta, pd.Timedelta):
        wn_series, tow_series = timedelta_2_weeks_and_seconds(pd.Series([time_delta]))
        return wn_series.iloc[0], tow_series.iloc[0]
    in_nanoseconds = time_delta / pd.Timedelta(1, "ns")
    weeks = np.floor(in_nanoseconds / constants.cNanoSecondsPerWeek)
    week_nanoseconds = in_nanoseconds - weeks * constants.cNanoSecondsPerWeek
    return weeks, week_nanoseconds.astype(np.float64) / constants.cNanoSecondsPerSecond


def week_and_seconds_2_timedelta(weeks, seconds):
    return pd.Timedelta(weeks * constants.cSecondsPerWeek + seconds, "seconds")


def timedelta_2_seconds(time_delta: pd.Timedelta):
    if pd.isnull(time_delta):
        return np.nan
    assert isinstance(time_delta, pd.Timedelta), (
        "time_delta must be of type pd.Timedelta"
    )
    integer_seconds = np.float64(round(time_delta.total_seconds()))
    fractional_seconds = (
        np.float64(
            timedelta_2_nanoseconds(time_delta)
            - integer_seconds * constants.cNanoSecondsPerSecond
        )
        / constants.cNanoSecondsPerSecond
    )
    return integer_seconds + fractional_seconds


def timedelta_2_nanoseconds(time_delta: pd.Timedelta):
    assert isinstance(time_delta, pd.Timedelta), (
        "time_delta must be of type pd.Timedelta"
    )
    return np.float64(time_delta / pd.Timedelta(1, "ns"))


def rinex_header_time_string_2_timestamp_ns(time_string: str) -> pd.Timestamp:
    elements = time_string.split()
    time_scale = elements[-1]
    assert time_scale == "GPS", "Time scales other than GPST not supported yet"
    year = int(elements[0])
    month = int(elements[1])
    day = int(elements[2])
    hour = int(elements[3])
    minute = int(elements[4])
    second = float(elements[5])
    datetime64_string = (
        f"{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:012.9f}"
    )
    timestamp = pd.Timestamp(np.datetime64(datetime64_string))
    return timestamp


def deg_2_rad(angle_deg):
    return angle_deg * np.pi / 180


# From https://stackoverflow.com/a/51253225
def convert_size_to_bytes(size_str):
    """Convert human filesizes to bytes.

    Special cases:
     - singular units, e.g., "1 byte"
     - byte vs b
     - yottabytes, zetabytes, etc.
     - with & without spaces between & around units.
     - floats ("5.2 mb")

    To reverse this, see hurry.filesize or the Django filesizeformat template
    filter.

    :param size_str: A human-readable string representing a file size, e.g.,
    "22 megabytes".
    :return: The number of bytes represented by the string.
    """
    multipliers = {
        "kilobyte": 1024,
        "megabyte": 1024**2,
        "gigabyte": 1024**3,
        "terabyte": 1024**4,
        "petabyte": 1024**5,
        "exabyte": 1024**6,
        "zetabyte": 1024**7,
        "yottabyte": 1024**8,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
        "pb": 1024**5,
        "eb": 1024**6,
        "zb": 1024**7,
        "yb": 1024**8,
    }

    for suffix in multipliers:
        size_str = size_str.lower().strip().strip("s")
        if size_str.lower().endswith(suffix):
            return int(float(size_str[0 : -len(suffix)]) * multipliers[suffix])
    if size_str.endswith("b"):
        size_str = size_str[0:-1]
    elif size_str.endswith("byte"):
        size_str = size_str[0:-4]
    return int(float(size_str))


def build_glonass_slot_dictionary(header_line):
    header_line_split = header_line.split()
    n_sat = int(header_line_split[0])
    glonass_slot_dict = {}
    for sat in range(1, n_sat + 1):
        # append entry to dict
        glonass_slot_dict |= {
            int(header_line_split[1 + 2 * (sat - 1)][1:]): int(
                header_line_split[1 + 2 * (sat - 1) + 1]
            )
        }
    return glonass_slot_dict


def satellite_id_2_system_time_scale(satellite_id):
    assert len(satellite_id) == 3, (
        f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    )
    return constants.constellation_2_system_time_scale[constellation(satellite_id)]


def constellation(satellite_id: str):
    assert len(satellite_id) == 3, (
        f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    )
    return satellite_id[0]


def compute_sagnac_effect(sat_pos_m, rx_pos_m):
    """compute the Sagnac effect (effect of the Earth's rotation during signal propagation°

    Input:
    - sat_pos_m: satellite ECEF position. np.ndarray of shape (n, 3)
    - rx_pos_m: satellite ECEF position. np.ndarray of shape (3,)

    Note:
    The computation uses small angle approximation of cos and sin.

    Reference:
    RTKLIB v2.4.2 manual, eq E.3.8b, p 140
    """
    sagnac_effect_m = (
        constants.cGpsOmegaDotEarth_rps / constants.cGpsSpeedOfLight_mps
    ) * (sat_pos_m[:, 0] * rx_pos_m[1] - sat_pos_m[:, 1] * rx_pos_m[0])
    return sagnac_effect_m


def compute_satellite_elevation_and_azimuth(sat_pos_ecef, receiver_pos_ecef):
    """
    sat_pos_ecef: np.array of shape (n, 3)
    receiver_pos_ecef: np.array of shape (3,)
    Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.9),(B.13),(B.14)
    """
    sat_pos_wrt_rx_pos_ecef = sat_pos_ecef - receiver_pos_ecef
    sat_pos_wrt_rx_pos_norm = np.linalg.norm(sat_pos_wrt_rx_pos_ecef, axis=1)[
        :, np.newaxis
    ]
    unit_vector_rx_satellite_ecef = sat_pos_wrt_rx_pos_ecef / sat_pos_wrt_rx_pos_norm
    [receiver_lat_rad, receiver_lon_rad, __] = ecef_2_geodetic(receiver_pos_ecef)
    unit_e_ecef = [-np.sin(receiver_lon_rad), np.cos(receiver_lon_rad), 0]
    unit_n_ecef = [
        -np.cos(receiver_lon_rad) * np.sin(receiver_lat_rad),
        -np.sin(receiver_lon_rad) * np.sin(receiver_lat_rad),
        np.cos(receiver_lat_rad),
    ]
    unit_u_ecef = [
        np.cos(receiver_lon_rad) * np.cos(receiver_lat_rad),
        np.sin(receiver_lon_rad) * np.cos(receiver_lat_rad),
        np.sin(receiver_lat_rad),
    ]
    elevation_rad = np.arcsin(np.dot(unit_vector_rx_satellite_ecef, unit_u_ecef))
    azimuth_rad = np.arctan2(
        np.dot(unit_vector_rx_satellite_ecef, unit_e_ecef),
        np.dot(unit_vector_rx_satellite_ecef, unit_n_ecef),
    )

    return elevation_rad, azimuth_rad


def geodetic_2_ecef(lat_rad, lon_rad, altitude_m):
    """Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.1),(B.2),(B.3)
    """
    n = constants.cWgs84EarthSemiMajorAxis_m / np.sqrt(
        1 - constants.cWgs84EarthEccentricity**2 * np.sin(lat_rad) ** 2
    )
    x = (n + altitude_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + altitude_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - constants.cWgs84EarthEccentricity**2) * n + altitude_m) * np.sin(lat_rad)
    return [x, y, z]


def ecef_2_geodetic(pos_ecef):
    """Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.4),(B.5),(B.6)
    """
    p = np.sqrt(pos_ecef[0] ** 2 + pos_ecef[1] ** 2)
    longitude_rad = np.arctan2(pos_ecef[1], pos_ecef[0])
    precision_m = 1e-3
    delta_h_m = 1  # initialization to a value larger than precision
    altitude_m = 0
    latitude_rad = np.arctan2(
        pos_ecef[2], p * (1 - constants.cWgs84EarthEccentricity**2)
    )
    while delta_h_m > precision_m:
        n = constants.cWgs84EarthSemiMajorAxis_m / np.sqrt(
            1 - constants.cWgs84EarthEccentricity**2 * np.sin(latitude_rad) ** 2
        )
        altitude_previous = altitude_m
        altitude_m = p / np.cos(latitude_rad) - n
        delta_h_m = np.abs(altitude_m - altitude_previous)
        latitude_rad = np.arctan2(
            pos_ecef[2],
            p * (1 - n * constants.cWgs84EarthEccentricity**2 / (n + altitude_m)),
        )
    return [latitude_rad, longitude_rad, altitude_m]

def ecef_2_satellite(pos_ecef: np.array, pos_sat_ecef: np.array, epoch: np.array):
    """
    Converts an ecef position to the satellite-fixed coordinate frame

    pos_sat_ecef: shape (n,3)
    pos_ecef: shape (n,3)
    epoch: shape (n,)
    """
    k = -pos_sat_ecef / np.linalg.norm(pos_sat_ecef, axis=1).reshape(-1, 1)
    pos_sun_ecef = compute_sun_ecef_position(epoch).T
    unit_vector_sun_ecef = (pos_sun_ecef - pos_sat_ecef) / np.linalg.norm(
        pos_sun_ecef - pos_sat_ecef, axis=1
    ).reshape(-1, 1)
    j = np.cross(k, unit_vector_sun_ecef)
    j = j / np.linalg.norm(j, axis=1).reshape(-1, 1)
    i = np.cross(j, k)

    rot_mat_ecef2sat = np.stack([i, j, k], axis=1)
    pos_sat_frame = np.stack(
        [
            rot_mat_ecef2sat[i, :, :] @ (pos_ecef[i, :] - pos_sat_ecef[i, :])
            for i in range(pos_sat_ecef.shape[0])
        ]
    )
    return pos_sat_frame, rot_mat_ecef2sat

def obs_dataset_to_obs_dataframe(ds: xarray.Dataset):
    # Flatten the xarray DataSet into a pandas DataFrame:
    logger.info("Converting Dataset into flat Dataframe of observations")
    flat_obs = pd.DataFrame()
    for obs_label, sat_time_obs_array in ds.data_vars.items():
        df = sat_time_obs_array.to_dataframe(name="obs_value").reset_index()
        df = df[df["obs_value"].notna()]
        df = df.assign(obs_type=lambda x: obs_label)
        flat_obs = pd.concat([flat_obs, df])
    return flat_obs


def get_gpst_utc_leap_seconds(rinex_file: Path):
    leap_seconds_astropy = compute_gps_utc_leap_seconds(
        yyyy=int(rinex_file.name[12:16]), doy=int(rinex_file.name[16:19])
    )

    # sanity check if leap second information is in the header of the RNX NAV file
    # compare astropy leap and RNX NAV leap seconds
    header = georinex.rinexheader(rinex_file)
    if "LEAP SECONDS" in header:
        ls_before = header["LEAP SECONDS"][0:6].strip()
        assert 0 < len(ls_before) < 3, (
            f"Unexpected leap seconds {ls_before} in {rinex_file}"
        )

        ls_after = header["LEAP SECONDS"][6:12].strip()
        if ls_after == "":
            return int(ls_before)
        assert 0 < len(ls_after) < 3, (
            f"Unexpected leap seconds {ls_after} in {rinex_file}"
        )

        assert ls_after == ls_before, (
            f"Leap second change announcement in {rinex_file}, this case is not tested, aborting."
        )

        leap_seconds_rnx = int(ls_before)
        assert leap_seconds_rnx == leap_seconds_astropy, (
            "leap second computed from astropy is different from RINEX NAV header"
        )

    return leap_seconds_astropy


def is_sorted(iterable):
    return all(iterable[i] <= iterable[i + 1] for i in range(len(iterable) - 1))


def compute_gps_utc_leap_seconds(yyyy: int, doy: int):
    timestamp = pd.Timestamp(year=yyyy, month=1, day=1) + pd.Timedelta(days=doy)
    if timestamp < constants.cGpstUtcEpoch:
        return np.nan
    ls_table = iers.LeapSeconds().auto_open()
    mjd_current = astrotime.Time(timestamp).mjd
    # check the ls_table in reverse order until mjd is lower than current mjd
    ls = np.nan
    for ind in range(len(ls_table) - 1, 0, -1):
        if ls_table[ind]["mjd"] - mjd_current <= 0:
            ls = ls_table[ind]["tai_utc"] - 19  # -19 to go back to GPS ref
            break
    assert ~np.isnan(ls), "GPS leap second could not be retrieved"
    return ls


def timestamp_to_gps_week_and_dow(ts: pd.Timestamp) -> tuple[int, int]:
    ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    delta = ts_utc - constants.cGpstUtcEpoch.tz_localize("UTC")
    gps_week = delta.days // 7
    dow = delta.days % 7  # day of week
    return gps_week, dow

def compute_sun_ecef_position(epochs: np.array) -> np.array:
    """
    Compute the Sun's ECEF position using the Astropy library
    """
    time = astrotime.Time(epochs, scale="utc")
    sun_gcrs = get_sun(time)
    sun_ecef = sun_gcrs.transform_to(ITRS(obstime=time))
    return sun_ecef.cartesian.xyz.to(astropy.units.meter).value