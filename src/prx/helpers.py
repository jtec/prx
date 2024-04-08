import platform
from pathlib import Path
import logging
from prx import constants
import numpy as np
import pandas as pd
import subprocess
import math
import joblib
import georinex
import imohash
import os

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)
log = logging.getLogger(__name__)


def parse_boolean_env_variable(env_variable_name: str, value_if_not_set: bool):
    # Based on https://stackoverflow.com/a/65407083/2567449
    var_string = os.environ.get(env_variable_name, None)
    if var_string is None:
        return value_if_not_set
    var_string = var_string.lower().strip()
    assert var_string in ("True", "true", "1", "False", "false", "0")
    return var_string in ("True", "true", "1")


disk_cache = joblib.Memory(
    Path(__file__).parent.joinpath("diskcache"), verbose=0, mmap_mode="r"
)
disable_caching = parse_boolean_env_variable("PRX_NO_CACHING", False)
if disable_caching:
    log.debug("Caching disabled by environment variable PRX_NO_CACHING, purging cache.")
    disk_cache.clear()


def get_logger(label):
    return logging.getLogger(label)


def prx_repository_root() -> Path:
    return Path(__file__).parents[2]


def hash_of_file_content(file: Path, use_sampling: bool = False):
    assert file.exists(), f"Looks like {file} does not exist."
    sample_threshhold = imohash.imohash.SAMPLE_THRESHOLD
    if not use_sampling:
        sample_threshhold = math.inf
    t0 = pd.Timestamp.now()
    hash_string = imohash.hashfile(
        file, hexdigest=True, sample_threshhold=sample_threshhold
    )
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want want to think about partially hashing the"
            f" file, e.g. using https://github.com/kalafut/py-imohash"
        )

    return hash_string


def timestamp_2_timedelta(timestamp: pd.Timestamp, time_scale):
    assert type(timestamp) == pd.Timestamp, "timestamp must be of type pd.Timestamp"
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


def timedelta_2_weeks_and_seconds(time_delta: pd.Timedelta):
    assert type(time_delta) == pd.Timedelta, "time_delta must be of type pd.Timedelta"
    in_nanoseconds = time_delta / pd.Timedelta(1, "ns")
    weeks = math.floor(in_nanoseconds / constants.cNanoSecondsPerWeek)
    week_nanoseconds = in_nanoseconds - weeks * constants.cNanoSecondsPerWeek
    return weeks, np.float64(week_nanoseconds) / constants.cNanoSecondsPerSecond


def week_and_seconds_2_timedelta(weeks, seconds):
    return pd.Timedelta(weeks * constants.cSecondsPerWeek + seconds, "seconds")


def timedelta_2_seconds(time_delta: pd.Timedelta):
    assert type(time_delta) == pd.Timedelta, "time_delta must be of type pd.Timedelta"
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
    assert type(time_delta) == pd.Timedelta, "time_delta must be of type pd.Timedelta"
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


def repair_with_gfzrnx(file):
    path_folder_gfzrnx = prx_repository_root().joinpath("tools", "gfzrnx")
    path_binary = path_folder_gfzrnx.joinpath(
        constants.gfzrnx_binary[platform.system()]
    )
    command = [
        str(path_binary),
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
        log.info(f"Ran gfzrnx file repair on {file}")
    else:
        log.info(f"gfzrnx file repair run failed: {result}")
        assert False
    return file


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
    assert (
        len(satellite_id) == 3
    ), f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    return constants.constellation_2_system_time_scale[constellation(satellite_id)]


def constellation(satellite_id: str):
    assert (
        len(satellite_id) == 3
    ), f"Satellite ID unexpectedly not three characters long: {satellite_id}"
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


def compute_relativistic_clock_effect(sat_pos_m: np.array, sat_vel_mps: np.array):
    """
    Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equation (5.19)

    Expects both arrays to be of shape (rows, columns) (n, 3)
    """
    relativistic_clock_effect_m = (
        -2
        * np.einsum("ij, ij->i", sat_pos_m, sat_vel_mps)
        / constants.cGpsSpeedOfLight_mps
    )

    return relativistic_clock_effect_m


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


def parse_rinex_obs_file(rinex_file: Path):
    @disk_cache.cache
    def cached_load(rinex_file: Path, file_hash: str):
        log.info(f"Parsing {rinex_file} ...")
        repair_with_gfzrnx(rinex_file)
        parsed = georinex.load(rinex_file)
        return parsed

    t0 = pd.Timestamp.now()
    file_content_hash = hash_of_file_content(rinex_file)
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want to partially hash the file"
        )
    return cached_load(rinex_file, file_content_hash)


def get_gpst_utc_leap_seconds_from_rinex_header(rinex_file: Path):
    header = georinex.rinexheader(rinex_file)
    assert "LEAP SECONDS" in header, "LEAP SECONDS not found in RINEX header"
    ls_before = header["LEAP SECONDS"][0:6].strip()
    assert (
        len(ls_before) > 0 and len(ls_before) < 3
    ), f"Unexpected leap seconds {ls_before} in {rinex_file}"
    ls_after = header["LEAP SECONDS"][6:12].strip()
    if ls_after == "":
        return int(ls_before)
    assert (
        len(ls_after) > 0 and len(ls_after) < 3
    ), f"Unexpected leap seconds {ls_after} in {rinex_file}"
    assert (
        ls_after == ls_before
    ), f"Leap second change annoucement in {rinex_file}, this case is not tested, aborting."
    return int(ls_before)


def is_sorted(iterable):
    return all(iterable[i] <= iterable[i + 1] for i in range(len(iterable) - 1))
