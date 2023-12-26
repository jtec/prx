import hashlib
from pathlib import Path
import logging
from . import constants
import numpy as np
import pandas as pd
import glob
import subprocess
import math
import cProfile


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


def get_logger(label):
    return logging.getLogger(label)


def prx_root() -> Path:
    return Path(__file__).parent.parent


# From https://stackoverflow.com/a/3431838
def md5_of_file_content(file: Path):
    assert file.exists(), f"Looks like {file} does not exist."
    t0 = pd.Timestamp.now()
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    hash_string = hash_md5.hexdigest()
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
    gfzrnx_binaries = glob.glob(
        str(prx_root().joinpath("prx/tools/gfzrnx/**gfzrnx**")), recursive=True
    )
    for gfzrnx_binary in gfzrnx_binaries:
        command = f" {gfzrnx_binary} -finp {file} -fout {file}  -chk -kv -f"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            log.info(f"Ran gfzrnx file repair on {file}")
            return file
    assert False, f"gdzrnx file repair run failed: {result}"


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
    constellation_2_system_time_scale = {
        "G": "GPST",
        "S": "SBAST",
        "E": "GST",
        "C": "BDT",
        "R": "GLONASST",
        "J": "QZSST",
        "I": "IRNSST",
    }
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
    """compute the sagnac effect (effect of the Earth rotation during signal propagation°

    Input:
    - sat_pos_m: satellite ECEF position. np.ndarray of shape (3,)
    - rx_pos_m: satellite ECEF position. np.ndarray of shape (3,)

    Note:
    The computation uses small angle approximation of cos and sin.

    Reference:
    RTKLIB v2.4.2 manual, eq E.3.8b, p 140
    """
    sagnac_effect_m = (
        constants.cEarthRotationRate_radps
        / constants.cGpsIcdSpeedOfLight_mps
        * (sat_pos_m[0] * rx_pos_m[1] - sat_pos_m[1] * rx_pos_m[0])
    )
    return sagnac_effect_m


def compute_relativistic_clock_effect(
    sat_pos_m: np.array(
        3,
    ),
    sat_vel_mps: np.array(
        3,
    ),
):
    """
    Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equation (5.19)
    """
    relativistic_clock_effect_m = (
        -2 * np.dot(sat_pos_m, sat_vel_mps) / constants.cGpsIcdSpeedOfLight_mps
    )

    return relativistic_clock_effect_m


def compute_satellite_elevation_and_azimuth(sat_pos_ecef, receiver_pos_ecef):
    """
    Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.9),(B.13),(B.14)
    """
    rho = (sat_pos_ecef - receiver_pos_ecef) / np.linalg.norm(
        sat_pos_ecef - receiver_pos_ecef
    )
    [lat, lon, __] = ecef_2_geodetic(receiver_pos_ecef)
    unit_e = [-np.sin(lon), np.cos(lon), 0]
    unit_n = [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)]
    unit_u = [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]
    elevation_rad = np.arcsin(np.dot(rho, unit_u))
    azimuth_rad = np.arctan2(np.dot(rho, unit_e), np.dot(rho, unit_n))
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
