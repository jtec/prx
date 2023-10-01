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
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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
        # TODO double-check the offset between BDT and GPST
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
    fractional_seconds = np.float64(
        timedelta_2_nanoseconds(time_delta)
        - integer_seconds * constants.cNanoSecondsPerSecond
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
        str(prx_root().joinpath("tools/gfzrnx/**gfzrnx**")), recursive=True
    )
    for gfzrnx_binary in gfzrnx_binaries:
        command = f" {gfzrnx_binary} -finp {file} -fout {file}  -chk -kv -f"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            log.info(f"Ran gfzrnx file repair on {file}")
            return file
    assert False, "gdzrnx file repair run failed!"


def deg_2_rad(angle_deg):
    return angle_deg * np.pi / 180


def profile_this(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile"
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper


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
