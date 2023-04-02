import hashlib
from pathlib import Path
import logging
import constants
import numpy as np
import pandas as pd
import glob
import subprocess

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


def timestamp_2_gpst_ns(ts: pd.Timestamp):
    delta = ts - constants.cGpstEpoch
    return delta.delta


def rinex_header_time_string_2_timestamp_ns(time_string: str) -> np.datetime64:
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
