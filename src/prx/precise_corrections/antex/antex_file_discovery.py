from ftplib import FTP
import ftplib
from pathlib import Path
import re

import georinex
import pandas as pd
import urllib
import fnmatch

from prx import converters
from prx import util
from prx.util import timestamp_to_mid_day

"""
Logic overview:
- The code first extracts the GPS week from the provided RINEX observation file.
- It then identifies the most recent ANTEX file available both locally and remotely.
- If both local and remote ANTEX files are the same, the local version is reused, as the local database is considered up to date.
- Otherwise, the latest remote file is downloaded and used.
- Note: If the most recent ANTEX file has a GPS week that is earlier than the observation file's GPS week 
(e.g., the remote database has not yet been updated), a warning is displayed to indicate that the most recent ANTEX file 
was still used despite being older than the observation file.
"""

log = util.get_logger(__name__)

atx_filename = f"igs20_????.atx"


def date_to_gps_week(date: pd.Timestamp):
    """
    Convert a Timestamp to GPS week

    GPS week starts on 01/06/1980 (sunday)
    """
    gps_epoch = pd.Timestamp("1980-01-06T00:00:00Z")
    delta = date.tz_localize("UTC") - gps_epoch
    gps_week = delta.days // 7
    return gps_week


def extract_gps_week(filename: str) -> int:
    """
    Extract GPS week from filenames like 'igsxx_WWWW.atx'.
    """
    match = re.search(r"igs...(\d{4})\.atx", filename)
    if match:
        return int(match.group(1))
    return -1


def atx_file_database_folder():
    """
    Returns the path to the folder where ATX database files are stored.
    """
    db_folder = (
        util.prx_repository_root() / "src/prx/precise_corrections/antex/atx_files"
    )
    db_folder.mkdir(exist_ok=True)
    return db_folder


def find_latest_local_antex_file(db_folder=atx_file_database_folder()):
    candidates = list(db_folder.glob(atx_filename))
    if not candidates:
        return None
    return max(candidates, key=lambda c: extract_gps_week(c))


def list_ftp_directory(server, folder):
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(folder)
    dir_list = []
    ftp.dir(dir_list.append)
    return [c.split()[-1].strip() for c in dir_list]


def fetch_latest_remote_antex_file():
    """
    List the ANTEX files available online and returns the latest
    """
    server = "gssc.esa.int"
    remote_folder = f"/igs/station/general"
    candidates = list_ftp_directory(server, remote_folder)
    candidates = [c for c in candidates if fnmatch.fnmatch(c, atx_filename)]
    if not candidates:
        return None
    return max(candidates, key=lambda c: extract_gps_week(c))


def check_online_availability(file: str, folder: Path) -> Path | None:
    """
    Need to keep the same inputs as try_downloading_sp3_ftp, in order to be able to use `unittest.mock.patch` in tests
    """
    server = "gssc.esa.int"
    remote_folder = f"/igs/station/general"
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(remote_folder)
    try:
        ftp.size(file)
        return folder.joinpath(file)
    except ftplib.error_perm:
        log.warning(f"{file} not available on {server}")
        return None


def try_downloading_atx_ftp(file: str, folder: Path):
    """
    Download the wanted remote file
    """
    server = "gssc.esa.int"
    remote_folder = f"/igs/station/general"
    ftp_file = f"ftp://{server}/{remote_folder}/{file}"
    local_file = folder / file
    urllib.request.urlretrieve(ftp_file, local_file)
    if not local_file.exists():
        log.warning(f"Could not download {ftp_file}")
        return None
    log.info(f"Downloaded ANTEX file {ftp_file}")
    return local_file


def get_atx_file(date: pd.Timestamp, db_folder=atx_file_database_folder()):
    gps_week = date_to_gps_week(date)
    latest_atx_local = find_latest_local_antex_file(db_folder)
    latest_atx_remote = fetch_latest_remote_antex_file()
    if latest_atx_remote == latest_atx_local:
        return latest_atx_local
    elif latest_atx_remote and (
        not latest_atx_local
        or extract_gps_week(latest_atx_remote) > extract_gps_week(latest_atx_local)
    ):
        # Download the latest file
        atx_file = try_downloading_atx_ftp(latest_atx_remote, db_folder)
    if atx_file is not None:
        if gps_week > extract_gps_week(atx_file.name):
            log.warning(
                f"No ANTEX file found for the target GPS week {gps_week} — using the most recent available instead."
            )
        return atx_file

    raise FileNotFoundError("No file ANTEX found locally or online.")


def discover_or_download_atx_file(observation_file_path=Path()):
    """
    Returns the path to a valid antes file (local or downloaded) corresponding to the observation file.
    """
    log.info(f"Finding auxiliary files for {observation_file_path} ...")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    header = georinex.rinexheader(rinex_3_obs_file)

    t_start = timestamp_to_mid_day(
        util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
        - pd.Timedelta(200, unit="milliseconds")
    )

    atx_file = get_atx_file(t_start)
    return atx_file
