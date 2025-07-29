import argparse
import logging
from ftplib import FTP
import os
from pathlib import Path
import re

import georinex
import pandas as pd
import urllib
import fnmatch

from prx import converters
from prx import util
import prx
from prx.util import timestamp_to_mid_day

log = util.get_logger(__name__)

atx_filename = f"igs[0-9][0-9]_????.atx"

def date_to_gps_week(date:pd.Timestamp):
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
    db_folder = util.prx_repository_root() / "src/prx/precise_corrections/antex/atx_files"
    db_folder.mkdir(exist_ok=True)
    return db_folder

def find_latest_local_antex_file(db_folder = atx_file_database_folder()):
    candidates = list(
        db_folder.glob(atx_filename)
    )
    if not candidates :
        return None
    return max(candidates, key=lambda c: extract_gps_week(c))

# def get_local_atx(gps_week : int, db_folder=atx_file_database_folder()):
#     candidates = list(
#         db_folder.glob(atx_filename)
#     )
#     candidates = [
#         c for c in candidates if len(c) == 14 and extract_gps_week(c) != -1
#         ]  
#     if len(candidates) == 0:
#         return None
#     if len(candidates) > 1:
#         candidates = sorted(
#             candidates, 
#             key=lambda c: extract_gps_week(c),
#             reverse=True
#         )
#         log.warning(
#                 f"Found more than one ANTEX file for {gps_week}: \n{candidates} \n Will use the first one."
#             )
#     log.info(f"Found the ANTEX local file : {candidates[0]}")
#     return candidates[0]

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
    server = 'gssc.esa.int'
    remote_folder = f"/igs/station/general"
    candidates = list_ftp_directory(server, remote_folder)
    candidates = list_ftp_directory(server, remote_folder)
    candidates = [
        c
        for c in candidates
        if fnmatch.fnmatch(c, atx_filename)
    ]
    if not candidates : 
        return None
    return max(candidates, key=lambda c:extract_gps_week(c))

def try_downloading_atx_ftp(remote_file : str, folder : Path):
    server = 'gssc.esa.int'
    remote_folder = f"/igs/station/general"
    candidates = list_ftp_directory(server, remote_folder)
    candidates = [
        c
        for c in candidates
        if fnmatch.fnmatch(c, remote_file)
    ]
    if len(candidates) == 0:
        log.warning(f"Could not find ANTEX file named {remote_file}")
        return None
    if len(candidates) > 1 :
        log.warning(
            f"Found more than one ANTEX file named {remote_file}: \n{candidates} \n Will use the first one."
        )
    file = candidates[0]
    ftp_file = f"ftp://{server}/{remote_folder}/{file}"
    local_file = folder / file
    urllib.request.urlretrieve(ftp_file, local_file)
    if not local_file.exists():
        log.warning(f"Could not download {ftp_file}")
        return None
    log.info(f"Downloaded ANTEX file {ftp_file}")
    return local_file

def get_atx_file(date: pd.Timestamp, db_folder = atx_file_database_folder()):
    gps_week = date_to_gps_week(date)
    atx_file = None
    while atx_file is None:
        latest_atx_local =  find_latest_local_antex_file(db_folder)
        latest_atx_remote = fetch_latest_remote_antex_file()
        if latest_atx_remote == latest_atx_local :
            return latest_atx_local
        elif latest_atx_remote and (not latest_atx_local or extract_gps_week(latest_atx_remote)> extract_gps_week(latest_atx_local)):
            # Download the latest file 
            atx_file = try_downloading_atx_ftp(latest_atx_remote, db_folder)
        if atx_file is not None :
            if gps_week > extract_gps_week(atx_file.name) :
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

    t_start = timestamp_to_mid_day(util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
        - pd.Timedelta(200, unit="milliseconds"))

    atx_file = get_atx_file(t_start) 
    return atx_file