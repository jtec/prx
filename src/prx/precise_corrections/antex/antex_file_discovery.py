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

atx_filename = f"igs*.atx"

def list_all_atx_files():
    server = 'gssc.esa.int'
    folder = '/igs/station/general'
    with FTP(server) as ftp:
        ftp.login()
        ftp.cwd(folder)
        files = ftp.nlst()
    return [f for f in files if f.lower().endswith('.atx')]

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

def get_local_atx(gps_week : int, db_folder=atx_file_database_folder()):
    candidates = list(
        db_folder.glob(atx_filename)
    )
    candidates = [
        c for c in candidates if len(c) == 14 and extract_gps_week(c) != -1
        ]  
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        candidates = sorted(
            candidates, 
            key=lambda c: extract_gps_week(c),
            reverse=True
        )
        log.warning(
                f"Found more than one ANTEX file for {gps_week}: \n{candidates} \n Will use the first one."
            )
    log.info(f"Found the ANTEX local file : {candidates[0]}")
    return candidates[0]

def list_ftp_directory(server, folder):
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(folder)
    dir_list = []
    ftp.dir(dir_list.append)
    return [c.split()[-1].strip() for c in dir_list]

def try_downloading_atx_ftp(gps_week : int, folder : Path):
    server = 'gssc.esa.int'
    remote_folder = f"/igs/station/general"
    candidates = list_ftp_directory(server, remote_folder)
    candidates = [
        c
        for c in candidates
        if fnmatch.fnmatch(c, atx_filename)
    ]
    candidates = [
        c for c in candidates if len(c) == 14 and extract_gps_week(c) != -1
    ]
    if len(candidates) == 0:
        log.warning(f"Could not find ANTEX file for {gps_week}")
        return None
    
    candidates = sorted(
            candidates, 
            key=lambda c: extract_gps_week(c),
            reverse=True
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
        atx_file = get_local_atx(gps_week, db_folder)
        if atx_file is None :
            atx_file = try_downloading_atx_ftp(gps_week, db_folder)
        if atx_file is not None :
            return atx_file
        gps_week -=1
    return None

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