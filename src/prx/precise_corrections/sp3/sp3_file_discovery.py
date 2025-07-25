import argparse
import logging
from ftplib import FTP
import os
from pathlib import Path

import georinex
import pandas as pd
import urllib
import fnmatch

from prx import converters, util, constants
from prx.util import timestamp_to_mid_day, timestamp_to_gps_week_and_dow

log = logging.getLogger(__name__)

### Since gps week 2238
priority = [
    ("COD", "FIN"),
    ("GRG", "FIN"),
    ("GFZ", "FIN"),
    ("ESA", "FIN"),
    ("WUM", "FIN"),
    ("JAX", "FIN"),
    ("JPL", "FIN"),
    ("MIT", "FIN"),
    ("COD", "RAP"),
    ("GRG", "RAP"),
    ("GFZ", "RAP"),
    ("ESA", "RAP"),
    ("WUM", "RAP"),
]
# WWWW/AAA0PPPTYP_YYYYDDDHHMM_LEN_SMP_CNT.FMT.gz
# PPP : MGX
# CNT : ORB
# FMT : SP3

# This priority list is applied starting from GPS week 2238, to select SP3 orbit and CLK files based on the preferred
# analysis centers and product types. Final ("FIN") products are prioritized due to their higher accuracy and reliability.
# When final products are unavailable, rapid ("RAP") products serve as a fallback.
# Among the analysis centers, COD, GRG, and GFZ are placed at the top based on their long-standing reputation for delivering
# precise and complete orbit solutions within the IGS community. ESA, WUM, JAX, JPL, and MIT follow, as they also provide
# high-quality products, but may differ slightly in availability, latency, or consistency.
# Before GPS week 2238, the same type of SP3 files can be found, but they are stored in /{gps_week}/mgex directories,
# requiring a different file discovery logic.


def get_index_of_priority_from_filename(filename: str):
    for i, p in enumerate(priority):
        if p[0] in filename and p[1] in filename:
            return i


def build_sp3_filename(date: pd.Timestamp, aaa_typ):
    # aaa_typ : tuple of str (aaa, typ)
    # aaa: IGS analysis center
    # typ: IGS product type (RAP or FIN)
    yyyy = date.year
    ddd = f"{date.day_of_year:03d}"
    aaa = aaa_typ[0]
    typ = aaa_typ[1]
    sp3_filename = f"{aaa}0MGX{typ}_{yyyy}{ddd}0000_01D_05M_ORB.SP3.gz"
    clk_filename = f"{aaa}0MGX{typ}_{yyyy}{ddd}0000_01D_30S_CLK.CLK.gz"
    return sp3_filename, clk_filename


def sp3_file_database_folder():
    db_folder = Path(__file__).parent / "test/datasets"
    db_folder.mkdir(exist_ok=True)
    return db_folder


def sp3_file_folder(day: pd.Timestamp, parent_folder: Path):
    folder = parent_folder / f"{day.year}/{day.day_of_year:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_local_sp3(day: pd.Timestamp, file, db_folder=sp3_file_database_folder()):
    candidates = list(sp3_file_folder(day, db_folder).glob(file))
    if len(candidates) == 0:
        return None
    log.info(f"Found the sp3 local file : {candidates[0]}")
    local_file = converters.compressed_to_uncompressed(candidates[0])
    return local_file.name


def list_ftp_directory(server, folder):
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(folder)
    dir_list = []
    ftp.dir(dir_list.append)
    return [c.split()[-1].strip() for c in dir_list]


def check_online_availability(gps_week, day: pd.Timestamp, folder: Path, file):
    server = "gssc.esa.int"
    if gps_week > 2237:
        remote_folder = f"/gnss/products/{gps_week}"
    else:
        remote_folder = f"/gnss/products/{gps_week}/mgex"
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(remote_folder)
    try:
        ftp.size(file)
        return Path(file).stem
    except:
        log.warning(f"{file} not available on {server}")
        return None


def try_downloading_sp3_ftp(gps_week, day: pd.Timestamp, folder: Path, file):
    server = "gssc.esa.int"
    if gps_week > 2237:
        remote_folder = f"/gnss/products/{gps_week}"
    else:
        remote_folder = f"/gnss/products/{gps_week}/mgex"
    ftp_file = f"ftp://{server}/{remote_folder}/{file}"
    local_compressed_file = folder / file
    urllib.request.urlretrieve(ftp_file, local_compressed_file)
    if not local_compressed_file.exists():
        log.warning(f"Could not download {ftp_file}")
        return None
    local_file = converters.compressed_to_uncompressed(local_compressed_file)
    os.remove(local_compressed_file)
    log.info(f"Downloaded sp3 file {ftp_file}")
    return local_file


def get_sp3_files(
    mid_day_start: pd.Timestamp,
    mid_day_end: pd.Timestamp,
    db_folder=sp3_file_database_folder(),
):
    sp3_files = []
    day = mid_day_start
    gps_week, _ = timestamp_to_gps_week_and_dow(day)
    while day <= mid_day_end:
        for p in priority:
            sp3_filename, clk_filename = build_sp3_filename(day, p)
            file_orb = get_local_sp3(day, sp3_filename, db_folder)
            file_clk = get_local_sp3(day, clk_filename, db_folder)
            if file_orb is None:
                file_orb = try_downloading_sp3_ftp(
                    gps_week, day, sp3_file_folder(day, db_folder), sp3_filename
                )
            if file_clk is None:
                file_clk = try_downloading_sp3_ftp(
                    gps_week, day, sp3_file_folder(day, db_folder), clk_filename
                )
            if file_orb is not None and file_clk is not None:
                sp3_files.append((file_orb, file_clk))
                break
            # If we reach the end of the priority list without success
            if file_orb is None and file_clk is None and p == priority[-1]:
                sp3_files.append((None, None))
        day += pd.Timedelta(1, unit="days")
    return sp3_files


def discover_or_download_sp3_file(observation_file_path=Path()):
    """
    Returns the path to a valid SP3 file (local or downloaded) corresponding to the observation file.
    Tries to respect a priority hierarchy: IGS FIN > COD FIN > GRG FIN > ... > IGS ULR.
    """
    log.info(f"Finding auxiliary files for {observation_file_path} ...")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    header = georinex.rinexheader(rinex_3_obs_file)

    t_start = timestamp_to_mid_day(
        util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
        - pd.Timedelta(200, unit="milliseconds")
    )
    t_end = timestamp_to_mid_day(
        util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])
    )

    sp3_files = get_sp3_files(t_start, t_end)
    return sp3_files
