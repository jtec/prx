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
    sp3_filename = f"{aaa}0MGX{typ}_{yyyy}{ddd}*_01D_*_ORB.SP3.gz"
    clk_filename = f"{aaa}0MGX{typ}_{yyyy}{ddd}*_01D_*_CLK.CLK.gz"
    return sp3_filename, clk_filename


def sp3_file_database_folder():
    db_folder = Path(__file__).parent / "test/datasets"
    db_folder.mkdir(exist_ok=True)
    return db_folder


def sp3_file_folder(day: pd.Timestamp, parent_folder: Path):
    folder = parent_folder / f"{day.year}/{day.day_of_year:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_local_sp3(day: pd.Timestamp, pattern, db_folder=sp3_file_database_folder()):
    candidates = list(sp3_file_folder(day, db_folder).glob(pattern))
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        log.warning(
            f"Found more than one sp3 file for {day}: \n{candidates} \n Will use the first one."
        )
    log.info(f"Found the sp3 local file : {candidates[0]}")
    return candidates[0]


def list_ftp_directory(server, folder):
    ftp = FTP(server)
    ftp.login()
    ftp.cwd(folder)
    dir_list = []
    ftp.dir(dir_list.append)
    return [c.split()[-1].strip() for c in dir_list]


def try_downloading_sp3_ftp(gps_week, day: pd.Timestamp, folder: Path, pattern):
    server = "gssc.esa.int"
    if gps_week > 2237:
        remote_folder = f"/gnss/products/{gps_week}"
    else:
        remote_folder = f"/gnss/products/{gps_week}/mgex"
    candidates = list_ftp_directory(server, remote_folder)
    candidates = [c for c in candidates if fnmatch.fnmatch(c, pattern)]
    if len(candidates) == 0:
        log.warning(f"Could not find sp3 file for {day}")
        return None
    candidates = sorted(
        candidates,
        key=lambda x: int("BRDC00" in x) + int("IGS" in x),
        reverse=True,
    )  #
    file = candidates[0]
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


def get_sp3_file(
    mid_day_start: pd.Timestamp,
    mid_day_end: pd.Timestamp,
    db_folder=sp3_file_database_folder(),
):
    day = mid_day_start
    gps_week, _ = timestamp_to_gps_week_and_dow(day)
    while day <= mid_day_end:
        for p in priority:
            sp3_filename, clk_filename = build_sp3_filename(day, p)
            sp3_file = get_local_sp3(day, sp3_filename, db_folder)
            clk_file = get_local_sp3(day, clk_filename, db_folder)
            if sp3_file is None:
                sp3_file = try_downloading_sp3_ftp(
                    gps_week, day, sp3_file_folder(day, db_folder), sp3_filename
                )
            if clk_file is None:
                clk_file = try_downloading_sp3_ftp(
                    gps_week, day, sp3_file_folder(day, db_folder), clk_filename
                )
            if sp3_file is not None and clk_file is not None:
                return sp3_file, clk_file
        day += pd.Timedelta(1, unit="days")
    return None, None


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

    sp3_file, clk_file = get_sp3_file(t_start, t_end)
    return sp3_file, clk_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="rinex_aux_files",
        description="rinex_aux_files discovers or downloads files needed to get started on positioning: "
        "broadcast ephemeris, precise ephemeris etc.",
    )
    parser.add_argument(
        "--observation_file_path", type=str, help="Observation file path", required=True
    )
    args = parser.parse_args()
    assert Path(args.observation_file_path).exists(), (
        f"Cannot find observation file {args.observation_file_path}"
    )
    discover_or_download_sp3_file(Path(args.observation_file_path))
