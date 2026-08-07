import ftplib
import logging
import urllib
from pathlib import Path

import pandas as pd

from prx import util

log = logging.getLogger(__name__)


def bia_file_database_folder():
    """
    Returns the path to the folder where ATX database files are stored.
    """
    db_folder = util.prx_src_directory() / "precise_corrections/bia/bia_files"
    db_folder.mkdir(exist_ok=True, parents=True)
    return db_folder


def build_bia_file_name(year: int, doy: int, analysis_center: str):
    return f"{analysis_center}0MGXFIN_{year}{doy:03d}0000_01D_01D_OSB.BIA.gz"


def bia_file_folder(year: int, doy: int):
    folder = bia_file_database_folder() / f"{year}/{doy:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_local_bia_file(year: int, doy: int, analysis_center: str) -> Path | None:
    local_file = bia_file_folder(year, doy) / build_bia_file_name(
        year, doy, analysis_center
    )
    if local_file.exists():
        return local_file
    else:
        return None


def check_online_availability(year: int, doy: int, analysis_center: str) -> Path | None:
    """
    Need to keep the same inputs as try_downloading_bia_ftp, in order to be able to use `unittest.mock.patch` in tests
    """
    server = "gssc.esa.int"
    gps_week, _ = util.timestamp_to_gps_week_and_dow(
        pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    )
    remote_folder = f"gnss/products/{gps_week}"
    file = build_bia_file_name(year, doy, analysis_center)
    ftp = ftplib.FTP(server)
    ftp.login()
    ftp.cwd(remote_folder)
    try:
        ftp.size(file)
        return bia_file_folder(year, doy) / build_bia_file_name(
            year, doy, analysis_center
        )
    except ftplib.error_perm:
        log.warning(f"{file} not available on {server}")
        return None


def try_downloading_bia_ftp(year: int, doy: int, analysis_center: str) -> Path | None:
    server = "gssc.esa.int"
    file = build_bia_file_name(year, doy, analysis_center)
    gps_week, _ = util.timestamp_to_gps_week_and_dow(
        pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    )
    remote_folder = f"gnss/products/{gps_week}"
    ftp_file = f"ftp://{server}/{remote_folder}/{file}"
    local_file = bia_file_folder(year, doy) / build_bia_file_name(
        year, doy, analysis_center
    )
    urllib.request.urlretrieve(ftp_file, local_file)
    if not local_file.exists():
        log.warning(f"Could not download {ftp_file}")
        return None
    log.info(f"Downloaded bia file: {ftp_file}")
    return local_file


def discover_or_download_bia_file(sp3_file_path: Path) -> Path | None:
    log.info(f"Finding bia files for {sp3_file_path} ...")
    year = int(sp3_file_path.name[11:15])
    doy = int(sp3_file_path.stem[15:18])
    analysis_center = sp3_file_path.name[0:3]

    local_file = get_local_bia_file(year, doy, analysis_center)
    if local_file:
        log.info(f"Found local bia file: {local_file}")
        return local_file

    downloaded_file = try_downloading_bia_ftp(year, doy, analysis_center)
    if downloaded_file:
        return downloaded_file

    return None
