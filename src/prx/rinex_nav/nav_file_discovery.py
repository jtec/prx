import argparse
import os
from pathlib import Path
import georinex
import urllib.request
import pandas as pd

from prx import converters, helpers
from prx.helpers import timestamp_to_mid_day
from prx.util import is_rinex_3_nav_file

log = helpers.get_logger(__name__)


def is_rinex_3_mixed_mgex_broadcast_ephemerides_file(file: Path):
    return str(file).endswith("MN.rnx")


def try_downloading_ephemerides_http(day: pd.Timestamp, local_destination_folder: Path):
    # IGS BKG Rinex 3.04 mixed file paths follow this pattern:
    # For files after 2022
    # https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2023/002/BRDC00IGS_R_20230020000_01D_MN.rnx.gz
    # For files before 2022
    # https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2021/365/BRDC00WRD_R_20213650000_01D_MN.rnx.gz
    if day.year < 2022:
        country_code = "WRD"
    else:
        country_code = "IGS"
    remote_file = Path(
        f"/{day.year}/{day.day_of_year:03}/BRDC00{country_code}_R_{day.year}{day.day_of_year:03}0000_01D_MN.rnx.gz"
    )
    local_compressed_file = local_destination_folder.joinpath(remote_file.name)
    url = "https://igs.bkg.bund.de/root_ftp/IGS/BRDC" + str(remote_file.as_posix())
    try:
        urllib.request.urlretrieve(url, local_compressed_file)
        local_file = converters.compressed_to_uncompressed(local_compressed_file)
        os.remove(local_compressed_file)
        log.info(f"Downloaded broadcast ephemerides file from {url}")
        helpers.repair_with_gfzrnx(local_file)
        return local_file
    except Exception:
        log.warning(f"Could not download broadcast ephemerides file from {url}")
        return None


def try_downloading_ephemerides_ftp(day: pd.Timestamp, folder: Path):
    root = "igs.ign.fr/pub/igs/data"
    file_name = f"BRDM00DLR_S_{day.year}{day.day_of_year:03}0000_01D_MN.rnx.gz"
    ftp_file_path = f"ftp://{root}/{day.year}/{day.day_of_year:03}/{file_name}"
    local_compressed_file = folder / file_name
    urllib.request.urlretrieve(ftp_file_path, local_compressed_file)
    if not local_compressed_file.exists():
        log.warning(f"Could not download {ftp_file_path}")
        return None
    local_file = converters.compressed_to_uncompressed(local_compressed_file)
    os.remove(local_compressed_file)
    log.info(f"Downloaded broadcast ephemerides file {ftp_file_path}")
    helpers.repair_with_gfzrnx(local_file)
    return local_file


def try_downloading_ephemerides(mid_day: pd.Timestamp, folder: Path):
    local_file = try_downloading_ephemerides_http(mid_day, folder)
    if not local_file:
        local_file = try_downloading_ephemerides_ftp(mid_day, folder)
    if not local_file:
        log.warning(f"Could not download broadcast ephemerides for {mid_day}")
    return local_file


def rinex_3_ephemerides_file_coverage_time(ephemerides_file: Path):
    parts = str(ephemerides_file).split("_")
    start_time = pd.to_datetime(parts[-3], format="%Y%j%H%M")
    assert (
        parts[-2][-1] == "D"
    ), f"Was expecting 'D' (days) as duration unit in Rinex ephemerides file name: {ephemerides_file}"
    duration = parts[-2][:-1]
    duration_unit = parts[-2][-1]
    end_time = start_time + pd.Timedelta(int(duration), duration_unit.lower())
    return start_time, end_time


def nav_file_folder(day: pd.Timestamp, parent_folder: Path):
    folder = parent_folder / f"{day.year}/{day.day_of_year:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def nav_file_database_folder():
    db_folder = Path(__file__).parent / "nav_files"
    db_folder.mkdir(exist_ok=True)
    return db_folder


def get_local_ephemerides(
    day: pd.Timestamp,
):
    candidates = list(
        nav_file_folder(day, nav_file_database_folder()).glob(
            f"*_{day.year}{day.day_of_year:03d}0000_01D_MN.rnx"
        )
    )
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        log.warning(
            f"Found more than one nav file for {day}: \n{candidates} \n Will use the first one."
        )
    return candidates[0]


def update_local_database(mid_day_start: pd.Timestamp, mid_day_end: pd.Timestamp):
    # Download IGS nav files for the given period to our local nav file folder
    day = mid_day_start
    while day <= mid_day_end:
        nav_file = get_local_ephemerides(day)
        if nav_file is None:
            try_downloading_ephemerides(
                day, nav_file_folder(day, nav_file_database_folder())
            )
        day += pd.Timedelta(1, unit="days")


def discover_or_download_ephemerides(
    t_start: pd.Timestamp, t_end: pd.Timestamp, folder, constellations
):
    # If there are any navigation file provided by the user, use them, if not, use IGS files.
    user_provided_nav_files = [
        f for f in folder.rglob("*.rnx") if is_rinex_3_nav_file(f)
    ]
    if len(user_provided_nav_files) > 0:
        return user_provided_nav_files
    # Ephemeris files cover at least a day, so first round time stamps to midday here
    t_start = timestamp_to_mid_day(t_start)
    t_end = timestamp_to_mid_day(t_end)
    # Update our local ephemeris database, fetching nav file from IGS servers for the days in question
    update_local_database(t_start, t_end)
    ephemerides_files = []
    day = t_start
    while day <= t_end:
        ephemerides_file = get_local_ephemerides(day)
        if ephemerides_file is not None:
            ephemerides_files.append(ephemerides_file)
        day += pd.Timedelta(1, unit="days")
    return ephemerides_files


def discover_or_download_auxiliary_files(observation_file_path=Path()):
    log.info(f"Finding auxiliary files for {observation_file_path} ...")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    header = georinex.rinexheader(rinex_3_obs_file)
    # Subtract 200 ms from the first obs epoch in receiver time
    # This accounts for
    #   - signal travel time to GEO satellite (~120 ms)
    #   + max sat clock offset (37 ms)
    #   + 43 ms for receiver clock offset + margin
    ephs = discover_or_download_ephemerides(
        helpers.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
        - pd.Timedelta(200, unit="milliseconds"),
        helpers.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"]),
        rinex_3_obs_file.parent,
        list(header["fields"].keys()),
    )
    # Note that ephs may be a list of paths, in case the observation file spans several days
    return {"broadcast_ephemerides": ephs}


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
    assert Path(
        args.observation_file_path
    ).exists(), f"Cannot find observation file {args.observation_file_path}"
    discover_or_download_auxiliary_files(Path(args.observation_file_path))
