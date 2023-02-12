import argparse
import json
import os
import shutil
from pathlib import Path
import georinex
from collections import defaultdict
import urllib.request
import numpy as np
import glob
import subprocess

import pandas as pd

import converters
import helpers
import prx

log = helpers.get_logger(__name__)


def is_rinex_3_mixed_mgex_broadcast_ephemerides_file(file: Path):
    return str(file).endswith("MN.rnx")


def repair_with_gfzrnx(file):
    gfzrnx_binaries = glob.glob(
        str(prx.prx_root().joinpath("tools/gfzrnx/**gfzrnx**")), recursive=True
    )
    for gfzrnx_binary in gfzrnx_binaries:
        command = f" {gfzrnx_binary} -finp {file} -fout {file}  -chk -kv -f"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            log.info(f"Ran file gfzrnx file repair on {file}")
            return file
    assert False, "gdzrnx file repair run failed!"


def try_downloading_ephemerides_from_bkg(
    t_start: pd.Timestamp, t_end: pd.Timestamp, folder: Path
):

    time = t_start
    files = []
    while True:
        # IGS BKG Rinex 3.04 mixed file paths follow this pattern:
        # https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2023/002/BRDC00IGS_R_20230020000_01D_MN.rnx.gz
        remote_file = Path(
            f"/{time.year}/{time.day_of_year:03}/BRDC00IGS_R_{time.year}{time.day_of_year:03}0000_01D_MN.rnx.gz"
        )
        local_compressed_file = folder.joinpath(remote_file.name)
        url = "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/" + str(remote_file)
        urllib.request.urlretrieve(url, local_compressed_file)
        local_file = converters.compressed_to_uncompressed(local_compressed_file)
        os.remove(local_compressed_file)
        log.info(f"Downloaded broadcast ephemerides file from {url}")
        local_file = repair_with_gfzrnx(local_file)
        files.append(local_file)
        # Assuming that the downloaded files cover the whole day:
        t_coverage_start, t_coverage_end = rinex_3_ephemerides_file_coverage_time(
            local_file
        )
        if t_coverage_end > t_end:
            return files
    return None


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


def discover_local_ephemerides(
    t_start: pd.Timestamp, t_end: pd.Timestamp, folder: Path
):
    candidates = glob.glob(str(folder.joinpath("**.rnx**")), recursive=True)
    nav_files = []
    start_end_end_times = []
    for candidate in candidates:
        nav_file = converters.compressed_to_uncompressed(Path(candidate))
        if nav_file is None:
            continue
        if not is_rinex_3_mixed_mgex_broadcast_ephemerides_file(nav_file):
            continue
        nav_files.append(nav_file)
        start_end_end_times.extend(rinex_3_ephemerides_file_coverage_time(nav_file))
    if len(nav_files) == 0:
        return None
    if min(start_end_end_times) > t_start or max(start_end_end_times) < t_end:
        return None
    return nav_files


def discover_or_download_ephemerides(
    t_start: pd.Timestamp, t_end: pd.Timestamp, folder, constellations
):
    sources = [discover_local_ephemerides, try_downloading_ephemerides_from_bkg]
    for source in sources:
        ephemerides_files = source(t_start, t_end, folder)
        if ephemerides_files is not None:
            return ephemerides_files


def get_on_it(observation_file_path=Path()):
    log.info(f"Finding auxiliary files for {observation_file_path} ...")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    header = georinex.rinexheader(rinex_3_obs_file)
    eph = discover_or_download_ephemerides(
        helpers.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]),
        helpers.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"]),
        rinex_3_obs_file.parent,
        list(header["fields"].keys()),
    )
    if len(eph) > 1:
        assert (
            False
        ), "Observations crossing day boundaries not handled yet, need to merge ephemeris files here"
    return {"broadcast-ephemerides": eph}


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
    get_on_it(Path(args.observation_file_path), args.output_format)
