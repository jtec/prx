import argparse
import json
from pathlib import Path
import converters
import helpers
import georinex
from collections import defaultdict
import urllib


log = helpers.get_logger(__name__)


def download_ephemerides(t_start_tai_ns, t_end_tai_ns):
    return None


def get_on_it(observation_file_path=Path()):
    log.info(f"Finding auxiliary files for {observation_file_path} ...")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    header = georinex.rinexheader(rinex_3_obs_file)
    eph = download_ephemerides(
        helpers.rinex_header_time_string_2_gpst_ns(header["TIME OF FIRST OBS"]),
        helpers.rinex_header_time_string_2_gpst_ns(header["TIME OF LAST OBS"]),
    )


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
