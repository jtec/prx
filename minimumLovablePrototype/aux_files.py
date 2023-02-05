import argparse
import json
from pathlib import Path
import converters
import helpers
import georinex
import parse_rinex
import constants
from collections import defaultdict
import git


log = helpers.get_logger(__name__)


def get_on_it(observation_file_path=Path()):
    log.info(f"Finding auxiliary files for {observation_file_path} ...")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='rinex_aux_files',
        description='rinex_aux_files discovers or downloads files needed to get started on positioning: broadcast ephemeris, precise ephemeris etc.',)
    parser.add_argument('--observation_file_path',
                        type=str,
                        help='Observation file path',
                        required=True)
    args = parser.parse_args()
    assert Path(args.observation_file_path).exists(), f"Cannot find observation file {args.observation_file_path}"
    get_on_it(Path(args.observation_file_path), args.output_format)
