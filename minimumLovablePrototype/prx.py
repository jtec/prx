import argparse
from pathlib import Path
import glob
import itertools
import subprocess
import logging
import converters

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)
def prx_root() -> Path:
    return Path(__file__).parent.parent


def process(observation_file_path):
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='prx',
        description='prx processes RINEX observations, computes a few useful things such as satellite position, '
                    'relativistic effects etc. and outputs everything to a text file in a convenient format.',
        epilog='P.S. GNSS rules!')
    parser.add_argument('--observation_file_path', type=str,
                        help='Observation file path', default=None)
    args = parser.parse_args()
    if args.observation_file_path is not None and Path(args.observation_file_path).exists():
        process(args.observation_file_path)
