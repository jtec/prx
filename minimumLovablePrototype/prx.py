import argparse
from pathlib import Path
import logging
import converters
import helpers

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)
def prx_root() -> Path:
    return Path(__file__).parent.parent


def write_prx_file(prx_content: dict, file: Path):
    with open(file, 'w', encoding='utf-8') as file:
        file.write("empty prx file")


def process(observation_file_path: Path):
    logger.info(f"Starting processing {observation_file_path.name} (full path {observation_file_path})")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    prx_header = {
        "input_files": [
            {
                "name": rinex_3_obs_file.name,
                "md5": helpers.md5(rinex_3_obs_file)
            }
        ]
    }
    prx_content = {
        "header": prx_header,
        "records": []
    }
    prx_file = str(rinex_3_obs_file).replace('.rnx', '.json')
    write_prx_file(prx_content, prx_file)


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