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


def prx_root() -> Path:
    return Path(__file__).parent.parent


def write_prx_file(prx_header: dict,
                   prx_content: dict,
                   file_name_without_extension: Path,
                   output_format: str):
    output_writers = {
        "jsonseq": write_json_text_sequence_file
    }
    output_writers[output_format](prx_header, prx_content, file_name_without_extension)


def write_json_text_sequence_file(prx_header: dict,
                                  prx_content: dict,
                                  file_name_without_extension: Path):
    indent = 2
    output_file = Path(f"{str(file_name_without_extension)}.{constants.cPrxJsonTextSequenceFileExtension}")
    with open(output_file, 'w',
              encoding='utf-8') as file:
        file.write("\u241E" + json.dumps(prx_header, ensure_ascii=False, indent=indent) + "\n")
    log.info(f"Generated JSON Text Sequence prx file: {file}")


# From RINEX Version 3.05, 1 December, 2020.
def carrier_frequencies_hz():
    cf = defaultdict(dict)
    # GPS
    cf["G"]["L1"] = 1575.42 * constants.cHzPerMhz
    cf["G"]["L2"] = 1227.60 * constants.cHzPerMhz
    cf["G"]["L5"] = 1176.45 * constants.cHzPerMhz
    # GLONASS
    # FDMA signals
    cf["R"]["L1"] = defaultdict(dict)
    cf["R"]["L2"] = defaultdict(dict)
    for frequency_slot in range(-7, 12 + 1):
        cf["R"]["L1"][frequency_slot] = (1602 + frequency_slot * 9 / 16) * constants.cHzPerMhz
        cf["R"]["L2"][frequency_slot] = (1246 + frequency_slot * 7 / 16) * constants.cHzPerMhz
    # CDMA signals
    cf["R"]["L4"] = 1600.995 * constants.cHzPerMhz
    cf["R"]["L3"] = 1202.025 * constants.cHzPerMhz
    # Galileo
    cf["E"]["L1"] = 1575.42 * constants.cHzPerMhz
    cf["E"]["L5"] = 1176.45 * constants.cHzPerMhz
    cf["E"]["L7"] = 1207.140 * constants.cHzPerMhz
    cf["E"]["L8"] = 1191.795 * constants.cHzPerMhz
    cf["E"]["L6"] = 1278.75 * constants.cHzPerMhz
    # SBAS
    cf["S"]["L1"] = 1575.42 * constants.cHzPerMhz
    cf["S"]["L5"] = 1176.45 * constants.cHzPerMhz
    # QZSS
    cf["J"]["L1"] = 1575.42 * constants.cHzPerMhz
    cf["J"]["L2"] = 1227.60 * constants.cHzPerMhz
    cf["J"]["L5"] = 1176.45 * constants.cHzPerMhz
    cf["J"]["L6"] = 1278.75 * constants.cHzPerMhz
    # Beidou
    cf["C"]["L1"] = 1575.42 * constants.cHzPerMhz
    cf["C"]["L2"] = 1561.098 * constants.cHzPerMhz
    cf["C"]["L5"] = 1176.45 * constants.cHzPerMhz
    cf["C"]["L7"] = 1207.140 * constants.cHzPerMhz
    cf["C"]["L6"] = 1268.52 * constants.cHzPerMhz
    cf["C"]["L8"] = 1191.795 * constants.cHzPerMhz
    # NavIC/IRNSS
    cf["I"]["L5"] = 1176.45 * constants.cHzPerMhz
    cf["I"]["S"] = 2492.028 * constants.cHzPerMhz
    return cf


def build_header(rinex_header, input_files):
    prx_header = {}
    prx_header["input_files"] = [{"name": file.name, "md5": helpers.md5(file)} for file in input_files]
    prx_header["speed_of_light_mps"] = constants.cGpsIcdSpeedOfLight_mps
    prx_header["reference_frame"] = constants.cPrxReferenceFrame
    prx_header["carrier_frequencies_hz"] = carrier_frequencies_hz()
    prx_header["prx_git_commit_id"] = git.Repo(search_parent_directories=True).head.object.hexsha;
    return prx_header


def process(observation_file_path: Path, output_format):
    log.info(f"Starting processing {observation_file_path.name} (full path {observation_file_path})")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    rinex_header = georinex.rinexheader(rinex_3_obs_file)
    rinex_obs = parse_rinex.load(rinex_3_obs_file, use_caching=True)
    prx_file = str(rinex_3_obs_file).replace('.rnx', "")
    write_prx_file(
        build_header(rinex_header, [rinex_3_obs_file]),
        rinex_obs,
        prx_file,
        output_format)


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
        process(args.observation_file_path, "jsonseq")
