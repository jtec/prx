import argparse
from pathlib import Path
import converters
import helpers
import georinex
import xarray
import parse_rinex
import constants
from collections import defaultdict

log = helpers.get_logger(__name__)


def prx_root() -> Path:
    return Path(__file__).parent.parent


def write_prx_file(prx_header: dict,
                   prx_content: dict,
                   file: Path):
    with open(file, 'w', encoding='utf-8') as file:
        file.write("empty prx file")


# From RINEX Version 3.05, 1 December, 2020.
def carrier_frequencies():
    cf = defaultdict(dict)
    # GPS
    cf["G"]["L1"] = 1575.42
    cf["G"]["L2"] = 1227.60
    cf["G"]["L5"] = 1176.45
    # GLONASS
    # FDMA signals
    for frequency_slot in range(-7, 12+1):
        cf["R"]["L1"][frequency_slot] = 1602 + frequency_slot*9/16
        cf["R"]["L2"][frequency_slot] = 1246 + frequency_slot*7/16
    # CDMA signals
    cf["R"]["L4"] = 1600.995
    cf["R"]["L3"] = 1202.025
    # Galileo
    cf["E"]["L1"] = 1575.42
    cf["E"]["L5"] = 1176.45
    cf["E"]["L7"] = 1207.140
    cf["E"]["L8"] = 1191.795
    cf["E"]["L6"] = 1278.75
    # SBAS
    cf["S"]["L1"] = 1575.42
    cf["S"]["L5"] = 1176.45
    # QZSS
    cf["J"]["L1"] = 1575.42
    cf["J"]["L2"] = 1227.60
    cf["J"]["L5"] = 1176.45
    cf["J"]["L6"] = 1278.75
    # Beidou
    cf["C"]["L1"] = 1575.42
    cf["C"]["L2"] = 1561.098
    cf["C"]["L5"] = 1176.45
    cf["C"]["L7"] = 1207.140
    cf["C"]["L6"] = 1268.52
    cf["C"]["L8"] = 1191.795
    # NavIC/IRNSS
    cf["I"]["L5"] = 1176.45
    cf["I"]["S"] = 2492.028
def build_header(rinex_header, input_files):
    prx_header = {}
    prx_header["input_files"] = [{"name": file.name, "md5": helpers.md5(file)} for file in input_files]
    prx_header["speed_of_light_mps"] = constants.cGpsIcdSpeedOfLight_mps
    prx_header["reference_frame"] = constants.cPrxReferenceFrame
    prx_header["carrier_frequencies_Hz"] = carrier_frequencies()

    return prx_header
def process(observation_file_path: Path):
    log.info(f"Starting processing {observation_file_path.name} (full path {observation_file_path})")
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    rinex_header = georinex.rinexheader(rinex_3_obs_file)
    rinex_obs = parse_rinex.load(rinex_3_obs_file, use_caching=True)
    prx_file = str(rinex_3_obs_file).replace('.rnx', '.json')
    write_prx_file(
        build_header(rinex_header, [rinex_3_obs_file]),
        rinex_obs,
        prx_file)


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
