import argparse
import json
from pathlib import Path
import georinex
import pandas as pd
import numpy as np

import parse_rinex
from collections import defaultdict
import git

import converters
import helpers
import constants
import aux_file_discovery as aux

log = helpers.get_logger(__name__)


def write_prx_file(
        prx_header: dict,
        prx_records: pd.DataFrame,
        file_name_without_extension: Path,
        output_format: str,
):
    output_writers = {"jsonseq": write_json_text_sequence_file, "csv": write_csv_file}
    if output_format not in output_writers.keys():
        assert (
            False
        ), f"Output format {output_format} not supported,  we can do {list(output_writers.keys())}"
    output_writers[output_format](prx_header, prx_records, file_name_without_extension)


def write_json_text_sequence_file(
        prx_header: dict, prx_records: pd.DataFrame, file_name_without_extension: Path
):
    indent = 2
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxJsonTextSequenceFileExtension}"
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            "\u241E" + json.dumps(prx_header, ensure_ascii=False, indent=indent) + "\n"
        )
    log.info(f"Generated JSON Text Sequence prx file: {output_file}")


def write_csv_file(
        prx_header: dict, prx_records: pd.DataFrame, file_name_without_extension: Path
):
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxCsvFileExtension}"
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"Empty so far." + "\n")
    log.info(f"Generated CSV prx file: {file}")


# From RINEX Version 3.05, 1 December, 2020.
def carrier_frequencies_hz():
    cf = defaultdict(dict)
    # GPS
    cf["G"]["L1"] = 1575.42 * constants.cHzPerMhz
    cf["G"]["L2"] = 1227.60 * constants.cHzPerMhz
    cf["G"]["L5"] = 1176.45 * constants.cHzPerMhz
    # GLONASS FDMA signals
    cf["R"]["L1"] = defaultdict(dict)
    cf["R"]["L2"] = defaultdict(dict)
    for frequency_slot in range(-7, 12 + 1):
        cf["R"]["L1"][frequency_slot] = (
                                                1602 + frequency_slot * 9 / 16
                                        ) * constants.cHzPerMhz
        cf["R"]["L2"][frequency_slot] = (
                                                1246 + frequency_slot * 7 / 16
                                        ) * constants.cHzPerMhz
    # Glonass CDMA signals
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


def build_header(input_files):
    prx_header = {}
    prx_header["input_files"] = [
        {"name": file.name, "md5": helpers.md5_of_file_content(file)}
        for file in input_files
    ]
    prx_header["speed_of_light_mps"] = constants.cGpsIcdSpeedOfLight_mps
    prx_header["reference_frame"] = constants.cPrxReferenceFrame
    prx_header["carrier_frequencies_hz"] = carrier_frequencies_hz()
    prx_header["prx_git_commit_id"] = git.Repo(
        search_parent_directories=True
    ).head.object.hexsha
    return prx_header


def check_assumptions(rinex_3_obs_file):
    obs_header = georinex.rinexheader(rinex_3_obs_file)
    if "RCV CLOCK OFFS APPL" in obs_header.keys():
        assert (
                obs_header["RCV CLOCK OFFS APPL"].strip() == "0"
        ), "Handling of 'RCV CLOCK OFFS APPL' != 0 not implemented yet."
    assert (
            obs_header["TIME OF FIRST OBS"].split()[-1].strip() == "GPS"
    ), "Handling of observation files using time scales other than GPST no implemented yet"


def build_records(rinex_3_obs_file, rinex_3_ephemerides_file):
    check_assumptions(rinex_3_obs_file)
    obs = parse_rinex.load(rinex_3_obs_file, use_caching=True)

    # Flatten the xarray DataSet into a pandas DataFrame:
    flat_obs = pd.DataFrame()
    for obs_label, sat_time_obs_array in obs.data_vars.items():
        df = sat_time_obs_array.to_dataframe(name="obs_value").reset_index()
        df = df[df["obs_value"].notna()]
        df = df.assign(obs_type=lambda x: obs_label)
        flat_obs = pd.concat([flat_obs, df])

    def format_flat_rows(row):
        return [
            helpers.timestamp_2_gpst_ns(pd.Timestamp(row[0])),
            str(row[1]),
            row[2],
            str(row[3]),
        ]

    flat_obs = flat_obs.apply(format_flat_rows, axis=1, result_type="expand")
    flat_obs.rename(
        columns={0: "time_of_reception_in_receiver_time_ns", 1: "satellite", 2: "observation_value",
                 3: "observation_type"},
        inplace=True,
    )

    nav = parse_rinex.load(rinex_3_ephemerides_file, use_caching=True)

    # Compute time-of-emission in satellite time (we don't have satellite clock offset from constellation time yet)
    def compute_time_of_emission_in_satellite_time(row):
        if row["observation_type"].startswith("C"):
            return row["time_of_reception_in_receiver_time_ns"] - constants.cNanoSecondsPerSecond * (
                        row["observation_value"] / constants.cGpsIcdSpeedOfLight_mps)
        return np.float64("NaN")

    flat_obs["time_of_emission_in_satellite_time_gpst_ns"] = flat_obs.apply(compute_time_of_emission_in_satellite_time, axis=1)

    # Compute broadcast satellite clock - constellation time offsets
    def compute_satellite_clock_offsets(row, nav):
        if row["observation_type"].startswith("C"):
            return row["time_of_reception_in_receiver_time_ns"] - constants.cNanoSecondsPerSecond * (
                        row["observation_value"] / constants.cGpsIcdSpeedOfLight_mps)
        return np.float64("NaN")
    flat_obs["satellite_clock_offset_ns"] = flat_obs.apply(compute_satellite_clock_offsets, axis=1, args=(nav,))


def process(observation_file_path: Path, output_format="jsonseq"):
    log.info(
        f"Starting processing {observation_file_path.name} (full path {observation_file_path})"
    )
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    prx_file = str(rinex_3_obs_file).replace(".rnx", "")
    aux_files = aux.discover_or_download_auxiliary_files(observation_file_path)
    write_prx_file(
        build_header([rinex_3_obs_file, aux_files["broadcast_ephemerides"]]),
        build_records(rinex_3_obs_file, aux_files["broadcast_ephemerides"]),
        prx_file,
        output_format,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="prx",
        description="prx processes RINEX observations, computes a few useful things such as satellite position, "
                    "relativistic effects etc. and outputs everything to a text file in a convenient format.",
        epilog="P.S. GNSS rules!",
    )
    parser.add_argument(
        "--observation_file_path", type=str, help="Observation file path", default=None
    )
    parser.add_argument(
        "--output_format",
        type=str,
        help="Output file format",
        choices=["jsonseq", "csv"],
        default="jsonseq",
    )
    args = parser.parse_args()
    if (
            args.observation_file_path is not None
            and Path(args.observation_file_path).exists()
    ):
        process(Path(args.observation_file_path), args.output_format)
