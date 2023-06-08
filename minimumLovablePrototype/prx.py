import argparse
import json
from pathlib import Path
import georinex
import pandas as pd
import numpy as np
from collections import defaultdict
import git

import parse_rinex
import converters
import helpers
import constants
import aux_file_discovery as aux
import process_ephemerides as eph

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
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxJsonTextSequenceFileExtension}"
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(
            "\u241E" + json.dumps(prx_header, ensure_ascii=False) + "\n"
        )
        drop_columns = ["time_of_reception_in_receiver_time", "satellite", "time_of_emission_in_satellite_time"]
        for epoch in prx_records["time_of_reception_in_receiver_time"].unique():
            epoch = pd.Timestamp(epoch)
            epoch_obs = prx_records[prx_records["time_of_reception_in_receiver_time"] == epoch]
            record = {"time_of_reception_in_receiver_time": epoch.strftime("%Y:%m:%dT%H:%M:%S.%f"), "satellites": {}}
            for idx, row in epoch_obs.iterrows():
                sat = row["satellite"]
                row = epoch_obs.iloc[[idx]].dropna(axis='columns')
                record["satellites"][sat] = {"observations": {}}
                for col in row.columns:
                    if len(col) == 3:
                        record["satellites"][sat]["observations"][col] = row[col].values[0]
                        continue
                    if col in drop_columns:
                        continue
                    if type(row[col].values[0]) is np.ndarray:
                        row[col].values[0] = row[col].values[0].tolist()
                    record["satellites"][sat][col] = row[col].values[0]
            file.write(
                "\u241E" + json.dumps(record, ensure_ascii=False) + "\n"
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


def build_header(input_files):
    prx_header = {}
    prx_header["input_files"] = [
        {"name": file.name, "md5": helpers.md5_of_file_content(file)}
        for file in input_files
    ]
    prx_header["speed_of_light_mps"] = constants.cGpsIcdSpeedOfLight_mps
    prx_header["carrier_frequencies_hz"] = constants.carrier_frequencies_hz()
    prx_header["prx_git_commit_id"] = git.Repo(
        search_parent_directories=True
    ).head.object.hexsha
    return prx_header


def check_assumptions(rinex_3_obs_file, rinex_3_nav_file):
    obs_header = georinex.rinexheader(rinex_3_obs_file)
    nav_header = georinex.rinexheader(rinex_3_nav_file)
    if "RCV CLOCK OFFS APPL" in obs_header.keys():
        assert (
            obs_header["RCV CLOCK OFFS APPL"].strip() == "0"
        ), "Handling of 'RCV CLOCK OFFS APPL' != 0 not implemented yet."
    assert (
        obs_header["TIME OF FIRST OBS"].split()[-1].strip() == "GPS"
    ), "Handling of observation files using time scales other than GPST not implemented yet."


def build_records(rinex_3_obs_file, rinex_3_ephemerides_file):
    check_assumptions(rinex_3_obs_file, rinex_3_ephemerides_file)
    obs = parse_rinex.load(rinex_3_obs_file, use_caching=True)

    # Flatten the xarray DataSet into a pandas DataFrame:
    log.info("Converting Dataset into flat Dataframe of observations")
    flat_obs = pd.DataFrame()
    for obs_label, sat_time_obs_array in obs.data_vars.items():
        df = sat_time_obs_array.to_dataframe(name="obs_value").reset_index()
        df = df[df["obs_value"].notna()]
        df = df.assign(obs_type=lambda x: obs_label)
        flat_obs = pd.concat([flat_obs, df])

    def format_flat_rows(row):
        return [
            pd.Timestamp(row[0]),
            str(row[1]),
            row[2],
            str(row[3]),
        ]

    flat_obs = flat_obs.apply(format_flat_rows, axis=1, result_type="expand")
    flat_obs.rename(
        columns={
            0: "time_of_reception_in_receiver_time",
            1: "satellite",
            2: "observation_value",
            3: "observation_type",
        },
        inplace=True,
    )

    # Compute time-of-emission in satellite time (we don't have satellite clock offset from constellation time yet)
    log.info("Computing times of emission in satellite time")
    per_sat = flat_obs.pivot(
        index=["time_of_reception_in_receiver_time", "satellite"],
        columns=["observation_type"],
        values="observation_value",
    ).reset_index()
    code_phase_columns = [c for c in per_sat.columns if c[0] == "C"]
    per_sat["time_of_emission_in_satellite_time"] = per_sat[
        "time_of_reception_in_receiver_time"
    ] - pd.to_timedelta(
        per_sat[code_phase_columns]
        .mean(axis=1, skipna=True)
        .divide(constants.cGpsIcdSpeedOfLight_mps),
        unit="s",
    )

    def compute_sat_state(row, ephemerides):
        (
            position_system_frame_m,
            velocity_system_frame_mps,
            clock_offset_m,
            clock_offset_rate_mps,
        ) = eph.compute_satellite_state(
            ephemerides,
            row["satellite"],
            helpers.timestamp_2_timedelta(
                row["time_of_emission_in_satellite_time"],
                eph.satellite_id_2_system_time_scale(row["satellite"]),
            ),
        )
        return pd.Series(
            [
                position_system_frame_m,
                velocity_system_frame_mps,
                clock_offset_m,
                clock_offset_rate_mps,
            ]
        )

    log.info("Computing satellite states")
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(rinex_3_ephemerides_file)
    per_sat[
        [
            "satellite_position_m",
            "satellite_velocity_mps",
            "satellite_clock_bias_m",
            "satellite_clock_bias_drift_mps",
        ]
    ] = per_sat.apply(compute_sat_state, axis=1, args=(ephemerides,))

    return per_sat


def process(observation_file_path: Path, output_format="jsonseq"):
    # We expect a Path, but might get a string here:
    observation_file_path = Path(observation_file_path)
    log.info(
        f"Starting processing {observation_file_path.name} (full path {observation_file_path})"
    )
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    prx_file = str(rinex_3_obs_file).replace(".rnx", "")
    aux_files = aux.discover_or_download_auxiliary_files(rinex_3_obs_file)
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
