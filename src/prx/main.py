import argparse
import json
import logging
import sys
from pathlib import Path
import georinex
import pandas as pd
import numpy as np
import git
import prx.util
from prx.util import add_range_column

from prx import atmospheric_corrections as atmo, util
from prx.constants import carrier_frequencies_hz
from prx.util import parse_rinex_obs_file
from prx.util import is_rinex_3_obs_file, is_rinex_3_nav_file
from prx.rinex_nav import nav_file_discovery
from prx import constants, converters, user
from prx.rinex_nav import evaluate as rinex_evaluate
from prx.rinex_nav.evaluate import parse_rinex_nav_file

log = util.get_logger(__name__)


@prx.util.timeit
def write_prx_file(
    prx_header: dict,
    prx_records: pd.DataFrame,
    file_name_without_extension: Path,
    output_format: str,
):
    output_writers = {"jsonseq": write_json_text_sequence_file, "csv": write_csv_file}
    if output_format not in output_writers.keys():
        assert False, f"Output format {output_format} not supported,  we can do {list(output_writers.keys())}"
    return output_writers[output_format](
        prx_header, prx_records, file_name_without_extension
    )


def write_json_text_sequence_file(
    prx_header: dict, prx_records: pd.DataFrame, file_name_without_extension: Path
):
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxJsonTextSequenceFileExtension}"
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\u241e" + json.dumps(prx_header, ensure_ascii=False) + "\n")
        drop_columns = [
            "time_of_reception_in_receiver_time",
            "satellite",
        ]
        for epoch in prx_records["time_of_reception_in_receiver_time"].unique():
            epoch = pd.Timestamp(epoch)
            epoch_obs = prx_records[
                prx_records["time_of_reception_in_receiver_time"] == epoch
            ]
            record = {
                "time_of_reception_in_receiver_time": epoch.strftime(
                    "%Y:%m:%dT%H:%M:%S.%f"
                ),
                "satellites": {},
            }
            for idx, row in epoch_obs.iterrows():
                sat = row["satellite"]
                row = row.dropna().to_frame().transpose()
                record["satellites"][sat] = {"observations": {}}
                for col in row.columns:
                    if len(col) == 3:
                        record["satellites"][sat]["observations"][col] = row[
                            col
                        ].values[0]
                        continue
                    if col in drop_columns:
                        continue
                    if type(row[col].values[0]) is np.ndarray:
                        row[col].values[0] = row[col].values[0].tolist()
                    record["satellites"][sat][col] = row[col].values[0]
            file.write("\u241e" + json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Generated JSON Text Sequence prx file: {output_file}")


def write_csv_file(
    prx_header: dict, flat_records: pd.DataFrame, file_name_without_extension: Path
):
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxCsvFileExtension}"
    )
    flat_records["sat_elevation_deg"] = np.rad2deg(
        flat_records.elevation_rad.to_numpy()
    )
    flat_records["sat_azimuth_deg"] = np.rad2deg(flat_records.azimuth_rad.to_numpy())
    flat_records = flat_records.drop(columns=["elevation_rad", "azimuth_rad"])
    # Re-arrange records to have one line per code observation, with the associated carrier phase and
    # Doppler observation, and auxiliary information such as satellite position, velocity, clock offset, etc.
    # write records
    # Start with code observations, as they have TGDs, and merge in other observation types one by one
    flat_records["tracking_id"] = flat_records.observation_type.str[1:3]
    records = flat_records.loc[flat_records.observation_type.str.startswith("C")]
    records["C_obs_m"] = records.observation_value
    records = records.drop(columns=["observation_value", "observation_type"])
    type_2_unit = {"D": "hz", "L": "cycles", "S": "dBHz", "C": "m"}
    for obs_type in ["D", "L", "S"]:
        obs = flat_records.loc[
            (flat_records.observation_type.str.startswith(obs_type))
            & (flat_records.observation_type.str.len() == 3)
        ][
            [
                "satellite",
                "time_of_reception_in_receiver_time",
                "observation_value",
                "tracking_id",
            ]
        ]
        obs = obs.rename(
            columns={"observation_value": f"{obs_type}_obs_{type_2_unit[obs_type]}"}
        )
        if obs_type == "L":
            # add LLI as new column
            obs_lli = flat_records.loc[
                flat_records.observation_type.str.contains("lli"),
                [
                    "satellite",
                    "time_of_reception_in_receiver_time",
                    "observation_value",
                    "tracking_id",
                ],
            ]
            obs = obs.merge(
                obs_lli,
                on=["satellite", "time_of_reception_in_receiver_time", "tracking_id"],
                how="left",
            )
            obs = obs.rename(columns={"observation_value": "LLI"})
        records = records.merge(
            obs,
            on=["satellite", "time_of_reception_in_receiver_time", "tracking_id"],
            how="left",
        )
    records["constellation"] = records.satellite.str[0]
    records["prn"] = records.satellite.str[1:]
    records = records.rename(columns={"tracking_id": "rnx_obs_identifier"})
    records = records.drop(
        columns=[
            "satellite",
            "time_of_emission_isagpst",
        ]
    )
    records = records.sort_values(
        by=[
            "time_of_reception_in_receiver_time",
            "constellation",
            "prn",
            "rnx_obs_identifier",
        ]
    )
    # Keep only records with valid sat states
    records = records[records.sat_clock_offset_m.notna()]
    # write header
    prx_header["processing_time"] = str(
        pd.Timestamp.now() - prx_header["processing_start_time"]
    )
    prx_header["processing_start_time"] = prx_header["processing_start_time"].strftime(
        "%Y-%m-%d %H:%M:%S.%f3"
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"# {json.dumps(prx_header)}\n")
    records.to_csv(
        path_or_buf=output_file,
        index=False,
        mode="a",
        float_format="%.6f",
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )
    log.info(f"Generated CSV prx file: {file}")
    return output_file


def build_metadata(input_files):
    # convert input_files to a list of files
    files = []
    files.append(input_files["obs_file"])
    files.extend([file for file in input_files["nav_file"]])

    prx_metadata = {}
    obs_header = georinex.rinexheader(input_files["obs_file"])
    prx_metadata["approximate_receiver_ecef_position_m"] = (
        np.fromstring(obs_header["APPROX POSITION XYZ"], sep=" ")
    ).tolist()
    if prx_metadata["approximate_receiver_ecef_position_m"] == [0, 0, 0]:
        # compute position with first epoch
        logging.info(
            "Compute approximate position from first epoch with at least 4 GPS satellites"
        )
        prx_metadata["approximate_receiver_ecef_position_m"] = (
            user.bootstrap_coarse_receiver_position(
                input_files["obs_file"], input_files["nav_file"]
            ).tolist()
        )

    prx_metadata["input_files"] = [
        {
            "name": file.name,
            "murmur3_hash": util.hash_of_file_content(file),
        }
        for file in files
    ]
    prx_metadata["prx_git_commit_id"] = git.Repo(
        search_parent_directories=True
    ).head.object.hexsha
    return prx_metadata


def check_assumptions(
    rinex_3_obs_file,
):
    obs_header = georinex.rinexheader(rinex_3_obs_file)
    if "RCV CLOCK OFFS APPL" in obs_header.keys():
        assert (
            obs_header["RCV CLOCK OFFS APPL"].strip() == "0"
        ), "Handling of 'RCV CLOCK OFFS APPL' != 0 not implemented yet."
    assert (
        obs_header["TIME OF FIRST OBS"].split()[-1].strip() == "GPS"
    ), "Handling of observation files using time scales other than GPST not implemented yet."


def parse_rinex_nav_or_obs_file(rinex_file_path: Path):
    if is_rinex_3_obs_file(rinex_file_path):
        return parse_rinex_obs_file(rinex_file_path)
    elif is_rinex_3_nav_file(rinex_file_path):
        return parse_rinex_nav_file(rinex_file_path)
    assert (
        False
    ), f"File {rinex_file_path} appears to be neither RINEX 3 OBS nor NAV file."


def warm_up_parser_cache(rinex_files):
    _ = [parse_rinex_nav_or_obs_file(file) for file in rinex_files]


def compute_ephemeris_discontinuities(
    rinex_3_ephemerides_files: list[Path],
    sat_states: pd.DataFrame,
    approximate_receiver_ecef_position_m: np.ndarray,
):
    # Extract observations right after a change in ephemeris
    sat_states = sat_states.sort_values("query_time_isagpst")
    sat_states["after_discontinuity"] = False
    for _, group_df in sat_states.groupby(["sv", "signal"]):
        after_discontinuity = group_df.ephemeris_hash != group_df.ephemeris_hash.shift(
            1
        )
        # The very first of a signal's sat states can't be after a discontinuity
        after_discontinuity.iloc[0] = False
        before_discontinuity = group_df.ephemeris_hash != group_df.ephemeris_hash.shift(
            -1
        )
        sat_states.loc[group_df.index, "after_discontinuity"] = (
            after_discontinuity.values
        )
        # The very last of a signal's sat states can't be before a discontinuity
        before_discontinuity.iloc[-1] = False
        sat_states.loc[group_df.index, "before_discontinuity"] = (
            before_discontinuity.values
        )
    # Satellite states with ephemeris after the discontinuity
    df_new_ephemeris = (
        sat_states[sat_states.after_discontinuity]
        .copy()
        .drop(columns=["after_discontinuity"])
        .reset_index(drop=True)
    )
    # Now compute satellite states after discontinuity with ephemeris before the discontinuity
    query = sat_states.loc[
        sat_states.before_discontinuity, ["sv", "signal", "query_time_isagpst"]
    ].drop_duplicates()
    # Use the query time from before the discontinuity to select the previous ephemeris
    query["ephemeris_selection_time_isagpst"] = query.query_time_isagpst
    # But still compute satellite states for after the discontinuity
    query.query_time_isagpst = df_new_ephemeris.query_time_isagpst.values
    query = query.sort_values("ephemeris_selection_time_isagpst")
    df_previous_ephemeris = rinex_evaluate.compute(
        rinex_3_ephemerides_files, query
    ).reset_index(drop=True)
    shared_columns = ["sv", "signal", "query_time_isagpst"]
    df_new_ephemeris = df_new_ephemeris.sort_values(by=shared_columns).reset_index(
        drop=True
    )
    df_previous_ephemeris = df_previous_ephemeris.sort_values(
        by=shared_columns
    ).reset_index(drop=True)
    df_new_ephemeris = add_range_column(
        df_new_ephemeris, approximate_receiver_ecef_position_m
    )
    df_previous_ephemeris = add_range_column(
        df_previous_ephemeris, approximate_receiver_ecef_position_m
    )
    assert df_new_ephemeris[shared_columns].equals(
        df_previous_ephemeris[shared_columns]
    )
    numeric_columns = df_new_ephemeris.select_dtypes(include=np.number).columns.tolist()
    dsc = df_new_ephemeris
    dsc[numeric_columns] -= df_previous_ephemeris[numeric_columns]
    dsc["previous_ephemeris_hash"] = df_previous_ephemeris.ephemeris_hash
    return dsc


@prx.util.timeit
def build_records(
    rinex_3_obs_file,
    rinex_3_ephemerides_files,
    approximate_receiver_ecef_position_m,
):
    warm_up_parser_cache([rinex_3_obs_file] + rinex_3_ephemerides_files)
    approximate_receiver_ecef_position_m = np.array(
        approximate_receiver_ecef_position_m
    )
    check_assumptions(rinex_3_obs_file)
    flat_obs = util.parse_rinex_obs_file(rinex_3_obs_file)

    flat_obs.time = pd.to_datetime(flat_obs.time, format="%Y-%m-%dT%H:%M:%S")
    flat_obs.obs_value = flat_obs.obs_value.astype(float)
    flat_obs[["sv", "obs_type"]] = flat_obs[["sv", "obs_type"]].astype(str)

    flat_obs = flat_obs.rename(
        columns={
            "time": "time_of_reception_in_receiver_time",
            "sv": "satellite",
            "obs_value": "observation_value",
            "obs_type": "observation_type",
        },
    )

    log.info("Computing times of emission in satellite time")
    per_sat = flat_obs.pivot(
        index=["time_of_reception_in_receiver_time", "satellite"],
        columns=["observation_type"],
        values="observation_value",
    ).reset_index()
    per_sat["time_scale"] = (
        per_sat["satellite"].str[0].map(constants.constellation_2_system_time_scale)
    )
    per_sat["system_time_scale_epoch"] = per_sat["time_scale"].map(
        constants.system_time_scale_rinex_utc_epoch
    )
    # When calling georinex.load() with useindicators=True, there are additional ssi columns such as C1Cssi.
    # To exclude them, we check the length of the column name
    code_phase_columns = [c for c in per_sat.columns if c[0] == "C" and len(c) == 3]
    # TODO Find a more self-explanatory name for the following variable
    #
    #  The following term contains, for each satellite
    #  - time-of-flight: around 70 ms for MEO orbits. This includes small
    #  terms (up to tens of meters in units of distance, ten meters correspond to 34 nanoseconds)
    #  such as satellite code bias and atmospheric delays
    #  - satellite clock offset w.r.t. its constellation time
    #  - the receiver clock offset w.r.t. the satellite's constellation time (GPST, GST, BDT etc.) modulo 1 second
    # The integer seconds of the receiver clock offset w.r.t. the satellite's constellation time (GPST, GST, BDT etc.)
    # are likely removed by the receiver to align all pseudoranges to the same order of magnitude.
    # By subtracting the term from the receiver time of reception, we get the time of emission in
    # the satellite's constellation time frame (integer-second aligned to the receiver time frame), plus the
    # satellite clock offset plus those small terms of a few tens of nanoseconds
    tof_dtrx = pd.to_timedelta(
        per_sat[code_phase_columns]
        .mean(axis=1, skipna=True)
        .divide(constants.cGpsSpeedOfLight_mps),
        unit="s",
    )
    # As error terms are tens of nanoseconds here, and the receiver clock is integer-second aligned to GPST, we
    # already have times-of-emission that are integer-second aligned GPST here.
    per_sat["time_of_emission_isagpst"] = (
        per_sat["time_of_reception_in_receiver_time"] - tof_dtrx
    )

    flat_obs = flat_obs.merge(
        per_sat[
            [
                "time_of_reception_in_receiver_time",
                "satellite",
                "time_of_emission_isagpst",
            ]
        ],
        on=["time_of_reception_in_receiver_time", "satellite"],
    )

    # Compute broadcast position, velocity, clock offset, clock offset rate and TGDs
    query = flat_obs[flat_obs["observation_type"].str.startswith("C")]
    query = query.rename(
        columns={
            "observation_type": "signal",
            "satellite": "sv",
            "time_of_emission_isagpst": "query_time_isagpst",
        },
    )

    sat_states = rinex_evaluate.compute_parallel(
        rinex_3_ephemerides_files,
        query,
    ).reset_index(drop=True)

    discontinuities = compute_ephemeris_discontinuities(
        rinex_3_ephemerides_files, sat_states, approximate_receiver_ecef_position_m
    )
    discontinuities = discontinuities[
        [
            "sv",
            "signal",
            "query_time_isagpst",
            "ephemeris_hash",
            "previous_ephemeris_hash",
            "range_m",
            "sat_clock_offset_m",
        ]
    ]
    discontinuities.query_time_isagpst = discontinuities.query_time_isagpst.dt.strftime(
        "%Y:%m:%dT%H:%M:%S.%f"
    )
    discontinuities = discontinuities.to_dict("records")

    sat_states = sat_states.rename(
        columns={
            "sv": "satellite",
            "signal": "observation_type",
            "query_time_isagpst": "time_of_emission_isagpst",
        },
    )
    # We need Timestamps to compute tropo delays
    sat_states = sat_states.merge(
        flat_obs[
            [
                "satellite",
                "time_of_emission_isagpst",
                "time_of_reception_in_receiver_time",
            ]
        ].drop_duplicates(),
        on=["satellite", "time_of_emission_isagpst"],
        how="left",
    )
    # Compute anything else that is satellite-specific
    sat_states["relativistic_clock_effect_m"] = util.compute_relativistic_clock_effect(
        sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(),
        sat_states[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]].to_numpy(),
    )
    sat_states["sagnac_effect_m"] = util.compute_sagnac_effect(
        sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(),
        approximate_receiver_ecef_position_m,
    )
    [latitude_user_rad, longitude_user_rad, height_user_m] = util.ecef_2_geodetic(
        approximate_receiver_ecef_position_m
    )
    days_of_year = np.array(
        sat_states["time_of_reception_in_receiver_time"]
        .apply(lambda element: element.timetuple().tm_yday)
        .to_numpy()
    )
    (
        sat_states["elevation_rad"],
        sat_states["azimuth_rad"],
    ) = util.compute_satellite_elevation_and_azimuth(
        sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(),
        approximate_receiver_ecef_position_m,
    )
    (
        tropo_delay_m,
        __,
        __,
        __,
        __,
    ) = atmo.compute_tropo_delay_unb3m(
        latitude_user_rad * np.ones(days_of_year.shape),
        height_user_m * np.ones(days_of_year.shape),
        days_of_year,
        sat_states.elevation_rad.to_numpy(),
    )
    sat_states["tropo_delay_m"] = tropo_delay_m
    # Merge sat states into observation dataframe. Due to Galileo's FNAV/INAV ephemerides
    # being signal-specific, we merge on the code identifier here and not only the satellite
    sat_states["code_id"] = sat_states["observation_type"].str[1:3]
    flat_obs["code_id"] = flat_obs["observation_type"].str[1:3]
    sat_states = sat_states.drop(
        columns=["observation_type", "time_of_reception_in_receiver_time"]
    )
    flat_obs = flat_obs.merge(
        sat_states,
        on=["satellite", "code_id", "time_of_emission_isagpst"],
        how="left",
    )
    # Fix code biases being merged into lines with signals that are not code signals
    flat_obs.loc[
        ~(flat_obs.observation_type.str.startswith("C")), "sat_code_bias_m"
    ] = np.nan
    # GLONASS satellites with both FDMA and CDMA signals have a frequency slot for FDMA signals,
    # for CDMA signals we use the common carrier frequency of those signals.
    glo_cdma = flat_obs[
        (flat_obs.satellite.str[0] == "R")
        & (flat_obs["observation_type"].str[1].astype(int) > 2)
    ]
    flat_obs.loc[glo_cdma.index, "frequency_slot"] = int(1)

    def assign_carrier_frequencies(flat_obs):
        freq_dict = pd.json_normalize(carrier_frequencies_hz(), sep="_").to_dict(
            orient="records"
        )[0]
        assignable = flat_obs.frequency_slot.notna()
        keys = (
            flat_obs.satellite[assignable].str[0]
            + "_L"
            + flat_obs["observation_type"][assignable].str[1]
            + "_"
            + flat_obs.frequency_slot[assignable].astype(int).astype(str)
        )
        flat_obs.loc[:, "carrier_frequency_hz"] = keys.map(freq_dict)
        return flat_obs

    flat_obs = assign_carrier_frequencies(flat_obs)
    # create a dictionary containing the headers of the different NAV files.
    # The keys are the "YYYYDDD" (year and day of year) and are located at
    # [12:19] of the file name using RINEX naming convention
    nav_header_dict = {
        file.name[12:19]: georinex.rinexheader(file)
        for file in rinex_3_ephemerides_files
    }

    for file in rinex_3_ephemerides_files:
        # get year and doy from NAV filename
        year = int(file.name[12:16])
        doy = int(file.name[16:19])

        # Selection criteria: time of emission belonging to the day of the current NAV file
        mask = (
            flat_obs.time_of_emission_isagpst
            >= pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        ) & (
            flat_obs.time_of_emission_isagpst
            < pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy)
        )
        if "IONOSPHERIC CORR" in nav_header_dict[f"{year:03d}" + f"{doy:03d}"]:
            log.info(f"Computing iono delay for {year}-{doy:03d}")
            time_of_emission_weeksecond_isagpst = (
                flat_obs.loc[mask]
                .apply(
                    lambda row: util.timedelta_2_weeks_and_seconds(
                        row.time_of_emission_isagpst
                        - constants.system_time_scale_rinex_utc_epoch["GPST"]
                    )[1],
                    axis=1,
                )
                .to_numpy()
            )

            flat_obs.loc[
                mask,
                "iono_delay_m",
            ] = atmo.compute_l1_iono_delay_klobuchar(
                time_of_emission_weeksecond_isagpst,
                nav_header_dict[f"{year:03d}" + f"{doy:03d}"]["IONOSPHERIC CORR"][
                    "GPSA"
                ],
                nav_header_dict[f"{year:03d}" + f"{doy:03d}"]["IONOSPHERIC CORR"][
                    "GPSB"
                ],
                flat_obs.loc[mask].elevation_rad,
                flat_obs.loc[mask].azimuth_rad,
                latitude_user_rad,
                longitude_user_rad,
            ) * (
                constants.carrier_frequencies_hz()["G"]["L1"][1] ** 2
                / flat_obs.loc[mask].carrier_frequency_hz ** 2
            )
        else:
            logging.warning(f"Missing iono model parameters for day {doy:03d}")
            flat_obs.loc[
                mask,
                "iono_delay_m",
            ] = np.nan

    return flat_obs, discontinuities


@prx.util.timeit
def process(observation_file_path: Path, output_format="csv"):
    t0 = pd.Timestamp.now()
    # We expect a Path, but might get a string here:
    observation_file_path = Path(observation_file_path)
    log.info(
        f"Starting processing {observation_file_path.name} (full path {observation_file_path})"
    )
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    rinex_3_obs_file = prx.util.repair_with_gfzrnx(rinex_3_obs_file)
    prx_file = rinex_3_obs_file.with_suffix("")
    aux_files = nav_file_discovery.discover_or_download_auxiliary_files(
        rinex_3_obs_file
    )
    metadata = build_metadata(
        {"obs_file": rinex_3_obs_file, "nav_file": aux_files["broadcast_ephemerides"]}
    )
    metadata["processing_start_time"] = t0
    prx.util.repair_with_gfzrnx(rinex_3_obs_file)
    records, discontinuities = build_records(
        rinex_3_obs_file,
        aux_files["broadcast_ephemerides"],
        metadata["approximate_receiver_ecef_position_m"],
    )
    metadata["discontinuities"] = discontinuities
    return write_prx_file(
        metadata,
        records,
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
        default="csv",
    )
    args = parser.parse_args()
    if args.observation_file_path is None:
        log.error("No observation file path provided.")
        sys.exit(1)
    if not Path(args.observation_file_path).exists():
        log.error(f"Observation file {args.observation_file_path} does not exist.")
        sys.exit(1)
    process(Path(args.observation_file_path), args.output_format)
