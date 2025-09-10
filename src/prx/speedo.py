# Imports at the top of the file
import logging
import numpy as np
import pandas as pd
import polars as pl
import georinex
from line_profiler import profile
from prx.util import (
    add_range_column,
    timedelta_2_seconds,
    timestamp_2_timedelta,
    timeit,
)
from prx import atmospheric_corrections as atmo, util
from prx.constants import carrier_frequencies_hz
from prx import constants
from prx.rinex_nav import evaluate as rinex_evaluate


@profile
def build_records_polars(
    rinex_3_obs_file,
    rinex_3_ephemerides_files,
    approximate_receiver_ecef_position_m,
):
    """
    Fast implementation using Polars for high-performance data processing.
    Returns same output as build_records in main.py but with Polars speed.
    """

    log = util.get_logger(__name__)

    # Ensure proper array format
    approximate_receiver_ecef_position_m = np.array(
        approximate_receiver_ecef_position_m
    )

    # Parse observation file (returns pandas DataFrame)
    flat_obs_pd = util.parse_rinex_obs_file(rinex_3_obs_file)

    # Convert to Polars for fast processing, ensuring nanosecond datetime precision
    flat_obs = pl.from_pandas(flat_obs_pd)

    # Convert datetime columns to nanosecond precision if needed
    if flat_obs["time"].dtype != pl.Datetime("ns"):
        flat_obs = flat_obs.with_columns([pl.col("time").dt.cast_time_unit("ns")])

    # Data type optimization using Polars
    # Check if time is already datetime or needs conversion
    if flat_obs["time"].dtype == pl.Utf8:
        flat_obs = flat_obs.with_columns(
            [pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="ns")]
        )

    flat_obs = flat_obs.with_columns(
        [
            pl.col("obs_value").cast(pl.Float64),
            pl.col("sv").cast(pl.Utf8),
            pl.col("obs_type").cast(pl.Utf8),
        ]
    )

    # Column renaming using Polars
    flat_obs = flat_obs.rename(
        {
            "time": "time_of_reception_in_receiver_time",
            "sv": "satellite",
            "obs_value": "observation_value",
            "obs_type": "observation_type",
        }
    )

    log.info("Computing times of emission in satellite time")

    # Polars pivot operation
    per_sat = flat_obs.pivot(
        index=["time_of_reception_in_receiver_time", "satellite"],
        columns="observation_type",
        values="observation_value",
        aggregate_function="mean",
    )

    # Time scale mapping - convert to pandas temporarily for complex mappings
    per_sat_pd = per_sat.to_pandas()
    per_sat_pd["time_scale"] = (
        per_sat_pd["satellite"].str[0].map(constants.constellation_2_system_time_scale)
    )
    per_sat_pd["system_time_scale_epoch"] = per_sat_pd["time_scale"].map(
        constants.system_time_scale_rinex_utc_epoch
    )
    per_sat = pl.from_pandas(per_sat_pd)

    # Find code phase columns efficiently
    code_phase_columns = [
        c for c in per_sat.columns if c.startswith("C") and len(c) == 3
    ]

    # Calculate time of flight using Polars operations
    if code_phase_columns:
        # Calculate mean of code phase columns
        per_sat = per_sat.with_columns(
            [
                pl.concat_list([pl.col(c) for c in code_phase_columns])
                .list.mean()
                .truediv(constants.cGpsSpeedOfLight_mps)
                .mul(1_000_000)  # Convert to microseconds
                .alias("tof_dtrx_us")
            ]
        )

        # Calculate time of emission using Polars duration
        per_sat = per_sat.with_columns(
            [
                (
                    pl.col("time_of_reception_in_receiver_time")
                    - pl.duration(microseconds=pl.col("tof_dtrx_us"))
                ).alias("time_of_emission_isagpst")
            ]
        )
    else:
        per_sat = per_sat.with_columns(
            [
                pl.col("time_of_reception_in_receiver_time").alias(
                    "time_of_emission_isagpst"
                )
            ]
        )

    # Merge using Polars join
    flat_obs = flat_obs.join(
        per_sat.select(
            [
                "time_of_reception_in_receiver_time",
                "satellite",
                "time_of_emission_isagpst",
            ]
        ),
        on=["time_of_reception_in_receiver_time", "satellite"],
        how="left",
    )

    # Ensure nanosecond precision for time_of_emission_isagpst
    flat_obs = flat_obs.with_columns(
        [pl.col("time_of_emission_isagpst").dt.cast_time_unit("ns")]
    )

    # Prepare query for satellite states computation - convert to pandas for compatibility
    query_pl = flat_obs.filter(pl.col("observation_type").str.starts_with("C"))
    query = query_pl.to_pandas()  # Convert to pandas for rinex_evaluate compatibility

    # Ensure datetime precision is nanoseconds for compatibility with rinex_evaluate
    if "time_of_emission_isagpst" in query.columns:
        query["time_of_emission_isagpst"] = (
            pd.to_datetime(query["time_of_emission_isagpst"])
            .dt.as_unit("ns")
            .dt.tz_localize(None)
        )

    query = query.rename(
        columns={
            "observation_type": "signal",
            "satellite": "sv",
            "time_of_emission_isagpst": "query_time_isagpst",
        }
    )

    # Compute satellite states using rinex_evaluate
    log.info("Computing satellite states")
    sat_states = rinex_evaluate.compute_parallel(
        rinex_3_ephemerides_files, query
    ).reset_index(drop=True)

    # Optimized discontinuities computation
    def compute_ephemeris_discontinuities_optimized(sat_states_df):
        sat_states_df = sat_states_df.sort_values("query_time_isagpst")
        sat_states_df["after_discontinuity"] = False

        # Use vectorized operations where possible
        for _, group_df in sat_states_df.groupby(["sv", "signal"]):
            after_discontinuity = (
                group_df.ephemeris_hash != group_df.ephemeris_hash.shift(1)
            )
            after_discontinuity.iloc[0] = False
            sat_states_df.loc[group_df.index, "after_discontinuity"] = (
                after_discontinuity.values
            )

        df_new_ephemeris = (
            sat_states_df[sat_states_df.after_discontinuity]
            .copy()
            .drop(columns=["after_discontinuity"])
            .reset_index(drop=True)
        )

        if df_new_ephemeris.empty:
            return pd.DataFrame(
                columns=[
                    "sv",
                    "signal",
                    "query_time_isagpst",
                    "ephemeris_hash",
                    "previous_ephemeris_hash",
                    "range_m",
                    "sat_clock_offset_m",
                ]
            )

        query_disc = sat_states_df.loc[
            sat_states_df.after_discontinuity, ["sv", "signal", "query_time_isagpst"]
        ].drop_duplicates()

        query_disc["ephemeris_selection_time_isagpst"] = (
            query_disc.query_time_isagpst - pd.Timedelta("10s")
        )
        query_disc = query_disc.sort_values("ephemeris_selection_time_isagpst")

        df_previous_ephemeris = rinex_evaluate.compute(
            rinex_3_ephemerides_files, query_disc
        ).reset_index(drop=True)

        shared_columns = ["query_time_isagpst", "sv", "signal"]
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

        if not df_new_ephemeris[shared_columns].equals(
            df_previous_ephemeris[shared_columns]
        ):
            return pd.DataFrame(
                columns=[
                    "sv",
                    "signal",
                    "query_time_isagpst",
                    "ephemeris_hash",
                    "previous_ephemeris_hash",
                    "range_m",
                    "sat_clock_offset_m",
                ]
            )

        numeric_columns = df_new_ephemeris.select_dtypes(
            include=np.number
        ).columns.tolist()
        dsc = df_new_ephemeris.copy()
        dsc[numeric_columns] -= df_previous_ephemeris[numeric_columns]
        dsc["previous_ephemeris_hash"] = df_previous_ephemeris.ephemeris_hash

        return dsc

    discontinuities = compute_ephemeris_discontinuities_optimized(sat_states)

    # Process discontinuities
    if not discontinuities.empty:
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
        discontinuities.query_time_isagpst = discontinuities.query_time_isagpst.apply(
            lambda ts: timedelta_2_seconds(timestamp_2_timedelta(ts, "GPST"))
        )
        discontinuities = discontinuities.to_dict("records")
    else:
        discontinuities = []

    # Convert sat_states to Polars for faster processing
    sat_states_pl = pl.from_pandas(sat_states)

    # Rename satellite states columns using Polars
    sat_states_pl = sat_states_pl.rename(
        {
            "sv": "satellite",
            "signal": "observation_type",
            "query_time_isagpst": "time_of_emission_isagpst",
        }
    )

    # Keep nanosecond precision to match pandas datetime precision

    # Merge timestamps for tropospheric delay computation using Polars
    sat_states_pl = sat_states_pl.join(
        flat_obs.select(
            [
                "satellite",
                "time_of_emission_isagpst",
                "time_of_reception_in_receiver_time",
            ]
        ).unique(),
        on=["satellite", "time_of_emission_isagpst"],
        how="left",
    )

    # Convert back to pandas for numpy operations that require it
    sat_states = sat_states_pl.to_pandas()

    # Compute satellite-specific effects efficiently
    pos_array = sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy()
    vel_array = sat_states[
        ["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]
    ].to_numpy()

    sat_states["relativistic_clock_effect_m"] = util.compute_relativistic_clock_effect(
        pos_array, vel_array
    )
    sat_states["sagnac_effect_m"] = util.compute_sagnac_effect(
        pos_array, approximate_receiver_ecef_position_m
    )

    # Geodetic conversion
    [latitude_user_rad, longitude_user_rad, height_user_m] = util.ecef_2_geodetic(
        approximate_receiver_ecef_position_m
    )

    # Vectorized day of year computation
    days_of_year = sat_states[
        "time_of_reception_in_receiver_time"
    ].dt.dayofyear.to_numpy()

    # Satellite elevation and azimuth
    (
        sat_states["elevation_rad"],
        sat_states["azimuth_rad"],
    ) = util.compute_satellite_elevation_and_azimuth(
        pos_array, approximate_receiver_ecef_position_m
    )

    # Tropospheric delay
    (tropo_delay_m, _, _, _, _) = atmo.compute_tropo_delay_unb3m(
        latitude_user_rad * np.ones(days_of_year.shape),
        height_user_m * np.ones(days_of_year.shape),
        days_of_year,
        sat_states.elevation_rad.to_numpy(),
    )
    sat_states["tropo_delay_m"] = tropo_delay_m

    # Convert both back to Polars for efficient joining
    sat_states_pl = pl.from_pandas(sat_states)

    # Add code_id columns using Polars string operations
    sat_states_pl = sat_states_pl.with_columns(
        [pl.col("observation_type").str.slice(1, 2).alias("code_id")]
    ).drop(["observation_type", "time_of_reception_in_receiver_time"])

    flat_obs = flat_obs.with_columns(
        [pl.col("observation_type").str.slice(1, 2).alias("code_id")]
    )

    # Merge using Polars join
    flat_obs = flat_obs.join(
        sat_states_pl,
        on=["satellite", "code_id", "time_of_emission_isagpst"],
        how="left",
    )

    # Fix code biases for non-code observations using Polars
    flat_obs = flat_obs.with_columns(
        [
            pl.when(~pl.col("observation_type").str.starts_with("C"))
            .then(None)
            .otherwise(pl.col("sat_code_bias_m"))
            .alias("sat_code_bias_m")
        ]
    )

    # GLONASS frequency slot handling using Polars
    flat_obs = flat_obs.with_columns(
        [
            pl.when(
                (pl.col("satellite").str.slice(0, 1) == "R")
                & (pl.col("observation_type").str.slice(1, 1).cast(pl.Int32) > 2)
            )
            .then(1)
            .otherwise(pl.col("frequency_slot"))
            .alias("frequency_slot")
        ]
    )

    # Optimized carrier frequency assignment using Polars
    # Convert to pandas temporarily for the complex frequency mapping
    flat_obs_pd = flat_obs.to_pandas()
    freq_dict = pd.json_normalize(carrier_frequencies_hz(), sep="_").to_dict(
        orient="records"
    )[0]
    assignable = flat_obs_pd.frequency_slot.notna()

    if assignable.sum() > 0:
        keys = (
            flat_obs_pd.satellite[assignable].str[0]
            + "_L"
            + flat_obs_pd["observation_type"][assignable].str[1]
            + "_"
            + flat_obs_pd.frequency_slot[assignable].astype(int).astype(str)
        )
        flat_obs_pd.loc[assignable, "carrier_frequency_hz"] = keys.map(freq_dict)

    # Convert back to Polars
    flat_obs = pl.from_pandas(flat_obs_pd)

    # Navigation file headers
    nav_header_dict = {
        file.name[12:19]: georinex.rinexheader(file)
        for file in rinex_3_ephemerides_files
    }

    # Convert to pandas for ionospheric corrections (complex operations)
    flat_obs_final = flat_obs.to_pandas()

    # Ionospheric corrections
    for file in rinex_3_ephemerides_files:
        year = int(file.name[12:16])
        doy = int(file.name[16:19])

        mask = (
            flat_obs_final.time_of_emission_isagpst
            >= pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        ) & (
            flat_obs_final.time_of_emission_isagpst
            < pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy)
        )

        if "IONOSPHERIC CORR" in nav_header_dict[f"{year:03d}{doy:03d}"]:
            log.info(f"Computing iono delay for {year}-{doy:03d}")
            time_of_emission_weeksecond_isagpst = util.timedelta_2_weeks_and_seconds(
                flat_obs_final.loc[mask].time_of_emission_isagpst
                - constants.system_time_scale_rinex_utc_epoch["GPST"]
            )[1].to_numpy()

            flat_obs_final.loc[mask, "iono_delay_m"] = (
                atmo.compute_l1_iono_delay_klobuchar(
                    time_of_emission_weeksecond_isagpst,
                    nav_header_dict[f"{year:03d}{doy:03d}"]["IONOSPHERIC CORR"]["GPSA"],
                    nav_header_dict[f"{year:03d}{doy:03d}"]["IONOSPHERIC CORR"]["GPSB"],
                    flat_obs_final.loc[mask].elevation_rad,
                    flat_obs_final.loc[mask].azimuth_rad,
                    latitude_user_rad,
                    longitude_user_rad,
                )
                * (
                    constants.carrier_frequencies_hz()["G"]["L1"][1] ** 2
                    / flat_obs_final.loc[mask].carrier_frequency_hz ** 2
                )
            )
        else:
            logging.warning(f"Missing iono model parameters for day {doy:03d}")
            flat_obs_final.loc[mask, "iono_delay_m"] = np.nan

    return flat_obs_final, discontinuities


@timeit
def select_ephemerides_polars(df, query):
    # 1. Filter out rows with missing reference times
    df = df.filter(pl.col("ephemeris_reference_time_isagpst").is_not_null())

    # 2. Sort both DataFrames
    query = query.sort("ephemeris_selection_time_isagpst")
    df = df.sort("ephemeris_reference_time_isagpst")

    # 3. Add FNAV/INAV indicator (minimal branching)
    query = query.with_columns(
        pl.when(
            (pl.col("constellation") == "E")
            & (pl.col("signal").cast(pl.Utf8).str.slice(1, 1) == "5")
        )
        .then(pl.lit("fnav"))
        .when(pl.col("constellation") == "E")
        .then(pl.lit("inav"))
        .otherwise(pl.lit(""))
        .alias("fnav_or_inav")
    )

    # 4. Create join_key (needed since Polars join_asof only supports one `by`)
    query = query.with_columns(
        (
            pl.col("constellation")
            + "_"
            + pl.col("sv").cast(pl.Utf8)
            + "_"
            + pl.col("fnav_or_inav")
        ).alias("join_key")
    )
    df = df.with_columns(
        (
            pl.col("constellation")
            + "_"
            + pl.col("sv").cast(pl.Utf8)
            + "_"
            + pl.lit("")
        ).alias("join_key")
    )

    # 5. Merge with asof join (fastest available in Polars)
    merged = query.join_asof(
        df,
        left_on="ephemeris_selection_time_isagpst",
        right_on="ephemeris_reference_time_isagpst",
        by="join_key",
        strategy="backward",
    )

    # 6. Compute time deltas directly
    merged = merged.with_columns(
        [
            (pl.col("query_time_isagpst") - pl.col("ephemeris_reference_time_isagpst"))
            .dt.total_seconds()
            .alias("query_time_wrt_ephemeris_reference_time_s"),
            (pl.col("query_time_isagpst") - pl.col("clock_reference_time_isagpst"))
            .dt.total_seconds()
            .alias("query_time_wrt_clock_reference_time_s"),
        ]
    )

    # 7. Ephemeris validity mask
    merged = merged.with_columns(
        (
            (pl.col("query_time_isagpst") < pl.col("validity_end"))
            & (pl.col("query_time_isagpst") > pl.col("validity_start"))
        ).alias("ephemeris_valid")
    )

    return merged
