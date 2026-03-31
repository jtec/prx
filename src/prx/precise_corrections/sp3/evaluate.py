import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import georinex
from pathlib import Path
import matplotlib.pyplot as plt

from prx import constants, util
from prx.precise_corrections.antex import antex_processing as atx_processing

log = util.get_logger(__name__)


def parse_sp3_file(file_path: Path):
    @util.disk_cache.cache(ignore=["file_path"])
    def cached_load(file_path: Path, file_hash: str):
        log.info(f"Parsing {file_path} (hash {file_hash}) ...")
        parsed = georinex.load(file_path)
        assert parsed is not None
        df = parsed.to_dataframe().reset_index()
        df.set_index([df["ECEF"], df.groupby("ECEF").cumcount()], inplace=True)
        df.drop(columns=["ECEF"], inplace=True)
        df = df.unstack(0)
        df.columns = [f"{x}_{y}" for x, y in df.columns]
        cols_to_drop = [
            col
            for col in df.columns
            if "x" not in col and "position" not in col and "velocity" not in col
        ]
        df.drop(cols_to_drop, axis=1, inplace=True)
        for col in df.columns:
            if "position" not in col and "velocity" not in col:
                df.rename(columns={col: col.replace("_x", "")}, inplace=True)
        # Convert timestamps to seconds since GPST epoch
        df["gpst_s"] = (df["time"] - constants.cGpstUtcEpoch).apply(
            util.timedelta_2_seconds
        )
        df.drop(columns=["time"], inplace=True)
        df["sat_clock_offset_m"] = (
            constants.cGpsSpeedOfLight_mps * df["clock"]
        ) / constants.cMicrosecondsPerSecond
        df["sat_clock_drift_mps"] = (
            constants.cGpsSpeedOfLight_mps * df["dclock"]
        ) / constants.cMicrosecondsPerSecond
        for axis in ["x", "y", "z"]:
            df["position_" + axis] = (
                df["position_" + axis] * constants.cMetersPerKilometer
            )
        # Give some columns more pithy names
        df.rename(
            columns={
                "position_x": "sat_pos_com_x_m",
                "position_y": "sat_pos_com_y_m",
                "position_z": "sat_pos_com_z_m",
            },
            inplace=True,
        )
        df.rename(
            columns={
                "velocity_x": "sat_vel_x_mps",
                "velocity_y": "sat_vel_y_mps",
                "velocity_z": "sat_vel_z_mps",
            },
            inplace=True,
        )
        df.drop(columns={"clock", "dclock"}, inplace=True)
        # Put timestamps first
        df.insert(0, "gpst_s", df.pop("gpst_s"))
        return df

    file_content_hash = util.hash_of_file_content(file_path)
    return cached_load(file_path, file_content_hash)


# def plot_lagrange_interpolation(
#     polynomial, times, samples, interpolation_time, interpolated_value, label
# ):
#     polynomial_times = np.linspace(min(times), max(times), 10 * times.size)
#     polynomial_samples = Polynomial(polynomial.coef[::-1])(polynomial_times)
#     plt.plot(times, samples, "o", label="samples")
#     plt.plot(polynomial_times, polynomial_samples, ".", label="samples")
#     plt.plot(
#         interpolation_time,
#         Polynomial(polynomial.coef[::-1])(interpolation_time),
#         "x",
#         label=label,
#     )
#     plt.legend()
#     plt.grid()
#     plt.show()


from scipy.interpolate import KroghInterpolator


def interpolate_krogh(t_sp3: np.array, x_sp3: np.array, t_query: np.array):
    """
    9-point Hermite/Krogh polynomial interpolation.
    Similar to Lagrange interpolation, but allows to easily compute derivative.

    t_sp3   : array of SP3 epoch times
    x_sp3   : array of SP3 states
    t_query : array of times at 1-second step
    """

    x_out = np.zeros_like(t_query, dtype=float)
    dx_out = np.zeros_like(t_query, dtype=float)

    for k, tq in enumerate(t_query):
        # Find nearest SP3 index
        idx = np.searchsorted(t_sp3, tq)

        # Choose 9 points centered around idx
        i0 = max(0, idx - 4)
        i1 = min(len(t_sp3), i0 + 9)
        i0 = i1 - 9  # adjust start properly

        t9 = t_sp3[i0:i1]
        x9 = x_sp3[i0:i1]

        # Build Krogh polynomials
        k_x = KroghInterpolator(t9, x9)

        # Evaluate position
        x_out[k] = k_x(tq)

        # Evaluate velocity (1st derivative)
        dx_out[k] = k_x.derivative(tq)

    return x_out, dx_out


# def interpolate(df, query_time_gpst_s, plot_interpolation=False):
#     n_samples_each_side = 4
#     assert df["sv"].unique().size == 1, "This function expects one satellite at a time"
#     closest_sample_index = np.argmin(np.abs(df["gpst_s"] - query_time_gpst_s))
#     start_index = closest_sample_index - n_samples_each_side
#     end_index = closest_sample_index + n_samples_each_side
#     assert start_index >= 0, (
#         f"We need at least {n_samples_each_side} before the sample closest to the query time to interpolate"
#     )
#     assert end_index < len(df.index), (
#         f"We need at least {n_samples_each_side} after the sample closest to the query time to interpolate"
#     )
#     columns_to_interpolate = [
#         "sat_pos_com_x_m",
#         "sat_pos_com_y_m",
#         "sat_pos_com_z_m",
#         "sat_clock_offset_m",
#     ]
#     interpolated = df[closest_sample_index : closest_sample_index + 1]
#     interpolated["gpst_s"] = query_time_gpst_s
#     for col in columns_to_interpolate:
#         interpolated[col] = float("nan")
#     for col in columns_to_interpolate:
#         times = df["gpst_s"].iloc[start_index : end_index + 1].to_numpy()
#         samples = df[col].iloc[start_index : end_index + 1].to_numpy()
#         # Improve numerical conditioning by subtracting the first sample
#         poly = lagrange(times - times[0], samples - samples[0])
#         interpolated[col] = (
#             Polynomial(poly.coef[::-1])(query_time_gpst_s - times[0]) + samples[0]
#         )
#         if plot_interpolation:
#             plot_lagrange_interpolation(
#                 poly,
#                 times - times[0],
#                 samples - samples[0],
#                 query_time_gpst_s - times[0],
#                 interpolated[col] - samples[0],
#                 f"{col} {df['sv'].unique()}",
#             )
#         first_derivative = Polynomial(poly.coef[::-1]).deriv(1)(
#             query_time_gpst_s - times[0]
#         )
#         match col:
#             case "sat_pos_com_x_m":
#                 interpolated["sat_vel_x_mps"] = first_derivative
#             case "sat_pos_com_y_m":
#                 interpolated["sat_vel_y_mps"] = first_derivative
#             case "sat_pos_com_z_m":
#                 interpolated["sat_vel_z_mps"] = first_derivative
#             case "sat_clock_offset_m":
#                 interpolated["sat_clock_drift_mps"] = first_derivative
#             case _:
#                 log.warning(f"{col} not recognized")
#     return interpolated


def compute(sp3_file_path, query, atx_file_path):
    """
    Inputs:
    - sp3_file_path: path to the SP3 file
    - query: a pd.DataFrame with columns ["sv", "signal", "query_time_isagpst"]
    - atx_file_path: path to the atx file
    """
    df = parse_sp3_file(sp3_file_path)
    df = df[df["sv"].isin(query["sv"])]

    # def interpolate_sat_states(row):
    #     samples = df[df["sv"] == row["sv"]]
    #     if len(samples.index) > 0:
    #         sat_pv = interpolate(
    #             samples,
    #             util.timedelta_2_seconds(
    #                 row["query_time_isagpst"] - constants.cGpstUtcEpoch
    #             ),
    #         )
    #         sat_pv["health_flag"] = [0]
    #     else:
    #         sat_pv = pd.DataFrame()
    #         sat_pv["gpst_s"] = [row.query_time_isagpst]
    #         sat_pv["sv"] = [row.sv]
    #         sat_pv["sat_pos_com_x_m"] = [np.nan]
    #         sat_pv["sat_pos_com_y_m"] = [np.nan]
    #         sat_pv["sat_pos_com_z_m"] = [np.nan]
    #         sat_pv["sat_clock_offset_m"] = [np.nan]
    #         sat_pv["relativistic_clock_correction_m"] = [0]
    #         sat_pv["sat_vel_x_mps"] = [np.nan]
    #         sat_pv["sat_vel_y_mps"] = [np.nan]
    #         sat_pv["sat_vel_z_mps"] = [np.nan]
    #         sat_pv["sat_clock_drift_mps"] = [np.nan]
    #         sat_pv["health_flag"] = [1]
    #     return pd.concat((row.drop("sv"), sat_pv.squeeze()))

    for sv, group in query.groupby("sv"):
        # Select SP3 rows for this satellite
        sp3 = df.loc[df["sv"] == sv]

        # Convert query times to GPST seconds
        t_query = np.array(
            [
                util.timedelta_2_seconds(t - constants.cGpstUtcEpoch)
                for t in group.query_time_isagpst
            ]
        )

        # correct query time by satellite clock offset
        t0 = t_query
        for _ in range(2):
            C, _ = interpolate_krogh(
                sp3.gpst_s.to_numpy(), sp3.sat_clock_offset_m.to_numpy(), t0
            )
            t0 = t_query - C / constants.cGpsSpeedOfLight_mps
            print(C[0])
        t_query = t_query - C / constants.cGpsSpeedOfLight_mps
        X, dX = interpolate_krogh(
            sp3.gpst_s.to_numpy(), sp3.sat_pos_com_x_m.to_numpy(), t_query
        )
        Y, dY = interpolate_krogh(
            sp3.gpst_s.to_numpy(), sp3.sat_pos_com_y_m.to_numpy(), t_query
        )
        Z, dZ = interpolate_krogh(
            sp3.gpst_s.to_numpy(), sp3.sat_pos_com_z_m.to_numpy(), t_query
        )
        C, dC = interpolate_krogh(
            sp3.gpst_s.to_numpy(), sp3.sat_clock_offset_m.to_numpy(), t_query
        )

        # Assign results back to the `query` dataframe
        query.loc[group.index, "sat_pos_com_x_m"] = X
        query.loc[group.index, "sat_pos_com_y_m"] = Y
        query.loc[group.index, "sat_pos_com_z_m"] = Z
        query.loc[group.index, "sat_clock_offset_m"] = C
        query.loc[group.index, "sat_vel_x_mps"] = dX
        query.loc[group.index, "sat_vel_y_mps"] = dY
        query.loc[group.index, "sat_vel_z_mps"] = dZ
        query.loc[group.index, "sat_clock_drift_mps"] = dC
        query.loc[group.index, "health_flag"] = 0

    # # interpolate sat states at query time
    # query = (
    #     query.apply(interpolate_sat_states, axis=1)
    #     .reset_index(drop=True)
    #     .drop(columns=["gpst_s", "t0"])
    # )

    # add phase center offset
    pco = atx_processing.compute_pco_sat(
        query=query,
        atx_df=atx_processing.parse_atx(atx_file_path),
    )

    # merge pco into query, special care due to using 'freq_id'
    query = (
        query.assign(freq_id=query.signal.str[1].astype(int))
        .merge(
            pco,
            left_on=["query_time_isagpst", "sv", "freq_id"],
            right_on=["query_time_isagpst", "sv", "freq_id"],
            how="left",
        )
        .drop(columns=["freq_id"])
    )

    # compute satellite antenna position
    query = query.assign(
        sat_pos_x_m=query["sat_pos_com_x_m"] + query["pco_sat_x_m"],
        sat_pos_y_m=query["sat_pos_com_y_m"] + query["pco_sat_y_m"],
        sat_pos_z_m=query["sat_pos_com_z_m"] + query["pco_sat_z_m"],
        # sp3 clocks are already corrected by relativistic effect
        relativistic_clock_effect_m=0,
    )

    return query


if __name__ == "main":
    pass
