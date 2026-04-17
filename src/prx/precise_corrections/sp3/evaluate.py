import numpy as np
import georinex
from pathlib import Path

from scipy.interpolate import KroghInterpolator
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


def interpolate_krogh(t_sp3: np.array, x_sp3: np.array, t_query: np.array, order=9):
    """
    Similar to Lagrange interpolation, but allows to easily compute derivative.

    t_sp3   : array of SP3 epoch times
    x_sp3   : array of SP3 states
    t_query : array of times at 1-second step
    order   : nb of SP3 points used for interpolation
    """

    x_out = np.zeros_like(t_query, dtype=float)
    dx_out = np.zeros_like(t_query, dtype=float)

    for k, tq in enumerate(t_query):
        # Find nearest SP3 index
        idx = np.searchsorted(t_sp3, tq)

        # Choose 9 points centered around idx
        i0 = max(0, idx - int((order - 1) / 2))
        i1 = min(len(t_sp3), i0 + order)
        i0 = i1 - order  # adjust start properly

        t9 = t_sp3[i0:i1]
        x9 = x_sp3[i0:i1]

        # Build Krogh polynomials
        k_x = KroghInterpolator(t9, x9)

        # Evaluate position
        x_out[k] = k_x(tq)

        # Evaluate velocity (1st derivative)
        dx_out[k] = k_x.derivative(tq)

    return x_out, dx_out


def compute(
    sp3_file_path, query, atx_file_path, is_query_corrected_by_sat_clock_offset=False
):
    """
    Inputs:
    - sp3_file_path: path to the SP3 file
    - query: a pd.DataFrame with columns ["sv", "signal", "query_time_isagpst"]
    - atx_file_path: path to the atx file
    """
    df_sp3 = parse_sp3_file(sp3_file_path)

    # filter query and sp3 data that have common sv
    df_sp3 = df_sp3[df_sp3["sv"].isin(query["sv"].unique())]
    query = query[query["sv"].isin(df_sp3["sv"].unique())]

    for sv, group in query.groupby("sv"):
        # Select SP3 rows for this satellite
        sp3 = df_sp3.loc[df_sp3["sv"] == sv]

        # Convert query times to GPST seconds
        t_query = np.array(
            [
                util.timedelta_2_seconds(t - constants.cGpstUtcEpoch)
                for t in group.query_time_isagpst
            ]
        )

        if not is_query_corrected_by_sat_clock_offset:
            # correct query time by satellite clock offset
            t0 = t_query
            for _ in range(2):
                C, _ = interpolate_krogh(
                    sp3.gpst_s.to_numpy(), sp3.sat_clock_offset_m.to_numpy(), t0, 2
                )
                t0 = t_query - C / constants.cGpsSpeedOfLight_mps
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
            sp3.gpst_s.to_numpy(), sp3.sat_clock_offset_m.to_numpy(), t_query, 2
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
        query.loc[group.index, "relativistic_clock_effect_m"] = np.array(
            [
                -2 * np.dot(p, v) / constants.cGpsSpeedOfLight_mps
                for p, v in zip(np.stack([X, Y, Z]).T, np.stack([dX, dY, dZ]).T)
            ]
        )

    # add phase center offset
    pco = atx_processing.compute_pco_sat(
        query=query,
        atx_df=atx_processing.parse_atx(atx_file_path),
    )

    # merge pco into query
    query = query.merge(
        pco,
        left_on=["query_time_isagpst", "sv", "signal"],
        right_on=["query_time_isagpst", "sv", "signal"],
        how="left",
    )

    # compute satellite antenna position
    query = query.assign(
        sat_pos_x_m=query["sat_pos_com_x_m"] + query["pco_sat_x_m"],
        sat_pos_y_m=query["sat_pos_com_y_m"] + query["pco_sat_y_m"],
        sat_pos_z_m=query["sat_pos_com_z_m"] + query["pco_sat_z_m"],
    )

    return query


if __name__ == "main":
    pass
