import json
from pathlib import Path

import numpy as np
import pandas as pd
import georinex as gr

from prx.constants import cGpsSpeedOfLight_mps
import prx.constants
import prx.rinex_nav.evaluate
import prx.helpers


def parse_prx_csv_file_metadata(prx_file: Path):
    with open(prx_file, "r") as f:
        metadata = json.loads(f.readline().replace("# ", ""))
    metadata["approximate_receiver_ecef_position_m"] = np.array(
        metadata["approximate_receiver_ecef_position_m"]
    )
    return metadata


def parse_prx_csv_file(prx_file: Path):
    df = pd.read_csv(prx_file, comment="#")
    df.time_of_reception_in_receiver_time = pd.to_datetime(
        df.time_of_reception_in_receiver_time
    )
    metadata = parse_prx_csv_file_metadata(prx_file)
    return df, metadata


def spp_vt_lsq(df, p_ecef_m):
    df = df[df.D_obs_hz.notna() & df.sat_clock_drift_mps.notna()].reset_index(drop=True)
    df["D_obs_mps"] = -df.D_obs_hz * cGpsSpeedOfLight_mps / df.carrier_frequency_hz
    # Remove satellite velocity projected onto line of sight
    rx_sat_vectors = (
        df[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy() - p_ecef_m.T
    )
    row_sums = np.linalg.norm(rx_sat_vectors, axis=1)
    unit_vectors = (rx_sat_vectors.T / row_sums).T
    # The next line computes the row-wise dot product of the two matrices
    df["satellite_los_velocities"] = np.sum(
        unit_vectors
        * df[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]].to_numpy(),
        axis=1,
    ).reshape(-1, 1)
    df["D_obs_corrected_mps"] = (
        df.D_obs_mps.to_numpy().reshape(-1, 1)
        + df.sat_clock_drift_mps.to_numpy().reshape(-1, 1)
        - df["satellite_los_velocities"].to_numpy().reshape(-1, 1)
    )
    # Jacobian of Doppler observation w.r.t. receiver clock offset drift w.r.t. constellation system clock
    H_dclock = np.zeros(
        (
            len(df.D_obs_corrected_mps),
            len(df.constellation.unique()),
        )
    )
    for i, constellation in enumerate(df.constellation.unique()):
        H_dclock[df.constellation == constellation, i] = 1
    H = np.hstack((-unit_vectors, H_dclock))
    x_lsq, residuals, _, _ = np.linalg.lstsq(H, df.D_obs_corrected_mps, rcond=None)
    return x_lsq.reshape(-1, 1)


def spp_pt_lsq(df, dx_convergence_l2=1e-6, max_iterations=10):
    df = df[df.C_obs_m.notna() & df.sat_clock_offset_m.notna()]
    df["C_obs_m_corrected"] = (
        df.C_obs_m
        + df.sat_clock_offset_m
        + df.relativistic_clock_effect_m
        - df.sagnac_effect_m
        - df.iono_delay_m
        - df.tropo_delay_m
        - df.sat_code_bias_m
    )
    # Jacobian of pseudorange observation w.r.t. receiver clock offset w.r.t. constellation system clock
    H_clock = np.zeros(
        (
            len(df.C_obs_m_corrected),
            len(df.constellation.unique()),
        )
    )
    for i, constellation in enumerate(df.constellation.unique()):
        H_clock[df.constellation == constellation, i] = 1
    # Initial linearization point
    x_linearization = np.zeros((3 + len(df.constellation.unique()), 1))
    solution_increment_l2 = np.inf
    n_iterations = 0
    while solution_increment_l2 > dx_convergence_l2:
        # Compute predicted pseudo-range as geometric distance + receiver clock bias, predicted at x_linearization
        C_obs_m_predicted = (
            np.linalg.norm(
                x_linearization[0:3].T
                - df[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(),
                axis=1,
            )  # geometric distance
            + np.squeeze(H_clock @ x_linearization[3:])
        )  # rx to constellation clock bias
        # compute jacobian matrix
        rx_sat_vectors = (
            df[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy()
            - x_linearization[:3].T
        )
        row_sums = np.linalg.norm(rx_sat_vectors, axis=1)
        unit_vectors = (rx_sat_vectors.T / row_sums).T
        # One clock offset per constellation
        H = np.hstack((-unit_vectors, H_clock))
        x_lsq, residuals, _, _ = np.linalg.lstsq(
            H, df.C_obs_m_corrected - C_obs_m_predicted, rcond=None
        )
        x_lsq = x_lsq.reshape(-1, 1)
        solution_increment_l2 = np.linalg.norm(x_lsq)
        x_linearization += x_lsq
        n_iterations += 1
        assert (
            n_iterations <= max_iterations
        ), "LSQ did not converge in allowed number of iterations"
    return x_linearization


def bootstrap_coarse_receiver_position(filepath_obs, filepath_nav):
    """Computes a position from the first epoch of a RNX OBS file

    This is useful if no APPROX POSITION XYZ field is present in the header.
    The function builds a minimal DataFrame with the required columns to call spp_pt_lsq
    All corrections involving positions are set to 0.
    """
    first_epoch = pd.Timestamp(gr.gettime(filepath_obs)[0])
    # select GPS C1C for first epoch
    n_obs = 0
    while n_obs < 4:  # loop until 4 observations are available
        obs = gr.load(
            filepath_obs,
            tlim=[
                first_epoch.isoformat(),
                (first_epoch + pd.Timedelta(seconds=1)).isoformat(),
            ],
            use={"G"},
            meas=["C1C"],
        )

        if "C1C" in obs:  # check if there is at least one GPS L1C/A observation
            obs = obs.isel(time=[0]).where(obs.isel(time=[0]).C1C.notnull(), drop=True)
            n_obs = len(obs.sv.values)
        else:
            n_obs = 0
        first_epoch = first_epoch + pd.Timedelta(seconds=1)

    time_of_flight = [
        pd.Timedelta(
            float(obs.C1C.isel(time=0, sv=i)) / prx.constants.cGpsSpeedOfLight_mps,
            unit="s",
        )
        for i in range(len(obs.sv))
    ]
    time_of_emission = [
        obs.time.values[0] - time_of_flight[i] for i in range(len(obs.sv))
    ]

    # build query df
    query = pd.DataFrame(
        data={
            "time_of_reception_in_receiver_time": obs.time.values[0],
            "observation_value": obs.isel(time=0).C1C.values,
            "signal": "C1C",
            "sv": obs.sv.values,
            "query_time_isagpst": time_of_emission,
        }
    )

    # compute satellite states
    # TODO find correct ephemeris file in list
    current_nav_file_index = 0
    sat_states = prx.rinex_nav.evaluate.compute(
        filepath_nav[current_nav_file_index], query
    )
    sat_states = sat_states.rename(
        columns={
            "signal": "observation_type",
            "query_time_isagpst": "time_of_emission_isagpst",
        }
    )
    sat_states["relativistic_clock_effect_m"] = (
        prx.helpers.compute_relativistic_clock_effect(
            sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(),
            sat_states[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]].to_numpy(),
        )
    )

    # create obs df by merging query and sat_states
    obs_df = query[
        [
            "time_of_reception_in_receiver_time",
            "sv",
            "observation_value",
        ]
    ].merge(
        sat_states[
            [
                "sat_code_bias_m",
                "sat_clock_offset_m",
                "sat_clock_drift_mps",
                "sv",
                "sat_pos_x_m",
                "sat_pos_y_m",
                "sat_pos_z_m",
                "sat_vel_x_mps",
                "sat_vel_y_mps",
                "sat_vel_z_mps",
                "relativistic_clock_effect_m",
            ]
        ],
        how="left",
        on="sv",
    )
    obs_df = obs_df.rename(columns={"observation_value": "C_obs_m"})
    # add missing columns with 0 value, since approximate position is unknown
    obs_df = pd.concat(
        [
            obs_df,
            pd.DataFrame(
                {
                    "sagnac_effect_m": np.zeros(len(obs.sv)),
                    "iono_delay_m": np.zeros(len(obs.sv)),
                    "tropo_delay_m": np.zeros(len(obs.sv)),
                    "constellation": "G",
                }
            ),
        ],
        axis=1,
    )

    solution = spp_pt_lsq(
        obs_df,
    )

    return solution[0:3].squeeze()


def compute_spp_base_obs(df_rover, p_base):
    df_base = df_rover.copy()
    range_obs = np.linalg.norm(
        df_base[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy() - p_base.T,
        axis=1,
    )
    range_obs += (
        -df_rover.sat_clock_offset_m
        - df_rover.relativistic_clock_effect_m
        + df_rover.sagnac_effect_m
        + df_rover.iono_delay_m
        + df_rover.tropo_delay_m
        + df_rover.sat_code_bias_m
    )
    df_base["C_obs_m"] = range_obs
    df_base["L_obs_cycles"] = range_obs / (
        df_rover.carrier_frequency_hz / cGpsSpeedOfLight_mps
    )
    return df_base


def solve_lsq(df):
    pass


def trajectory_pvt_lsq(
    df_rover,
    p_base,
    df_base=None,
):
    if df_base is None:
        df_base = compute_spp_base_obs(df_rover, p_base)
    for _df in [df_rover, df_base]:
        _df["time_of_reception_rounded"] = (
            _df.time_of_reception_in_receiver_time.dt.round("50ms")
        )
        _df["sv"] = _df["constellation"].astype(str) + _df["prn"].astype(str).str.pad(
            2, fillchar="0"
        )
        _df["broadcast_range"] = np.linalg.norm(
            df_base[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy()
            - p_base.T,
            axis=1,
        )
        # Remove fast-changing satellite terms
        _df["lambda"] = _df.carrier_frequency_hz / cGpsSpeedOfLight_mps
        _df.C_obs_m = _df.C_obs_m - _df.broadcast_range + _df.sat_clock_offset_m
        _df.L_obs_cycles = (
            _df.L_obs_cycles
            - (_df.broadcast_range - _df.sat_clock_offset_m) / _df["lambda"]
        )
    df = df_rover.merge(
        df_base,
        on=["time_of_reception_rounded", "sv", "rnx_obs_identifier", "ephemeris_hash"],
        suffixes=("_rover", "_base"),
    ).reset_index(drop=True)
    df["C_obs_m_sd"] = df.C_obs_m_rover - df.C_obs_m_base
    df[["u_x", "u_y", "u_z"]] = (
        df[["sat_pos_x_m_base", "sat_pos_y_m_base", "sat_pos_z_m_base"]].to_numpy()
        - p_base.T
    ) / df["broadcast_range_base"].to_numpy().reshape(-1, 1)
    df = df.rename(columns={"constellation_rover": "constellation"})
    df = df[
        [
            "time_of_reception_rounded",
            "u_x",
            "u_y",
            "u_z",
            "C_obs_m_sd",
            "constellation",
        ]
    ]

    epochs = list(df.groupby("time_of_reception_rounded"))[0]
    solve_lsq(
        df[["time_of_reception_in_receiver_time_rover", "u_x", "u_y", "u_z"]].to_numpy()
    )
    return None
