import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import georinex as gr

from prx.constants import cGpsSpeedOfLight_mps
import prx.constants
import prx.rinex_nav.evaluate
import prx.util


def parse_prx_csv_file_metadata(prx_file: Path):
    with open(prx_file, "r") as f:
        metadata = json.loads(f.readline().replace("# ", ""))
    return metadata


def parse_prx_csv_file(prx_file: Path):
    df = pl.read_csv(
        prx_file, comment_prefix="#", schema_overrides={"ephemeris_hash": pl.String}
    ).to_pandas()
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
        assert n_iterations <= max_iterations, (
            "LSQ did not converge in allowed number of iterations"
        )
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
        filepath_nav[current_nav_file_index], pl.from_pandas(query)
    ).to_pandas()
    sat_states = sat_states.rename(
        columns={
            "signal": "observation_type",
            "query_time_isagpst": "time_of_emission_isagpst",
        }
    )
    sat_states["relativistic_clock_effect_m"] = (
        prx.util.compute_relativistic_clock_effect(
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
