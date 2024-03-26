import json
from pathlib import Path
import numpy as np
import pandas as pd

from prx.constants import cGpsSpeedOfLight_mps


def parse_prx_csv_file_metadata(prx_file: Path):
    with open(prx_file, "r") as f:
        metadata = json.loads(f.readline().replace("# ", ""))
    return metadata


def parse_prx_csv_file(prx_file: Path):
    return pd.read_csv(prx_file, comment="#"), parse_prx_csv_file_metadata(prx_file)


def spp_vt_lsq(df, p_ecef_m):
    df = df[df.D_obs_hz.notna()].reset_index(drop=True)
    df["D_obs_mps"] = -df.D_obs_hz * cGpsSpeedOfLight_mps / df.carrier_frequency_hz
    # Remove satellite velocity projected onto line of sight
    rx_sat_vectors = df[["x_m", "y_m", "z_m"]].to_numpy() - p_ecef_m.T
    row_sums = np.linalg.norm(rx_sat_vectors, axis=1)
    unit_vectors = (rx_sat_vectors.T / row_sums).T
    # The next line computes the row-wise dot product of the two matrices
    df["satellite_los_velocities"] = np.sum(
        unit_vectors * df[["dx_mps", "dy_mps", "dz_mps"]].to_numpy(), axis=1
    ).reshape(-1, 1)
    df["D_obs_corrected_mps"] = (
        df.D_obs_mps.to_numpy().reshape(-1, 1)
        + df.dclock_mps.to_numpy().reshape(-1, 1)
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
    x_lsq, residuals, _, _ = np.linalg.lstsq(H, df.D_obs_corrected_mps, rcond="warn")
    return x_lsq.reshape(-1, 1)


def spp_pt_lsq(df, dx_convergence_l2=1e-6, max_iterations=10):
    df = df[df.C_obs.notna()]
    df["C_obs_corrected"] = (
        df.C_obs
        + df.clock_m
        + df.relativistic_clock_effect_m
        - df.sagnac_effect_m
        - df.code_iono_delay_klobuchar_m
        - df.tropo_delay_m
        - df.group_delay_m
    )
    # Jacobian of pseudorange observation w.r.t. receiver clock offset w.r.t. constellation system clock
    H_clock = np.zeros(
        (
            len(df.C_obs_corrected),
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
        C_obs_predicted = (
            np.linalg.norm(
                x_linearization[0:3].T - df[["x_m", "y_m", "z_m"]].to_numpy(), axis=1
            )
            + np.squeeze(  # geometric distance
                H_clock @ x_linearization[3:]
            )
        )  # rx to constellation clock bias
        # compute jacobian matrix
        rx_sat_vectors = df[["x_m", "y_m", "z_m"]].to_numpy() - x_linearization[:3].T
        row_sums = np.linalg.norm(rx_sat_vectors, axis=1)
        unit_vectors = (rx_sat_vectors.T / row_sums).T
        # One clock offset per constellation
        H = np.hstack((-unit_vectors, H_clock))
        x_lsq, residuals, _, _ = np.linalg.lstsq(
            H, df.C_obs_corrected - C_obs_predicted, rcond="warn"
        )
        x_lsq = x_lsq.reshape(-1, 1)
        solution_increment_l2 = np.linalg.norm(x_lsq)
        x_linearization += x_lsq
        n_iterations += 1
        assert (
            n_iterations <= max_iterations
        ), "LSQ did not converge in allowed number of iterations"
    return x_linearization
