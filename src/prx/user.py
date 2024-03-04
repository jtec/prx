import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_prx_csv_file_metadata(prx_file: Path):
    with open(prx_file, "r") as f:
        metadata = json.loads(f.readline().replace("# ", ""))
    return metadata


def parse_prx_csv_file(prx_file: Path):
    return pd.read_csv(prx_file, comment="#"), parse_prx_csv_file_metadata(prx_file)


def spp_pt_lsq(df):
    df["C_obs_corrected"] = (
        df.C_obs
        + df.clock_m
        + df.relativistic_clock_effect_m
        - df.sagnac_effect_m
        - df.code_iono_delay_klobuchar_m
        - df.tropo_delay_m
        - df.group_delay_m
    )
    # Determine the constellation selection matrix of each obs
    H_clock = np.zeros(
        (
            len(df.C_obs_corrected),
            len(df.constellation.unique()),
        )
    )
    for i, constellation in enumerate(df.constellation.unique()):
        H_clock[df.constellation == constellation, i] = 1
    # initial linearization point
    x_linearization = np.zeros((3 + len(df.constellation.unique()), 1))
    solution_increment_sos = np.inf
    n_iterations = 0
    while solution_increment_sos > 1e-6:
        # compute predicted pseudo-range as geometric distance + clock bias, predicted at x_linearization
        C_obs_predicted = np.linalg.norm(
            x_linearization[0:3].T - df[["x_m", "y_m", "z_m"]].to_numpy(), axis=1
        ) + np.squeeze(  # geometric distance
            H_clock @ x_linearization[3:]
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
        solution_increment_sos = np.linalg.norm(x_lsq)
        x_linearization += x_lsq
        n_iterations += 1
    assert n_iterations < 10
    return x_linearization
