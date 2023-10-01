from functools import lru_cache

import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import georinex
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from .. import helpers
from .. import constants

memory = joblib.Memory(Path(__file__).parent.joinpath("diskcache"), verbose=0)
log = helpers.get_logger(__name__)


def parse_sp3_file(file_path: Path):
    @lru_cache
    @memory.cache
    def cached_load(file_path: Path, file_hash: str):
        log.info(f"Parsing {file_path} ...")
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
            helpers.timedelta_2_seconds
        )
        df.drop("time", axis=1, inplace=True)
        df["clock"] = df["clock"] / constants.cMicrosecondsPerSecond
        df["dclock"] = df["dclock"] / constants.cMicrosecondsPerSecond
        for axis in ["x", "y", "z"]:
            df["position_" + axis] = (
                df["position_" + axis] * constants.cMetersPerKilometer
            )
        # Give some columns more pithy names
        df.rename(
            columns={"position_x": "x_m", "position_y": "y_m", "position_z": "z_m"},
            inplace=True,
        )
        df.rename(
            columns={
                "velocity_x": "dx_mps",
                "velocity_y": "dy_mps",
                "velocity_z": "dz_mps",
            },
            inplace=True,
        )
        df.rename(columns={"clock": "clock_s", "dclock": "dclock_sps"}, inplace=True)
        # Put timestamps first
        df.insert(0, "gpst_s", df.pop("gpst_s"))
        return df

    t0 = pd.Timestamp.now()
    file_content_hash = helpers.md5_of_file_content(file_path)
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want to partially hash the file"
        )
    return cached_load(file_path, file_content_hash)


def plot_lagrange_interpolation(
    polynomial, times, samples, interpolation_time, interpolated_value, label
):
    polynomial_times = np.linspace(min(times), max(times), 10 * times.size)
    polynomial_samples = Polynomial(polynomial.coef[::-1])(polynomial_times)
    plt.plot(times, samples, "o", label="samples")
    plt.plot(polynomial_times, polynomial_samples, ".", label="samples")
    plt.plot(
        interpolation_time,
        Polynomial(polynomial.coef[::-1])(interpolation_time),
        "x",
        label=label,
    )
    plt.legend()
    plt.grid()
    plt.show()


def interpolate(df, query_time_gpst_s):
    n_samples_each_side = 4
    assert df["sv"].unique().size == 1, "This function expects one satellite at a time"
    closest_sample_index = np.argmin(np.abs(df["gpst_s"] - query_time_gpst_s))
    start_index = closest_sample_index - n_samples_each_side
    end_index = closest_sample_index + n_samples_each_side
    assert (
        start_index >= 0
    ), f"We need at least {n_samples_each_side} before the sample closest to the query time to interpolate"
    assert end_index < len(
        df.index
    ), f"We need at least {n_samples_each_side} after the sample closest to the query time to interpolate"
    columns_to_interpolate = ["x_m", "y_m", "z_m", "clock_s"]
    interpolated = df[closest_sample_index : closest_sample_index + 1]
    interpolated["gpst_s"] = query_time_gpst_s
    for col in columns_to_interpolate:
        interpolated[col] = float("nan")
    for col in columns_to_interpolate:
        times = df["gpst_s"].iloc[start_index : end_index + 1].to_numpy()
        samples = df[col].iloc[start_index : end_index + 1].to_numpy()
        # Improve numerical conditioning by subtracting the first sample
        poly = lagrange(times - times[0], samples - samples[0])
        interpolated[col] = (
            Polynomial(poly.coef[::-1])(query_time_gpst_s - times[0]) + samples[0]
        )
        """
        plot_lagrange_interpolation(poly,
                                    times - times[0],
                                    samples - samples[0],
                                    query_time_gpst -times[0],
                                    interpolated[col] - samples[0], f"{col} {df['sv'].unique()}")
        """
        first_derivative = Polynomial(poly.coef[::-1]).deriv(1)(
            query_time_gpst_s - times[0]
        )
        if col in ["x_m", "y_m", "z_m"]:
            interpolated[f"d{col}ps"] = first_derivative
        elif col == "clock_s":
            interpolated["dclock_sps"] = first_derivative

    return interpolated


def compute(sp3_file_path, query_time_gpst):
    df = parse_sp3_file(sp3_file_path)
    interpolated = pd.DataFrame()
    for sv, sv_df in df.groupby(by="sv"):
        interpolated = pd.concat(
            [
                interpolated,
                interpolate(
                    sv_df.reset_index().drop(columns=["index"]),
                    helpers.timedelta_2_seconds(query_time_gpst),
                ),
            ]
        )
    return interpolated


if __name__ == "main":
    pass
