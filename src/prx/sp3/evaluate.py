from functools import lru_cache
import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import georinex
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from prx import helpers
from prx import constants

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
            columns={"position_x": "sat_pos_x_m", "position_y": "sat_pos_y_m", "position_z": "sat_pos_z_m"},
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

    file_content_hash = helpers.hash_of_file_content(file_path)
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


def interpolate(df, query_time_gpst_s, plot_interpolation=False):
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
    columns_to_interpolate = ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m", "sat_clock_offset_m"]
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
        if plot_interpolation:
            plot_lagrange_interpolation(
                poly,
                times - times[0],
                samples - samples[0],
                query_time_gpst_s - times[0],
                interpolated[col] - samples[0],
                f"{col} {df['sv'].unique()}",
            )
        first_derivative = Polynomial(poly.coef[::-1]).deriv(1)(
            query_time_gpst_s - times[0]
        )
        if col in ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]:
            interpolated[f"d{col}ps"] = first_derivative
        elif col == "sat_clock_offset_m":
            interpolated["sat_clock_drift_mps"] = first_derivative

    return interpolated


def compute(sp3_file_path, query):
    df = parse_sp3_file(sp3_file_path)
    df = df[df["sv"].isin(query["sv"])]

    def interpolate_sat_states(row):
        sat_pv = interpolate(
            df[df["sv"] == row["sv"]],
            helpers.timedelta_2_seconds(row["query_time_isagpst"]),
        )

        return pd.concat((row.drop("sv"), sat_pv.squeeze()))

    query = query.apply(interpolate_sat_states, axis=1).reset_index()
    return query.drop(columns=["gpst_s", "index", "t0"])


if __name__ == "main":
    pass
