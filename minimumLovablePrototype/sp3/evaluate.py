import math
import subprocess
from functools import lru_cache

import pandas as pd
import numpy as np
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
import glob
import georinex
from pathlib import Path
import joblib

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
        df = df.set_index([df['ECEF'], df.groupby('ECEF').cumcount()]).drop('ECEF', 1).unstack(0)
        df.columns = [f'{x}_{y}' for x, y in df.columns]
        cols_to_drop = [col for col in df.columns if 'x' not in col and 'position' not in col and 'velocity' not in col]
        df.drop(cols_to_drop, axis=1, inplace=True)
        for col in df.columns:
            if 'position' not in col and 'velocity' not in col:
                df.rename(columns={col: col.replace('_x', '')}, inplace=True)
        df['gpst_s'] = df['time'] - constants.cGpstUtcEpoch
        df.drop('time', axis=1, inplace=True)
        df['clock'] = df['clock'] / constants.cMicrosecondsPerSecond
        df['dclock'] = df['dclock'] / constants.cMicrosecondsPerSecond
        for axis in ['x', 'y', 'z']:
            df['position_' + axis] = df['position_' + axis] * constants.cMetersPerKilometer
        return df

    t0 = pd.Timestamp.now()
    file_content_hash = helpers.md5_of_file_content(file_path)
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want to partially hash the file"
        )
    return cached_load(file_path, file_content_hash)


def interpolate(df, query_time_gpst):
    n_samples_each_side = 4
    assert df['sv'].unique().size == 1, "This function expects one satellite at a time"
    closest_sample_index = np.argmin(np.abs(df['gpst_s'] - query_time_gpst))
    start_index = closest_sample_index - n_samples_each_side
    end_index = closest_sample_index + n_samples_each_side
    assert start_index >= 0, f"We need at least {n_samples_each_side} before the sample closest to the query time to interpolate"
    assert end_index < len(df.index), f"We need at least {n_samples_each_side} after the sample closest to the query time to interpolate"
    columns_to_interpolate = [col for col in df.columns if 'position' in col or col == 'clock']
    interpolated = df.iloc[closest_sample_index, :]
    for col in columns_to_interpolate:
        times = df['gpst_s'].iloc[start_index:end_index+1].to_numpy()
        samples = df[col].iloc[start_index:end_index+1].to_numpy()
        # Improve numerical conditioning by subtracting the first sample
        poly = lagrange(times - times[0], samples - samples[0])
        interpolated[col] = Polynomial(poly.coef[::-1])(query_time_gpst - times[0]) + samples[0]
        first_derivative = Polynomial(poly.coef[::-1]).deriv(1)(query_time_gpst - times[0])
        if 'position' in col:
            interpolated[col.replace('position', 'velocity')] = first_derivative
        elif col == 'clock':
            interpolated['dclock'] = first_derivative

    return interpolated.to_frame().T


def compute(sp3_file_path, query_time_gpst):
    df = parse_sp3_file(sp3_file_path)
    df['gpst_s'] = df['gpst_s'].apply(helpers.timedelta_2_seconds)
    interpolated = pd.DataFrame()
    for sv, sv_df in df.groupby(by="sv"):
        interpolated = pd.concat([interpolated, interpolate(sv_df.reset_index(), helpers.timedelta_2_seconds(query_time_gpst))])
    pass


if __name__ == "main":
    pass
