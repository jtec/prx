import logging
from datetime import timedelta

import numpy as np

import pandas as pd
import polars as pl

from prx.constants import cSecondsPerDay
from prx.util import timedelta_2_seconds

log = logging.getLogger(__name__)


def test_timedelta_2_seconds():
    expected_timedelta_s = cSecondsPerDay +  1.23456789
    assert np.isclose(timedelta_2_seconds(pd.Timedelta(days=1, seconds=1.23456789)), expected_timedelta_s, atol=1e-9)
    assert np.isclose(timedelta_2_seconds(pd.Series([pd.Timedelta(days=1, seconds=1.23456789)])).iloc[0], expected_timedelta_s, atol=1e-9)
    assert np.isclose(timedelta_2_seconds(pl.Series(values=[timedelta(days=1, seconds=1.23456789)], dtype=pl.Duration(time_unit="ns")))[0], expected_timedelta_s, atol=1e-9)