
import os
from pathlib import Path
import shutil
import georinex
import pandas as pd
import pytest
from unittest.mock import patch 

from prx import util
from prx.precise_corrections.antex import antex_file_discovery as atx


@pytest.fixture
def set_up_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_obs_file = test_directory.joinpath("TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz")
    # local_database = atx.atx_file_folder(pd.Timestamp('2023-01-01'), Path('src/prx/precise_corrections/atx/test/datasets'))

    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_obs_file.name}",
        test_obs_file,
    )
    assert test_obs_file.exists()
   
    yield {"test_obs_file": test_obs_file}
    shutil.rmtree(test_directory)

def test_get_atx_file(set_up_test):
    """
    Assert that prx will download the latest ATX file available if it is not present locally. 
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")

    db_folder = set_up_test["test_obs_file"].parent
    downloaded_atx = atx.get_atx_file(t_start, db_folder)
    
    assert downloaded_atx is not None
    assert downloaded_atx.exists()
    
def test_fallback_to_latest_existing_atx(set_up_test):
    """
    This test checks that if no ANTEX file is available for the target GPS week (e.g. the week is too recent), 
    the latest available ANTEX file from a previous GPS week is used instead.
    """

    db_folder = set_up_test["test_obs_file"].parent
    now = pd.Timestamp.now()
    gps_week_future = atx.date_to_gps_week(now) + 10
    atx_filename = f'igs*{str(gps_week_future)}.atx'
    with patch('prx.precise_corrections.antex.antex_file_discovery.atx_filename', atx_filename):
        downloaded_atx = atx.get_atx_file(now, db_folder)

    