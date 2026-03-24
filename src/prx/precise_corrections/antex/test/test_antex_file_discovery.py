import os
from pathlib import Path
import shutil
import georinex
import pandas as pd
import pytest
from unittest.mock import patch

import prx
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


def test_extract_gps_week():
    filenames = [
        "igs20_2134.atx",
        "igs_10.atx",
        "igs14_abc.atx",
    ]
    expected_returns = [2134, -1, -1]
    gps_week = [atx.extract_gps_week(f) for f in filenames]
    assert gps_week == expected_returns


def test_download_if_not_local(set_up_test):
    """
    Tests that the ANTEX file is downloaded when no local file is available,
    but a valid remote file exists.

    Scenario :
    - No local ANTEX file available
    - A remote ANTEX file exists
    -> The function should trigger the download of the remote file.
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(
        header["TIME OF FIRST OBS"]
    ) - pd.Timedelta(200, unit="milliseconds")

    db_folder = set_up_test["test_obs_file"].parent
    downloaded_atx = atx.get_atx_file(t_start, db_folder)

    # Ensure the file was downloaded and exists
    assert downloaded_atx is not None
    assert downloaded_atx.exists()


def test_download_if_remote_is_newer(set_up_test):
    """
    Tests that the ANTEX file is downloaded when the remote file has a newer GPS week
    than the local one.

    Scenario :
    - local file = igs20_2370.atx
    - remote file = igs20_2375.atx
    -> The function should download igs20_2375.atx
    """
    latest_local = "igs20_2370.atx"
    latest_remote = "igs20_2375.atx"
    date = pd.Timestamp("2025-01-01 12:00:00")
    db_folder = set_up_test["test_obs_file"].parent

    with (
        patch(  # replace download function by online availability check
            "prx.precise_corrections.antex.antex_file_discovery.try_downloading_atx_ftp",
            new=prx.precise_corrections.antex.antex_file_discovery.check_online_availability,
        ),
        patch(  # simulate that the latest local file found is 'igs20_2370.atx'
            "prx.precise_corrections.antex.antex_file_discovery.find_latest_local_antex_file",
            return_value=latest_local,
        ),
        patch(  # simulate that the latest remote file found is 'igs20_2375.atx'
            "prx.precise_corrections.antex.antex_file_discovery.fetch_latest_remote_antex_file",
            return_value=latest_remote,
        ),
    ):
        result = atx.get_atx_file(date, db_folder)
        assert result is not None
        assert result.name == latest_remote


def test_skip_download_if_same_week(set_up_test):
    """
    Tests that the ANTEX file is not downloaded when the remote and local files
    are from the same GPS week.

    Scenario:
    - local file = igs20_2375.atx
    - remote file = igs20_2375.atx
    → The function should skip download and return the local file
    """
    latest_local = "igs20_2375.atx"
    latest_remote = "igs20_2375.atx"
    date = pd.Timestamp("2025-01-01 12:00:00")
    db_folder = set_up_test["test_obs_file"].parent

    with (
        patch(  # mock download function
            "prx.precise_corrections.antex.antex_file_discovery.try_downloading_atx_ftp"
        ) as mock_download,
        patch(  # simulate that the latest local file found is 'igs20_2375.atx'
            "prx.precise_corrections.antex.antex_file_discovery.find_latest_local_antex_file",
            return_value=latest_local,
        ),
        patch(  # simulate that the latest remote file found is 'igs20_2375.atx'
            "prx.precise_corrections.antex.antex_file_discovery.fetch_latest_remote_antex_file",
            return_value=latest_remote,
        ),
    ):
        result = atx.get_atx_file(date, db_folder)

        assert result is not None
        # Ensure that the download function was not called
        mock_download.assert_not_called()
        # Ensure the returned file is the local one
        assert result == latest_local
