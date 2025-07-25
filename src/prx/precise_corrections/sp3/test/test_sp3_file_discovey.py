import os
from pathlib import Path
import shutil
import georinex
import pytest

import pandas as pd
from unittest.mock import patch
import prx
import prx.util
from prx import util
from prx.precise_corrections.sp3 import sp3_file_discovery as sp3


@pytest.fixture
def set_up_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_obs_file = test_directory.joinpath("TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz")

    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_obs_file.name}",
        test_obs_file,
    )
    assert test_obs_file.exists()
    test_date = pd.Timestamp("2023-01-01")
    sp3_subfolder = sp3.sp3_file_folder(test_date, test_directory)

    # copy all sp3 files
    for file in (
        util.prx_repository_root()
        / "src"
        / "prx"
        / "precise_corrections"
        / "sp3"
        / "test"
        / "datasets"
        / "2023"
        / "001"
    ).glob("*"):
        test_sp3_file = sp3_subfolder / file.name
        shutil.copy(
            util.prx_repository_root()
            / f"src/prx/precise_corrections/sp3/test/datasets/2023/001/{test_sp3_file.name}",
            test_sp3_file,
        )
        assert test_sp3_file.exists()

    yield {
        "test_obs_file": test_obs_file,
    }
    shutil.rmtree(test_directory)


def test_get_index_of_priority_from_filename():
    index_priority = []
    for sp3_filename in [
        "COD0MGXFIN_20230010000_01D_05M_ORB.SP3.gz",
        "GFZ0MGXRAP_20230010000_01D_05M_ORB.SP3.gz",
        "JAX0MGXFIN_20230010000_01D_30S_CLK.CLK.gz",
        "WUM0MGXFIN_20230010000_01D_05M_ORB.SP3.gz",
        "WUM0MGXFIN_20230010000_01D_30S_CLK.CLK.gz",
    ]:
        index_priority.append(sp3.get_index_of_priority_from_filename(sp3_filename))
    assert index_priority == [0, 10, 5, 4, 4]


def test_download_single_sp3():
    assert False


def test_get_sp3_files(set_up_test):
    """
    Assert that prx will download the first SP3 file in the priority list if not present.
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
    t_end = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])

    with (
        patch(  # replace download function by online availability check
            "prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp",
            new=prx.precise_corrections.sp3.sp3_file_discovery.check_online_availability,
        ),
        patch(  # simulate that local database is empty
            "prx.precise_corrections.sp3.sp3_file_discovery.get_local_sp3",
            return_value=None,
        ),
    ):
        sp3_files = sp3.get_sp3_files(t_start, t_end)

    file_orb = sp3_files[0][0]
    file_clk = sp3_files[0][1]
    assert file_orb is not None
    assert file_clk is not None
    assert sp3.get_index_of_priority_from_filename(
        file_orb
    ) == sp3.get_index_of_priority_from_filename(file_clk)


def test_get_sp3_files_multiple_days(set_up_test):
    """
    Assert that prx will download the first SP3 file in the priority list if not present.
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
    t_end = t_start + pd.Timedelta(days=1)

    downloaded_files = []
    with (
        patch(  # replace download function by online availability check
            "prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp",
            new=prx.precise_corrections.sp3.sp3_file_discovery.check_online_availability,
        ),
        patch(  # simulate that local database is empty
            "prx.precise_corrections.sp3.sp3_file_discovery.get_local_sp3",
            return_value=None,
        ),
    ):
        sp3_files = sp3.get_sp3_files(t_start, t_end)

    assert len(sp3_files) == 2
    for ind_day in range(2):
        file_orb = sp3_files[ind_day][0]
        file_clk = sp3_files[ind_day][1]
        assert file_orb is not None
        assert file_clk is not None
        assert sp3.get_index_of_priority_from_filename(
            file_orb
        ) == sp3.get_index_of_priority_from_filename(file_clk)


def test_download_FIN_when_local_RAP_is_available(set_up_test):
    """
    The tested scenario is a local database containing the RAP (rapid) SP3 products, but the FIN (final) products are
    available online.
    The FIN products should be downloaded
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
    t_end = t_start
    local_db = set_up_test["test_obs_file"].parent

    # remove FIN products from local db
    for file in local_db.glob("**/*FIN*"):
        file.unlink()

    with (
        patch(  # replace download function by online availability check
            "prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp",
            new=prx.precise_corrections.sp3.sp3_file_discovery.check_online_availability,
        ),
    ):
        sp3_files = sp3.get_sp3_files(t_start, t_end, local_db)

    file_orb = sp3_files[0][0]
    file_clk = sp3_files[0][1]
    assert "FIN" in file_orb
    assert "FIN" in file_clk
    assert sp3.get_index_of_priority_from_filename(
        file_orb
    ) == sp3.get_index_of_priority_from_filename(file_clk)


def test_match_CLK_and_ORB(set_up_test):
    """
    The tested scenario is a local database containing an ORB file, but not the matching CLK file.
    The missing CLK file should be downloaded.
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"])
    t_end = t_start
    local_db = set_up_test["test_obs_file"].parent

    # find an ORB file without matching CLK file
    list_orb = list(local_db.glob("**/*ORB*"))
    for ind_orb, local_orb in enumerate(list_orb):
        idx_priority = sp3.get_index_of_priority_from_filename(local_orb.name)
        _, file_clk_expected = sp3.build_sp3_filename(
            t_start, sp3.priority[idx_priority]
        )
        if not list(local_db.glob("**/file_clk_expected")):
            break

    with (
        patch(  # replace priority list with single element
            "prx.precise_corrections.sp3.sp3_file_discovery.priority",
            new=[sp3.priority[idx_priority]],
        ),
        patch(  # replace download function by online availability check
            "prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp",
            new=prx.precise_corrections.sp3.sp3_file_discovery.check_online_availability,
        ),
    ):
        sp3_files = sp3.get_sp3_files(t_start, t_end, local_db)

    file_orb = sp3_files[0][0]
    file_clk = sp3_files[0][1]
    assert sp3.get_index_of_priority_from_filename(
        file_orb
    ) == sp3.get_index_of_priority_from_filename(file_clk)
