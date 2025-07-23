import os
from pathlib import Path
import shutil
import georinex
import pytest

import pandas as pd
from unittest.mock import patch 
import prx
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
    # local_database = sp3.sp3_file_folder(pd.Timestamp('2023-01-01'), Path('src/prx/precise_corrections/sp3/test/datasets'))

    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_obs_file.name}",
        test_obs_file,
    )
    assert test_obs_file.exists()
    test_date = pd.Timestamp("2023-01-01")
    sp3_subfolder = sp3.sp3_file_folder(test_date, test_directory)
    test_sp3_file = sp3_subfolder / "COD0OPSRAP_20230010000_01D_05M_ORB.SP3.gz"
    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/precise_corrections/sp3/test/datasets/2023/001/{test_sp3_file.name}",
        test_sp3_file,
    )
    assert test_sp3_file.exists()

    yield {"test_obs_file": test_obs_file, "test_sp3_file" : test_sp3_file}
    shutil.rmtree(test_directory)

def test_get_index_of_priority_from_filename(set_up_test):
    obs_file = set_up_test["test_obs_file"]
    sp3_filename = set_up_test["test_sp3_file"].name
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    index_in_priority = sp3.get_index_of_priority_from_filename(sp3_filename, t_start)

    assert index_in_priority == 6

def test_get_sp3_file(set_up_test):
    """
    Assert that prx will download the first SP3 file in the priority list if not present.
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    t_end = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])

    db_folder = set_up_test["test_obs_file"].parent

    sp3.get_sp3_file(t_start, t_end, db_folder)

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(t_start)
    expected_priority = (
        sp3.priority_before_gps_week_2237
        if gps_week < 2238
        else sp3.priority_since_gps_week_2238
    )
    sp3_filename, clk_filename = sp3.build_sp3_filename(gps_week, dow, t_start, expected_priority[0], expected_priority)
    sp3_filename = str(Path(sp3_filename).with_suffix(""))
    clk_filename = str(Path(clk_filename).with_suffix(""))

    downloaded_sp3 = sp3.get_local_sp3(t_start, sp3_filename, db_folder)
    downloaded_clk = sp3.get_local_sp3(t_start, clk_filename, db_folder)
    assert downloaded_sp3 is not None
    assert downloaded_clk is not None 
    assert downloaded_sp3.exists()
    assert downloaded_clk.exists()

def test_priority_since_2238():
    '''
    This test verifies that the function `get_sp3_file` correctly iterates through the
    full priority list defined in `priority_since_gps_week_2238` when no local SP3 file
    is found and all FTP download attempts fail.

    The test uses patches to:
    - Mock `build_sp3_filename` to track the exact sequence of priority values used.
    - Force `get_local_sp3` to return None, simulating the absence of a local file.
    - Force `try_downloading_sp3_ftp` to return None, simulating a failed download.

    It asserts that `build_sp3_filename` is called with the correct arguments and in
    the expected order, ensuring the priority logic is followed as intended. 
    '''
    test_date = pd.Timestamp("2023-12-15 12:00:00", tz="UTC")
    priority = sp3.priority_since_gps_week_2238

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(test_date)

    with patch("prx.precise_corrections.sp3.sp3_file_discovery.build_sp3_filename") as mock_build_name, \
         patch("prx.precise_corrections.sp3.sp3_file_discovery.get_local_sp3", return_value=None), \
         patch("prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp", return_value=None):

        sp3.get_sp3_file(test_date, test_date)

        expected_calls = [
            ((gps_week, dow, test_date, priority[i], priority),) for i in range (0, len(priority))
        ]
        actual_calls = mock_build_name.call_args_list
        print(mock_build_name.call_args_list)
        assert actual_calls == expected_calls, f"Expected call order:\n{expected_calls}\nbut got:\n{actual_calls}"
    
def test_priority_before_2237():
    '''
    This test verifies that the function `get_sp3_file` correctly iterates through the
    full priority list defined in `priority_before_gps_week_2237` when no local SP3 file
    is found and all FTP download attempts fail.

    The test uses patches to:
    - Mock `build_sp3_filename` to track the exact sequence of priority values used.
    - Force `get_local_sp3` to return None, simulating the absence of a local file.
    - Force `try_downloading_sp3_ftp` to return None, simulating a failed download.

    It asserts that `build_sp3_filename` is called with the correct arguments and in
    the expected order, ensuring the priority logic is followed as intended. 
    '''
    test_date = pd.Timestamp("2021-12-15 12:00:00", tz="UTC")
    priority = sp3.priority_before_gps_week_2237

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(test_date)

    with patch("prx.precise_corrections.sp3.sp3_file_discovery.build_sp3_filename") as mock_build_name, \
         patch("prx.precise_corrections.sp3.sp3_file_discovery.get_local_sp3", return_value=None), \
         patch("prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp", return_value=None):

        sp3.get_sp3_file(test_date, test_date)

        expected_calls = [
            ((gps_week, dow, test_date, priority[i], priority),) for i in range (0, len(priority))
        ]
        actual_calls = mock_build_name.call_args_list
        print(mock_build_name.call_args_list)
        assert actual_calls == expected_calls, f"Expected call order:\n{expected_calls}\nbut got:\n{actual_calls}"

def test_priority_local_file_found_midway_since_2238(set_up_test):
    '''
    This test verifies that `get_sp3_file` iterates through the priority list in order,
    attempting to retrieve each SP3 file (locally or via FTP) until it finds a valid one.

    In this scenario, all files before a given priority fail (local = None, FTP = None),
    and a local file is found only at a specific index in the list.

    The test asserts that:
    - All previous priority options were attempted (get_local + download).
    - The function stops once the local file is found.
    - The priority logic is respected in call order.
    '''
    sp3_filename = set_up_test['test_sp3_file'].name
    obs_file = set_up_test["test_obs_file"]

    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    t_end = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])

    db_folder = set_up_test["test_obs_file"].parent

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(t_start)
    expected_priority = (
        sp3.priority_before_gps_week_2237
        if gps_week < 2238
        else sp3.priority_since_gps_week_2238
    )
    index_found_file_priority = sp3.get_index_of_priority_from_filename(sp3_filename,t_start)
    calls = 0

    for i in range(index_found_file_priority+1):
        with patch("prx.precise_corrections.sp3.sp3_file_discovery.priority_since_gps_week_2238", expected_priority[:i+1]), \
             patch("prx.precise_corrections.sp3.sp3_file_discovery.priority_before_gps_week_2237", expected_priority[:i+1]):
            for j in range(i+1):
                if i != j and i != index_found_file_priority:
                    with patch("prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp", return_value = None):
                        sp3_file = sp3.get_sp3_file(t_start, t_end, db_folder) 
                    assert sp3_file is None
                elif i == index_found_file_priority :
                    with patch("prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp", return_value = None):
                        sp3_file = sp3.get_sp3_file(t_start, t_end, db_folder) 
                    assert sp3_file is not None
                else : 
                    sp3_file = sp3.get_sp3_file(t_start, t_end, db_folder)
                    calls +=1
                    assert sp3_file is not None
                    if sp3_file.exists() :
                        sp3_file.unlink() 
    assert calls == index_found_file_priority

def test_download_all(set_up_test):
    """
    At least one file was successfully downloaded 
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    t_end = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])

    db_folder = set_up_test["test_obs_file"].parent

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(t_start)
    if gps_week < 2238 :
        priority = sp3.priority_before_gps_week_2237
    else : priority = sp3.priority_since_gps_week_2238

    downloaded_sp3_files = []
    downloaded_clk_files = []
    for p in priority:
        sp3_filename, clk_filename = sp3.build_sp3_filename(gps_week, dow, t_start, p, priority)
        downloaded_sp3 = sp3.try_downloading_sp3_ftp(gps_week, t_start, db_folder, sp3_filename)
        downloaded_clk = sp3.try_downloading_sp3_ftp(gps_week, t_start, db_folder, clk_filename)
        
        if downloaded_sp3 is not None :
            downloaded_sp3_files.append(downloaded_sp3)
        if downloaded_clk is not None :
            downloaded_clk_files.append(downloaded_clk)
        
    assert downloaded_sp3_files
    assert downloaded_clk_files




