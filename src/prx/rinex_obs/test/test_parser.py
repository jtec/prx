from pathlib import Path
import pytest
import shutil
import os
import pandas as pd
from prx.rinex_obs.parser import parse as prx_obs_parse
from prx import converters, helpers


@pytest.fixture
def input_for_test_tlse():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    datasets_directory = Path(__file__).parent / "datasets"
    compressed_compact_rinex_file = (
        datasets_directory / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx.gz"
    )
    shutil.copy(
        compressed_compact_rinex_file,
        test_directory / compressed_compact_rinex_file.name,
    )

    yield test_directory / compressed_compact_rinex_file.name
    shutil.rmtree(test_directory)


def test_compare_to_georinex():
    file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx_slice_0.24h.rnx.gz"
    )
    prx_output = (
        prx_obs_parse(file)
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    georinex_output = (
        helpers.parse_rinex_obs_file(file)
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    assert georinex_output.equals(prx_output)


def test_basic_check_on_rinex(input_for_test_tlse):
    prx_output = prx_obs_parse(input_for_test_tlse)
    test_cases = [
        # (timestamp, svid, obs_type, expected_value)
        (
            pd.Timestamp("2022-01-01 00:00:00"),
            "G07",
            "C1C",
            24998288.344,
        ),  # field number 1
        (
            pd.Timestamp("2022-01-01 00:00:30"),
            "G32",
            "C2X",
            24604446.773,
        ),  # field number 3
        (pd.Timestamp("2022-01-01 04:03:00"), "R18", "S2P", 49.600),  # field number 16
        (
            pd.Timestamp("2022-01-01 10:32:30"),
            "C10",
            "C2I",
            39217569.273,
        ),  # field number 1
        (
            pd.Timestamp("2022-01-01 14:43:00"),
            "C05",
            "C6I",
            40095505.355,
        ),  # field number 2
        (
            pd.Timestamp("2022-01-01 16:04:00"),
            "E02",
            "L1X",
            133855042.539,
        ),  # field number 9
        (pd.Timestamp("2022-01-01 17:53:00"), "E36", "D5X", 1871.932),  # field number 6
        (
            pd.Timestamp("2022-01-01 18:57:00"),
            "S36",
            "C1C",
            37796707.203,
        ),  # field number 1
        (pd.Timestamp("2022-01-01 18:57:30"), "S36", "S5I", 46.600),  # field number 8
    ]
    for timestamp, svid, obs_type, value in test_cases:
        assert (
            prx_output[
                (prx_output["time"] == timestamp)
                & (prx_output["sv"] == svid)
                & (prx_output["obs_type"] == obs_type)
            ].obs_value.values[0]
            == value
        )
    test_cases_missing = [
        (pd.Timestamp("2022-01-01 00:00:00"), "C01", "C2I", None),  # missing satellite
        (pd.Timestamp("2022-01-01 00:00:00"), "C01", "C5X", None),  # missing obs type
        (pd.Timestamp("2022-01-01 00:00:00"), "G07", "C2W", None),  # missing obs
        (pd.Timestamp("2023-01-01 00:00:30"), "G32", "C2X", None),  # missing epoch
    ]
    for timestamp, svid, obs_type, value in test_cases_missing:
        assert (
            len(
                prx_output[
                    (prx_output["time"] == timestamp)
                    & (prx_output["sv"] == svid)
                    & (prx_output["obs_type"] == obs_type)
                ].obs_value
            )
            == 0
        )
