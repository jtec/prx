from pathlib import Path
import pytest
import shutil
import os
import pandas as pd

from prx.helpers import repair_with_gfzrnx, obs_dataset_to_obs_dataframe
from prx.rinex_obs.parser import parse as prx_obs_parse
from prx import converters
import georinex


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
    repair_with_gfzrnx(file)
    prx_output = prx_obs_parse(file).sort_values(by=["time", "sv", "obs_type"])
    # remove lli from prx outputs
    drop_lli = [
        obs_type for obs_type in prx_output.obs_type.unique() if "lli" in obs_type
    ]
    prx_output = prx_output.loc[~prx_output.obs_type.isin(drop_lli), :]
    prx_output = prx_output.reset_index(drop=True)

    georinex_output = (
        obs_dataset_to_obs_dataframe(georinex.load(file))
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    assert georinex_output.equals(prx_output)


def test_compare_to_georinex_with_lli(input_for_test_tlse):
    file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx_slice_0.24h.rnx.gz"
    )
    repair_with_gfzrnx(file)
    prx_output = (
        prx_obs_parse(file)
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )

    georinex_output = (
        obs_dataset_to_obs_dataframe(georinex.load(file, useindicators=True))
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )

    drop_ssi = [
        obs_type for obs_type in georinex_output.obs_type.unique() if "ssi" in obs_type
    ]
    georinex_output = georinex_output.loc[~georinex_output.obs_type.isin(drop_ssi), :]

    print("prx lli list:")
    print([type for type in prx_output.obs_type.unique() if len(type) > 3])
    print("georinex lli list:")
    print([type for type in georinex_output.obs_type.unique() if len(type) > 3])

    geo_lli_list = [type for type in georinex_output.obs_type.unique() if "lli" in type]
    for lli in geo_lli_list:
        geo_lli = georinex_output.loc[georinex_output.obs_type == lli, :].reset_index(
            drop=True
        )
        prx_lli = prx_output.loc[prx_output.obs_type == lli, :].reset_index(drop=True)
        assert geo_lli.equals(prx_lli)

    # Negative assertion: I suspect a bug in georinex, that is not parsing all lli. If both prx and georinex parses
    # the same lli, this assertion will fail and we may want to update this test to remove it.
    prx_lli_list = [type for type in prx_output.obs_type.unique() if "lli" in type]
    assert (
        set(geo_lli_list) != set(prx_lli_list)
    ), "prx and georinex parse the same LLI! If georinex has been updated, you may want to remove this asssertion!!"
    for lli in prx_lli_list:
        if lli not in geo_lli_list:
            prx_lli = prx_output.loc[prx_output.obs_type == lli, :].reset_index(
                drop=True
            )
            geo_lli = georinex_output.loc[
                georinex_output.obs_type == lli, :
            ].reset_index(drop=True)
            print(f"prx      has parsed {len(prx_lli)} CSs for obs {lli[0:3]}")
            print(f"georinex has parsed {len(geo_lli)} CSs for obs {lli[0:3]}")


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
