import shutil
import polars as pl

import pytest

from prx import util, constants
import prx.precise_corrections.bia.bia_processing as bia


@pytest.fixture(scope="session")
def input_for_test(tmp_path_factory):
    test_directory = tmp_path_factory.mktemp("test_inputs")
    test_files = {
        "bia": test_directory / "COD0OPSFIN_20230010000_01D_01D_OSB.BIA.gz",
    }
    for key, test_file_path in test_files.items():
        shutil.copy(
            util.prx_src_directory().joinpath(
                "test", "datasets", "TLSE_2023001", test_file_path.name
            ),
            test_file_path,
        )
        assert test_file_path.exists()
    yield test_files
    shutil.rmtree(test_directory)


def test_bia_parsing(input_for_test):
    bia_df = bia.parse_bia_file(input_for_test["bia"])
    # manual check in file
    assert (
        bia_df.filter(pl.col("sat_id") == "G01").get_column("C1C").item()
        == 9.2018 / constants.cNanoSecondsPerSecond * constants.cGpsSpeedOfLight_mps
    )
    assert (
        bia_df.filter(pl.col("sat_id") == "G01").get_column("C1W").item()
        == 10.5943 / constants.cNanoSecondsPerSecond * constants.cGpsSpeedOfLight_mps
    )
