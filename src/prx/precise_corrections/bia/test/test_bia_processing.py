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
        bia_df.filter((pl.col("sat_id") == "G01") & (pl.col("obs_id") == "C1C")).item(
            0, "sat_hw_bias_m"
        )
        == 9.2018 / constants.cNanoSecondsPerSecond * constants.cGpsSpeedOfLight_mps
    )
    assert (
        bia_df.filter((pl.col("sat_id") == "G01") & (pl.col("obs_id") == "C1W")).item(
            0, "sat_hw_bias_m"
        )
        == 10.5943 / constants.cNanoSecondsPerSecond * constants.cGpsSpeedOfLight_mps
    )


def test_iono_free_code_bias(input_for_test):
    """
    The iono-free combination of the satellite hardware code biases should be close to 0
    (or close to a constant across all satellites from the same constellation)
    """
    threshold = 1e-4
    bia_df = bia.parse_bia_file(input_for_test["bia"]).pivot(
        on="obs_id", index="sat_id", values="sat_hw_bias_m"
    )
    # Iono-free combination of COD for GPS uses C1W and C2W
    f1 = constants.carrier_frequencies_hz()["G"]["L1"][1]
    f2 = constants.carrier_frequencies_hz()["G"]["L2"][1]
    bia_if_gps = bia_df.filter(pl.col("sat_id").str.starts_with("G")).select(
        pl.col("sat_id"),
        ((f1**2 * pl.col("C1W") - f2**2 * pl.col("C2W")) / (f1**2 - f2**2)).alias(
            "if_code_bias"
        ),
    )
    print(f"Maximum iono-free code bias for GPS: {bia_if_gps['if_code_bias'].max()} m")
    assert (bia_if_gps["if_code_bias"] < threshold).all()
    # Iono-free combination of COD for Galileo uses C1W and C2W
    f1 = constants.carrier_frequencies_hz()["E"]["L1"][1]
    f2 = constants.carrier_frequencies_hz()["E"]["L5"][1]
    bia_if_gal = bia_df.filter(pl.col("sat_id").str.starts_with("E")).select(
        pl.col("sat_id"),
        ((f1**2 * pl.col("C1C") - f2**2 * pl.col("C5Q")) / (f1**2 - f2**2)).alias(
            "if_code_bias"
        ),
    )
    print(f"Maximum iono-free code bias for GAL: {bia_if_gal['if_code_bias'].max()} m")
    assert (bia_if_gal["if_code_bias"] < threshold).all()
