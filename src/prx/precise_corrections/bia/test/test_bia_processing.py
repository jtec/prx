import shutil

import numpy as np
import pandas as pd
import polars as pl

import pytest

from prx import util, constants
import prx.precise_corrections.bia.bia_processing as bia


@pytest.fixture(scope="session")
def input_for_test(tmp_path_factory):
    temp_directory = tmp_path_factory.mktemp("test_inputs")
    src_dir = util.prx_src_directory()
    test_dir = src_dir.joinpath("test", "datasets")
    test_files = {
        "cod": temp_directory / "COD0MGXFIN_20230010000_01D_01D_OSB.BIA.gz",
        "gfz": temp_directory / "GFZ0MGXRAP_20230010000_01D_01D_OSB.BIA.gz",
        "grg": temp_directory / "GRG0MGXFIN_20240010000_01D_01D_OSB.BIA.gz",
        "wum": temp_directory / "WUM0MGXRAP_20240010000_01D_01D_OSB.BIA.gz",
    }
    shutil.copy(
        test_dir.joinpath("TLSE_2023001", test_files["cod"].name), test_files["cod"]
    )
    shutil.copy(
        test_dir.joinpath("TLSE_2023001", test_files["gfz"].name), test_files["gfz"]
    )
    shutil.copy(
        test_dir.joinpath("TLSE00FRA_R_2024001", test_files["grg"].name),
        test_files["grg"],
    )
    shutil.copy(
        test_dir.joinpath("TLSE00FRA_R_2024001", test_files["wum"].name),
        test_files["wum"],
    )
    for test_file_path in test_files.values():
        assert test_file_path.exists()
    yield test_files
    shutil.rmtree(temp_directory)


def test_bia_parsing(input_for_test):
    bia_df = bia.parse_bia_file(input_for_test["cod"])
    # manual check in file
    assert (
        bia_df.filter((pl.col("sat_id") == "G01") & (pl.col("obs_id") == "C1C")).item(
            0, "sat_hw_bias_m"
        )
        == -1.4198 / constants.cNanoSecondsPerSecond * constants.cGpsSpeedOfLight_mps
    )
    assert (
        bia_df.filter((pl.col("sat_id") == "G01") & (pl.col("obs_id") == "C1W")).item(
            0, "sat_hw_bias_m"
        )
        == -0.0000 / constants.cNanoSecondsPerSecond * constants.cGpsSpeedOfLight_mps
    )


def test_iono_free_code_bias(input_for_test):
    """
    The iono-free combination of the satellite hardware code biases should be close to 0
    (or close to a constant across all satellites from the same constellation)
    """
    threshold = 1e-4
    obs_used_for_if = {
        "cod": {
            "G": ["C1W", "C2W"],
            "E": ["C1C", "C5Q"],
            "C": ["C2I", "C6I"],
            "J": ["C1C", "C2L"],
        },
        "gfz": {
            "G": ["C1W", "C2W"],
            "E": ["C1C", "C5Q"],
            "C": ["C2I", "C6I"],
            # "J": ["C1C", "C2L"],
        },
        "grg": {
            "G": ["L1W", "L2W"],
            "E": ["L1X", "L5X"],
            "C": ["L2I", "L6I"],
        },
        "wum": {
            "G": ["C1W", "C2W"],
            "R": ["C1P", "C2P"],
            "E": ["C1C", "C5Q"],
            "C": ["C2I", "C6I"],
            # "J": ["C1X", "C2X"],
        },
    }
    for ac in obs_used_for_if:
        print(f"=== Testing analysis center {ac} ===")
        bia_df = (
            bia.parse_bia_file(input_for_test[ac])
            .filter(pl.col("start") == pl.col("start").unique().min())
            .pivot(on="obs_id", index="sat_id", values="sat_hw_bias_m")
        )
        for const in obs_used_for_if[ac]:
            # Iono-free combination of COD for GPS uses C1W and C2W
            freq_id1 = obs_used_for_if[ac][const][0][1]
            freq_id2 = obs_used_for_if[ac][const][1][1]
            f1 = constants.carrier_frequencies_hz()[const]["L" + freq_id1][1]
            f2 = constants.carrier_frequencies_hz()[const]["L" + freq_id2][1]
            bia_if_gps = bia_df.filter(pl.col("sat_id").str.starts_with(const)).select(
                pl.col("sat_id"),
                (
                    (
                        f1**2 * pl.col(obs_used_for_if[ac][const][0])
                        - f2**2 * pl.col(obs_used_for_if[ac][const][1])
                    )
                    / (f1**2 - f2**2)
                ).alias("if_code_bias"),
            )
            print(
                f"Maximum iono-free code bias for {const}: {bia_if_gps['if_code_bias'].max()} m"
            )
            assert (bia_if_gps["if_code_bias"] < threshold).all()


def test_retrieve_satellite_biases(input_for_test):
    # choose a query where 2 different biases exists for the same sat/sig at different times
    query = pd.DataFrame(
        {
            "sv": ["G03", "G03"],
            "signal": ["C5Q", "C5Q"],
            "query_time_isagpst": [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 00:15:00"),
            ],
        }
    )

    sat_bias = bia.compute_sat_hw_biases(
        query, bia.parse_bia_file(input_for_test["wum"])
    )

    assert "sat_code_bias_m" in sat_bias.columns
    assert "sat_carrier_bias_m" in sat_bias.columns
    # manual check in bia file
    assert sat_bias[
        ["sat_code_bias_m", "sat_carrier_bias_m"]
    ].to_numpy() == pytest.approx(
        np.array([[1.84671646, 0.38192426], [1.84671646, 0.39412953]])
    )
