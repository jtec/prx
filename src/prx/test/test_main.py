import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from prx import util
from prx import main
from prx import user
from prx.precise_corrections.antex.antex_file_discovery import atx_file_database_folder
from prx.precise_corrections.sp3.sp3_file_discovery import sp3_file_database_folder
from prx.rinex_nav import nav_file_discovery

log = util.get_logger(__name__)


# This function sets up a temporary directory, copies a rinex observations file into that directory
# and returns its path. The @pytest.fixture annotation allows us to pass the function as an input
# to test functions. When running a test function, pytest will then first run this function, pass
# whatever is passed to `yield` to the test function, and run the code after `yield` after the test,
# even if the test crashes.
@pytest.fixture
def input_for_test_tlse(tmp_path_factory):
    test_directory = tmp_path_factory.mktemp("test_inputs")
    print(test_directory)
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    datasets_directory = Path(__file__).parent / "datasets"
    # Also provide ephemerides on disk so the test does not have to download them:
    compressed_crx = (
        datasets_directory / "TLSE_2023001/TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    )
    rnx_nav = datasets_directory / "TLSE_2023001/BRDC00IGS_R_20230010000_01D_MN.rnx.gz"
    for file in [compressed_crx, rnx_nav]:
        shutil.copy(
            file,
            test_directory / file.name,
        )

    # copy and uncompress precise correction files to local database
    os.makedirs(sp3_file_database_folder() / "2023/001/", exist_ok=True)
    sp3_orb = (
        datasets_directory / "TLSE_2023001/COD0MGXFIN_20230010000_01D_05M_ORB.SP3.gz"
    )
    sp3_orb_local = shutil.copy(
        sp3_orb, sp3_file_database_folder() / "2023/001" / sp3_orb.name
    )
    assert sp3_orb_local.exists()

    sp3_clk = (
        datasets_directory / "TLSE_2023001/COD0MGXFIN_20230010000_01D_30S_CLK.CLK.gz"
    )
    sp3_clk_local = shutil.copy(
        sp3_clk, sp3_file_database_folder() / "2023/001" / sp3_clk.name
    )
    assert sp3_clk_local.exists()

    os.makedirs(atx_file_database_folder(), exist_ok=True)
    atx = datasets_directory / "igs20_2408_reduced_size.atx"
    atx_local = shutil.copy(atx, atx_file_database_folder() / atx.name)
    assert atx_local.exists()

    yield test_directory / compressed_crx.name
    shutil.rmtree(test_directory)


@pytest.fixture
def input_for_test_tlse_2024(tmp_path_factory):
    test_directory = tmp_path_factory.mktemp("test_inputs")
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    datasets_directory = Path(__file__).parent / "datasets"
    # Also provide ephemerides on disk so the test does not have to download them:
    compressed_compact_rinex_file = (
        datasets_directory
        / "TLSE00FRA_R_2024001"
        / "TLSE00FRA_R_20240011800_01H_30S_MO.crx.gz"
    )
    ephemerides_file = (
        datasets_directory / "TLSE00FRA_R_2024001/BRDC00IGS_R_20240010000_01D_MN.rnx.gz"
    )
    for file in [compressed_compact_rinex_file, ephemerides_file]:
        shutil.copy(
            file,
            test_directory / file.name,
        )

    yield test_directory / compressed_compact_rinex_file.name
    shutil.rmtree(test_directory)


@pytest.fixture
def input_for_test_nist(tmp_path_factory):
    test_directory = tmp_path_factory.mktemp("test_inputs")
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "NIST00USA_R_20230010100_05M_30S_MO.crx.gz"
    test_file = test_directory.joinpath(compressed_compact_rinex_file)
    shutil.copy(
        Path(__file__).parent
        / f"datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_file,
    )
    assert test_file.exists()
    # Also provide ephemerides so the test does not have to download them:
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.gz"
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{ephemerides_file}",
        test_file.parent.joinpath(ephemerides_file),
    )
    assert test_file.parent.joinpath(ephemerides_file).exists()

    yield test_file
    shutil.rmtree(test_file.parent)


@pytest.fixture
def input_for_test_with_first_epoch_at_midnight(tmp_path_factory):
    # Having a first epoch at midnight requires to have the NAV data from the previous day, because we are computing
    # the time of emission as (time of reception - pseudorange/celerity)
    test_directory = tmp_path_factory.mktemp("test_inputs")
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    filepath_to_obs_file = "TLSE00FRA_R_20230010000_30M_30S_GO.crx.gz"
    test_obs_file = test_directory.joinpath(filepath_to_obs_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{filepath_to_obs_file}",
        test_obs_file,
    )
    assert test_obs_file.exists()

    # nav data from same day
    shutil.copy(
        Path(__file__).parent
        / "datasets/TLSE_2023001/BRDC00IGS_R_20230010000_01D_MN.rnx.gz",
        test_directory.joinpath("BRDC00IGS_R_20230010000_01D_MN.rnx.gz"),
    )
    # nav data from previous day
    shutil.copy(
        Path(__file__).parent
        / "datasets/TLSE_2023001/BRDC00IGS_R_20223650000_01D_MN.rnx.gz",
        test_directory.joinpath("BRDC00IGS_R_20223650000_01D_MN.rnx.gz"),
    )

    yield {
        "obs_file": test_obs_file,
    }
    shutil.rmtree(test_directory)


def test_prx_command_line_call(input_for_test_tlse):
    test_file = input_for_test_tlse
    prx_path = util.prx_repository_root() / "src/prx/main.py"
    command = f"uv run python {prx_path} --observation_file_path {test_file}"
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    assert result.returncode == 0
    assert expected_prx_file.exists()


def test_prx_function_call(input_for_test_tlse):
    test_file = input_for_test_tlse
    main.process(observation_file_path=test_file, prx_level=2)
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()
    df = pd.read_csv(expected_prx_file, comment="#")
    assert not df.empty
    assert util.is_sorted(df.time_of_reception_in_receiver_time)
    # Elevation sanity check
    assert (
        df[(df.prn == 14) & (df.constellation == "C")].sat_elevation_deg - 34.86
    ).abs().max() < 0.3
    # In such as short period we expect at most one transition from one ephemeris to the next one
    for signal, group_df in df.groupby(["constellation", "prn", "rnx_obs_identifier"]):
        n_ephemeris_changes = sum(
            group_df.ephemeris_hash.astype(float).diff().dropna().abs() > 0
        )
        assert n_ephemeris_changes <= 1


def test_prx_lli_parsing(input_for_test_tlse):
    test_file = input_for_test_tlse
    main.process(observation_file_path=test_file)
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    df = pd.read_csv(expected_prx_file, comment="#")
    # LLI check - CS expected (LLI = 1)
    assert (
        df.loc[
            (df.time_of_reception_in_receiver_time == "2023-01-01 01:00:01.000000")
            & (df.rnx_obs_identifier == "7D")
            & (df.constellation == "C")
            & (df.prn == 30),
            "LLI",
        ]
        == 1
    ).all()
    assert (
        df.loc[
            (df.time_of_reception_in_receiver_time == "2023-01-01 01:00:07.000000")
            & (df.rnx_obs_identifier.isin(["2W", "1C"]))
            & (df.constellation == "G")
            & (df.prn == 17),
            "LLI",
        ]
        == 1
    ).all()
    # LLI check - no CS (LLI = 0)
    assert (
        df.loc[
            (df.time_of_reception_in_receiver_time == "2023-01-01 01:00:07.000000")
            & (df.rnx_obs_identifier == "2I")
            & (df.constellation == "C")
            & (df.prn == 5),
            "LLI",
        ]
        == 0
    ).all()
    # LLI check - no phase tracking (L_obs_cycles = NaN and LLI = NaN)
    assert df.loc[df.L_obs_cycles.isnull(), "LLI"].isnull().all()


def test_prx_function_call_for_obs_file_across_two_days(
    input_for_test_with_first_epoch_at_midnight,
):
    test_file = input_for_test_with_first_epoch_at_midnight["obs_file"]
    main.process(observation_file_path=test_file)
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()


def run_rinex_through_prx(rinex_obs_file: Path, prx_level: int = 2):
    main.process(observation_file_path=rinex_obs_file, prx_level=prx_level)
    expected_prx_file = Path(str(rinex_obs_file).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()
    records, metadata = user.parse_prx_csv_file(expected_prx_file)
    records = pd.read_csv(expected_prx_file, comment="#")
    assert not records.empty
    assert metadata
    records.sat_code_bias_m = records.sat_code_bias_m.fillna(0)
    records = records[records.C_obs_m.notna() & records.sat_pos_x_m.notna()]
    return records, metadata


def test_spp_lsq_nist(input_for_test_nist):
    df, metadata = run_rinex_through_prx(input_for_test_nist)
    assert (df.iono_delay_m > 0).all()
    assert (df.tropo_delay_m > 0).all()
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        df.time_of_reception_in_receiver_time
        == df.time_of_reception_in_receiver_time.min()
    ]
    for constellations_to_use in [
        (
            "G",
            "E",
            "C",
        ),
        ("G", "S"),
        ("G",),
        ("E",),
        ("C",),
        ("R",),
    ]:
        obs = df_first_epoch[df.constellation.isin(constellations_to_use)]
        pt_lsq = user.spp_pt_lsq(obs)
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        log.info(
            f"Using constellations: {constellations_to_use}, {len(obs.sv.unique())} SVs"
        )
        log.info(f"Position offset: {position_offset}")
        assert (
            np.max(np.abs(position_offset)) < 2e1
        )  # relaxed position offset (instead of 1e1)


def test_spp_lsq_tlse(input_for_test_tlse):
    df, metadata = run_rinex_through_prx(input_for_test_tlse)
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        (
            df.time_of_reception_in_receiver_time
            == df.time_of_reception_in_receiver_time.min()
        )
        & (df.sat_elevation_deg > 10)
    ]
    for constellations_to_use in [
        (
            "G",
            "E",
            "C",
        ),
        ("G", "S"),
        ("G",),
        ("E",),
        ("C",),
        ("R",),
    ]:
        obs = df_first_epoch[df.constellation.isin(constellations_to_use)]
        pt_lsq = user.spp_pt_lsq(obs)
        vt_lsq = user.spp_vt_lsq(obs, p_ecef_m=pt_lsq[0:3, :])
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        # Static receiver, so:
        velocity_offset = vt_lsq[0:3, :]
        log.info(
            f"Using constellations: {constellations_to_use}, {len(obs.sv.unique())} SVs"
        )
        log.info(f"Position offset: {position_offset}")
        log.info(f"Velocity offset: {velocity_offset}")
        assert np.max(np.abs(position_offset)) < 1e1
        assert np.max(np.abs(velocity_offset)) < 1e-1


def test_spp_lsq_tlse_single_freq(input_for_test_tlse):
    obs_filter = {"G": ["1C"], "E": ["1X"], "C": ["2I"], "R": ["1C"]}
    df, metadata = run_rinex_through_prx(input_for_test_tlse, prx_level=2)
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        (
            df.time_of_reception_in_receiver_time
            == df.time_of_reception_in_receiver_time.min()
        )
        & (df.sat_elevation_deg > 10)
    ]
    for constellations_to_use in [
        ("G",),
        ("E",),
        ("C",),
        ("R",),
        ("G", "E", "C", "R"),
    ]:
        query_filter = " or ".join(
            [
                f"((constellation == '{const}') and (rnx_obs_identifier in {signal}))"
                for const, signal in obs_filter.items()
                if const in constellations_to_use
            ]
        )
        obs = df_first_epoch.query(query_filter)
        pt_lsq = user.spp_pt_lsq(obs)
        vt_lsq = user.spp_vt_lsq(obs, p_ecef_m=pt_lsq[0:3, :])
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        # Static receiver, so:
        velocity_offset = vt_lsq[0:3, :]
        log.info(
            f"Using constellations: {constellations_to_use}, {len(obs.sv.unique())} SVs"
        )
        log.info(f"Position offset: {np.squeeze(position_offset)}")
        log.info(f"Velocity offset: {np.squeeze(velocity_offset)}")
        assert np.max(np.abs(position_offset)) < 3
        assert np.max(np.abs(velocity_offset)) < 3e-2


def test_spp_lsq_tlse_with_precise_corrections(input_for_test_tlse):
    """
    Use iono-free combinations considered by IGS conventions (CODE Analysis Center):
    | Constellation | Frequency pair             |
    | --------------|----------------------------|
    | GPS           | L1, L2                     |
    | GLONASS       | L1, L2 (slot‑dependent)    |
    | Galileo       | E1, E5a                    |
    | BeiDou        | B1I, B2I                   |
    | QZSS          [ L1, L2                     |
    | IRNSS         | Not standardized           |

    Ref:
    - https://www.bernese.unibe.ch/publist/2015/pres/EO_BSW2015_MGEX_clock_determination_at_CODE.pdf (slide 3)
    - https://www.mdpi.com/2072-4292/12/9/1415 (table 2)
    """
    obs_filter = {
        "G": ["1C", "2X"],
        "E": ["1X", "5X"],
        "C": ["2I", "7I", "7D"],  # some BDS3 sats use 7D instead of 7I
        "R": ["1C", "2C"],
    }
    df, metadata = run_rinex_through_prx(input_for_test_tlse, prx_level=3)
    df_first_epoch = df[
        (
            df.time_of_reception_in_receiver_time
            == df.time_of_reception_in_receiver_time.min()
        )
        & (df.sat_elevation_deg > 10)
    ]
    for constellations_to_use in [
        ("G",),
        ("E",),
        ("C",),
        ("R",),
        ("G", "E", "C", "R"),
    ]:
        current_obs_filter = {
            const: signal
            for const, signal in obs_filter.items()
            if const in constellations_to_use
        }
        query_filter = " or ".join(
            [
                f"((constellation == '{const}') and (rnx_obs_identifier in {signal}))"
                for const, signal in current_obs_filter.items()
            ]
        )
        obs = user.compute_iono_free_code_obs(
            df_first_epoch.query(query_filter), current_obs_filter
        )
        pt_lsq = user.spp_pt_lsq(obs)
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)

        log.info(
            f"Using constellations: {constellations_to_use}, {obs.pipe(lambda d: obs.constellation + obs.prn.astype(str)).nunique()} SVs"
        )
        log.info(f"Position offset: {np.squeeze(position_offset)}")
        assert np.max(np.abs(position_offset)) < 1e1


def test_spp_lsq_tlse_2024(input_for_test_tlse_2024):
    """
    Calculates the position offset twice.
    Once including a faulty satellite (G27) and once only with satellites with a health_flag of 0.

    Allows to assess the faulty satellite's impact on positioning accuracy
    """
    df, metadata = run_rinex_through_prx(input_for_test_tlse_2024)
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        (
            df.time_of_reception_in_receiver_time
            == df.time_of_reception_in_receiver_time.min()
        )
        & (df.sat_elevation_deg > 10)
    ]
    constellation_to_use = ["G"]

    # observation with every sat
    obs = df_first_epoch[df.constellation.isin(constellation_to_use)]
    pt_lsq = user.spp_pt_lsq(obs)
    position_offset = pt_lsq[0:3, :] - np.array(
        metadata["approximate_receiver_ecef_position_m"]
    ).reshape(-1, 1)

    log.info(
        f"Using constellations: {constellation_to_use}, {len(obs.sv.unique())} SVs"
    )
    log.info(f"Position offset: {position_offset}")

    # observation without G27, using health_flag
    obs_without_G27 = obs.loc[obs.health_flag == 0]  # keep only healthy satellites
    pt_lsq_without_G27 = user.spp_pt_lsq(obs_without_G27)
    position_offset_without_G27 = pt_lsq_without_G27[0:3, :] - np.array(
        metadata["approximate_receiver_ecef_position_m"]
    ).reshape(-1, 1)
    log.info(
        f"Using constellations: {constellation_to_use}, {len(obs.sv.unique())} SVs (healthy)"
    )
    log.info(f"Position offset: {position_offset_without_G27}")

    # Verification: the position error significantly decreases when excluding G27
    assert np.linalg.norm(position_offset_without_G27) < np.linalg.norm(position_offset)
    # Verification: the position error without G27 is within tolerance
    assert np.max(np.abs(position_offset_without_G27)) < 1e1


def test_spp_lsq_for_obs_file_across_two_days(
    input_for_test_with_first_epoch_at_midnight,
):
    df, metadata = run_rinex_through_prx(
        input_for_test_with_first_epoch_at_midnight["obs_file"],
        prx_level=2,
    )
    df_first_epoch = df[
        df.time_of_reception_in_receiver_time
        == df.time_of_reception_in_receiver_time.min()
    ]
    for constellations_to_use in [
        ("G",),
    ]:
        obs = df_first_epoch[df.constellation.isin(constellations_to_use)]
        pt_lsq = user.spp_pt_lsq(obs)
        vt_lsq = user.spp_vt_lsq(obs, p_ecef_m=pt_lsq[0:3, :])
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        # Static receiver, so:
        velocity_offset = vt_lsq[0:3, :]
        log.info(f"Position offset: {position_offset}")
        log.info(f"Velocity offset: {velocity_offset}")
        assert np.max(np.abs(position_offset)) < 1e1
        assert np.max(np.abs(velocity_offset)) < 1e-1


def test_prx_level_1(input_for_test_tlse):
    main.process(observation_file_path=input_for_test_tlse, prx_level=1)
    expected_prx_file = Path(str(input_for_test_tlse).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()

    # Read first line of file, containing meta-data
    metadata = user.parse_prx_csv_file_metadata(expected_prx_file)
    assert metadata["prx_level"] == 1

    # Read the CSV file
    df = pd.read_csv(expected_prx_file, comment="#")

    # Expected CSV column names
    expected_column_names = {
        "time_of_reception_in_receiver_time",
        "C_obs_m",
        "D_obs_hz",
        "L_obs_cycles",
        "S_obs_dBHz",
        "rnx_obs_identifier",
        "constellation",
        "prn",
        "carrier_frequency_hz",
        "health_flag",
        "LLI",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "relativistic_clock_effect_m",
        "sat_elevation_deg",
        "sat_azimuth_deg",
        "ephemeris_hash",
    }

    # Checking if all renamed parameters exist in the dataframe columns
    assert set(df.columns) == expected_column_names, (
        f"Additional columns in computed prx file: {set(df.columns).difference(expected_column_names)}"
    )


def test_prx_level_2(input_for_test_tlse):
    main.process(observation_file_path=input_for_test_tlse, prx_level=2)
    expected_prx_file = Path(str(input_for_test_tlse).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()

    # Read first line of file, containing meta-data
    metadata = user.parse_prx_csv_file_metadata(expected_prx_file)
    assert metadata["prx_level"] == 2

    # Read the CSV file
    df = pd.read_csv(expected_prx_file, comment="#")

    # Expected CSV column names
    expected_column_names = {
        "time_of_reception_in_receiver_time",
        "C_obs_m",
        "D_obs_hz",
        "L_obs_cycles",
        "S_obs_dBHz",
        "rnx_obs_identifier",
        "constellation",
        "prn",
        "carrier_frequency_hz",
        "health_flag",
        "LLI",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "relativistic_clock_effect_m",
        "sagnac_effect_m",
        "tropo_delay_m",
        "sat_code_bias_m",
        "iono_delay_m",
        "sat_elevation_deg",
        "sat_azimuth_deg",
        "ephemeris_hash",
    }

    # Checking if all renamed parameters exist in the dataframe columns
    assert set(df.columns) == expected_column_names, (
        f"Additional columns in computed prx file: {set(df.columns).difference(expected_column_names)}"
    )


def test_prx_level_3(input_for_test_tlse):
    main.process(observation_file_path=input_for_test_tlse, prx_level=3)
    expected_prx_file = Path(str(input_for_test_tlse).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()

    # Read first line of file, containing meta-data
    metadata = user.parse_prx_csv_file_metadata(expected_prx_file)
    assert metadata["prx_level"] == 3

    # Read the CSV file
    df = pd.read_csv(expected_prx_file, comment="#")

    # Expected CSV column names
    expected_column_names = {
        "time_of_reception_in_receiver_time",
        "C_obs_m",
        "D_obs_hz",
        "L_obs_cycles",
        "S_obs_dBHz",
        "rnx_obs_identifier",
        "constellation",
        "prn",
        "carrier_frequency_hz",
        "health_flag",
        "LLI",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_pos_com_x_m",
        "sat_pos_com_y_m",
        "sat_pos_com_z_m",
        "pco_sat_x_m",
        "pco_sat_y_m",
        "pco_sat_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "relativistic_clock_effect_m",
        "sagnac_effect_m",
        "tropo_delay_m",
        "iono_delay_m",
        "sat_elevation_deg",
        "sat_azimuth_deg",
        "sat_code_bias_m",
        "sat_carrier_bias_m",
    }

    # Checking if all renamed parameters exist in the dataframe columns
    assert set(df.columns) == expected_column_names, (
        f"Additional columns in computed prx file: {set(df.columns).difference(expected_column_names)}"
    )


def test_function_call_with_alternative_tropo(input_for_test_tlse):
    expected_prx_file_saas = main.process(
        observation_file_path=input_for_test_tlse,
        prx_level=2,
        model_tropo="saastamoinen",
    )
    assert expected_prx_file_saas.exists()
    df_saas = pd.read_csv(expected_prx_file_saas, comment="#")

    expected_prx_file_unb3m = main.process(
        observation_file_path=input_for_test_tlse, prx_level=2, model_tropo="unb3m"
    )
    assert expected_prx_file_unb3m.exists()
    df_unb3 = pd.read_csv(expected_prx_file_unb3m, comment="#")

    expected_prx_file_default = main.process(
        observation_file_path=input_for_test_tlse,
        prx_level=2,
    )
    assert expected_prx_file_default.exists()
    df_default = pd.read_csv(expected_prx_file_default, comment="#")

    # Verify that the tropo delays in each dataframe are equal or close, i.e. that the default model is Saastamoinen
    np.testing.assert_array_equal(df_default.tropo_delay_m, df_saas.tropo_delay_m)
    # Differences are large especially for low elevation. The comparison is done after applying an elevation mask
    np.testing.assert_allclose(
        df_unb3.tropo_delay_m.loc[df_default.sat_elevation_deg > 10],
        df_saas.tropo_delay_m.loc[df_default.sat_elevation_deg > 10],
        atol=1,  # 1 m difference
    )

    assert True


def test_bootstrap_coarse_receiver_position(input_for_test_tlse):
    # Download RNX NAV file
    aux_files = nav_file_discovery.discover_or_download_auxiliary_files(
        input_for_test_tlse
    )

    # Compute solution from first epoch with more than 4 GPS L1C/A observations, without minimum corrections
    solution = user.bootstrap_coarse_receiver_position(
        input_for_test_tlse, aux_files["broadcast_ephemerides"]
    )

    # Retrieve approximate position from RINEX OBS header
    metadata = main.build_metadata(
        {
            "obs_file": input_for_test_tlse,
            "nav_file": aux_files["broadcast_ephemerides"],
        }
    )

    # Compute position error
    position_error_ecef_m = (
        solution[0:3] - metadata["approximate_receiver_ecef_position_m"]
    )

    # Verify position error magnitude
    # A tolerance of 100 m is accepted due to missing corrections (no atmospheric corrections, Sagnac effect)
    # and the use of only one constellation and signal
    assert np.linalg.norm(position_error_ecef_m) < 100
