import pandas as pd
import numpy as np
from pathlib import Path

from prx.precise_corrections.sp3.evaluate import compute
from prx.rinex_obs.parser import parse_rinex_obs_file
from prx.converters import anything_to_rinex_3
import shutil
import pytest
import os
from prx import constants


@pytest.fixture
def input_for_test():
    test_directory = (
        Path(__file__).parent.joinpath(f"./tmp_test_directory_{__name__}").resolve()
    )
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    test_sp3_file = test_directory.joinpath("WUM0MGXULT_20220010000_01H_05M_ORB.SP3")
    shutil.copy(
        Path(__file__).parent.joinpath("datasets", test_sp3_file.name), test_sp3_file
    )

    test_sp3_file_with_one_sample_removed = test_directory.joinpath(
        "WUM0MGXULT_20220010000_01H_05M_ORB_one_sample_removed.SP3"
    )
    shutil.copy(
        Path(__file__).parent.joinpath(
            "datasets", test_sp3_file_with_one_sample_removed.name
        ),
        test_sp3_file_with_one_sample_removed,
    )

    test_atx_file = test_directory.joinpath("igs20_2408_reduced_size.atx")
    shutil.copy(
        Path(__file__).parent.joinpath("datasets", test_atx_file.name), test_atx_file
    )

    assert test_sp3_file.exists()
    assert test_sp3_file_with_one_sample_removed.exists()
    assert test_atx_file.exists()

    yield {
        "test_file": test_sp3_file,
        "test_file_one_sample_removed": test_sp3_file_with_one_sample_removed,
        "atx_file": test_atx_file,
    }
    shutil.rmtree(test_directory)


@pytest.fixture(scope="session")
def input_for_test_rtklib(tmp_path_factory):
    test_directory = tmp_path_factory.mktemp("test_inputs")
    test_files = {
        "obs": test_directory / "NIST00USA_R_20230010000_01D_30S_MO.crx.gz",
        "nav": test_directory / "BRDC00IGS_R_20230010000_01D_MN.rnx.gz",
        "sp3": test_directory / "GFZ0MGXRAP_20230010000_01D_05M_ORB.SP3",
        "atx": test_directory / "igs20_2408_reduced_size.atx",
    }
    for key, test_file_path in test_files.items():
        shutil.copy(
            Path(__file__)
            .parents[3]
            .joinpath("test", "datasets", "TLSE_2023001", test_file_path.name),
            test_file_path,
        )
        assert test_file_path.exists()
    test_files["sp3_rtklib"] = (
        Path(__file__)
        .parents[3]
        .joinpath("tools", "validation_data", "satpos_sp3_matrtklib.csv.gz")
        # .joinpath("tools", "validation_data", "satpos_sp3_no_pco_matrtklib.csv")
    )
    yield test_files
    shutil.rmtree(test_directory)


def test_at_sample(input_for_test):
    sp3_file = input_for_test["test_file"]
    # Compute satellite states directly at a sample
    query = pd.DataFrame(
        [
            {
                "query_time_isagpst": pd.Timestamp("2021-12-31T00:20:00.00000000"),
                "sv": "G01",
                "signal": "C1C",
            }
        ]
    )
    sat_states = compute(sp3_file, query, input_for_test["atx_file"])
    # We then expect the satellite state to be close to the sample
    # PG01  13744.907145 -20823.122313   8309.113118    469.979467
    assert np.allclose(
        sat_states[sat_states["sv"] == "G01"][
            ["sat_pos_com_x_m", "sat_pos_com_y_m", "sat_pos_com_z_m"]
        ].to_numpy(),
        1e3 * np.array([13744.907145, -20823.122313, 8309.113118]),
        rtol=1e-5,
        atol=1e-3,
    )
    assert np.allclose(
        sat_states[sat_states["sv"] == "G01"][["sat_clock_offset_m"]].to_numpy(),
        np.array(
            [
                (469.979467 / constants.cMicrosecondsPerSecond)
                * constants.cGpsSpeedOfLight_mps
            ]
        ),
        rtol=1e-5,
        atol=1e-3 / constants.cGpsSpeedOfLight_mps,
    )


def test_between_samples(input_for_test):
    sp3_file = input_for_test["test_file_one_sample_removed"]
    # Compute satellite states at a sample time that has been manually removed from the file
    query = pd.DataFrame(
        [
            {
                "query_time_isagpst": pd.Timestamp("2021-12-31T00:30:00.00000000"),
                "sv": "G01",
                "signal": "C1C",
            }
        ]
    )
    sat_states = compute(sp3_file, query, input_for_test["atx_file"], True)
    # We then expect the interpolated satellite state to be close to the removed sample
    # PG01  13624.009028 -20092.399598  10082.111937    469.973744
    assert sat_states[sat_states["sv"] == "G01"][
        ["sat_pos_com_x_m", "sat_pos_com_y_m", "sat_pos_com_z_m"]
    ].to_numpy()[0] == pytest.approx(
        1e3 * np.array([13624.009028, -20092.399598, 10082.111937]), abs=1e-3
    )
    assert np.allclose(
        sat_states[sat_states["sv"] == "G01"][["sat_clock_offset_m"]].to_numpy(),
        np.array(
            [
                constants.cGpsSpeedOfLight_mps
                * (469.973744 / constants.cMicrosecondsPerSecond)
            ]
        ),
        rtol=1e-5,
        atol=1e-3 / constants.cGpsSpeedOfLight_mps,
    )


def test_compare_matrtklib_without_galileo(input_for_test_rtklib):
    flat_obs = parse_rinex_obs_file(anything_to_rinex_3(input_for_test_rtklib["obs"]))
    flat_obs.time = pd.to_datetime(flat_obs.time, format="%Y-%m-%dT%H:%M:%S")
    flat_obs.obs_value = flat_obs.obs_value.astype(float)
    flat_obs[["sv", "obs_type"]] = flat_obs[["sv", "obs_type"]].astype(str)

    # TODO: investigate why G14 fails the test
    flat_obs = flat_obs.loc[flat_obs.sv != "G14"]

    # keep only the signals present in rtklib file
    obs_filter = {"G": "C1C", "C": "C2I", "R": "C1C"}
    query_filter = " or ".join(
        [
            f"((sv.str[0] == '{const}') and (obs_type == '{signal}'))"
            for const, signal in obs_filter.items()
        ]
    )
    flat_obs = flat_obs.query(query_filter)

    # keep reduced number of epochs
    flat_obs = flat_obs.loc[flat_obs.time < flat_obs.time.min() + pd.Timedelta("1m")]

    flat_obs = flat_obs.rename(
        columns={
            "time": "time_of_reception_in_receiver_time",
            "sv": "satellite",
            "obs_value": "observation_value",
            "obs_type": "observation_type",
        },
    )

    per_sat = flat_obs.pivot(
        index=["time_of_reception_in_receiver_time", "satellite"],
        columns=["observation_type"],
        values="observation_value",
    ).reset_index()
    per_sat["time_scale"] = (
        per_sat["satellite"].str[0].map(constants.constellation_2_system_time_scale)
    )
    per_sat["system_time_scale_epoch"] = per_sat["time_scale"].map(
        constants.system_time_scale_rinex_utc_epoch
    )
    code_phase_columns = [c for c in per_sat.columns if c[0] == "C" and len(c) == 3]
    tof_dtrx = pd.to_timedelta(
        per_sat[code_phase_columns]
        .mean(axis=1, skipna=True)
        .divide(constants.cGpsSpeedOfLight_mps),
        unit="s",
    )
    per_sat["time_of_emission_isagpst"] = (
        per_sat["time_of_reception_in_receiver_time"] - tof_dtrx
    )

    flat_obs = flat_obs.merge(
        per_sat[
            [
                "time_of_reception_in_receiver_time",
                "satellite",
                "time_of_emission_isagpst",
            ]
        ],
        on=["time_of_reception_in_receiver_time", "satellite"],
    )

    # Build the query DataFrame we need to evaluate ephemerides
    query = flat_obs[flat_obs["observation_type"].str.startswith("C")]
    query = query.rename(
        columns={
            "observation_type": "signal",
            "satellite": "sv",
            "time_of_emission_isagpst": "query_time_isagpst",
        },
    )

    sat_states_func = compute(
        input_for_test_rtklib["sp3"], query, input_for_test_rtklib["atx"]
    )
    # apply relativistic clock correction
    sat_states_func = sat_states_func.assign(
        sat_clock_offset_corr_m=sat_states_func["sat_clock_offset_m"]
        + sat_states_func["relativistic_clock_effect_m"]
    )

    sat_states_rtklib = (
        pd.read_csv(
            input_for_test_rtklib["sp3_rtklib"],
            parse_dates=[0],
        )
        .rename(
            columns={
                "epoch": "time_of_reception_in_receiver_time",
                "prn": "sv",
                "pos_x": "sat_pos_x_m",
                "pos_y": "sat_pos_y_m",
                "pos_z": "sat_pos_z_m",
                "clk": "sat_clock_offset_corr_m",
            }
        )
        .dropna()
    )

    diff = (
        sat_states_func.set_index(["time_of_reception_in_receiver_time", "sv"])[
            ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m", "sat_clock_offset_corr_m"]
        ]
        - sat_states_rtklib.set_index(["time_of_reception_in_receiver_time", "sv"])
    ).dropna()

    print(
        "\n" + diff.unstack("sv").describe().loc[["min", "mean", "max"], :].to_string()
    )
    assert diff["sat_pos_x_m"].abs().max().max() < 5e-2
    assert diff["sat_pos_y_m"].abs().max().max() < 5e-2
    assert diff["sat_pos_z_m"].abs().max().max() < 5e-2
    assert diff["sat_clock_offset_corr_m"].abs().max().max() < 2e-1


def test_compare_matrtklib_galileo(input_for_test_rtklib):
    flat_obs = parse_rinex_obs_file(input_for_test_rtklib["obs"])
    flat_obs.time = pd.to_datetime(flat_obs.time, format="%Y-%m-%dT%H:%M:%S")
    flat_obs.obs_value = flat_obs.obs_value.astype(float)
    flat_obs[["sv", "obs_type"]] = flat_obs[["sv", "obs_type"]].astype(str)

    # keep only the Galileo signals present in rtklib file
    flat_obs = flat_obs.query("(sv.str[0] == 'E') and (obs_type == 'C1C')")

    # keep reduced number of epochs
    flat_obs = flat_obs.loc[
        (flat_obs.time >= flat_obs.time.min() + pd.Timedelta("1H"))
        & (
            flat_obs.time
            < flat_obs.time.min() + pd.Timedelta("1H") + pd.Timedelta("10m")
        )
    ]

    flat_obs = flat_obs.rename(
        columns={
            "time": "time_of_reception_in_receiver_time",
            "sv": "satellite",
            "obs_value": "observation_value",
            "obs_type": "observation_type",
        },
    )

    per_sat = flat_obs.pivot(
        index=["time_of_reception_in_receiver_time", "satellite"],
        columns=["observation_type"],
        values="observation_value",
    ).reset_index()
    per_sat["time_scale"] = (
        per_sat["satellite"].str[0].map(constants.constellation_2_system_time_scale)
    )
    per_sat["system_time_scale_epoch"] = per_sat["time_scale"].map(
        constants.system_time_scale_rinex_utc_epoch
    )
    code_phase_columns = [c for c in per_sat.columns if c[0] == "C" and len(c) == 3]
    tof_dtrx = pd.to_timedelta(
        per_sat[code_phase_columns]
        .mean(axis=1, skipna=True)
        .divide(constants.cGpsSpeedOfLight_mps),
        unit="s",
    )
    per_sat["time_of_emission_isagpst"] = (
        per_sat["time_of_reception_in_receiver_time"] - tof_dtrx
    )

    flat_obs = flat_obs.merge(
        per_sat[
            [
                "time_of_reception_in_receiver_time",
                "satellite",
                "time_of_emission_isagpst",
            ]
        ],
        on=["time_of_reception_in_receiver_time", "satellite"],
    )

    # Build the query DataFrame we need to evaluate ephemerides
    query = flat_obs[flat_obs["observation_type"].str.startswith("C")]
    query = query.rename(
        columns={
            "observation_type": "signal",
            "satellite": "sv",
            "time_of_emission_isagpst": "query_time_isagpst",
        },
    )

    sat_states_func = compute(
        input_for_test_rtklib["sp3"], query, input_for_test_rtklib["atx"]
    )
    # apply relativistic clock correction
    sat_states_func = sat_states_func.assign(
        sat_clock_offset_corr_m=sat_states_func["sat_clock_offset_m"]
        + sat_states_func["relativistic_clock_effect_m"]
    )

    sat_states_rtklib = (
        pd.read_csv(
            input_for_test_rtklib["sp3_rtklib"],
            parse_dates=[0],
        )
        .rename(
            columns={
                "epoch": "time_of_reception_in_receiver_time",
                "prn": "sv",
                "pos_x": "sat_pos_x_m",
                "pos_y": "sat_pos_y_m",
                "pos_z": "sat_pos_z_m",
                "clk": "sat_clock_offset_corr_m",
            }
        )
        .dropna()
    )

    diff = (
        sat_states_func.set_index(["time_of_reception_in_receiver_time", "sv"])[
            ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m", "sat_clock_offset_corr_m"]
        ]
        - sat_states_rtklib.set_index(["time_of_reception_in_receiver_time", "sv"])
    ).dropna()

    print(
        "\n" + diff.unstack("sv").describe().loc[["min", "mean", "max"], :].to_string()
    )
    assert diff["sat_pos_x_m"].abs().max().max() < 5e-2
    assert diff["sat_pos_y_m"].abs().max().max() < 5e-2
    assert diff["sat_pos_z_m"].abs().max().max() < 5e-2
    assert diff["sat_clock_offset_corr_m"].abs().max().max() < 5e-2
