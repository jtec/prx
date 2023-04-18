import math
import pandas as pd
import numpy as np
import parse_rinex
import constants
import helpers

log = helpers.get_logger(__name__)


def convert_rnx3_nav_file_to_dataframe(path):
    # parse RNX3 NAV file using georinex module
    nav_ds = parse_rinex.load(path, use_caching=True)
    nav_df = convert_nav_dataset_to_dataframe(nav_ds)
    return nav_df


constellation_2_system_time_scale = {
    "G": "GPST",
    "S": "SBAST",
    "E": "GST",
    "C": "BDT",
    "R": "GLONASST",
    "J": "QZSST",
    "I": "IRNWT",
}


def satellite_id_2_system_time_scale(satellite_id):
    return constellation_2_system_time_scale[satellite_id[0]]


def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py"""
    nav_df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    nav_df.dropna(how="all", inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df["source"] = nav_ds.filename

    nav_df["time_scale"] = nav_df.apply(
        lambda row: satellite_id_2_system_time_scale(row["sv"]), axis=1
    )

    # TODO Can we be sure that this is always GPST?
    gpst_s = (
            pd.to_numeric(nav_df["time"] - constants.cGpstEpoch)
            / constants.cNanoSecondsPerSecond
    )
    # Week second:
    nav_df["t_oc"] = gpst_s - constants.cSecondsPerWeek * np.floor(
        gpst_s / constants.cSecondsPerWeek
    )

    nav_df.rename(
        columns={
            "M0": "M_0",
            "Eccentricity": "e",
            "Toe": "t_oe",
            "DeltaN": "deltaN",
            "Cuc": "C_uc",
            "Cus": "C_us",
            "Cic": "C_ic",
            "Crc": "C_rc",
            "Cis": "C_is",
            "Crs": "C_rs",
            "Io": "i_0",
            "Omega0": "Omega_0",
        },
        inplace=True,
    )
    return nav_df


def select_nav_ephemeris(nav_dataframe, satellite_id, gpst_datetime, obs_type=None):
    """select an ephemeris from a RNX3 nav dataframe for a particular sv and time, and return the ephemeris.

    Input examples:
    nav_dataset = convert_nav_dataset_to_dataframe(path_to_rnx3_nav_file)
    satellite_id = np.array('G01', dtype='<U3') # satellite ID for a single satellite,
    gpst_datetime = np.datetime64('2022-01-01T00:00:00.000'), np.datetime64(tow_to_datetime(gps_week, gps_tow))
    obs_type = "C1C"

    Output:
    nav_dataframe: a pandas.dataframe containing the selected ephemeris
    """
    ephemerides_of_requested_sat = nav_dataframe[
        nav_dataframe.sv.str.contains(satellite_id)]

    # if the considered satellite is Galileo, there is a need to check which type of ephemeris has to be retrieved (
    # F/NAV or I/NAV)
    if not (obs_type is None) and satellite_id[0] == 'E':
        frequency_letter = obs_type[1]
        match frequency_letter:
            case '1' | '7':  # DataSrc >= 512
                ephemerides_of_requested_sat = \
                    ephemerides_of_requested_sat.loc[ephemerides_of_requested_sat.DataSrc >= 512]
            case '5':  # DataSrc < 512
                ephemerides_of_requested_sat = \
                    ephemerides_of_requested_sat.loc[ephemerides_of_requested_sat.DataSrc < 512]
            case _:  # other galileo signals not supported in rnx3
                log.info(f"Could not retrieve ephemeris for satellite id: {satellite_id} and obs: {obs_type}")
        # in case of sv values such as 'E25_1', replace by 'E25'
        ephemerides_of_requested_sat.sv = ephemerides_of_requested_sat.sv.values[0][0:3]

    # Find first ephemeris before time of interest
    ephemerides_of_requested_sat = ephemerides_of_requested_sat.sort_values(by=["time"])
    ephemerides_of_requested_sat_before_requested_time = (
        ephemerides_of_requested_sat.loc[
            ephemerides_of_requested_sat["time"] <= gpst_datetime
            ]
    )
    assert (
            ephemerides_of_requested_sat_before_requested_time.shape[0] > 0
    ), f"Did not find ephemeris with timestamp before {gpst_datetime}"

    return ephemerides_of_requested_sat_before_requested_time.iloc[[-1]]


def compute_satellite_clock_offset_and_clock_offset_rate(
        parsed_rinex_3_nav_file: pd.DataFrame,
        satellite: str,
        time_constellation_time_ns: pd.Timestamp,
):
    ephemeris_df = select_nav_ephemeris(
        parsed_rinex_3_nav_file, satellite, time_constellation_time_ns.to_datetime64()
    )
    # Convert to float64 seconds here, as pandas.Timedelta has only nanosecond resolution
    time_wrt_ephemeris_epoch_s = pd.Timedelta(
        time_constellation_time_ns - ephemeris_df["time"].iloc[0]
    ).total_seconds()
    if satellite[0] == "R":
        offset_at_epoch_s = ephemeris_df["SVclockBias"].iloc[0]
        offset_rate_at_epoch_sps = ephemeris_df["SVrelFreqBias"].iloc[0]
        offset_acceleration_sps2 = 0
    elif satellite[0] == "S":
        # TODO RINEX 3.05 mentions a W0 time offset term for SBAS, where do we get that from?
        offset_at_epoch_s = ephemeris_df["SVclockBias"].iloc[0]
        offset_rate_at_epoch_sps = ephemeris_df["SVrelFreqBias"].iloc[0]
        offset_acceleration_sps2 = 0
    else:
        offset_at_epoch_s = ephemeris_df["SVclockBias"].iloc[0]
        offset_rate_at_epoch_sps = ephemeris_df["SVclockDrift"].iloc[0]
        offset_acceleration_sps2 = ephemeris_df["SVclockDriftRate"].iloc[0]

    offset_s = (
            offset_at_epoch_s
            + offset_rate_at_epoch_sps * time_wrt_ephemeris_epoch_s
            + offset_acceleration_sps2 * math.pow(time_wrt_ephemeris_epoch_s, 2)
    )
    offset_rate_sps = (
            offset_rate_at_epoch_sps
            + 2 * offset_acceleration_sps2 * time_wrt_ephemeris_epoch_s
    )

    return (
        constants.cGpsIcdSpeedOfLight_mps * offset_s,
        constants.cGpsIcdSpeedOfLight_mps * offset_rate_sps,
    )


def compute_total_group_delay(
        parsed_rinex_3_nav_file: pd.DataFrame,
        time_constellation_time_ns: pd.Timestamp,
        satellite: str,
        obs_type: str,
):
    """compute the total group delay from a parsed rnx3 file, for a specific satellite, time and observation type

    Input examples:
    parsed_rinex_3_nav_file = ph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file)
    satellite = "G01" # satellite ID for a single satellite,
    time_constellation_time_ns = pd.Timestamp(np.datetime64("2022-01-01T01:00:00.000000000"))
    obs_type = "C1C" # RNX3 observation code

    Output:
    nav_dataframe: a pandas.dataframe containing the selected ephemeris

    Reference:
    - GPS: IS-GPS-200N.pdf, §20.3.3.3.3.2
    - Galileo: Galileo_OS_SIS_ICD_v2.0.pdf, §5.1.5
    """
    ephemeris_df = select_nav_ephemeris(
        parsed_rinex_3_nav_file, satellite, time_constellation_time_ns.to_datetime64(), obs_type=obs_type
    )

    # compute the scale factor, depending on the constellation and frequency
    constellation = satellite[0]
    frequency_code = obs_type[1]
    match constellation:
        case "G":
            group_delay = ephemeris_df.TGD.values[0]
            match frequency_code:
                case "1":
                    gamma = 1
                case "2":
                    gamma = (constants.carrier_frequencies_hz()["G"]["L1"] / constants.carrier_frequencies_hz()["G"][
                        "L2"]) ** 2
                case _:
                    gamma = np.nan
        case "E":
            match frequency_code:
                case "1":
                    group_delay = ephemeris_df.BGDe5b.values[0]
                    gamma = 1
                case "5":
                    group_delay = ephemeris_df.BGDe5a.values[0]
                    gamma = (constants.carrier_frequencies_hz()["E"]["L1"] / constants.carrier_frequencies_hz()["E"][
                        "L5"]) ** 2
                case "7":
                    group_delay = ephemeris_df.BGDe5b.values[0]
                    gamma = (constants.carrier_frequencies_hz()["E"]["L1"] / constants.carrier_frequencies_hz()["E"][
                        "L7"]) ** 2
        case _:
            gamma = np.nan

    if np.isnan(gamma):
        log.info(f"Could not retrieve total group delay for satellite id: {satellite} and obs: {obs_type}")

    return group_delay * gamma
