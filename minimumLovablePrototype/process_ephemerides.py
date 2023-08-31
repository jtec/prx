import math

from gnss_lib_py.utils.sim_gnss import find_sat
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
    "I": "IRNSST",
}


def satellite_id_2_system_time_scale(satellite_id):
    assert (
            len(satellite_id) == 3
    ), f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    return constellation_2_system_time_scale[constellation(satellite_id)]


def glonass_xdot(x, a):
    p = x[0:3]
    v = x[3:6]
    mu = 398600.44 * 1e9
    a_e = 6378136.0
    J_2 = 1082625.7 * 1e-9
    omega = 0.7292115 * 1e-5
    pdot = v
    vdot = np.zeros(3)
    r = np.linalg.norm(p)
    c1 = -mu / math.pow(p[0], 3) - (3 / 2) * math.pow(J_2, 2) * (
            mu * math.pow(a_e, 2) / math.pow(r, 5)
    ) * (1 - 5 * math.pow(p[2] / r, 2))
    vdot[0] = c1 * p[0] + math.pow(omega, 2) * p[0] + 2 * omega * v[1] + a[0]
    vdot[1] = c1 * p[1] + math.pow(omega, 2) * p[1] - 2 * omega * v[0] + a[1]
    vdot[2] = c1 * p[2] + a[2]
    return np.concatenate((pdot, vdot))


def compute_glonass_pv(sat_ephemeris: pd.DataFrame, t_system_time: pd.Timedelta):
    """Compute GLONASS satellite position and velocity from ephemerides"""
    toe = sat_ephemeris["ephemeris_reference_time_system_time"].values[0]
    p = sat_ephemeris[["X", "Y", "Z"]].values.flatten()
    v = sat_ephemeris[["dX", "dY", "dZ"]].values.flatten()
    x = np.concatenate((p, v))
    a = sat_ephemeris[["dX2", "dY2", "dZ2"]].values.flatten()
    t = toe

    assert t_system_time >= t, (
        f"Time for which orbit is to be computed {t_system_time} is before "
        f"ephemeris reference time {t}, should we be we propagating GLONASS orbits backwards in time?"
    )
    while abs((t - t_system_time).delta) > 1:
        max_time_step_s = 60
        h = min(
            max_time_step_s,
            float((t_system_time - t).delta) / constants.cNanoSecondsPerSecond,
        )
        k1 = glonass_xdot(x, a)
        k2 = glonass_xdot(x + k1 * h / 2, a)
        k3 = glonass_xdot(x + k2 * h / 2, a)
        k4 = glonass_xdot(x + k3 * h, a)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += pd.Timedelta(h, "seconds")

    return x[0:3], x[3:6]


def compute_kepler_pv(sat_ephemeris: pd.DataFrame, t_system_time: pd.Timedelta):
    system_time_week, system_time_week_second = helpers.timedelta_2_weeks_and_seconds(
        t_system_time
    )
    # use gnss_lib_py as a stopgap until we have our own ephemeris computation. Pretend that this is a GPS ephemeris:
    satellite = sat_ephemeris["sv"].iloc[0]
    if constellation(satellite) == "E":
        sat_ephemeris["GPSWeek"] = sat_ephemeris["GALWeek"]
    if constellation(satellite) == "C":
        sat_ephemeris["GPSWeek"] = sat_ephemeris["BDTWeek"]

    sv_posvel = find_sat(
        sat_ephemeris, np.array([float(system_time_week_second)]), system_time_week
    )
    position_system_frame_m = sv_posvel[["x", "y", "z"]].values.flatten()
    velocity_system_frame_mps = sv_posvel[["vx", "vy", "vz"]].values.flatten()
    return position_system_frame_m, velocity_system_frame_mps


def constellation(satellite_id: str):
    assert (
            len(satellite_id) == 3
    ), f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    return satellite_id[0]


def compute_satellite_state(
        ephemerides: pd.DataFrame, satellite_id: str, t_system_time_ns: pd.Timedelta
):
    sat_ephemeris = select_nav_ephemeris(ephemerides, satellite_id, t_system_time_ns)
    if (
            constellation(satellite_id) == "G"
            or constellation(satellite_id) == "E"
            or constellation(satellite_id) == "C"
    ):
        position_system_frame_m, velocity_system_frame_mps = compute_kepler_pv(
            sat_ephemeris, t_system_time_ns
        )
        relativistic_clock_effect_m = compute_relativistic_clock_effect(
            position_system_frame_m, velocity_system_frame_mps
        )
    elif constellation(satellite_id) == "R":
        position_system_frame_m, velocity_system_frame_mps = compute_glonass_pv(
            sat_ephemeris, t_system_time_ns
        )
        # relativistic clock effect are already taken into account in Glonass navigation message
        relativistic_clock_effect_m = 0
    else:
        position_system_frame_m = np.full(3, np.nan)
        velocity_system_frame_mps = np.full(3, np.nan)
        relativistic_clock_effect_m = np.nan
        # assert False, f"Constellation of {satellite_id} not supported"

    (
        clock_offset_m,
        clock_offset_rate_mps,
    ) = compute_satellite_clock_offset_and_clock_offset_rate(
        ephemerides, satellite_id, t_system_time_ns
    )

    return (
        position_system_frame_m,
        velocity_system_frame_mps,
        clock_offset_m,
        clock_offset_rate_mps,
        relativistic_clock_effect_m,
    )


def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py"""
    nav_df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    nav_df.dropna(how="all", inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df["source"] = nav_ds.filename
    # georinex adds suffixes to satellite IDs if it sees multiple ephemerides (e.g. F/NAV, I/NAV) for the same
    # satellite and the same timstamp.
    # The downstream code expects three-letter satellite IDs, to remove suffixes.
    nav_df["sv"] = nav_df.apply(lambda row: row["sv"][:3], axis=1)

    nav_df["time_scale"] = nav_df.apply(
        lambda row: satellite_id_2_system_time_scale(row["sv"]), axis=1
    )
    nav_df["time"] = nav_df.apply(
        lambda row: helpers.timestamp_2_timedelta(row["time"], row["time_scale"]),
        axis=1,
    )

    def extract_toe(row):
        week_field = {
            "GPST": "GPSWeek",
            "GST": "GALWeek",
            "BDT": "BDTWeek",
            "SBAST": "GPSWeek",
            "QZSST": "GPSWeek",
            "IRNSST": "GPSWeek",
        }
        if row["time_scale"] != "GLONASST":
            full_seconds = (
                    row[week_field[row["time_scale"]]] * constants.cSecondsPerWeek
                    + row["Toe"]
            )
            return pd.Timedelta(full_seconds, "seconds")
        if row["time_scale"] == "GLONASST":
            return pd.Timedelta(row["time"])

    nav_df["ephemeris_reference_time_system_time"] = nav_df.apply(extract_toe, axis=1)

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


def select_nav_ephemeris(
        nav_dataframe: pd.DataFrame,
        satellite_id: str,
        t_system: pd.Timedelta,
        obs_type=None,
):
    """
    select an ephemeris from a RINEX 3 ephemeris dataframe for a particular sv and time, and return the ephemeris.
    """
    assert (
            type(t_system) == pd.Timedelta
    ), f"t_system is not a pandas.Timedelta, but {type(t_system)}"
    ephemerides_of_requested_sat = nav_dataframe.loc[
        (nav_dataframe["sv"] == satellite_id)
    ]
    # if the considered satellite is Galileo, there is a need to check which type of ephemeris has to be retrieved (
    # F/NAV or I/NAV)
    if obs_type is not None and constellation(satellite_id) == "E":
        frequency_letter = obs_type[1]
        match frequency_letter:
            case "1" | "7":  # DataSrc >= 512
                ephemerides_of_requested_sat = ephemerides_of_requested_sat.loc[
                    ephemerides_of_requested_sat.DataSrc
                    >= constants.cGalileoFnavDataSourceIndicator
                    ]
            case "5":  # DataSrc < 512
                ephemerides_of_requested_sat = ephemerides_of_requested_sat.loc[
                    ephemerides_of_requested_sat.DataSrc
                    < constants.cGalileoFnavDataSourceIndicator
                    ]
            case _:  # other galileo signals not supported in rnx3
                log.info(
                    f"Could not retrieve ephemeris for satellite id: {satellite_id} and obs: {obs_type}"
                )

    # Find first ephemeris before time of interest
    ephemerides_of_requested_sat = ephemerides_of_requested_sat.sort_values(by=["time"])
    ephemerides_of_requested_sat_before_requested_time = (
        ephemerides_of_requested_sat.loc[
            ephemerides_of_requested_sat["time"] <= t_system
            ]
    )
    assert (
            ephemerides_of_requested_sat_before_requested_time.shape[0] > 0
    ), f"Did not find ephemeris with timestamp before {t_system}"
    return ephemerides_of_requested_sat_before_requested_time.iloc[[-1]]


def compute_satellite_clock_offset_and_clock_offset_rate(
        parsed_rinex_3_nav_file: pd.DataFrame,
        satellite: str,
        time_constellation_time_ns: pd.Timedelta,
):
    ephemeris_df = select_nav_ephemeris(
        parsed_rinex_3_nav_file, satellite, time_constellation_time_ns
    )
    # Convert to 64-bit float seconds here, as pandas.Timedelta has only nanosecond resolution
    time_wrt_ephemeris_epoch_s = helpers.timedelta_2_seconds(
        time_constellation_time_ns - ephemeris_df["time"].iloc[0]
    )
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


def compute_total_group_delay_rnx3(
        parsed_rinex_3_nav_file: pd.DataFrame,
        time_constellation_time_ns: pd.Timedelta,
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
    - Beidou B1I, B2I, B3I: Beidou_ICD_B1I_v3.0.pdf, §5.2.4.10
    - Beidou B1Cp, B1Cd: Beidou_ICD_B1C_v1.0.pdf, §7.6.2 (not supported by rnx3)
    - Beidou B2bi: Beidou_ICD_B2b_v1.0.pdf, §7.5.2 (not supported by rnx3)

    Note: rinex v3 nav files only support a subset of observations.
    """
    ephemeris_df = select_nav_ephemeris(
        parsed_rinex_3_nav_file,
        satellite,
        time_constellation_time_ns,
        obs_type=obs_type,
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
                    gamma = (
                                    constants.carrier_frequencies_hz()["G"]["L1"]
                                    / constants.carrier_frequencies_hz()["G"]["L2"]
                            ) ** 2
                case _:
                    gamma = np.nan
        case "E":
            match frequency_code:
                case "1":
                    group_delay = ephemeris_df.BGDe5b.values[0]
                    gamma = 1
                case "5":
                    group_delay = ephemeris_df.BGDe5a.values[0]
                    gamma = (
                                    constants.carrier_frequencies_hz()["E"]["L1"]
                                    / constants.carrier_frequencies_hz()["E"]["L5"]
                            ) ** 2
                case "7":
                    group_delay = ephemeris_df.BGDe5b.values[0]
                    gamma = (
                                    constants.carrier_frequencies_hz()["E"]["L1"]
                                    / constants.carrier_frequencies_hz()["E"]["L7"]
                            ) ** 2
                case _:
                    group_delay = np.nan
                    gamma = np.nan
        case "C":
            match obs_type:
                case "C2I":  # called B1I in Beidou ICD
                    group_delay = ephemeris_df.TGD1.values[0]
                    gamma = 1
                case "C7I":  # called B2I in Beidou ICD
                    group_delay = ephemeris_df.TGD2.values[0]
                    gamma = 1
                case "C6I":  # called B3I in Beidou ICD
                    group_delay = 0
                    gamma = 1
                case _:
                    group_delay = np.nan
                    gamma = np.nan
        case _:
            group_delay = np.nan
            gamma = np.nan

    if np.isnan(gamma):
        log.info(
            f"Could not retrieve total group delay for satellite id: {satellite} and obs: {obs_type}"
        )

    return group_delay * gamma


def compute_sagnac_effect(sat_pos_m, rx_pos_m):
    """compute the sagnac effect (effect of the Earth rotation during signal propagation°

        Input:
        - sat_pos_m: satellite ECEF position. np.ndarray of shape (3,)
        - rx_pos_m: satellite ECEF position. np.ndarray of shape (3,)

        Note:
        The computation uses small angle approximation of cos and sin.

        Reference:
        RTKLIB v2.4.2 manual, eq E.3.8b, p 140
    """
    sagnac_effect_m = constants.cEarthRotationRate_radps / constants.cGpsIcdSpeedOfLight_mps \
                      * (sat_pos_m[0] * rx_pos_m[1] - sat_pos_m[1] * rx_pos_m[0])
    return sagnac_effect_m


def compute_relativistic_clock_effect(
        sat_pos_m: np.array(3, ),
        sat_vel_mps: np.array(3, )
):
    """
    Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equation (5.19)
    """
    relativistic_clock_effect_m = (
            -2 * np.dot(sat_pos_m, sat_vel_mps) / constants.cGpsIcdSpeedOfLight_mps
    )

    return relativistic_clock_effect_m


def compute_satellite_elevation_and_azimuth(sat_pos_ecef, receiver_pos_ecef):
    """
    Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.9),(B.13),(B.14)
    """
    rho = (sat_pos_ecef - receiver_pos_ecef) / np.linalg.norm(sat_pos_ecef - receiver_pos_ecef)
    [lat, lon, __] = ecef_2_geodetic(receiver_pos_ecef)
    unit_e = [-np.sin(lon), np.cos(lon), 0]
    unit_n = [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)]
    unit_u = [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]
    elevation_rad = np.arcsin(np.dot(rho, unit_u))
    azimuth_rad = np.arctan2(np.dot(rho, unit_e), np.dot(rho, unit_n))
    return elevation_rad, azimuth_rad


def ecef_2_geodetic(pos_ecef):
    """Reference:
    GNSS Data Processing, Vol. I: Fundamentals and Algorithms. Equations (B.4),(B.5),(B.6)
    """
    p = np.sqrt(pos_ecef[0] ** 2 + pos_ecef[1] ** 2)
    longitude_rad = np.arctan2(pos_ecef[1], pos_ecef[0])
    precision_m = 1e-3
    delta_h_m = 1  # initialization to a value larger than precision
    altitude_m = 0
    latitude_rad = np.arctan2(pos_ecef[2], p * (1 - constants.cWgs84EarthEccentricity ** 2))
    while delta_h_m > precision_m:
        n = constants.cWgs84EarthSemiMajorAxis_m / np.sqrt(
            1 - constants.cWgs84EarthEccentricity ** 2 * np.sin(latitude_rad) ** 2)
        altitude_previous = altitude_m
        altitude_m = p / np.cos(latitude_rad) - n
        delta_h_m = np.abs(altitude_m - altitude_previous)
        latitude_rad = np.arctan2(pos_ecef[2], p * (1 - n * constants.cWgs84EarthEccentricity ** 2 / (n + altitude_m)))
    return [latitude_rad, longitude_rad, altitude_m]
