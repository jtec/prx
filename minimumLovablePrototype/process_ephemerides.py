import math
from gnss_lib_py.utils.sim_gnss import find_sat
import pandas as pd
import numpy as np
import parse_rinex
import constants
import helpers


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
    assert len(satellite_id) == 3, f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    return constellation_2_system_time_scale[constellation(satellite_id)]


def glonass_xdot(x, a):
    p = x[0:3]
    v = x[3:6]
    mu = 398600.44 * 1e9
    a_e = 6378136.0
    J_2 = 1082625.7 * 1e-9
    omega = .7292115 * 1e-5
    pdot = v
    vdot = np.zeros(3)
    r = np.linalg.norm(p)
    c1 = -mu/math.pow(p[0], 3) - (3/2)*math.pow(J_2, 2) * (mu*math.pow(a_e, 2)/math.pow(r, 5)) * (1 - 5*math.pow(p[2]/r, 2))
    vdot[0] = c1 * p[0] + math.pow(omega, 2)*p[0] + 2*omega*v[1] + a[0]
    vdot[1] = c1 * p[1] + math.pow(omega, 2)*p[1] - 2*omega*v[0] + a[1]
    vdot[2] = c1 * p[2] + a[2]
    return np.concatenate((pdot, vdot))


def compute_glonass_pv(sat_ephemeris: pd.DataFrame, t_system_time: pd.Timedelta):
    """Compute GLONASS satellite position and velocity from ephemerides"""
    toe = sat_ephemeris["ephemeris_reference_time_system_time"].values[0]
    p = sat_ephemeris[['X', 'Y', 'Z']].values.flatten()
    v = sat_ephemeris[['dX', 'dY', 'dZ']].values.flatten()
    x = np.concatenate((p, v))
    a = sat_ephemeris[['dX2', 'dY2', 'dZ2']].values.flatten()
    t = toe

    assert t_system_time >= t, f"Time for which orbit is to be computed {t_system_time} is before ephemeris reference time {t}, should we be we propagating GLONASS orbits backwards in time?"
    while abs((t - t_system_time).delta) > 1:
        max_time_step_s = 60
        h = min(max_time_step_s, float((t_system_time - t).delta) / constants.cNanoSecondsPerSecond)
        k1 = glonass_xdot(x, a)
        k2 = glonass_xdot(x + k1 * h/2, a)
        k3 = glonass_xdot(x + k2 * h/2, a)
        k4 = glonass_xdot(x + k3 * h, a)
        x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += pd.Timedelta(h, 'seconds')

    return x[0:3], x[3:6]


def constellation(satellite_id: str):
    assert len(satellite_id) == 3, f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    return satellite_id[0]


def compute_satellite_state(ephemerides: pd.DataFrame, satellite_id: str, t_system_time: pd.Timedelta):
    sat_ephemeris = select_nav_ephemeris(ephemerides, satellite_id, t_system_time)
    if constellation(satellite_id) == "G":
        week, week_second = helpers.timedelta_2_weeks_and_seconds(t_system_time)
        sv_posvel = find_sat(sat_ephemeris, week_second, week)
        return sv_posvel[["x", "y", "z"]].values.flatten(), sv_posvel[["vx", "vy", "vz"]].values.flatten()
    if constellation(satellite_id) == "R":
        # Note that this satellite state is w.r.t. PZ -90.02
        p, v = compute_glonass_pv(sat_ephemeris, t_system_time)
        return p, v
    assert False, f"Constellation of {satellite_id} not supported"


def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py"""
    nav_df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    nav_df.dropna(how="all", inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df["source"] = nav_ds.filename
    # georinex adds suffixes to satellite IDs if it sees multiple ephemerides for the same satellite and the same timstamp.
    # The downstream code expects three-letter satellite IDs, to remove suffixes.
    nav_df["sv"] = nav_df.apply(
        lambda row: row["sv"][:3], axis=1
    )

    nav_df["time_scale"] = nav_df.apply(
        lambda row: satellite_id_2_system_time_scale(row["sv"]), axis=1
    )
    nav_df["time"] = nav_df.apply(
        lambda row: helpers.timestamp_2_timedelta(row["time"], row["time_scale"]), axis=1
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
            full_seconds = row[week_field[row["time_scale"]]] * constants.cSecondsPerWeek + row["Toe"]
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


def select_nav_ephemeris(nav_dataframe: pd.DataFrame, satellite_id: str, t_system: pd.Timedelta):
    """
    select an ephemeris from a RINEX 3 ephemeris dataframe for a particular sv and time, and return the ephemeris.
    """
    assert type(t_system) == pd.Timedelta, f"t_system is not a pandas.Timedelta, but {type(t_system)}"
    ephemerides_of_requested_sat = nav_dataframe.loc[
        (nav_dataframe["sv"] == satellite_id)
    ]
    # Find first ephemeris before time of interest
    ephemerides_of_requested_sat = ephemerides_of_requested_sat.sort_values(by=["time"])
    ephemerides_of_requested_sat_before_requested_time = (
        ephemerides_of_requested_sat.loc[
            ephemerides_of_requested_sat["time"] < t_system
        ]
    )
    assert (
        ephemerides_of_requested_sat_before_requested_time.shape[0] > 0
    ), f"Did not find ephemeris with timestamp before {t_system}"
    return ephemerides_of_requested_sat_before_requested_time.iloc[[-1]]


def compute_satellite_clock_offset_and_clock_offset_rate(
    parsed_rinex_3_nav_file: pd.DataFrame,
    satellite: str,
    time_constellation_time_ns: pd.Timestamp,
):
    ephemeris_df = select_nav_ephemeris(
        parsed_rinex_3_nav_file, satellite, time_constellation_time_ns.to_datetime64()
    )
    # Convert to 64-bi float seconds here, as pandas.Timedelta has only nanosecond resolution
    time_wrt_ephemeris_epoch_s = pd.Timedelta(
        time_constellation_time_ns - ephemeris_df["time"].iloc[0]
    ).total_seconds()
    # Clock offset is sub-second, and float64 has roughly 1e-15 precision at 1s, so we get roughly 10 micrometers floating-point
    # error here.
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
