import functools
import math
import subprocess
import pandas as pd
import numpy as np
import glob
from functools import lru_cache
import georinex
from pathlib import Path
import joblib

from .. import helpers
from .. import constants

memory = joblib.Memory(Path(__file__).parent.joinpath("diskcache"), verbose=0)
log = helpers.get_logger(__name__)


def repair_with_gfzrnx(file):
    gfzrnx_binaries = glob.glob(
        str(Path(__file__).parent.joinpath("tools/gfzrnx/**gfzrnx**")), recursive=True
    )
    for gfzrnx_binary in gfzrnx_binaries:
        command = f" {gfzrnx_binary} -finp {file} -fout {file}  -chk -kv -f"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            log.info(f"Ran gfzrnx file repair on {file}")
            log.debug(result.stdout)
            log.info(f"gfzrnx binary used: {gfzrnx_binary}")
            return file
    assert False, "gdzrnx file repair run failed!"


# Can speed up RINEX parsing by using parsing results previously obtained and saved to disk.
def parse_rinex_nav_file(rinex_file: Path):
    @memory.cache
    def cached_load(rinex_file: Path, file_hash: str):
        repair_with_gfzrnx(rinex_file)
        log.info(f"Parsing {rinex_file} ...")
        parsed = georinex.load(rinex_file)
        return parsed

    t0 = pd.Timestamp.now()
    file_content_hash = helpers.md5_of_file_content(rinex_file)
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want to partially hash the file"
        )
    return cached_load(rinex_file, file_content_hash)


def convert_rnx3_nav_file_to_dataframe(path):
    # parse RNX3 NAV file using georinex module
    nav_ds = parse_rinex_nav_file(path)
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

# Validity interval w.r.t. ephemeris reference time, e.g. GPS's ToE.
constellation_2_ephemeris_validity_interval = {
    "G": [pd.Timedelta(-2, "hours"), pd.Timedelta(2, "hours")],
    "S": [pd.Timedelta(-2, "hours"), pd.Timedelta(2, "hours")],
    "E": [pd.Timedelta(-2, "hours"), pd.Timedelta(2, "hours")],
    "C": [pd.Timedelta(-1, "hours"), pd.Timedelta(1, "hours")],
    "R": [pd.Timedelta(0, "hours"), pd.Timedelta(2 * 0.5, "hours")],
    "J": [pd.Timedelta(-1, "hours"), pd.Timedelta(1, "hours")],
    "I": [pd.Timedelta(-1, "hours"), pd.Timedelta(1, "hours")],
}

system_time_scale_2_rinex_utc_epoch = {
    "GPST": constants.cGpstUtcEpoch,
    "SBAST": constants.cGpstUtcEpoch,
    "GST": constants.cGpstUtcEpoch,
    "BDT": (
        constants.cGpstUtcEpoch
        + pd.Timedelta(1356 * constants.cSecondsPerWeek, "seconds")
        + pd.Timedelta(14, "seconds")
    ),
    "GLONASST": constants.cArbitraryGlonassUtcEpoch,
    "QZSST": constants.cGpstUtcEpoch,
    "IRNSST": constants.cGpstUtcEpoch,
}


def time_scale_integer_second_offset(time_scale_a, time_scale_b):
    offset = (
        system_time_scale_2_rinex_utc_epoch[time_scale_a]
        - system_time_scale_2_rinex_utc_epoch[time_scale_b]
    )
    offset = offset.round("s")
    return offset


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
    """Compute GLONASS satellite position and velocity from ephemeris"""
    toe = sat_ephemeris["ephemeris_reference_time_system_time"].values[0]
    p = sat_ephemeris[["X", "Y", "Z"]].values.flatten()
    v = sat_ephemeris[["dX", "dY", "dZ"]].values.flatten()
    x = np.concatenate((p, v))
    a = sat_ephemeris[["dX2", "dY2", "dZ2"]].values.flatten()
    t = toe

    assert t_system_time >= t, (
        f"Time for which orbit is to be computed {t_system_time} is before "
        f"ephemeris reference time {t}, should we be propagating GLONASS orbits backwards in time?"
    )
    while abs(helpers.timedelta_2_nanoseconds(t - t_system_time)) > 1:
        max_time_step_s = 60
        h = min(
            max_time_step_s,
            float(helpers.timedelta_2_nanoseconds(t_system_time - t))
            / constants.cNanoSecondsPerSecond,
        )
        k1 = glonass_xdot(x, a)
        k2 = glonass_xdot(x + k1 * h / 2, a)
        k3 = glonass_xdot(x + k2 * h / 2, a)
        k4 = glonass_xdot(x + k3 * h, a)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += pd.Timedelta(h, "seconds")

    return x[0:3], x[3:6]


# Adapted from gnss_lib_py's _compute_eccentric_anomoly() (sic!)
def eccentric_anomaly(M, e, tol=1e-5, max_iter=10):
    E = M
    for _ in np.arange(0, max_iter):
        f = M - E + e * np.sin(E)
        dfdE = e * np.cos(E) - 1.0
        dE = -f / dfdE
        E = E + dE

    if any(dE.iloc[:] > tol):
        print("Eccentric Anomaly may not have converged: dE = ", dE)

    return E


# Adapted from gnss_lib_py's find_sat()
def compute_kepler_orbit_position_and_velocity(ephem):
    # Extract parameters
    c_is = ephem["C_is"]
    c_ic = ephem["C_ic"]
    c_rs = ephem["C_rs"]
    c_rc = ephem["C_rc"]
    c_uc = ephem["C_uc"]
    c_us = ephem["C_us"]
    M_0 = ephem["M_0"]
    dN = ephem["deltaN"]

    ecc = ephem["e"]  # eccentricity
    omega = ephem["omega"]  # argument of perigee
    omega_0 = ephem["Omega_0"]
    sqrt_sma = ephem["sqrtA"]  # sqrt of semi-major axis
    sma = sqrt_sma**2  # semi-major axis

    sqrt_mu_A = (
        np.sqrt(constants.cMuEarth_m3ps2) * sqrt_sma**-3
    )  # mean angular motion

    sv_posvel = pd.DataFrame()
    sv_posvel.loc[:, "sv"] = ephem.index
    sv_posvel.set_index("sv", inplace=True)

    dt = ephem["query_time_wrt_ephemeris_reference_time_s"]

    # Calculate the mean anomaly with corrections
    M_corr = dN * dt
    M = M_0 + (sqrt_mu_A * dt) + M_corr

    # Compute Eccentric Anomaly
    E = eccentric_anomaly(M, ecc, tol=1e-5)

    cos_E = np.cos(E)
    sin_E = np.sin(E)
    e_cos_E = 1 - ecc * cos_E

    # Calculate the true anomaly from the eccentric anomaly
    sin_nu = np.sqrt(1 - ecc**2) * (sin_E / e_cos_E)
    cos_nu = (cos_E - ecc) / e_cos_E
    nu = np.arctan2(sin_nu, cos_nu)

    # Calculate the argument of latitude iteratively
    phi_0 = nu + omega
    phi = phi_0
    for i in range(5):
        cos_to_phi = np.cos(2.0 * phi)
        sin_to_phi = np.sin(2.0 * phi)
        phi_corr = c_uc * cos_to_phi + c_us * sin_to_phi
        phi = phi_0 + phi_corr

    # Calculate the longitude of ascending node with correction
    omega_corr = ephem["OmegaDot"] * dt

    # Also correct for the rotation since the beginning of the system time scale week for
    # which the Omega0 is defined.
    # TODO Is this valid for all constellations?
    week_second = ephem["query_time_isagpst"].apply(
        lambda t: helpers.timedelta_2_weeks_and_seconds(t)[1]
    )
    omega = omega_0 - (constants.cOmegaDotEarth_rps * (week_second)) + omega_corr

    # Calculate orbital radius with correction
    r_corr = c_rc * cos_to_phi + c_rs * sin_to_phi
    r = sma * e_cos_E + r_corr

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    dE = (sqrt_mu_A + dN) / e_cos_E
    dphi = np.sqrt(1 - ecc**2) * dE / e_cos_E
    # Changed from the paper
    dr = (sma * ecc * dE * sin_E) + 2 * (c_rs * cos_to_phi - c_rc * sin_to_phi) * dphi

    # Calculate the inclination with correction
    i_corr = c_ic * cos_to_phi + c_is * sin_to_phi + ephem["IDOT"] * dt
    i = ephem["i_0"] + i_corr

    ############################################
    ######  Lines added for velocity (2)  ######
    ############################################
    di = 2 * (c_is * cos_to_phi - c_ic * sin_to_phi) * dphi + ephem["IDOT"]

    # Find the position in the orbital plane
    xp = r * np.cos(phi)
    yp = r * np.sin(phi)

    ############################################
    ######  Lines added for velocity (3)  ######
    ############################################
    du = (1 + 2 * (c_us * cos_to_phi - c_uc * sin_to_phi)) * dphi
    dxp = dr * np.cos(phi) - r * np.sin(phi) * du
    dyp = dr * np.sin(phi) + r * np.cos(phi) * du
    # Find satellite position in ECEF coordinates
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    sv_posvel.loc[:, "x"] = xp * cos_omega - yp * cos_i * sin_omega
    sv_posvel.loc[:, "y"] = xp * sin_omega + yp * cos_i * cos_omega
    sv_posvel.loc[:, "z"] = yp * sin_i

    ############################################
    ######  Lines added for velocity (4)  ######
    ############################################
    omega_dot = ephem["OmegaDot"] - constants.cOmegaDotEarth_rps
    sv_posvel.loc[:, "vx"] = (
        dxp * cos_omega
        - dyp * cos_i * sin_omega
        + yp * sin_omega * sin_i * di
        - (xp * sin_omega + yp * cos_i * cos_omega) * omega_dot
    )

    sv_posvel.loc[:, "vy"] = (
        dxp * sin_omega
        + dyp * cos_i * cos_omega
        - yp * sin_i * cos_omega * di
        + (xp * cos_omega - (yp * cos_i * sin_omega)) * omega_dot
    )

    sv_posvel.loc[:, "vz"] = dyp * sin_i + yp * cos_i * di
    return pd.concat([ephem, sv_posvel], axis="columns")


def constellation(satellite_id: str):
    assert (
        len(satellite_id) == 3
    ), f"Satellite ID unexpectedly not three characters long: {satellite_id}"
    return satellite_id[0]


#@memory.cache
def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame"""
    df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    df.dropna(how="all", inplace=True)
    df.reset_index(inplace=True)
    df["source"] = nav_ds.filename
    # georinex adds suffixes to satellite IDs if it sees multiple ephemerides (e.g. F/NAV, I/NAV) for the same
    # satellite and the same timestamp.
    # The downstream code expects three-letter satellite IDs, so remove suffixes.
    df["sv"] = df.apply(lambda row: row["sv"][:3], axis=1)
    df["constellation"] = df["sv"].str[0]
    df["time_scale"] = df["constellation"].replace(constellation_2_system_time_scale)

    def compute_ephemeris_and_clock_offset_reference_times(group):
        week_field = {
            "GPST": "GPSWeek",
            "GST": "GALWeek",
            "BDT": "BDTWeek",
            "SBAST": "GPSWeek",
            "QZSST": "GPSWeek",
            "IRNSST": "GPSWeek",
        }
        group_time_scale = group["time_scale"].iloc[0]
        group_constellation = group["constellation"].iloc[0]
        if group_time_scale not in ["GLONASST", "SBAST"]:
            full_seconds = (
                group[week_field[group_time_scale]] * constants.cSecondsPerWeek
                + group["Toe"]
            )
            group["ephemeris_reference_time_system_time"] = pd.to_timedelta(
                full_seconds, unit="seconds"
            )
        else:
            # For SBAS and GLONASS there are no separate ephemeris reference time fields
            group["ephemeris_reference_time_system_time"] = (
                group["time"] - constants.cArbitraryGlonassUtcEpoch
            )
            # The first derivative of the clock offset is in a different field for SBAS and GLONASS
            group["SVclockDrift"] = group["SVrelFreqBias"]
            # And the second derivative is zero, i.e. the constellation ground segment uses a fist-order clock model
            group["SVclockDriftRate"] = 0
        group["ephemeris_reference_time_isagpst"] = group[
            "ephemeris_reference_time_system_time"
        ] + time_scale_integer_second_offset(group_time_scale, "GPST")
        group["clock_offset_reference_time_system_time"] = (
            group["time"] - system_time_scale_2_rinex_utc_epoch[group_time_scale]
        )
        group["clock_reference_time_isagpst"] = group[
            "clock_offset_reference_time_system_time"
        ] + time_scale_integer_second_offset(group_time_scale, "GPST")

        group["validity_start"] = (
            group["ephemeris_reference_time_isagpst"]
            + constellation_2_ephemeris_validity_interval[group_constellation][0]
        )
        group["validity_end"] = (
            group["ephemeris_reference_time_isagpst"]
            + constellation_2_ephemeris_validity_interval[group_constellation][1]
        )
        return group

    df = df.groupby("constellation").apply(
        compute_ephemeris_and_clock_offset_reference_times
    )
    df.rename(
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
    return df


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
    # If the considered satellite is Galileo, there is a need to check which type of ephemeris has to be retrieved (
    # F/NAV or I/NAV)
    # The 8th and 9th bit of the `data source` parameter in the Galileo navigation message allows to identify the type of message (F/NAV vs I/NAV)
    cGalileoFnavDataSourceIndicator = 512
    if obs_type is not None and constellation(satellite_id) == "E":
        frequency_letter = obs_type[1]
        match frequency_letter:
            case "1" | "7":  # DataSrc >= 512
                ephemerides_of_requested_sat = ephemerides_of_requested_sat.loc[
                    ephemerides_of_requested_sat.DataSrc
                    >= cGalileoFnavDataSourceIndicator
                ]
            case "5":  # DataSrc < 512
                ephemerides_of_requested_sat = ephemerides_of_requested_sat.loc[
                    ephemerides_of_requested_sat.DataSrc
                    < cGalileoFnavDataSourceIndicator
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


def to_isagpst(time: pd.Timedelta, timescale: str):
    return time + time_scale_integer_second_offset(timescale, "GPST")


def compute(rinex_nav_file_path, query_times_isagpst):
    rinex_nav_file_path = Path(rinex_nav_file_path)
    ds = parse_rinex_nav_file(rinex_nav_file_path)
    df = convert_nav_dataset_to_dataframe(ds)
    df = df[df["sv"].isin(query_times_isagpst.keys())]
    df["query_time_isagpst"] = df["sv"].apply(lambda sat: query_times_isagpst[sat])
    # Kick out all ephemerides that are not valid at the time of interest
    df = df[df["query_time_isagpst"] > df["validity_start"]]
    df = df[df["query_time_isagpst"] < df["validity_end"]]
    df["query_time_wrt_ephemeris_reference_time_s"] = (
        df["query_time_isagpst"] - df["ephemeris_reference_time_isagpst"]
    ).apply(helpers.timedelta_2_seconds)
    df["query_time_wrt_clock_reference_time_s"] = (
        df["query_time_isagpst"] - df["clock_reference_time_isagpst"]
    ).apply(helpers.timedelta_2_seconds)
    # TODO Can the reference time be in the future in a live navigation message?
    # TODO We only consider ephemeris reference time here to select which ephemeris a live user would use, what about
    #      clock offset reference time?
    df = df[df["query_time_wrt_ephemeris_reference_time_s"] > 0.0]
    df = (
        df.sort_values("query_time_wrt_ephemeris_reference_time_s")
        .groupby(["sv"])
        .first()
    )
    df.reset_index(inplace=True)
    df["clock_offset_m"] = constants.cGpsIcdSpeedOfLight_mps * (
        df["SVclockBias"]
        + df["SVclockDrift"] * df["query_time_wrt_clock_reference_time_s"]
        + df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"]**2
    )
    df["clock_offset_rate_mps"] = constants.cGpsIcdSpeedOfLight_mps * (
            df["SVclockDrift"]
            + 2 * df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"]
    )
    df = compute_kepler_orbit_position_and_velocity(df)
    df = df[["sv", "x", "y", "z", "vx", "vy", "vz", "clock_offset_m", "clock_offset_rate_mps", "query_time_isagpst"]]
    return df


if __name__ == "main":
    pass
