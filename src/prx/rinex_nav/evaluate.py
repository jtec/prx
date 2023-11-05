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
import scipy

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
    xdot = x * float("nan")
    xdot[["X", "Y", "Z"]] = x[["dX", "dY", "dZ"]]
    r = np.linalg.norm(p)
    c1 = (-mu / p[0]**3) - (3 / 2) * J_2**2 * (mu * (a_e**2 / r**5)) * (1 - 5 * (p[2] / r)**2)
    vdot[0] = c1 * p[0] + omega**2 * p[0] + 2 * omega * v[1] + a[0]
    vdot[1] = c1 * p[1] + math.pow(omega, 2) * p[1] - 2 * omega * v[0] + a[1]
    vdot[2] = c1 * p[2] + a[2]
    return np.concatenate((pdot, vdot))


def compute_propagated_position_and_velocity(df):
    toes = df["ephemeris_reference_time_isagpst"]
    pv = df[["X", "Y", "Z", "dX", "dY", "dZ"]]
    a = df[["dX2", "dY2", "dZ2"]]
    #p = df[["X", "Y", "Z"]].values.flatten()
    #v = df[["dX", "dY", "dZ"]].values.flatten()
    #x = np.concatenate((p, v))
    #a = df[["dX2", "dY2", "dZ2"]].values.flatten()
    ts = df["query_time_wrt_ephemeris_reference_time_s"] * 0
    t_ends = df["query_time_wrt_ephemeris_reference_time_s"]

    while True:
        # We integrate in fixed steps until the last step, which is the time between the next-to-last integrated state
        # and the query time.
        fixed_integration_time_step = 60
        time_steps = t_ends - ts
        time_steps[time_steps > fixed_integration_time_step] = fixed_integration_time_step
        time_steps[time_steps < 0] = 0
        if np.all(time_steps == 0):
            break
        k1 = glonass_xdot(pv, a)
        k2 = glonass_xdot(pv + k1 * time_steps / 2, a)
        k3 = glonass_xdot(pv + k2 * time_steps / 2, a)
        k4 = glonass_xdot(pv + k3 * time_steps, a)
        pv = pv + time_steps / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ts = ts + time_steps

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


def is_bds_geo(inclination_rad, semi_major_axis_m):
    # IGSO and MEO satellites have an inclination of 55 degrees, so we can
    # use that as a threshold to distinguish GEO from IGSO satellites.
    # TODO Ok to ignore eccentricity here?
    # From BDS-SIS-ICD-2.0, 2013-12, section 3.1
    inclination_igso_and_meo_rad = helpers.deg_2_rad(55)
    geo_and_igso_approximate_radius_m = 35786 * constants.cMetersPerKilometer + constants.cBdsCgcs2000SmiMajorAxis_m
    meo_approximate_radius_m = 21528 * constants.cMetersPerKilometer + constants.cBdsCgcs2000SmiMajorAxis_m
    radius_threshold_m = (meo_approximate_radius_m - geo_and_igso_approximate_radius_m) / 2
    inclination_threshold_rad = inclination_igso_and_meo_rad / 2
    is_geo = (semi_major_axis_m > radius_threshold_m) & (inclination_rad < inclination_threshold_rad)
    return is_geo


# Adapted from gnss_lib_py's find_sat()
def bds_orbit_position_and_velocity(eph):
    # Semi-major axis
    A = eph.sqrtA**2
    # Computed mean motion
    n_0 = (
        np.sqrt(constants.cBdsMuEarth_m3ps2) * eph.sqrtA**-3
    )
    # Time since ephemeris reference epch
    eph['t_k'] = eph.query_time_wrt_ephemeris_reference_time_s
    # Corrected mean motion
    n = n_0 + eph.deltaN
    # Computed mean anomaly
    M_k = eph.M_0 + (n * eph.t_k)
    # Eccentric Anomaly
    E_k = eccentric_anomaly(M_k, eph.e, tol=1e-5)
    # Computed true anomaly
    sin_nu = np.sqrt(1 - eph.e**2) * (np.sin(E_k) / (1 - eph.e * np.cos(E_k)))
    cos_nu = (np.cos(E_k) - eph.e) / (1 - eph.e * np.cos(E_k))
    nu_k = np.arctan2(sin_nu, cos_nu)
    # Computed argument of latitude
    phi_k = nu_k + eph.omega
    # Argument of latitude correction
    delta_u_k = eph.C_us * np.sin(2 * phi_k) + eph.C_uc * np.cos(2 * phi_k)
    # Radius correction
    delta_r_k = eph.C_rs * np.sin(2 * phi_k) + eph.C_rc * np.cos(2 * phi_k)
    # Inclination correction
    delta_i_k = eph.C_is * np.sin(2 * phi_k) + eph.C_ic * np.cos(2 * phi_k)
    # Corrected argument of latitude
    u_k = phi_k + delta_u_k
    # Corrected radius
    r_k = A * (1 - eph.e * np.cos(E_k)) + delta_r_k
    # Corrected inclination
    eph['i_k'] = eph.i_0 + eph.IDOT * eph.t_k + delta_i_k
    # Satellite positions in the orbital plane
    eph['x_k'] = r_k * np.cos(u_k)
    eph['y_k'] = r_k * np.sin(u_k)
    # Derivatives for velocity computation
    dE_k = n_0 + eph.deltaN / (1 - eph.e * np.cos(E_k))
    dphi_k = np.sqrt(1 - eph.e**2) * dE_k / (1 - eph.e * np.cos(E_k))
    dr_k = (A * eph.e * dE_k * np.sin(E_k)) + 2 * (eph.C_rs * np.cos(2.0 * phi_k) - eph.C_rc * np.sin(2.0 * phi_k)) * dphi_k
    eph['di_k'] = 2 * (eph.C_is * np.cos(2.0 * phi_k) - eph.C_ic * np.sin(2.0 * phi_k)) * dphi_k + eph.IDOT
    du_k = (1 + 2 * (eph.C_us * np.cos(2.0 * phi_k) - eph.C_uc * np.cos(2.0 * phi_k))) * dphi_k
    eph['dx_k'] = dr_k * np.cos(phi_k) - r_k * np.sin(phi_k) * du_k
    eph['dy_k'] = dr_k * np.sin(phi_k) + r_k * np.cos(phi_k) * du_k

    # GEO orbits are handled differently
    eph['is_geo_orbit'] = is_bds_geo(eph.i_0, A)

    def compute_bdcs_position(sub_df):
        if not sub_df['is_geo_orbit'].iloc[0]:
            # Corrected longitude of ascending node in BDCS
            Omega_k = sub_df.Omega_0 + (sub_df.OmegaDot - constants.cBdsOmegaDotEarth_rps) * sub_df.t_k - constants.cBdsOmegaDotEarth_rps * sub_df.t_oe
            Omega_k_dot = sub_df.OmegaDot - constants.cBdsOmegaDotEarth_rps
            # Satellite positions in BDCS
            sub_df['X_k'] = sub_df['x_k'] * np.cos(Omega_k) - sub_df['y_k'] * np.cos(sub_df['i_k']) * np.sin(Omega_k)
            sub_df['Y_k'] = sub_df['x_k'] * np.sin(Omega_k) + sub_df['y_k'] * np.cos(sub_df['i_k']) * np.cos(Omega_k)
            sub_df['Z_k'] = sub_df['y_k'] * np.sin(sub_df['i_k'])

            sub_df["dX_k"] = (
                    dxp * cos_omega
                    - dyp * cos_i * sin_omega
                    + yp * sin_omega * sin_i * di
                    - (xp * sin_omega + yp * cos_i * cos_omega) * omega_dot
            )
        else:
            # Corrected longitude of ascending node in BDCS for GEO satellites
            Omega_k = sub_df.Omega_0 + sub_df.OmegaDot*sub_df.t_k - constants.cBdsOmegaDotEarth_rps * sub_df.t_oe
            Omega_k_dot = sub_df.OmegaDot
            # Satellite positions in inertial frame
            X_GK = sub_df['x_k'] * np.cos(Omega_k) - sub_df['y_k'] * np.cos(sub_df['i_k']) * np.sin(Omega_k)
            Y_GK = sub_df['x_k'] * np.sin(Omega_k) + sub_df['y_k'] * np.cos(sub_df['i_k']) * np.cos(Omega_k)
            Z_GK = sub_df['y_k'] * np.sin(sub_df['i_k'])
            P_GK = np.transpose(np.array([X_GK, Y_GK, Z_GK]))
            # Do special rotation for Beidou GEO satellites, see Beidou_ICD_B3I_v1.0.pdf, Table 5-11
            z_angles = constants.cBdsOmegaDotEarth_rps * sub_df.t_k
            rotation_matrices = []
            for i, z_angle in enumerate(z_angles):
                x_angle = helpers.deg_2_rad(-5.0)
                Rx = np.array(
                    [[1, 0, 0], [0, np.cos(x_angle), np.sin(x_angle)], [0, -np.sin(x_angle), np.cos(x_angle)]])
                Rz = np.array(
                    [[np.cos(z_angle), np.sin(z_angle), 0], [-np.sin(z_angle), np.cos(z_angle), 0], [0, 0, 1]])
                rotation_matrices.append(np.matmul(Rz, Rx))
            R = scipy.linalg.block_diag(*rotation_matrices)
            P_K = np.matmul(R, np.reshape(P_GK, (-1, 1)))
            P_K = np.reshape(P_K, (-1, 3))
            sub_df['X_k'] = P_K[:, 0]
            sub_df['Y_k'] = P_K[:, 1]
            sub_df['Z_k'] = P_K[:, 2]
        return sub_df
    eph.groupby('is_geo_orbit').apply(compute_bdcs_position)

    ############################################
    ######  Lines added for velocity (1)  ######
    ############################################
    dE = (n0 + delta_n) / e_cos_E
    dphi = np.sqrt(1 - e**2) * dE / e_cos_E
    # Changed from the paper
    dr = (sma * e * dE * sin_E) + 2 * (c_rs * cos_to_phi - c_rc * sin_to_phi) * dphi

    # Calculate the inclination with correction
    di = 2 * (c_is * cos_to_phi - c_ic * sin_to_phi) * dphi + ephem["IDOT"]

    du = (1 + 2 * (c_us * cos_to_phi - c_uc * sin_to_phi)) * dphi
    dxp = dr * np.cos(phi) - r * np.sin(phi) * du
    dyp = dr * np.sin(phi) + r * np.cos(phi) * du

    omega_dot = ephem["OmegaDot"] - constants.cBdsOmegaDotEarth_rps
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


# @memory.cache
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
    df["clock_offset_m"] = constants.cGpsSpeedOfLight_mps * (
        df["SVclockBias"]
        + df["SVclockDrift"] * df["query_time_wrt_clock_reference_time_s"]
        + df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"] ** 2
    )
    df["clock_offset_rate_mps"] = constants.cGpsSpeedOfLight_mps * (
        df["SVclockDrift"]
        + 2 * df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"]
    )

    def evaluate_orbit(sub_df):
        match sub_df["constellation"].iloc[0]:
            case "C":
                sub_df = bds_orbit_position_and_velocity(sub_df)
            case "R":
                sub_df = compute_propagated_position_and_velocity(sub_df)
            case other:
                assert False, f"Unexpected orbit type: {other}"
        return sub_df
    df = df.groupby("constellation").apply(evaluate_orbit)
    df = df[
        [
            "sv",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "clock_offset_m",
            "clock_offset_rate_mps",
            "query_time_isagpst",
        ]
    ]
    return df


if __name__ == "main":
    pass
