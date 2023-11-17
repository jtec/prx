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
    @lru_cache
    @memory.cache
    def cached_load(rinex_file: Path, file_hash: str):
        repair_with_gfzrnx(rinex_file)
        log.info(f"Parsing {rinex_file} ...")
        ds = georinex.load(rinex_file)
        df = convert_nav_dataset_to_dataframe(ds)
        df["orbit_type"] = df.constellation.map(
            {
                "C": "kepler",
                "G": "kepler",
                "E": "kepler",
                "J": "kepler",
                "S": "sbas",
                "R": "glonass",
                "I": "irnss",
            }
        )
        return df

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
    c1 = (-mu / p[0] ** 3) - (3 / 2) * J_2**2 * (mu * (a_e**2 / r**5)) * (
        1 - 5 * (p[2] / r) ** 2
    )
    vdot[0] = c1 * p[0] + omega**2 * p[0] + 2 * omega * v[1] + a[0]
    vdot[1] = c1 * p[1] + math.pow(omega, 2) * p[1] - 2 * omega * v[0] + a[1]
    vdot[2] = c1 * p[2] + a[2]
    return np.concatenate((pdot, vdot))


def compute_propagated_position_and_velocity(df):
    toes = df["ephemeris_reference_time_isagpst"]
    pv = df[["X", "Y", "Z", "dX", "dY", "dZ"]]
    a = df[["dX2", "dY2", "dZ2"]]
    # p = df[["X", "Y", "Z"]].values.flatten()
    # v = df[["dX", "dY", "dZ"]].values.flatten()
    # x = np.concatenate((p, v))
    # a = df[["dX2", "dY2", "dZ2"]].values.flatten()
    ts = df["query_time_wrt_ephemeris_reference_time_s"] * 0
    t_ends = df["query_time_wrt_ephemeris_reference_time_s"]

    while True:
        # We integrate in fixed steps until the last step, which is the time between the next-to-last integrated state
        # and the query time.
        fixed_integration_time_step = 60
        time_steps = t_ends - ts
        time_steps[
            time_steps > fixed_integration_time_step
        ] = fixed_integration_time_step
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


def eccentric_anomaly(M, e, tol=1e-5, max_iter=10):
    E = M.copy()
    for iterations in range(0, max_iter):
        delta_E = -(E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E += delta_E
        if np.max(np.abs(delta_E)) < tol and iterations > 1:
            return E

    assert False, f"Eccentric Anomaly may not have converged: delta_E = {delta_E}"


def is_bds_geo(constellation, inclination_rad, semi_major_axis_m):
    # IGSO and MEO satellites have an inclination of 55 degrees, so we can
    # use that as a threshold to distinguish GEO from IGSO satellites.
    # TODO Ok to ignore eccentricity here?
    # From BDS-SIS-ICD-2.0, 2013-12, section 3.1
    inclination_igso_and_meo_rad = helpers.deg_2_rad(55)
    geo_and_igso_approximate_radius_m = (
        35786 * constants.cMetersPerKilometer + constants.cBdsCgcs2000SmiMajorAxis_m
    )
    meo_approximate_radius_m = (
        21528 * constants.cMetersPerKilometer + constants.cBdsCgcs2000SmiMajorAxis_m
    )
    radius_threshold_m = (
        meo_approximate_radius_m - geo_and_igso_approximate_radius_m
    ) / 2
    inclination_threshold_rad = inclination_igso_and_meo_rad / 2
    is_geo = (
        (semi_major_axis_m > radius_threshold_m)
        & (inclination_rad < inclination_threshold_rad)
        & (constellation == "C")
    )
    return is_geo


def position_in_orbital_plane(eph):
    # Semi-major axis
    eph["A"] = eph.sqrtA**2
    # Computed mean motion
    n_0 = np.sqrt(eph.MuEarthIcd_m3ps2 / eph.A**3)
    # Time since ephemeris reference epch
    eph["t_k"] = eph.query_time_wrt_ephemeris_reference_time_s
    # Corrected mean motion
    eph["n"] = n_0 + eph.deltaN
    # Computed mean anomaly
    M_k = eph.M_0 + (eph.n * eph.t_k)
    # Eccentric Anomaly
    eph["E_k"] = eccentric_anomaly(M_k.to_numpy(), eph.e.to_numpy())
    # Computed true anomaly
    sin_nu = np.sqrt(1 - eph.e**2) * (np.sin(eph.E_k) / (1 - eph.e * np.cos(eph.E_k)))
    cos_nu = (np.cos(eph.E_k) - eph.e) / (1 - eph.e * np.cos(eph.E_k))
    eph["nu_k"] = np.arctan2(sin_nu, cos_nu)
    # Computed argument of latitude
    eph["phi_k"] = eph.nu_k + eph.omega
    # Argument of latitude correction
    delta_u_k = eph.C_us * np.sin(2 * eph.phi_k) + eph.C_uc * np.cos(2 * eph.phi_k)
    # Radius correction
    delta_r_k = eph.C_rs * np.sin(2 * eph.phi_k) + eph.C_rc * np.cos(2 * eph.phi_k)
    # Inclination correction
    delta_i_k = eph.C_is * np.sin(2 * eph.phi_k) + eph.C_ic * np.cos(2 * eph.phi_k)
    # Corrected argument of latitude
    eph["u_k"] = eph.phi_k + delta_u_k
    # Corrected radius
    eph["r_k"] = eph.A * (1 - eph.e * np.cos(eph.E_k)) + delta_r_k
    # Corrected inclination
    eph["i_k"] = eph.i_0 + eph.IDOT * eph.t_k + delta_i_k
    # Satellite positions in the orbital plane
    eph["x_k"] = eph.r_k * np.cos(eph.u_k)
    eph["y_k"] = eph.r_k * np.sin(eph.u_k)
    # Derivatives for velocity computation, from
    # IS-GPS-200N, Table 20-IV
    eph["dE_k"] = eph.n / (1 - eph.e * np.cos(eph.E_k))
    eph["dnu_k"] = eph.dE_k * np.sqrt(1 - eph.e**2) / (1 - eph.e * np.cos(eph.E_k))
    eph["di_k"] = eph.IDOT + 2 * eph.dnu_k * (
        eph.C_is * np.cos(2.0 * eph.phi_k) - eph.C_ic * np.sin(2.0 * eph.phi_k)
    )
    eph["du_k"] = eph.dnu_k + 2 * eph.dnu_k * (
        eph.C_us * np.cos(2.0 * eph.phi_k) - eph.C_uc * np.sin(2.0 * eph.phi_k)
    )
    eph["dr_k"] = (eph.e * eph.A * eph.dE_k * np.sin(eph.E_k)) + 2 * eph.dnu_k * (
        eph.C_rs * np.cos(2.0 * eph.phi_k) - eph.C_rc * np.sin(2.0 * eph.phi_k)
    )
    eph["dx_k"] = eph.dr_k * np.cos(eph.phi_k) - eph.r_k * np.sin(eph.phi_k) * eph.du_k
    eph["dy_k"] = eph.dr_k * np.sin(eph.phi_k) + eph.r_k * np.cos(eph.phi_k) * eph.du_k
    # We need to know which orbits are Beidou GEOs later on
    eph["is_bds_geo"] = is_bds_geo(eph.constellation, eph.i_k, eph.A)


def orbital_plane_to_earth_centered_cartesian(eph):
    # Corrected longitude of ascending node in ECEF
    eph["Omega_k"] = (
        eph.Omega_0
        + (eph.OmegaDot - eph.OmegaEarthIcd_rps) * eph.t_k
        - eph.OmegaEarthIcd_rps * eph.t_oe
    )
    eph[eph.is_bds_geo]["Omega_k"] = (
        eph[eph.is_bds_geo].Omega_0
        + eph[eph.is_bds_geo].OmegaDot * eph[eph.is_bds_geo].t_k
        - eph[eph.is_bds_geo].OmegaEarthIcd_rps * eph[eph.is_bds_geo].t_oe
    )
    eph["Omega_k_dot"] = eph.OmegaDot - eph.OmegaEarthIcd_rps
    # Satellite positions in cartesian frame (for BDS GEOs this is an inertial frame, for others the system ECEF frame)
    # For BDS GEOs we apply an additional rotation later on.
    eph["X_k"] = eph.x_k * np.cos(eph.Omega_k) - eph.y_k * np.cos(eph.i_k) * np.sin(
        eph.Omega_k
    )
    eph["Y_k"] = eph.x_k * np.sin(eph.Omega_k) + eph.y_k * np.cos(eph.i_k) * np.cos(
        eph.Omega_k
    )
    eph["Z_k"] = eph.y_k * np.sin(eph.i_k)
    # ECEF velocity, from
    # IS-GPS-200N, Table 20-IV
    eph["dX_k"] = -eph.x_k * eph.Omega_k_dot * np.sin(eph.Omega_k)
    +eph.dx_k * np.cos(eph.Omega_k)
    -eph.dy_k * np.sin(eph.Omega_k) * np.cos(eph.i_k)
    -eph.y_k * eph.Omega_k_dot * np.cos(eph.Omega_k) * np.cos(eph.i_k)
    +eph.y_k * eph.di_k * np.sin(eph.Omega_k) * np.sin(eph.i_k)

    eph["dY_k"] = eph.x_k * eph.Omega_k_dot * np.cos(eph.Omega_k)
    -eph.y_k * (
        eph.Omega_k_dot * np.sin(eph.Omega_k) * np.cos(eph.i_k)
        + eph.di_k * np.cos(eph.Omega_k) * np.sin(eph.i_k)
    )
    +eph.dx_k * np.sin(eph.Omega_k)
    +eph.dy_k * np.cos(eph.Omega_k) * np.cos(eph.i_k)
    eph["dZ_k"] = eph.y_k * eph.di_k * np.cos(eph.i_k) + eph.dy_k * np.sin(eph.i_k)


def handle_bds_geos(eph):
    # Do special rotation from inertial to BDCS (ECEF) frame for Beidou GEO satellites, see
    # Beidou_ICD_B3I_v1.0.pdf, Table 5-11
    geos = eph[eph.is_bds_geo]
    P_GK = np.transpose(geos[["X_k", "Y_k", "Z_k"]].to_numpy())
    z_angles = geos.OmegaEarthIcd_rps * geos.t_k
    rotation_matrices = []
    for i, z_angle in enumerate(z_angles):
        x_angle = helpers.deg_2_rad(-5.0)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(x_angle), np.sin(x_angle)],
                [0, -np.sin(x_angle), np.cos(x_angle)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(z_angle), np.sin(z_angle), 0],
                [-np.sin(z_angle), np.cos(z_angle), 0],
                [0, 0, 1],
            ]
        )
        rotation_matrices.append(np.matmul(Rz, Rx))
    R = scipy.linalg.block_diag(*rotation_matrices)
    P_K = np.matmul(R, np.reshape(P_GK, (-1, 1)))
    P_K = np.reshape(P_K, (-1, 3))
    geos["X_k"] = P_K[:, 0]
    geos["Y_k"] = P_K[:, 1]
    geos["Z_k"] = P_K[:, 2]
    eph[eph.is_bds_geo] = geos


# Adapted from gnss_lib_py's find_sat()
def kepler_orbit_position_and_velocity(eph):
    # BDS GEO orbits are handled a little  differently
    eph["OmegaEarthIcd_rps"] = eph.constellation.map(
        {
            "C": constants.cBdsOmegaDotEarth_rps,
            "G": constants.cGpsOmegaDotEarth_rps,
            "E": constants.cGalOmegaDotEarth_rps,
            "J": constants.cQzssOmegaDotEarth_rps,
        }
    )
    eph["MuEarthIcd_m3ps2"] = eph.constellation.map(
        {
            "C": constants.cBdsMuEarth_m3ps2,
            "G": constants.cGpsMuEarth_m3ps2,
            "E": constants.cGalMuEarth_m3ps2,
            "J": constants.cQzssMuEarth_m3ps2,
        }
    )
    position_in_orbital_plane(eph)
    orbital_plane_to_earth_centered_cartesian(eph)
    handle_bds_geos(eph)
    eph.rename(
        columns={
            "X_k": "x_m",
            "Y_k": "y_m",
            "Z_k": "z_m",
            "dX_k": "vx_mps",
            "dY_k": "vy_mps",
            "dZ_k": "vz_mps",
        },
        inplace=True,
    )
    return eph

    eph_1 = eph.copy()
    eph_1 = eph_1.select_dtypes(include=np.number)
    eph_2 = eph_1.copy()
    dt = 1e-3
    eph_2["query_time_wrt_ephemeris_reference_time_s"] += dt
    position_in_orbital_plane(eph_1)
    position_in_orbital_plane(eph_2)
    orbital_plane_to_earth_centered_cartesian(eph_1)
    orbital_plane_to_earth_centered_cartesian(eph_2)

    d_eph = (eph_2 - eph_1) / dt


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


def to_isagpst(time: pd.Timedelta, timescale: str):
    return time + time_scale_integer_second_offset(timescale, "GPST")


def select_ephemerides(df, query_time_isagpst):
    df = df[df["sv"].isin(query_time_isagpst.keys())]
    df["query_time_isagpst"] = df["sv"].apply(lambda sat: query_time_isagpst[sat])
    # Kick out all ephemerides that are not valid at the time of interest
    df = df[df["query_time_isagpst"] > df["validity_start"]]
    df = df[df["query_time_isagpst"] < df["validity_end"]]
    df["query_time_wrt_ephemeris_reference_time_s"] = (
        df["query_time_isagpst"] - df["ephemeris_reference_time_isagpst"]
    ).apply(helpers.timedelta_2_seconds)
    df["query_time_wrt_clock_reference_time_s"] = (
        df["query_time_isagpst"] - df["clock_reference_time_isagpst"]
    ).apply(helpers.timedelta_2_seconds)
    df = df[df["query_time_wrt_ephemeris_reference_time_s"] >= 0.0]
    df = (
        df.sort_values("query_time_wrt_ephemeris_reference_time_s")
        .groupby(["sv"])
        .first()
    )
    df.reset_index(inplace=True)
    return df


def compute_clock_offsets(df):
    df["clock_offset_m"] = constants.cGpsSpeedOfLight_mps * (
        df["SVclockBias"]
        + df["SVclockDrift"] * df["query_time_wrt_clock_reference_time_s"]
        + df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"] ** 2
    )
    df["clock_offset_rate_mps"] = constants.cGpsSpeedOfLight_mps * (
        df["SVclockDrift"]
        + 2 * df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"]
    )


def compute(rinex_nav_file_path, query_times_isagpst):
    rinex_nav_file_path = Path(rinex_nav_file_path)
    df = parse_rinex_nav_file(rinex_nav_file_path)
    df = select_ephemerides(df, query_times_isagpst)
    compute_clock_offsets(df)

    def evaluate_orbit(sub_df):
        orbit_type = sub_df["orbit_type"].iloc[0]
        if orbit_type == "kepler":
            sub_df = kepler_orbit_position_and_velocity(sub_df)
        else:
            assert (
                False
            ), f"Ephemeris evaluation not implemented or under development for {sub_df['constellation'].iloc[0]}"
        return sub_df

    df = df.groupby("orbit_type").apply(evaluate_orbit)
    df = df[
        [
            "sv",
            "x_m",
            "y_m",
            "z_m",
            "vx_mps",
            "vy_mps",
            "vz_mps",
            "clock_offset_m",
            "clock_offset_rate_mps",
            "query_time_isagpst",
        ]
    ]
    return df


if __name__ == "main":
    pass
