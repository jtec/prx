import logging
import multiprocessing

import pandas as pd
import numpy as np
from pathlib import Path
import scipy
from joblib import Parallel, delayed
import georinex
from prx import util
from prx import constants
from prx.util import timeit, repair_with_gfzrnx

log = logging.getLogger(__name__)


@timeit
def parse_rinex_nav_file(rinex_file: Path):
    @util.disk_cache.cache(ignore=["rinex_file_path"])
    def cached_load(rinex_file_path: Path, file_hash: str):
        repair_with_gfzrnx(rinex_file)
        ds = georinex.load(rinex_file)
        ds.attrs["utc_gpst_leap_seconds"] = util.get_gpst_utc_leap_seconds(
            rinex_file_path
        )
        df = convert_nav_dataset_to_dataframe(ds)
        df["ephemeris_hash"] = pd.util.hash_pandas_object(df, index=False).astype(str)
        return df

    t0 = pd.Timestamp.now()

    file_content_hash = util.hash_of_file_content(rinex_file)
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want to partially hash the file"
        )
    return cached_load(rinex_file, file_content_hash)


def time_scale_integer_second_offset_wrt_gpst(time_scale, utc_gpst_leap_seconds=None):
    if time_scale in ["GPST", "SBAST", "QZSST", "IRNSST", "GST"]:
        return pd.Timedelta(seconds=0)
    if time_scale == "BDT":
        return pd.Timedelta(seconds=-14)
    if time_scale == "GLONASST":
        assert (
            utc_gpst_leap_seconds is not None
        ), "Need GPST-UTC leap seconds to compute GLONASST integer second offset w.r.t. GPST"
        return pd.Timedelta(seconds=-utc_gpst_leap_seconds)
    assert False, f"Unexpected time scale: {time_scale}"


def glonass_xdot_rtklib(x, acc_sun_moon):
    p = x[["X", "Y", "Z"]]
    v = x[["dX", "dY", "dZ"]]
    GM_e = 398600.4418 * 1e9
    R_e = 6378136.0
    J_2 = 1.0826257 * 1e-3
    omega_e = 7.292115 * 1e-5
    xdot = x.copy() * np.nan
    xdot[["X", "Y", "Z"]] = x[["dX", "dY", "dZ"]]
    r = np.linalg.norm(p.to_numpy(), axis=1)
    # How
    # https://github.com/tomojitakasu/RTKLIB/blob/71db0ffa0d9735697c6adfd06fdf766d0e5ce807/src/ephemeris.c#L261
    # computes it:
    a = 1.5 * J_2 * GM_e * (R_e**2 / r**5)
    b = 5 * p.loc[:, "Z"] ** 2 / r**2
    c = -GM_e / r**3 - a * (1 - b)
    xdot.loc[:, "dX"] = (
        (c + omega_e**2) * p.loc[:, "X"]
        + 2 * omega_e * v.loc[:, "dY"]
        + acc_sun_moon.loc[:, "dX2"]
    )
    xdot.loc[:, "dY"] = (
        (c + omega_e**2) * p.loc[:, "Y"]
        - 2 * omega_e * v.loc[:, "dX"]
        + acc_sun_moon.loc[:, "dY2"]
    )
    xdot.loc[:, "dZ"] = (c - 2 * a) * p.loc[:, "Z"] + acc_sun_moon.loc[:, "dZ2"]
    return xdot


def glonass_xdot_montenbruck(x, acc_sun_moon):
    p = x[["X", "Y", "Z"]]
    v = x[["dX", "dY", "dZ"]]
    GM_e = 398600.4418 * 1e9
    R_e = 6378136.0
    J_2 = 1.0826257 * 1e-3
    omega_e = 7.292115 * 1e-5
    xdot = x.copy() * np.nan
    xdot[["X", "Y", "Z"]] = x[["dX", "dY", "dZ"]]
    r = np.linalg.norm(p.to_numpy(), axis=1)
    # How Montenbruck, 2017, Handbook of GNSS, section 3.3.3 computes it:
    c1 = -GM_e / r**3
    c2 = -(3 / 2) * J_2 * GM_e * (R_e**2 / r**5) * (1 - (5 * p.loc[:, "Z"] ** 2) / r**2)
    xdot.loc[:, "dX"] = (
        c1 * p.loc[:, "X"]
        + c2 * p.loc[:, "X"]
        + omega_e**2 * p.loc[:, "X"]
        + 2 * omega_e * v.loc[:, "dY"]
        + acc_sun_moon.loc[:, "dX2"]
    )
    xdot.loc[:, "dY"] = (
        c1 * p.loc[:, "Y"]
        + c2 * p.loc[:, "Y"]
        + omega_e**2 * p.loc[:, "Y"]
        - 2 * omega_e * v.loc[:, "dX"]
        + acc_sun_moon.loc[:, "dY2"]
    )
    xdot.loc[:, "dZ"] = (
        c1 * p.loc[:, "Z"] + c2 * p.loc[:, "Z"] + acc_sun_moon.loc[:, "dZ2"]
    )
    return xdot


def sbas_orbit_position_and_velocity(df):
    # Based on Montenbruck, 2017, Handbook of GNSS, section 3.3.3, eq. 3.59
    t_query = df["query_time_wrt_ephemeris_reference_time_s"].values.reshape(-1, 1)
    df[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]] = (
        df[["X", "Y", "Z"]].values
        + df[["dX", "dY", "dZ"]].mul(t_query, axis=0).values
        + 0.5 * df[["dX2", "dY2", "dZ2"]].mul(t_query**2, axis=0).values
    )
    df[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]] = (
        df[["dX", "dY", "dZ"]].values
        + df[["dX2", "dY2", "dZ2"]].mul(t_query, axis=0).values
    )
    return df


def glonass_orbit_position_and_velocity(df):
    # Based on Montenbruck, 2017, Handbook of GNSS, section 3.3.3
    pv = df[["X", "Y", "Z", "dX", "dY", "dZ"]]
    a = df[["dX2", "dY2", "dZ2"]]
    t = df["query_time_wrt_ephemeris_reference_time_s"] * 0
    t_query = df["query_time_wrt_ephemeris_reference_time_s"]

    while True:
        # We integrate in fixed steps until the last step, which is the time between the next-to-last integrated state
        # and the query time.
        fixed_integration_time_step = 60
        h = (t_query - t).clip(0, fixed_integration_time_step)
        if np.all(h == 0):
            df[
                [
                    "sat_pos_x_m",
                    "sat_pos_y_m",
                    "sat_pos_z_m",
                    "sat_vel_x_mps",
                    "sat_vel_y_mps",
                    "sat_vel_z_mps",
                ]
            ] = pv

            return df
        # One step of 4th order Runge-Kutta integration:
        glonass_xdot = glonass_xdot_rtklib
        k1 = glonass_xdot(pv, a)
        k2 = glonass_xdot(pv + k1.mul(h / 2, axis=0), a)
        k3 = glonass_xdot(pv + k2.mul(h / 2, axis=0), a)
        k4 = glonass_xdot(pv + k3.mul(h, axis=0), a)
        pv = pv + (k1 + 2 * k2 + 2 * k3 + k4).mul(h / 6, axis=0)
        t = t + h


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
    inclination_igso_and_meo_rad = util.deg_2_rad(55)
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
    eph["nu_k"] = 2 * np.arctan(
        np.sqrt((1 + eph.e) / (1 - eph.e)) * np.tan(eph.E_k / 2)
    )
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
    eph["dx_k"] = eph.dr_k * np.cos(eph.u_k) - eph.r_k * eph.du_k * np.sin(eph.u_k)
    eph["dy_k"] = eph.dr_k * np.sin(eph.u_k) + eph.r_k * eph.du_k * np.cos(eph.u_k)
    # We need to know which orbits are Beidou GEOs later on
    eph["is_bds_geo"] = is_bds_geo(eph.constellation, eph.i_k, eph.A)


def orbital_plane_to_earth_centered_cartesian(eph):
    # Corrected longitude of ascending node in ECEF
    eph["Omega_k"] = (
        eph.Omega_0
        + (eph.OmegaDot - eph.OmegaEarthIcd_rps) * eph.t_k
        - eph.OmegaEarthIcd_rps * eph.t_oe
    )
    # Note that eph.loc[eph.is_bds_geo]["Omega_k"] would modify a copy of the GEO slice, not the original DataFrame.
    eph.loc[eph.is_bds_geo, "Omega_k"] = (
        eph[eph.is_bds_geo].Omega_0
        + eph[eph.is_bds_geo].OmegaDot * eph[eph.is_bds_geo].t_k
        - eph[eph.is_bds_geo].OmegaEarthIcd_rps * eph[eph.is_bds_geo].t_oe
    )
    eph["dOmega_k"] = eph.OmegaDot - eph.OmegaEarthIcd_rps
    eph.loc[eph.is_bds_geo, "dOmega_k"] = eph[eph.is_bds_geo].OmegaDot
    # Satellite positions in cartesian frame (for BDS GEOs this is a particular inertial
    # frame, for others the system ECEF frame)
    # For BDS GEOs we apply an additional rotation later-on to compute their position in Beidou's ECEF frame.
    eph["X_k"] = eph.x_k * np.cos(eph.Omega_k) - eph.y_k * np.cos(eph.i_k) * np.sin(
        eph.Omega_k
    )
    eph["Y_k"] = eph.x_k * np.sin(eph.Omega_k) + eph.y_k * np.cos(eph.i_k) * np.cos(
        eph.Omega_k
    )
    eph["Z_k"] = eph.y_k * np.sin(eph.i_k)
    # ECEF velocity, from
    # IS-GPS-200N, Table 20-IV
    eph["dX_k"] = (
        -eph.x_k * eph.dOmega_k * np.sin(eph.Omega_k)
        + eph.dx_k * np.cos(eph.Omega_k)
        - eph.dy_k * np.sin(eph.Omega_k) * np.cos(eph.i_k)
        - eph.y_k * eph.dOmega_k * np.cos(eph.Omega_k) * np.cos(eph.i_k)
        + eph.y_k * eph.di_k * np.sin(eph.Omega_k) * np.sin(eph.i_k)
    )
    eph["dY_k"] = (
        eph.x_k * eph.dOmega_k * np.cos(eph.Omega_k)
        - eph.y_k * eph.dOmega_k * np.sin(eph.Omega_k) * np.cos(eph.i_k)
        - eph.y_k * eph.di_k * np.cos(eph.Omega_k) * np.sin(eph.i_k)
        + eph.dx_k * np.sin(eph.Omega_k)
        + eph.dy_k * np.cos(eph.Omega_k) * np.cos(eph.i_k)
    )

    eph["dZ_k"] = eph.y_k * eph.di_k * np.cos(eph.i_k) + eph.dy_k * np.sin(eph.i_k)
    pass


def handle_bds_geos(eph):
    # Do special rotation from inertial to BDCS (ECEF) frame for Beidou GEO satellites, see
    # Beidou_ICD_B3I_v1.0, Table 5-11
    geos = eph[eph.is_bds_geo]
    if geos.empty:
        return
    P_GK = np.reshape(geos[["X_k", "Y_k", "Z_k"]].to_numpy(), (-1, 1))
    V_GK = np.reshape(geos[["dX_k", "dY_k", "dZ_k"]].to_numpy(), (-1, 1))
    z_angles = geos.OmegaEarthIcd_rps * geos.t_k
    rotation_matrices = []
    for i, z_angle in enumerate(z_angles):
        x_angle = util.deg_2_rad(-5.0)
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
    P_K = np.matmul(R, P_GK)
    P_K = np.reshape(P_K, (-1, 3))
    geos["X_k"] = P_K[:, 0]
    geos["Y_k"] = P_K[:, 1]
    geos["Z_k"] = P_K[:, 2]
    # Velocity in inertial frame that coincides with BDCS at this time, ie a "frozen" ECEF frame
    V_K_frozen = np.matmul(R, V_GK)
    V_K_frozen = np.reshape(V_K_frozen, (-1, 3))
    geos["dX_k"] = V_K_frozen[:, 0]
    geos["dY_k"] = V_K_frozen[:, 1]
    geos["dZ_k"] = V_K_frozen[:, 2]

    # Add term due to ECEFs angular velocity w.r.t. the frozen frame

    def frozen_to_rotating_bdcs(row):
        p = np.array([row["X_k"], row["Y_k"], row["Z_k"]])
        v_frozen = np.array([row["dX_k"], row["dY_k"], row["dZ_k"]])
        v_rotating = v_frozen + np.cross(np.array([0, 0, -row.OmegaEarthIcd_rps]), p)
        row[["dX_k", "dY_k", "dZ_k"]] = v_rotating
        return row

    geos = geos.apply(frozen_to_rotating_bdcs, axis=1)
    eph[eph.is_bds_geo] = geos


# Adapted from gnss_lib_py's find_sat()
def kepler_orbit_position_and_velocity(eph):
    eph["gps_week"] = eph["GPSWeek"]
    eph["gnss_id"] = eph["constellation"]
    eph["sv_id"] = eph["sv"]

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
    eph = eph.rename(
        columns={
            "X_k": "sat_pos_x_m",
            "Y_k": "sat_pos_y_m",
            "Z_k": "sat_pos_z_m",
            "dX_k": "sat_vel_x_mps",
            "dY_k": "sat_vel_y_mps",
            "dZ_k": "sat_vel_z_mps",
        },
    )
    return eph


def set_time_of_validity(df):
    def set_for_one_constellation(group):
        group_constellation = group["constellation"].iloc[0]
        group["validity_start"] = (
            group["ephemeris_reference_time_isagpst"]
            + constants.constellation_2_ephemeris_validity_interval[
                group_constellation
            ][0]
        )
        group["validity_end"] = (
            group["ephemeris_reference_time_isagpst"]
            + constants.constellation_2_ephemeris_validity_interval[
                group_constellation
            ][1]
        )
        return group

    df = df.groupby("constellation").apply(set_for_one_constellation)
    return df


def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame"""
    df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    df = df.dropna(how="all")
    df = df.reset_index()
    df["source"] = nav_ds.filename
    # georinex adds suffixes to satellite IDs if it sees multiple ephemerides (e.g. F/NAV, I/NAV) for the same
    # satellite and the same timestamp.
    # The downstream code expects three-letter satellite IDs, so remove suffixes.
    df["sv"] = df.sv.str[:3]
    df["constellation"] = df["sv"].str[0]
    df["time_scale"] = df["constellation"].replace(
        constants.constellation_2_system_time_scale
    )

    # Keep only SBAS ephemerides with low URA, these are based on SBAS message type 9, while those with large URA are
    # based on message type 17 which only contains almanac-like coarse satellite position estimates.
    if "URA" in df.columns:
        df = df[~((df.sv.str.startswith("S")) & (df.URA >= constants.cSbasURALimit))]

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
        if group_time_scale not in ["GLONASST", "SBAST"]:
            full_seconds = (
                group[week_field[group_time_scale]] * constants.cSecondsPerWeek
                + group["Toe"]
            )
            group["ephemeris_reference_time_system_time"] = (
                constants.system_time_scale_rinex_utc_epoch[group_time_scale]
                + pd.to_timedelta(full_seconds, unit="seconds")
            )
        else:
            # For SBAS and GLONASS there are no separate ephemeris reference time fields
            group["ephemeris_reference_time_system_time"] = group["time"]
            # The first derivative of the clock offset is in a different field for SBAS and GLONASS
            group["SVclockDrift"] = group["SVrelFreqBias"]
            # And the second derivative is zero, i.e. the constellation ground segment uses a fist-order clock model
            group["SVclockDriftRate"] = 0
        group["ephemeris_reference_time_isagpst"] = to_isagpst(
            group["ephemeris_reference_time_system_time"],
            group_time_scale,
            int(nav_ds.attrs["utc_gpst_leap_seconds"]),
        )
        group["clock_offset_reference_time_system_time"] = group["time"]
        group["clock_reference_time_isagpst"] = to_isagpst(
            group["clock_offset_reference_time_system_time"],
            group_time_scale,
            int(nav_ds.attrs["utc_gpst_leap_seconds"]),
        )
        return group

    df = (
        df.groupby("constellation")
        .apply(compute_ephemeris_and_clock_offset_reference_times)
        .reset_index(drop=True)
    )
    df = set_time_of_validity(df)
    df = df.rename(
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
    )
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
    df = df.reset_index(drop=True)
    df = compute_gal_inav_fnav_indicators(df)
    df["frequency_slot"] = df.FreqNum.where(df.sv.str[0] == "R", 1).astype(int)
    df.attrs["ionospheric_corr_GPS"] = nav_ds.ionospheric_corr_GPS
    return df


def compute_gal_inav_fnav_indicators(df):
    """
    Based on RINEX 3.05, section A8
    """
    df["fnav_or_inav"] = ""
    is_gal = df.sv.str[0] == "E"
    df.loc[is_gal, "fnav_or_inav_indicator"] = np.bitwise_and(
        df[is_gal].DataSrc.astype(np.uint).to_numpy(), 0b111
    )
    # We expect only the following navigation message types for Galileo:
    indicators = set(df[is_gal].fnav_or_inav_indicator.unique())
    assert len(indicators.intersection({1, 2, 4, 5})) == len(
        indicators
    ), f"Unexpected Galileo navigation message type: {indicators}"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 1), "fnav_or_inav"] = "inav"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 2), "fnav_or_inav"] = "fnav"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 4), "fnav_or_inav"] = "inav"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 5), "fnav_or_inav"] = "inav"
    return df


def to_isagpst(time, timescale, gpst_utc_leapseconds):
    if (isinstance(time, pd.Timedelta) or isinstance(time, pd.Series)) and isinstance(
        timescale, str
    ):
        return time - time_scale_integer_second_offset_wrt_gpst(
            timescale, gpst_utc_leapseconds
        )
    if isinstance(time, pd.Series) and isinstance(timescale, pd.Series):
        return time - timescale.apply(
            lambda element: time_scale_integer_second_offset_wrt_gpst(
                element, gpst_utc_leapseconds
            )
        )

    assert (
        False
    ), f"Unexpected types: time is {type(time)}, timescale is {type(timescale)}"


@timeit
def select_ephemerides(df, query):
    df = df[df.ephemeris_reference_time_isagpst.notna()]
    query = query.sort_values(by="ephemeris_selection_time_isagpst")
    df = df.sort_values(by="ephemeris_reference_time_isagpst")
    # Add fnav/inav indicator to query for to select the FNAV ephemeris for E5b signals, and INAV for other signals
    query["fnav_or_inav"] = ""
    query.loc[
        (query.sv.str[0] == "E") & (query.signal.str[1] == "5"), "fnav_or_inav"
    ] = "fnav"
    query.loc[
        (query.sv.str[0] == "E") & (query.signal.str[1] != "5"), "fnav_or_inav"
    ] = "inav"
    query = pd.merge_asof(
        query,
        df,
        left_on="ephemeris_selection_time_isagpst",
        right_on="ephemeris_reference_time_isagpst",
        by=["sv", "fnav_or_inav"],
        direction="backward",
    )
    # Compute times w.r.t. orbit and clock reference times used by downstream computations
    query["query_time_wrt_ephemeris_reference_time_s"] = (
        query["query_time_isagpst"] - query["ephemeris_reference_time_isagpst"]
    ).apply(util.timedelta_2_seconds)
    query["query_time_wrt_clock_reference_time_s"] = (
        query["query_time_isagpst"] - query["clock_reference_time_isagpst"]
    ).apply(util.timedelta_2_seconds)
    query["ephemeris_valid"] = (query["query_time_isagpst"] < query["validity_end"]) & (
        query["query_time_isagpst"] > query["validity_start"]
    )
    return query


def compute_clock_offsets(df):
    df["sat_clock_offset_m"] = constants.cGpsSpeedOfLight_mps * (
        df["SVclockBias"]
        + df["SVclockDrift"] * df["query_time_wrt_clock_reference_time_s"]
        + df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"] ** 2
    )
    df["sat_clock_drift_mps"] = constants.cGpsSpeedOfLight_mps * (
        df["SVclockDrift"]
        + 2 * df["SVclockDriftRate"] * df["query_time_wrt_clock_reference_time_s"]
    )
    return df


def compute_parallel(rinex_nav_file_paths, per_signal_query):
    if not isinstance(rinex_nav_file_paths, list):
        rinex_nav_file_paths = [rinex_nav_file_paths]
    # Warm up nav file parser cache so that we don't parse the file multiple times
    for rinex_nav_file_path in rinex_nav_file_paths:
        _ = parse_rinex_nav_file(rinex_nav_file_path)
    parallel = Parallel(n_jobs=round(multiprocessing.cpu_count() / 2), return_as="list")
    # split dataframe into `n_chunks` smaller dataframes
    n_chunks = min(len(per_signal_query.index), 4)
    chunks = np.array_split(per_signal_query, n_chunks)
    processed_chunks = parallel(
        delayed(compute)(rinex_nav_file_paths, chunk) for chunk in chunks
    )
    return pd.concat(processed_chunks)


def compute(rinex_nav_file_paths, per_signal_query):
    if not isinstance(rinex_nav_file_paths, list):
        rinex_nav_file_paths = [rinex_nav_file_paths]
    query_columns = per_signal_query.columns
    # per_signal_query is a pd.DataFrame with the following columns
    #   - time_of_reception_in_receiver_time
    #   - observation_value
    #   - signal
    #   - sv
    #   - query_time_isagpst
    rinex_nav_file_paths = [Path(path) for path in rinex_nav_file_paths]
    ephemeris_blocks = []
    for path in rinex_nav_file_paths:
        block = parse_rinex_nav_file(path)
        # Not using iono model parameters here, removing them from dataframe attributes to
        # enable concatenation
        block.attrs.pop("ionospheric_corr_GPS", None)
        ephemeris_blocks.append(block)
    ephemerides = pd.concat(ephemeris_blocks)
    # Group delays and clock offsets can be signal-specific, so we need to match ephemerides to code signals,
    # not only to satellites
    # Example: Galileo transmits E5a clock and group delay parameters in the F/NAV message, but parameters for other
    # signals in the I/NAV message
    if "ephemeris_selection_time_isagpst" not in per_signal_query.columns:
        per_signal_query["ephemeris_selection_time_isagpst"] = per_signal_query[
            "query_time_isagpst"
        ]
    per_signal_query = select_ephemerides(ephemerides, per_signal_query)
    per_signal_query = compute_clock_offsets(per_signal_query)
    # Compute orbital states for each (satellite,ephemeris) pair only once:
    per_sat_eph_query = (
        per_signal_query.groupby(["sv", "query_time_isagpst", "ephemeris_hash"])
        .first()
        .reset_index()
    )
    per_sat_eph_query = per_sat_eph_query.drop(
        columns=["sat_clock_offset_m", "sat_clock_drift_mps"]
    )

    def evaluate_orbit(sub_df):
        orbit_type = sub_df["orbit_type"].iloc[0]
        if orbit_type == "kepler":
            sub_df = kepler_orbit_position_and_velocity(sub_df)
        elif orbit_type == "glonass":
            sub_df = glonass_orbit_position_and_velocity(sub_df)
        elif orbit_type == "sbas":
            sub_df = sbas_orbit_position_and_velocity(sub_df)
        else:
            log.info(
                f"Ephemeris evaluation not implemented or under development for constellation {sub_df['constellation'].iloc[0]}, skipping"
            )
            sub_df[["x_m", "y_m", "z_m", "dx_mps", "dy_mps", "dz_mps"]] = np.nan
        return sub_df

    per_sat_eph_query = per_sat_eph_query.groupby("orbit_type").apply(evaluate_orbit)
    per_sat_eph_query = per_sat_eph_query.reset_index(drop=True)
    columns_to_keep = [
        "sv",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "query_time_isagpst",
        "ephemeris_hash",
    ]
    per_sat_eph_query = per_sat_eph_query[columns_to_keep]
    # Merge the computed satellite states into the larger signal-specific query dataframe
    per_signal_query = per_signal_query.merge(
        per_sat_eph_query, on=["sv", "query_time_isagpst", "ephemeris_hash"]
    )
    columns_to_keep = [
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
    ] + columns_to_keep
    per_signal_query = compute_total_group_delays(per_signal_query)

    if "signal" in per_signal_query.columns:
        columns_to_keep = ["signal", "sat_code_bias_m"] + columns_to_keep
    columns_to_keep.append("frequency_slot")
    computed_columns_to_keep = [
        col for col in columns_to_keep if col not in query_columns
    ]
    # per_signal_query.loc[
    #    ~per_signal_query.ephemeris_valid, computed_columns_to_keep
    # ] = np.nan
    per_signal_query = per_signal_query[columns_to_keep].reset_index(drop=True)
    return per_signal_query


def compute_total_group_delays(
    query,
):
    """
    This computes TGD terms, not ISCs, which are not captured by RINEX 3 - we'll have them with RINEX 4.
    References:
    - GPS: IS-GPS-200N, §20.3.3.3.3.2, IS-GPS-705J, §20.3.3.3.1.2.
    - Galileo: Galileo_OS_SIS_ICD_v2.0, §5.1.5
    - QZSS: is-qzss-pnt-005 §5.8.2
    - Beidou B1I, B2I, B3I: Beidou_ICD_B1I_v3.0, §5.2.4.10
    - Beidou B1Cp, B1Cd: Beidou_ICD_B1C_v1.0, §7.6.2 (not supported by rnx3)
    - Beidou B2bi: Beidou_ICD_B2b_v1.0, §7.5.2 (not supported by rnx3)
    """
    if "signal" not in query.columns:
        query["group_delay"] = np.nan
        return query
    query["constellation"] = query["sv"].str[0]
    query["frequency_code"] = query["signal"].str[1]
    query["speedOfLightIcd_mps"] = query.constellation.map(
        {
            "C": constants.cBdsSpeedOfLight_mps,
            "G": constants.cGpsSpeedOfLight_mps,
            "E": constants.cGalSpeedOfLight_mps,
            "J": constants.cQzssSpeedOfLight_mps,
        }
    )

    def compute_tgds(df):
        assert len(df.constellation.unique()) == 1
        assert len(df.signal.unique()) == 1

        df["gamma"] = np.nan
        df["tgd"] = np.nan

        match df.at[df.index[0], "constellation"]:
            case "G":
                df.tgd = df.TGD.values
                match df.at[df.index[0], "frequency_code"]:
                    case "1":
                        df.gamma = 1
                    case "2":
                        df.gamma = (
                            constants.carrier_frequencies_hz()["G"]["L1"][1]
                            / constants.carrier_frequencies_hz()["G"]["L2"][1]
                        ) ** 2
            case "J":
                df.tgd = df.TGD.values
                df.gamma = 1
            case "E":
                match df.at[df.index[0], "frequency_code"]:
                    case "1":
                        df.tgd = df.BGDe5b.values
                        df.gamma = 1
                    case "5":
                        df.tgd = df.BGDe5a.values
                        df.gamma = (
                            constants.carrier_frequencies_hz()["E"]["L1"][1]
                            / constants.carrier_frequencies_hz()["E"]["L5"][1]
                        ) ** 2
                    case "7":
                        df.tgd = df.BGDe5b.values
                        df.gamma = (
                            constants.carrier_frequencies_hz()["E"]["L1"][1]
                            / constants.carrier_frequencies_hz()["E"]["L7"][1]
                        ) ** 2
            case "C":
                df.gamma = 1
                match df.at[df.index[0], "signal"]:
                    case "C2I":  # called B1I in Beidou ICD
                        df.tgd = df.TGD1.values
                    case "C7I":  # called B2I in Beidou ICD
                        df.tgd = df.TGD2.values
                    case "C6I":  # called B3I in Beidou ICD
                        df.tgd = 0
        df["sat_code_bias_m"] = df.tgd * df.gamma * df.speedOfLightIcd_mps
        return df

    query = query.groupby(["signal", "constellation"]).apply(compute_tgds)
    return query


if __name__ == "main":
    pass
