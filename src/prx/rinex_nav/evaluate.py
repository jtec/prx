import logging
from typing import Union, List, Optional, Any

import pandas as pd
import numpy as np
from pathlib import Path
from prx import util
from prx import constants
from prx.rinex_nav.evaluate_pandas import parse_rinex_nav_file
from prx.util import timeit
import polars as pl

log = logging.getLogger(__name__)


def time_scale_integer_second_offset_wrt_gpst(
    time_scale: str, utc_gpst_leap_seconds: Optional[int] = None
) -> pd.Timedelta:
    if time_scale in ["GPST", "SBAST", "QZSST", "IRNSST", "GST"]:
        return pd.Timedelta(seconds=0)
    if time_scale == "BDT":
        return pd.Timedelta(seconds=-14)
    if time_scale == "GLONASST":
        assert utc_gpst_leap_seconds is not None, (
            "Need GPST-UTC leap seconds to compute GLONASST integer second offset w.r.t. GPST"
        )
        return pd.Timedelta(seconds=-utc_gpst_leap_seconds)
    assert False, f"Unexpected time scale: {time_scale}"


@timeit
def glonass_xdot_rtklib(
    x: Union[pd.DataFrame, pl.DataFrame],
    acc_sun_moon: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    # Use polars operations directly
    if hasattr(x, "to_pandas"):
        # Pure polars implementation
        GM_e = 398600.4418 * 1e9
        R_e = 6378136.0
        J_2 = 1.0826257 * 1e-3
        omega_e = 7.292115 * 1e-5

        # Calculate radius using polars
        r = (x.select(["X", "Y", "Z"]).to_numpy() ** 2).sum(axis=1) ** 0.5

        # Compute coefficients
        a = 1.5 * J_2 * GM_e * (R_e**2 / r**5)
        b = 5 * x.select("Z").to_numpy().flatten() ** 2 / r**2
        c = -GM_e / r**3 - a * (1 - b)

        # Create result dataframe with computed derivatives
        xdot = x.with_columns(
            [
                x.select("dX").to_series().alias("X"),
                x.select("dY").to_series().alias("Y"),
                x.select("dZ").to_series().alias("Z"),
                pl.lit(
                    (
                        (c + omega_e**2) * x.select("X").to_numpy().flatten()
                        + 2 * omega_e * x.select("dY").to_numpy().flatten()
                        + acc_sun_moon.select("dX2").to_numpy().flatten()
                    )
                ).alias("dX"),
                pl.lit(
                    (
                        (c + omega_e**2) * x.select("Y").to_numpy().flatten()
                        - 2 * omega_e * x.select("dX").to_numpy().flatten()
                        + acc_sun_moon.select("dY2").to_numpy().flatten()
                    )
                ).alias("dY"),
                pl.lit(
                    (
                        (c - 2 * a) * x.select("Z").to_numpy().flatten()
                        + acc_sun_moon.select("dZ2").to_numpy().flatten()
                    )
                ).alias("dZ"),
            ]
        )
        return xdot
    else:
        # Fallback to pandas for non-polars input
        p = x[["X", "Y", "Z"]]
        v = x[["dX", "dY", "dZ"]]
        GM_e = 398600.4418 * 1e9
        R_e = 6378136.0
        J_2 = 1.0826257 * 1e-3
        omega_e = 7.292115 * 1e-5
        xdot = x.copy() * np.nan
        xdot[["X", "Y", "Z"]] = x[["dX", "dY", "dZ"]]
        r = np.linalg.norm(p.to_numpy(), axis=1)
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


@timeit
def glonass_xdot_montenbruck(
    x: Union[pd.DataFrame, pl.DataFrame],
    acc_sun_moon: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    # Use polars operations directly
    if hasattr(x, "to_pandas"):
        # Pure polars implementation
        GM_e = 398600.4418 * 1e9
        R_e = 6378136.0
        J_2 = 1.0826257 * 1e-3
        omega_e = 7.292115 * 1e-5

        # Calculate radius using polars
        r = (x.select(["X", "Y", "Z"]).to_numpy() ** 2).sum(axis=1) ** 0.5

        # Extract position arrays
        X = x.select("X").to_numpy().flatten()
        Y = x.select("Y").to_numpy().flatten()
        Z = x.select("Z").to_numpy().flatten()
        dY = x.select("dY").to_numpy().flatten()
        dX = x.select("dX").to_numpy().flatten()

        # Compute coefficients
        c1 = -GM_e / r**3
        c2 = -(3 / 2) * J_2 * GM_e * (R_e**2 / r**5) * (1 - (5 * Z**2) / r**2)

        # Create result dataframe with computed derivatives
        xdot = x.with_columns(
            [
                x.select("dX").to_series().alias("X"),
                x.select("dY").to_series().alias("Y"),
                x.select("dZ").to_series().alias("Z"),
                pl.lit(
                    (
                        c1 * X
                        + c2 * X
                        + omega_e**2 * X
                        + 2 * omega_e * dY
                        + acc_sun_moon.select("dX2").to_numpy().flatten()
                    )
                ).alias("dX"),
                pl.lit(
                    (
                        c1 * Y
                        + c2 * Y
                        + omega_e**2 * Y
                        - 2 * omega_e * dX
                        + acc_sun_moon.select("dY2").to_numpy().flatten()
                    )
                ).alias("dY"),
                pl.lit(
                    (c1 * Z + c2 * Z + acc_sun_moon.select("dZ2").to_numpy().flatten())
                ).alias("dZ"),
            ]
        )
        return xdot
    else:
        # Fallback to pandas for non-polars input
        p = x[["X", "Y", "Z"]]
        v = x[["dX", "dY", "dZ"]]
        GM_e = 398600.4418 * 1e9
        R_e = 6378136.0
        J_2 = 1.0826257 * 1e-3
        omega_e = 7.292115 * 1e-5
        xdot = x.copy() * np.nan
        xdot[["X", "Y", "Z"]] = x[["dX", "dY", "dZ"]]
        r = np.linalg.norm(p.to_numpy(), axis=1)
        c1 = -GM_e / r**3
        c2 = (
            -(3 / 2)
            * J_2
            * GM_e
            * (R_e**2 / r**5)
            * (1 - (5 * p.loc[:, "Z"] ** 2) / r**2)
        )
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


@timeit
def sbas_orbit_position_and_velocity(
    df: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    # Based on Montenbruck, 2017, Handbook of GNSS, section 3.3.3, eq. 3.59
    if hasattr(df, "to_pandas"):
        # Pure polars implementation
        t_query = (
            df.select("query_time_wrt_ephemeris_reference_time_s").to_numpy().flatten()
        )

        # Position calculation
        X = df.select("X").to_numpy().flatten()
        Y = df.select("Y").to_numpy().flatten()
        Z = df.select("Z").to_numpy().flatten()
        dX = df.select("dX").to_numpy().flatten()
        dY = df.select("dY").to_numpy().flatten()
        dZ = df.select("dZ").to_numpy().flatten()
        dX2 = df.select("dX2").to_numpy().flatten()
        dY2 = df.select("dY2").to_numpy().flatten()
        dZ2 = df.select("dZ2").to_numpy().flatten()

        return df.with_columns(
            [
                pl.lit(X + dX * t_query + 0.5 * dX2 * t_query**2).alias("sat_pos_x_m"),
                pl.lit(Y + dY * t_query + 0.5 * dY2 * t_query**2).alias("sat_pos_y_m"),
                pl.lit(Z + dZ * t_query + 0.5 * dZ2 * t_query**2).alias("sat_pos_z_m"),
                pl.lit(dX + dX2 * t_query).alias("sat_vel_x_mps"),
                pl.lit(dY + dY2 * t_query).alias("sat_vel_y_mps"),
                pl.lit(dZ + dZ2 * t_query).alias("sat_vel_z_mps"),
            ]
        )
    else:
        # Fallback to pandas for non-polars input
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


@timeit
def glonass_orbit_position_and_velocity(
    df: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    # Based on Montenbruck, 2017, Handbook of GNSS, section 3.3.3
    if hasattr(df, "to_pandas"):
        # For complex Runge-Kutta integration, we need to work with pandas temporarily
        # This is because the integration requires complex operations that are hard to vectorize in polars
        df_pd = df.to_pandas()

        pv = df_pd[["X", "Y", "Z", "dX", "dY", "dZ"]]
        a = df_pd[["dX2", "dY2", "dZ2"]]
        t = df_pd["query_time_wrt_ephemeris_reference_time_s"] * 0
        t_query = df_pd["query_time_wrt_ephemeris_reference_time_s"]

        while True:
            # We integrate in fixed steps until the last step, which is the time between the next-to-last integrated state
            # and the query time.
            fixed_integration_time_step = 60
            h = (t_query - t).clip(0, fixed_integration_time_step)
            if np.all(h == 0):
                # Convert back to polars with computed results
                result = df.with_columns(
                    [
                        pl.lit(pv["X"].values).alias("sat_pos_x_m"),
                        pl.lit(pv["Y"].values).alias("sat_pos_y_m"),
                        pl.lit(pv["Z"].values).alias("sat_pos_z_m"),
                        pl.lit(pv["dX"].values).alias("sat_vel_x_mps"),
                        pl.lit(pv["dY"].values).alias("sat_vel_y_mps"),
                        pl.lit(pv["dZ"].values).alias("sat_vel_z_mps"),
                    ]
                )
                return result
            # One step of 4th order Runge-Kutta integration:
            glonass_xdot = glonass_xdot_rtklib
            k1 = glonass_xdot(pv, a)
            k2 = glonass_xdot(pv + k1.mul(h / 2, axis=0), a)
            k3 = glonass_xdot(pv + k2.mul(h / 2, axis=0), a)
            k4 = glonass_xdot(pv + k3.mul(h, axis=0), a)
            pv = pv + (k1 + 2 * k2 + 2 * k3 + k4).mul(h / 6, axis=0)
            t = t + h
    else:
        # Fallback to pandas for non-polars input
        pv = df[["X", "Y", "Z", "dX", "dY", "dZ"]]
        a = df[["dX2", "dY2", "dZ2"]]
        t = df["query_time_wrt_ephemeris_reference_time_s"] * 0
        t_query = df["query_time_wrt_ephemeris_reference_time_s"]

        while True:
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
            glonass_xdot = glonass_xdot_rtklib
            k1 = glonass_xdot(pv, a)
            k2 = glonass_xdot(pv + k1.mul(h / 2, axis=0), a)
            k3 = glonass_xdot(pv + k2.mul(h / 2, axis=0), a)
            k4 = glonass_xdot(pv + k3.mul(h, axis=0), a)
            pv = pv + (k1 + 2 * k2 + 2 * k3 + k4).mul(h / 6, axis=0)
            t = t + h


def compute_eccentric_anomaly_expr(max_iter: int = 10) -> pl.Expr:
    """Create polars expression to compute eccentric anomaly using Newton-Raphson iteration"""
    # For polars, we need to create an iterative expression
    # Start with M as initial guess for E
    e_expr = pl.col("M_k")

    # Perform Newton-Raphson iterations
    for i in range(max_iter):
        # E_new = E - (E - e*sin(E) - M) / (1 - e*cos(E))
        e_expr = e_expr - (
            (e_expr - pl.col("e") * e_expr.sin() - pl.col("M_k"))
            / (1 - pl.col("e") * e_expr.cos())
        )

    return e_expr


def is_bds_geo_polars() -> pl.Expr:
    """Create polars expression to identify Beidou GEO satellites"""
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
        meo_approximate_radius_m + geo_and_igso_approximate_radius_m
    ) / 2
    inclination_threshold_rad = inclination_igso_and_meo_rad / 2

    return (
        (pl.col("A") > radius_threshold_m)
        & (pl.col("i_k") < inclination_threshold_rad)
        & (pl.col("constellation") == "C")
    )


@timeit
def position_in_orbital_plane(eph: pl.DataFrame) -> pl.DataFrame:
    return (
        eph.with_columns(
            [
                # Semi-major axis
                (pl.col("sqrtA") ** 2).alias("A"),
            ]
        )
        .with_columns(
            [
                # Computed mean motion
                (pl.col("MuEarthIcd_m3ps2") / pl.col("A") ** 3).sqrt().alias("n_0"),
                # Time since ephemeris reference epoch
                pl.col("query_time_wrt_ephemeris_reference_time_s").alias("t_k"),
            ]
        )
        .with_columns(
            [
                # Corrected mean motion
                (pl.col("n_0") + pl.col("deltaN")).alias("n"),
            ]
        )
        .with_columns(
            [
                # Computed mean anomaly
                (pl.col("M_0") + pl.col("n") * pl.col("t_k")).alias("M_k"),
            ]
        )
        .with_columns(
            [
                # Eccentric Anomaly using iterative Newton-Raphson
                compute_eccentric_anomaly_expr().alias("E_k"),
            ]
        )
        .with_columns(
            [
                # Computed true anomaly
                (
                    2
                    * (
                        (((1 + pl.col("e")) / (1 - pl.col("e"))).sqrt())
                        * (pl.col("E_k") / 2).tan()
                    ).arctan()
                ).alias("nu_k"),
            ]
        )
        .with_columns(
            [
                # Computed argument of latitude
                (pl.col("nu_k") + pl.col("omega")).alias("phi_k"),
            ]
        )
        .with_columns(
            [
                # Corrections
                (
                    pl.col("C_us") * (2 * pl.col("phi_k")).sin()
                    + pl.col("C_uc") * (2 * pl.col("phi_k")).cos()
                ).alias("delta_u_k"),
                (
                    pl.col("C_rs") * (2 * pl.col("phi_k")).sin()
                    + pl.col("C_rc") * (2 * pl.col("phi_k")).cos()
                ).alias("delta_r_k"),
                (
                    pl.col("C_is") * (2 * pl.col("phi_k")).sin()
                    + pl.col("C_ic") * (2 * pl.col("phi_k")).cos()
                ).alias("delta_i_k"),
            ]
        )
        .with_columns(
            [
                # Corrected values
                (pl.col("phi_k") + pl.col("delta_u_k")).alias("u_k"),
                (
                    pl.col("A") * (1 - pl.col("e") * pl.col("E_k").cos())
                    + pl.col("delta_r_k")
                ).alias("r_k"),
                (
                    pl.col("i_0") + pl.col("IDOT") * pl.col("t_k") + pl.col("delta_i_k")
                ).alias("i_k"),
            ]
        )
        .with_columns(
            [
                # Satellite positions in orbital plane
                (pl.col("r_k") * pl.col("u_k").cos()).alias("x_k"),
                (pl.col("r_k") * pl.col("u_k").sin()).alias("y_k"),
                # Derivatives for velocity computation
                (pl.col("n") / (1 - pl.col("e") * pl.col("E_k").cos())).alias("dE_k"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("dE_k")
                    * (1 - pl.col("e") ** 2).sqrt()
                    / (1 - pl.col("e") * pl.col("E_k").cos())
                ).alias("dnu_k"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("IDOT")
                    + 2
                    * pl.col("dnu_k")
                    * (
                        pl.col("C_is") * (2.0 * pl.col("phi_k")).cos()
                        - pl.col("C_ic") * (2.0 * pl.col("phi_k")).sin()
                    )
                ).alias("di_k"),
                (
                    pl.col("dnu_k")
                    + 2
                    * pl.col("dnu_k")
                    * (
                        pl.col("C_us") * (2.0 * pl.col("phi_k")).cos()
                        - pl.col("C_uc") * (2.0 * pl.col("phi_k")).sin()
                    )
                ).alias("du_k"),
                (
                    pl.col("e") * pl.col("A") * pl.col("dE_k") * pl.col("E_k").sin()
                    + 2
                    * pl.col("dnu_k")
                    * (
                        pl.col("C_rs") * (2.0 * pl.col("phi_k")).cos()
                        - pl.col("C_rc") * (2.0 * pl.col("phi_k")).sin()
                    )
                ).alias("dr_k"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("dr_k") * pl.col("u_k").cos()
                    - pl.col("r_k") * pl.col("du_k") * pl.col("u_k").sin()
                ).alias("dx_k"),
                (
                    pl.col("dr_k") * pl.col("u_k").sin()
                    + pl.col("r_k") * pl.col("du_k") * pl.col("u_k").cos()
                ).alias("dy_k"),
            ]
        )
        .with_columns(
            [
                # Determine if this is a Beidou GEO satellite
                is_bds_geo_polars().alias("is_bds_geo"),
            ]
        )
    )


@timeit
def orbital_plane_to_earth_centered_cartesian(eph: pl.DataFrame) -> pl.DataFrame:
    return (
        eph.with_columns(
            [
                # Corrected longitude of ascending node in ECEF
                # Different calculation for BDS GEO vs others
                pl.when(pl.col("is_bds_geo"))
                .then(
                    pl.col("Omega_0")
                    + pl.col("OmegaDot") * pl.col("t_k")
                    - pl.col("OmegaEarthIcd_rps") * pl.col("t_oe")
                )
                .otherwise(
                    pl.col("Omega_0")
                    + (pl.col("OmegaDot") - pl.col("OmegaEarthIcd_rps")) * pl.col("t_k")
                    - pl.col("OmegaEarthIcd_rps") * pl.col("t_oe")
                )
                .alias("Omega_k"),
                # dOmega_k calculation
                pl.when(pl.col("is_bds_geo"))
                .then(pl.col("OmegaDot"))
                .otherwise(pl.col("OmegaDot") - pl.col("OmegaEarthIcd_rps"))
                .alias("dOmega_k"),
            ]
        )
        .with_columns(
            [
                # Satellite positions in cartesian frame
                (
                    pl.col("x_k") * pl.col("Omega_k").cos()
                    - pl.col("y_k") * pl.col("i_k").cos() * pl.col("Omega_k").sin()
                ).alias("X_k"),
                (
                    pl.col("x_k") * pl.col("Omega_k").sin()
                    + pl.col("y_k") * pl.col("i_k").cos() * pl.col("Omega_k").cos()
                ).alias("Y_k"),
                (pl.col("y_k") * pl.col("i_k").sin()).alias("Z_k"),
            ]
        )
        .with_columns(
            [
                # ECEF velocity components
                (
                    -pl.col("x_k") * pl.col("dOmega_k") * pl.col("Omega_k").sin()
                    + pl.col("dx_k") * pl.col("Omega_k").cos()
                    - pl.col("dy_k") * pl.col("Omega_k").sin() * pl.col("i_k").cos()
                    - pl.col("y_k")
                    * pl.col("dOmega_k")
                    * pl.col("Omega_k").cos()
                    * pl.col("i_k").cos()
                    + pl.col("y_k")
                    * pl.col("di_k")
                    * pl.col("Omega_k").sin()
                    * pl.col("i_k").sin()
                ).alias("dX_k"),
                (
                    pl.col("x_k") * pl.col("dOmega_k") * pl.col("Omega_k").cos()
                    - pl.col("y_k")
                    * pl.col("dOmega_k")
                    * pl.col("Omega_k").sin()
                    * pl.col("i_k").cos()
                    - pl.col("y_k")
                    * pl.col("di_k")
                    * pl.col("Omega_k").cos()
                    * pl.col("i_k").sin()
                    + pl.col("dx_k") * pl.col("Omega_k").sin()
                    + pl.col("dy_k") * pl.col("Omega_k").cos() * pl.col("i_k").cos()
                ).alias("dY_k"),
                (
                    pl.col("y_k") * pl.col("di_k") * pl.col("i_k").cos()
                    + pl.col("dy_k") * pl.col("i_k").sin()
                ).alias("dZ_k"),
            ]
        )
    )


@timeit
def handle_bds_geos(eph: pl.DataFrame) -> pl.DataFrame:
    """Handle special rotation for Beidou GEO satellites using polars expressions"""
    # Do special rotation from inertial to BDCS (ECEF) frame for Beidou GEO satellites, see
    # Beidou_ICD_B3I_v1.0, Table 5-11
    x_angle = util.deg_2_rad(-5.0)
    cos_x = np.cos(x_angle)
    sin_x = np.sin(x_angle)

    return (
        eph.with_columns(
            [
                # Calculate z_angle for each satellite
                (pl.col("OmegaEarthIcd_rps") * pl.col("t_k")).alias("z_angle"),
            ]
        )
        .with_columns(
            [
                # For BDS GEO satellites, apply combined rotation Rz * Rx
                # Rx rotates by -5 degrees around X axis, Rz rotates by z_angle around Z axis
                # Combined rotation matrix elements for position
                pl.when(pl.col("is_bds_geo"))
                .then(
                    pl.col("X_k") * pl.col("z_angle").cos()
                    + pl.col("Y_k") * pl.col("z_angle").sin() * cos_x
                    + pl.col("Z_k") * pl.col("z_angle").sin() * sin_x
                )
                .otherwise(pl.col("X_k"))
                .alias("X_k_rotated"),
                pl.when(pl.col("is_bds_geo"))
                .then(
                    -pl.col("X_k") * pl.col("z_angle").sin()
                    + pl.col("Y_k") * pl.col("z_angle").cos() * cos_x
                    + pl.col("Z_k") * pl.col("z_angle").cos() * sin_x
                )
                .otherwise(pl.col("Y_k"))
                .alias("Y_k_rotated"),
                pl.when(pl.col("is_bds_geo"))
                .then(-pl.col("Y_k") * sin_x + pl.col("Z_k") * cos_x)
                .otherwise(pl.col("Z_k"))
                .alias("Z_k_rotated"),
                # Apply same rotation to velocity (frozen frame)
                pl.when(pl.col("is_bds_geo"))
                .then(
                    pl.col("dX_k") * pl.col("z_angle").cos()
                    + pl.col("dY_k") * pl.col("z_angle").sin() * cos_x
                    + pl.col("dZ_k") * pl.col("z_angle").sin() * sin_x
                )
                .otherwise(pl.col("dX_k"))
                .alias("dX_k_frozen"),
                pl.when(pl.col("is_bds_geo"))
                .then(
                    -pl.col("dX_k") * pl.col("z_angle").sin()
                    + pl.col("dY_k") * pl.col("z_angle").cos() * cos_x
                    + pl.col("dZ_k") * pl.col("z_angle").cos() * sin_x
                )
                .otherwise(pl.col("dY_k"))
                .alias("dY_k_frozen"),
                pl.when(pl.col("is_bds_geo"))
                .then(-pl.col("dY_k") * sin_x + pl.col("dZ_k") * cos_x)
                .otherwise(pl.col("dZ_k"))
                .alias("dZ_k_frozen"),
            ]
        )
        .with_columns(
            [
                # Replace original coordinates with rotated ones
                pl.col("X_k_rotated").alias("X_k"),
                pl.col("Y_k_rotated").alias("Y_k"),
                pl.col("Z_k_rotated").alias("Z_k"),
                # Add term due to ECEF angular velocity w.r.t. frozen frame (cross product)
                # v_rotating = v_frozen + cross([0, 0, -OmegaEarth], position)
                pl.when(pl.col("is_bds_geo"))
                .then(
                    pl.col("dX_k_frozen")
                    + pl.col("OmegaEarthIcd_rps") * pl.col("Y_k_rotated")
                )
                .otherwise(pl.col("dX_k_frozen"))
                .alias("dX_k"),
                pl.when(pl.col("is_bds_geo"))
                .then(
                    pl.col("dY_k_frozen")
                    - pl.col("OmegaEarthIcd_rps") * pl.col("X_k_rotated")
                )
                .otherwise(pl.col("dY_k_frozen"))
                .alias("dY_k"),
                pl.col("dZ_k_frozen").alias("dZ_k"),
            ]
        )
        .drop(
            [
                "z_angle",
                "X_k_rotated",
                "Y_k_rotated",
                "Z_k_rotated",
                "dX_k_frozen",
                "dY_k_frozen",
                "dZ_k_frozen",
            ]
        )
    )


@timeit
# Adapted from gnss_lib_py's find_sat()
def kepler_orbit_position_and_velocity(eph: pl.DataFrame) -> pl.DataFrame:
    # Add required columns and constants
    eph = eph.with_columns(
        [
            pl.col("GPSWeek").alias("gps_week"),
            pl.col("constellation").alias("gnss_id"),
            pl.col("sv").alias("sv_id"),
            pl.col("constellation")
            .map_elements(
                lambda x: {
                    "C": constants.cBdsOmegaDotEarth_rps,
                    "G": constants.cGpsOmegaDotEarth_rps,
                    "E": constants.cGalOmegaDotEarth_rps,
                    "J": constants.cQzssOmegaDotEarth_rps,
                }.get(x, float("nan")),
                return_dtype=pl.Float64,
            )
            .alias("OmegaEarthIcd_rps"),
            pl.col("constellation")
            .map_elements(
                lambda x: {
                    "C": constants.cBdsMuEarth_m3ps2,
                    "G": constants.cGpsMuEarth_m3ps2,
                    "E": constants.cGalMuEarth_m3ps2,
                    "J": constants.cQzssMuEarth_m3ps2,
                }.get(x, float("nan")),
                return_dtype=pl.Float64,
            )
            .alias("MuEarthIcd_m3ps2"),
        ]
    )

    # Apply orbital calculations
    eph = position_in_orbital_plane(eph)
    eph = orbital_plane_to_earth_centered_cartesian(eph)
    eph = handle_bds_geos(eph)

    # Rename final coordinate columns
    return eph.rename(
        {
            "X_k": "sat_pos_x_m",
            "Y_k": "sat_pos_y_m",
            "Z_k": "sat_pos_z_m",
            "dX_k": "sat_vel_x_mps",
            "dY_k": "sat_vel_y_mps",
            "dZ_k": "sat_vel_z_mps",
        }
    )


@timeit
def set_time_of_validity(df: pd.DataFrame) -> pd.DataFrame:
    def set_for_one_constellation(group):
        group_constellation = group["constellation"].iat[0]
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

    df = (
        df.groupby("constellation")[df.columns]
        .apply(set_for_one_constellation)
        .reset_index(drop=True)
    )
    return df


@timeit
def convert_nav_dataset_to_dataframe(nav_ds: Any) -> pd.DataFrame:
    """convert ephemerides from xarray.Dataset to pandas.DataFrame"""
    df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    df = df.dropna(how="all")
    df = df.reset_index()
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


@timeit
def compute_gal_inav_fnav_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    assert len(indicators.intersection({1, 2, 4, 5})) == len(indicators), (
        f"Unexpected Galileo navigation message type: {indicators}"
    )
    df.loc[is_gal & (df.fnav_or_inav_indicator == 1), "fnav_or_inav"] = "inav"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 2), "fnav_or_inav"] = "fnav"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 4), "fnav_or_inav"] = "inav"
    df.loc[is_gal & (df.fnav_or_inav_indicator == 5), "fnav_or_inav"] = "inav"
    return df


def to_isagpst(
    time: Union[pd.Timedelta, pd.Series],
    timescale: Union[str, pd.Series],
    gpst_utc_leapseconds: int,
) -> Union[pd.Timedelta, pd.Series]:
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

    assert False, (
        f"Unexpected types: time is {type(time)}, timescale is {type(timescale)}"
    )


@timeit
def select_ephemerides(df: pl.DataFrame, query: pl.DataFrame) -> pl.DataFrame:
    """Polars implementation of ephemeris selection"""
    # Filter out ephemerides with missing reference times
    df = df.filter(pl.col("ephemeris_reference_time_isagpst").is_not_null())

    # Sort both dataframes by grouping columns first, then by time columns
    # This is required for efficient join_asof with 'by' parameter
    query = query.sort(["constellation", "sv", "ephemeris_selection_time_isagpst"])
    df = df.sort(["constellation", "sv", "ephemeris_reference_time_isagpst"])

    # Add fnav/inav indicator for Galileo signals
    query = query.with_columns([pl.lit("").alias("fnav_or_inav")])

    # Set INAV for Galileo satellites
    query = query.with_columns(
        [
            pl.when(pl.col("constellation") == "E")
            .then(pl.lit("inav"))
            .otherwise(pl.col("fnav_or_inav"))
            .alias("fnav_or_inav")
        ]
    )

    # Set FNAV for E5b signals (signal[1] == "5")
    query = query.with_columns(
        [
            pl.when(
                (pl.col("constellation") == "E")
                & (pl.col("signal").str.slice(1, 1) == "5")
            )
            .then(pl.lit("fnav"))
            .otherwise(pl.col("fnav_or_inav"))
            .alias("fnav_or_inav")
        ]
    )

    # Sort both dataframes again after adding fnav_or_inav column for optimal join_asof performance
    query = query.sort(
        ["constellation", "sv", "fnav_or_inav", "ephemeris_selection_time_isagpst"]
    )
    df = df.sort(
        ["constellation", "sv", "fnav_or_inav", "ephemeris_reference_time_isagpst"]
    )

    # Use join_asof for the ephemeris matching
    query = query.join_asof(
        df,
        left_on="ephemeris_selection_time_isagpst",
        right_on="ephemeris_reference_time_isagpst",
        by=["constellation", "sv", "fnav_or_inav"],
        strategy="backward",
    )

    # Compute time differences
    query = query.with_columns(
        [
            (
                (
                    pl.col("query_time_isagpst")
                    - pl.col("ephemeris_reference_time_isagpst")
                ).dt.total_seconds()
            ).alias("query_time_wrt_ephemeris_reference_time_s"),
            (
                (
                    pl.col("query_time_isagpst")
                    - pl.col("clock_reference_time_isagpst")
                ).dt.total_seconds()
            ).alias("query_time_wrt_clock_reference_time_s"),
            (
                (pl.col("query_time_isagpst") < pl.col("validity_end"))
                & (pl.col("query_time_isagpst") > pl.col("validity_start"))
            ).alias("ephemeris_valid"),
        ]
    )

    return query


@timeit
def extract_health_flag_from_query(query: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts the health flag for each row of a query from a `query` DataFrame containing ephemeris data.

    Args:
    query (pl.DataFrame): DataFrame containing at least the 'sv' column and the constellation-specific health flag columns.

    Returns:
    pl.DataFrame: DataFrame with health_flag column added
    """
    # get health flag, according to constellations
    """
    Health flag according to constellations :
        "G", "E", "S", "R", "J", "I" : "health"
        "C" : "SatH1"
    """

    return query.with_columns(
        [
            pl.col("sv").str.slice(0, 1).alias("constellation"),
            pl.when(pl.col("sv").str.slice(0, 1) == "C")
            .then(pl.col("SatH1"))
            .otherwise(pl.col("health"))
            .alias("health_flag"),
        ]
    )


@timeit
def compute_clock_offsets(
    df: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    if hasattr(df, "with_columns"):
        # Pure polars implementation
        return df.with_columns(
            [
                (
                    constants.cGpsSpeedOfLight_mps
                    * (
                        pl.col("SVclockBias")
                        + pl.col("SVclockDrift")
                        * pl.col("query_time_wrt_clock_reference_time_s")
                        + pl.col("SVclockDriftRate")
                        * pl.col("query_time_wrt_clock_reference_time_s") ** 2
                    )
                ).alias("sat_clock_offset_m"),
                (
                    constants.cGpsSpeedOfLight_mps
                    * (
                        pl.col("SVclockDrift")
                        + 2
                        * pl.col("SVclockDriftRate")
                        * pl.col("query_time_wrt_clock_reference_time_s")
                    )
                ).alias("sat_clock_drift_mps"),
            ]
        )
    else:
        # Fallback to pandas for non-polars input
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


@timeit
def merge_per_signal_query_with_ephemerides(
    rinex_nav_file_paths: Union[Path, List[Path]], per_signal_query: pl.DataFrame
) -> pl.DataFrame:
    if not isinstance(rinex_nav_file_paths, list):
        rinex_nav_file_paths = [rinex_nav_file_paths]
    rinex_nav_file_paths = [Path(path) for path in rinex_nav_file_paths]
    ephemeris_blocks = []
    for path in rinex_nav_file_paths:
        block = parse_rinex_nav_file(path)
        ephemeris_blocks.append(block)
    ephemerides = pd.concat(ephemeris_blocks)

    # Convert pandas ephemerides to polars for consistent processing
    ephemerides_pl = pl.from_pandas(ephemerides)

    # Group delays and clock offsets can be signal-specific, so we need to match ephemerides to code signals,
    # not only to satellites
    # Example: Galileo transmits E5a clock and group delay parameters in the F/NAV message, but parameters for other
    # signals in the I/NAV message
    if "ephemeris_selection_time_isagpst" not in per_signal_query.columns:
        per_signal_query = per_signal_query.with_columns(
            pl.col("query_time_isagpst").alias("ephemeris_selection_time_isagpst")
        )

    # Use polars implementation for ephemeris selection
    result = select_ephemerides(ephemerides_pl, per_signal_query)

    return result


def compute(
    rinex_nav_file_paths: Union[Path, List[Path]],
    per_signal_query: pl.DataFrame,
    is_query_corrected_by_sat_clock_offset: bool = False,
) -> pl.DataFrame:
    if "constellation" not in per_signal_query.columns:
        per_signal_query = per_signal_query.with_columns(
            pl.col("sv").str.slice(0, 1).alias("constellation")
        )
    return _compute(
        merge_per_signal_query_with_ephemerides(rinex_nav_file_paths, per_signal_query),
        is_query_corrected_by_sat_clock_offset,
    )


def _compute(
    per_signal_query_with_ephemerides: pl.DataFrame,
    is_query_corrected_by_sat_clock_offset: bool = False,
) -> pl.DataFrame:
    # compute satellite clock bias using polars
    if is_query_corrected_by_sat_clock_offset:
        per_signal_query_with_ephemerides = compute_clock_offsets(
            per_signal_query_with_ephemerides
        )
    else:  # compute satellite clock offset iteratively
        t = per_signal_query_with_ephemerides.select(
            "query_time_wrt_clock_reference_time_s"
        )
        for _ in range(2):
            per_signal_query_with_ephemerides = compute_clock_offsets(
                per_signal_query_with_ephemerides
            )
            per_signal_query_with_ephemerides = (
                per_signal_query_with_ephemerides.with_columns(
                    (
                        t.to_series()
                        - pl.col("sat_clock_offset_m") / constants.cGpsSpeedOfLight_mps
                    ).alias("query_time_wrt_clock_reference_time_s")
                )
            )
        # Apply sat clock correction to the query time for satellite position computation
        per_signal_query_with_ephemerides = (
            per_signal_query_with_ephemerides.with_columns(
                (
                    pl.col("query_time_wrt_ephemeris_reference_time_s")
                    - pl.col("sat_clock_offset_m") / constants.cGpsSpeedOfLight_mps
                ).alias("query_time_wrt_ephemeris_reference_time_s")
            )
        )

    # Compute orbital states for each (satellite,ephemeris) pair only once:
    # Use polars groupby operations
    per_sat_eph_query = (
        per_signal_query_with_ephemerides.group_by(
            ["sv", "query_time_isagpst", "ephemeris_hash"]
        )
        .first()
        .drop(["sat_clock_offset_m", "sat_clock_drift_mps"])
    )

    # Evaluate orbits by type using pure polars
    # Define required output columns that all orbit functions should produce
    required_orbit_columns = [
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
    ]

    # Process each orbit type separately
    kepler_mask = per_sat_eph_query.filter(pl.col("orbit_type") == "kepler")
    glonass_mask = per_sat_eph_query.filter(pl.col("orbit_type") == "glonass")
    sbas_mask = per_sat_eph_query.filter(pl.col("orbit_type") == "sbas")
    other_mask = per_sat_eph_query.filter(
        ~pl.col("orbit_type").is_in(["kepler", "glonass", "sbas"])
    )

    # Process each orbit type and select consistent columns
    orbit_results = []
    base_columns = [
        col for col in per_sat_eph_query.columns if col not in required_orbit_columns
    ]
    keep_columns = base_columns + required_orbit_columns

    if kepler_mask.shape[0] > 0:
        kepler_result = kepler_orbit_position_and_velocity(kepler_mask)
        # Select only the columns we need to keep consistency
        kepler_result = kepler_result.select(
            [col for col in keep_columns if col in kepler_result.columns]
        )
        orbit_results.append(kepler_result)

    if glonass_mask.shape[0] > 0:
        glonass_result = glonass_orbit_position_and_velocity(glonass_mask)
        # Select only the columns we need to keep consistency
        glonass_result = glonass_result.select(
            [col for col in keep_columns if col in glonass_result.columns]
        )
        orbit_results.append(glonass_result)

    if sbas_mask.shape[0] > 0:
        sbas_result = sbas_orbit_position_and_velocity(sbas_mask)
        # Select only the columns we need to keep consistency
        sbas_result = sbas_result.select(
            [col for col in keep_columns if col in sbas_result.columns]
        )
        orbit_results.append(sbas_result)

    if other_mask.shape[0] > 0:
        # For unsupported orbit types, add NaN columns and select consistent columns
        other_result = other_mask.with_columns(
            [
                pl.lit(float("nan")).alias("sat_pos_x_m"),
                pl.lit(float("nan")).alias("sat_pos_y_m"),
                pl.lit(float("nan")).alias("sat_pos_z_m"),
                pl.lit(float("nan")).alias("sat_vel_x_mps"),
                pl.lit(float("nan")).alias("sat_vel_y_mps"),
                pl.lit(float("nan")).alias("sat_vel_z_mps"),
            ]
        ).select(
            [col for col in keep_columns if col in other_mask.columns]
            + required_orbit_columns
        )
        orbit_results.append(other_result)

    # Combine all results - now they should have consistent columns
    if orbit_results:
        per_sat_eph_query = pl.concat(orbit_results, how="diagonal_relaxed")

    # Add health flag using polars
    per_sat_eph_query = extract_health_flag_from_query(per_sat_eph_query)
    columns_to_keep = [
        "sv",
        "constellation",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "query_time_isagpst",
        "ephemeris_hash",
        "health_flag",
        "frequency_slot",
    ]
    per_sat_eph_query = per_sat_eph_query.select(columns_to_keep)

    # Merge the computed satellite states into the larger signal-specific query dataframe using polars
    per_signal_query_with_ephemerides = per_signal_query_with_ephemerides.join(
        per_sat_eph_query,
        on=["constellation", "sv", "query_time_isagpst", "ephemeris_hash"],
        how="left",
    )
    # Compute group delays using pure polars
    per_signal_query_with_ephemerides = compute_total_group_delays(
        per_signal_query_with_ephemerides
    )

    # Define the exact column order to match reference CSV
    reference_column_order = [
        "signal",
        "sat_code_bias_m",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "sv",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "query_time_isagpst",
        "ephemeris_hash",
        "health_flag",
        "frequency_slot",
    ]

    # Select only the columns that exist and in the correct order
    final_columns = [
        col
        for col in reference_column_order
        if col in per_signal_query_with_ephemerides.columns
    ]

    # Create general-purpose sorting that works for any input
    # Sort by: signal, constellation, satellite number, timestamp
    per_signal_query_with_ephemerides = per_signal_query_with_ephemerides.with_columns([
        # Extract satellite number as integer for proper numeric sorting
        pl.col("sv").str.slice(1, 2).cast(pl.Int32, strict=False).alias("_sat_number")
    ]).sort([
        "signal",           # Sort by signal first (C1C, C2I, etc.)
        "constellation",    # Then by constellation (C, E, G, J, R, S)  
        "_sat_number",      # Then by satellite number (1, 2, 3, etc.)
        "query_time_isagpst" # Finally by timestamp
    ]).drop(["_sat_number"])  # Remove temporary column

    # Return the result with the correct column order and sorting - pure polars
    return per_signal_query_with_ephemerides.select(final_columns)


def compute_total_group_delays(
    query: pl.DataFrame,
) -> pl.DataFrame:
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
        return query.with_columns(pl.lit(float("nan")).alias("group_delay"))

    # Calculate frequency_code and speedOfLightIcd_mps
    query = query.with_columns(
        [
            pl.col("signal").str.slice(1, 1).alias("frequency_code"),
            pl.col("constellation")
            .map_elements(
                lambda x: {
                    "C": constants.cBdsSpeedOfLight_mps,
                    "G": constants.cGpsSpeedOfLight_mps,
                    "E": constants.cGalSpeedOfLight_mps,
                    "J": constants.cQzssSpeedOfLight_mps,
                }.get(x, float("nan")),
                return_dtype=pl.Float64,
            )
            .alias("speedOfLightIcd_mps"),
        ]
    )

    # GPS constellation
    gps_l1_l2_ratio_sq = (
        constants.carrier_frequencies_hz()["G"]["L1"][1]
        / constants.carrier_frequencies_hz()["G"]["L2"][1]
    ) ** 2

    # Galileo constellation
    gal_l1_l5_ratio_sq = (
        constants.carrier_frequencies_hz()["E"]["L1"][1]
        / constants.carrier_frequencies_hz()["E"]["L5"][1]
    ) ** 2
    gal_l1_l7_ratio_sq = (
        constants.carrier_frequencies_hz()["E"]["L1"][1]
        / constants.carrier_frequencies_hz()["E"]["L7"][1]
    ) ** 2

    # Compute gamma and tgd using nested when expressions
    query = query.with_columns(
        [
            # Gamma calculation
            pl.when(pl.col("constellation") == "G")
            .then(
                pl.when(pl.col("frequency_code") == "1")
                .then(1.0)
                .when(pl.col("frequency_code") == "2")
                .then(gps_l1_l2_ratio_sq)
                .otherwise(float("nan"))
            )
            .when(pl.col("constellation") == "J")
            .then(1.0)
            .when(pl.col("constellation") == "E")
            .then(
                pl.when(pl.col("frequency_code") == "1")
                .then(1.0)
                .when(pl.col("frequency_code") == "5")
                .then(gal_l1_l5_ratio_sq)
                .when(pl.col("frequency_code") == "7")
                .then(gal_l1_l7_ratio_sq)
                .otherwise(float("nan"))
            )
            .when(pl.col("constellation") == "C")
            .then(1.0)
            .otherwise(float("nan"))
            .alias("gamma"),
            # TGD calculation
            pl.when(pl.col("constellation") == "G")
            .then(pl.col("TGD"))
            .when(pl.col("constellation") == "J")
            .then(pl.col("TGD"))
            .when(pl.col("constellation") == "E")
            .then(
                pl.when(pl.col("frequency_code") == "1")
                .then(pl.col("BGDe5b"))
                .when(pl.col("frequency_code") == "5")
                .then(pl.col("BGDe5a"))
                .when(pl.col("frequency_code") == "7")
                .then(pl.col("BGDe5b"))
                .otherwise(float("nan"))
            )
            .when(pl.col("constellation") == "C")
            .then(
                pl.when(pl.col("signal") == "C2I")
                .then(pl.col("TGD1"))
                .when(pl.col("signal") == "C7I")
                .then(pl.col("TGD2"))
                .when(pl.col("signal") == "C6I")
                .then(0.0)
                .otherwise(float("nan"))
            )
            .otherwise(float("nan"))
            .alias("tgd"),
        ]
    )

    # Calculate sat_code_bias_m
    return query.with_columns(
        [
            (pl.col("tgd") * pl.col("gamma") * pl.col("speedOfLightIcd_mps")).alias(
                "sat_code_bias_m"
            )
        ]
    )


if __name__ == "main":
    pass
