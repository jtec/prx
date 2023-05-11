import numpy as np
from helpers import deg_2_rad
import constants


def compute_klobuchar_l1_correction(tow_s, gps_a, gps_b, elevation_rad, azimuth_rad, lat_user_rad, lon_user_rad):
    """compute the ionospheric corrections using the Klobuchar model

    Inputs:
    tow_s: (numpy.array) GPS time of week in seconds
    gps_a, gps_b: (numpy.array) Klobuchar model parameters, read from a RNX NAV file's header
    elevation_rad: (numpy.array) satellite elevation in radians
    azimuth_rad: (numpy.array) satellite azimuth in radians
    lat_user_rad: (scalar) user latitude in radians
    lon_user_rad: (scalar) user longitude in radians

    Reference: "GNSS DATA PROCESSING, Volume I: Fundamentals and Algorithms", J. Sanz Subirana, J.M. Juan Zornoza and M. Hernández-Pajares p 129
    """
    Re = 6378e3  # Earth radius (m)
    h = 350e3  # Height of the iono layer (m) for GPS. Shall be 375 km for Beidou
    phi_p = deg_2_rad(78.3)  # latitude of the geomagnetic pole (radians)
    lambda_p = deg_2_rad(291)  # longitude of the geomagnetic pole (radians)

    # Earth-centered angle of the Ionospheric Pierce Point
    psi = np.pi / 2 - elevation_rad - np.arcsin(Re / (Re + h) * np.cos(elevation_rad))

    # latitude of Ionospheric Pierce Point
    phi_i = np.arcsin(np.sin(lat_user_rad) * np.cos(psi) + np.cos(lat_user_rad) * np.sin(psi) * np.cos(azimuth_rad))

    # longitude of the Ionospheric Pierce Point
    lambda_i = lon_user_rad + psi * np.sin(azimuth_rad) / np.cos(phi_i)

    # geomagnetic latitude of the Ionospheric Pierce Point
    phi_m = np.arcsin(
        np.sin(phi_i) * np.sin(phi_p) + np.cos(phi_i) * np.cos(phi_p) * np.cos(lambda_i - lambda_p))

    # local time at the Ionospheric Pierce Point
    t = np.mod(43200 * lambda_i / np.pi + tow_s, 86400)

    # amplitude of the iono delay
    A_i = gps_a[0] + gps_a[1] * phi_m / np.pi + gps_a[2] * (phi_m / np.pi) ** 2 + gps_a[3] * (phi_m / np.pi) ** 3
    A_i = np.where(A_i < 0, 0, A_i)

    # period of iono delay
    P_i = gps_b[0] + gps_b[1] * phi_m / np.pi + gps_b[2] * (phi_m / np.pi) ** 2 + gps_b[3] * (phi_m / np.pi) ** 3
    P_i = np.where(P_i < 72000, 72000, P_i)

    # phase of the iono delay
    X_i = 2 * np.pi * (t - 50400) / P_i

    # slant factor
    F = 1 / np.sqrt(1 - (Re / (Re + h) * np.cos(elevation_rad)) ** 2)

    # iono time delay, in m
    iono_correction_l1_m = np.where(np.fabs(X_i) < np.pi / 2,
                                    constants.cGpsIcdSpeedOfLight_mps * (5e-9 + A_i * np.cos(X_i)) * F,
                                    constants.cGpsIcdSpeedOfLight_mps * 5e-9 * F)

    return iono_correction_l1_m


def compute_unb3m_correction(lat_rad, height_m, day_of_year, elevation_rad):
    unb3m_correction_m = np.nan
    return unb3m_correction_m
