import numpy as np


def compute_klobuchar_correction(tow, gps_a, gps_b, el_s, az_s, phi_u, lambda_u):
    """compute the ionospheric corrections using the Klobuchar model

    Inputs:
    tow: (numpy.array) GPS time of week in seconds
    gps_a, gps_b: (numpy.array) Klobuchar model parameters, read from a RNX NAV file's header
    el_s: (numpy.array) satellite elevation in radians
    az_s: (numpy.array) satellite azimuth in radians
    phi_u: (scalar) user latitude in radians
    lambda_u: (scalar) user longitude in radians

    Reference: "GNSS DATA PROCESSING, Volume I: Fundamentals and Algorithms", J. Sanz Subirana, J.M. Juan Zornoza and M. Hernández-Pajares p 129
    """
    Re = 6371e3  # Earth radius(m)
    h = 350e3  # Height of the iono layer(m)
    phi_p = 78.3 * np.pi / 180  # latitude of the geomagnetic pole
    lambda_p = 291 * np.pi / 180  # longitude of the geomagnetic pole
    c = 299792458

    # Earth - centered angle
    psi = np.pi / 2 - el_s - np.arcsin(Re / (Re + h) * np.cos(el_s))

    # latitude of IPP
    phi_i = np.arcsin(np.sin(phi_u) * np.cos(psi) + np.cos(phi_u) * np.sin(psi) * np.cos(az_s))

    # longitude of the IPP
    lambda_i = lambda_u + psi * np.sin(az_s) / np.cos(phi_i)

    # geomagnetic latitude of the IPP
    phi_m = np.arcsin(
        np.sin(phi_i) * np.sin(phi_p) + np.cos(phi_i) * np.cos(phi_p) * np.cos(lambda_i - lambda_p))

    # local time at the IPP
    t = np.mod(43200 * lambda_i / np.pi + tow, 86400)

    # amplitude of the iono delay
    A_i = gps_a[0] + gps_a[1] * phi_m / np.pi + gps_a[2] * (phi_m / np.pi) ** 2 + gps_a[3] * (phi_m / np.pi) ** 3
    A_i = np.where(A_i < 0, 0, A_i)

    # period of iono delay
    P_i = gps_b[0] + gps_b[1] * phi_m / np.pi + gps_b[2] * (phi_m / np.pi) ** 2 + gps_b[3] * (phi_m / np.pi) ** 3
    P_i = np.where(P_i < 72000, 72000, P_i)

    # phase of the iono delay
    X_i = 2 * np.pi * (t - 50400) / P_i

    # slant factor
    F = 1 / np.sqrt(1 - (Re / (Re + h) * np.cos(el_s)) ** 2)

    # iono time delay, in m
    iono_correction_l1_m = np.where(np.fabs(X_i) < np.pi / 2, c * (5e-9 + A_i * np.cos(X_i)) * F, c * 5e-9 * F)

    return iono_correction_l1_m
