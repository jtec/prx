import math


def compute_klobuchar_correction(tow, gps_a, gps_b, el_s, az_s, phi_u, lambda_u):
    """compute the ionospheric corrections using the Klobuchar model

    Inputs:
    tow:  GPS time of week in seconds
    gps_a, gps_b: Klobuchar model parameters, read from a RNX NAV file's header
    el_s: satellite elevation in radians
    az_s: satellite azimuth in radians
    phi_u: user latitude in radians
    lambda_u: user longitude in radians

    Reference: ESA Book, p 129
    """
    Re = 6371e3  # Earth radius(m)
    h = 350e3  # Height of the iono layer(m)
    phi_p = 78.3 * math.pi / 180  # latitude of the geomagnetic pole
    lambda_p = 291 * math.pi / 180  # longitude of the geomagnetic pole
    c = 299792458

    # Earth - centered angle
    psi = math.pi / 2 - el_s - math.asin(Re / (Re + h) * math.cos(el_s))

    # latitude of IPP
    phi_i = math.asin(math.sin(phi_u) * math.cos(psi) + math.cos(phi_u) * math.sin(psi) * math.cos(az_s))

    # longitude of the IPP
    lambda_i = lambda_u + psi * math.sin(az_s) / math.cos(phi_i)

    # geomagnetic latitude of the IPP
    phi_m = math.asin(
        math.sin(phi_i) * math.sin(phi_p) + math.cos(phi_i) * math.cos(phi_p) * math.cos(lambda_i - lambda_p))

    # local time at the IPP
    t = 43200 * lambda_i / math.pi + tow
    if t >= 86400: t -= 86400
    if t < 0: t += 86400

    # amplitude of the iono delay
    A_i = gps_a(0) + gps_a(1) * phi_m / math.pi + gps_a(2) * (phi_m / math.pi) ** 2 + gps_a(3) * (phi_m / math.pi) ** 3
    if A_i < 0: A_i = 0

    # period of iono delay
    P_i = gps_b(0) + gps_b(1) * phi_m / math.pi + gps_b(2) * (phi_m / math.pi) ** 2 + gps_b(3) * (phi_m / math.pi) ** 3
    if P_i < 72000: P_i = 72000

    # phase of the iono delay
    X_i = 2 * math.pi * (t - 50400) / P_i

    # slant factor
    F = 1 / math.sqrt(1 - (Re / (Re + h) * math.cos(el_s)) ** 2)

    # iono time delay, in m
    if math.fabs(X_i) < math.pi / 2:
        iono_correction_l1_s = c * (5e-9 + A_i * math.cos(X_i)) * F
    else:
        iono_correction_l1_s = c * 5e-9 * F

    return iono_correction_l1_s
