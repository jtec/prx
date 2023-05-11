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

    Reference: "GNSS DATA PROCESSING, Volume I: Fundamentals and Algorithms", J. Sanz Subirana, J.M. Juan Zornoza and
    M. Hernández-Pajares p 129
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


def compute_unb3m_correction(LATRAD, HEIGHTM, DAYOFYEAR, ELEVRAD):
    # This function is the python version of the matlab function UNB3M.m provided in the UNB3m_pack [1]
    #
    # INPUTS: numpy arrays of the same shape are expected
    #
    # OUTPUTS:
    # - HZD  : Hydrostatic zenith delay (m)
    # - HMF  : Hydrostatic Niell mapping function
    # - WZD  : Non-hyd. zenith delay (m)
    # - WMF  :  Non-hyd. Niell mapping function
    # - RTROP: Total slant delay (m)
    #
    # [1] https://gge.ext.unb.ca/Resources/unb3m/unb3m.html

    assert LATRAD.shape == HEIGHTM.shape, ">> atmospheric_corrections.compute_unb3m_correction: input arguments " \
                                          "should be arrays of the same shape "
    assert LATRAD.shape == DAYOFYEAR.shape, ">> atmospheric_corrections.compute_unb3m_correction: input arguments " \
                                            "should be arrays of the same shape "
    assert LATRAD.shape == ELEVRAD.shape, ">> atmospheric_corrections.compute_unb3m_correction: input arguments " \
                                          "should be arrays of the same shape "

    # Initialize UNB3m look-up table
    AVG = np.array([[15.0, 1013.25, 299.65, 75.00, 6.30, 2.77],
                    [30.0, 1017.25, 294.15, 80.00, 6.05, 3.15],
                    [45.0, 1015.75, 283.15, 76.00, 5.58, 2.57],
                    [60.0, 1011.75, 272.15, 77.50, 5.39, 1.81],
                    [75.0, 1013.00, 263.65, 82.50, 4.53, 1.55]])
    AMP = np.array([[15.0,  0.00,  0.00,  0.00, 0.00, 0.00],
                    [30.0, -3.75,  7.00,  0.00, 0.25, 0.33],
                    [45.0, -2.25, 11.00, -1.00, 0.32, 0.46],
                    [60.0, -1.75, 15.00, -2.50, 0.81, 0.74],
                    [75.0, -0.50, 14.50,  2.50, 0.62, 0.30]])
    EXCEN2 = 6.6943799901413e-03
    MD = 28.9644
    MW = 18.0152
    K1 = 77.604
    K2 = 64.79
    K3 = 3.776e5
    R = 8314.34
    C1 = 2.2768e-03
    K2PRIM = K2 - K1 * (MW / MD)
    RD = R / MD
    # DTR = 1.745329251994329e-02
    DOY2RAD = 0.31415926535897935601e01 * 2 / 365.25

    # initialize NMF tables
    ABC_AVG = np.array([[15.0, 1.2769934e-3, 2.9153695e-3, 62.610505e-3],
                        [30.0, 1.2683230e-3, 2.9152299e-3, 62.837393e-3],
                        [45.0, 1.2465397e-3, 2.9288445e-3, 63.721774e-3],
                        [60.0, 1.2196049e-3, 2.9022565e-3, 63.824265e-3],
                        [75.0, 1.2045996e-3, 2.9024912e-3, 64.258455e-3]])
    ABC_AMP = np.array([[15.0, 0.0,          0.0,          0.0],
                        [30.0, 1.2709626e-5, 2.1414979e-5, 9.0128400e-5],
                        [45.0, 2.6523662e-5, 3.0160779e-5, 4.3497037e-5],
                        [60.0, 3.4000452e-5, 7.2562722e-5, 84.795348e-5],
                        [75.0, 4.1202191e-5, 11.723375e-5, 170.37206e-5]])
    A_HT = 2.53e-5
    B_HT = 5.49e-3
    C_HT = 1.14e-3
    HT_TOPCON = 1 + A_HT / (1 + B_HT / (1 + C_HT))

    ABC_W2P0 = np.array([[15.0, 5.8021897e-4, 1.4275268e-3, 4.3472961e-2],
                         [30.0, 5.6794847e-4, 1.5138625e-3, 4.6729510e-2],
                         [45.0, 5.8118019e-4, 1.4572752e-3, 4.3908931e-2],
                         [60.0, 5.9727542e-4, 1.5007428e-3, 4.4626982e-2],
                         [75.0, 6.1641693e-4, 1.7599082e-3, 5.4736038e-2]])

    LATDEG = np.rad2deg(LATRAD)

    # Deal with southern hemisphere and yearly variation
    TD_O_Y = DAYOFYEAR
    TD_O_Y = np.where(LATDEG < 0, TD_O_Y + 182.625, TD_O_Y)

    COSPHS = np.cos((TD_O_Y - 28) * DOY2RAD)

    # Initialize pointers to lookup table
    LAT = np.abs(LATDEG)
    P1 = np.fix((LAT - 15) / 15) + 1 - 1
    P1 = np.where(LAT >= 75, 4, P1)
    P1 = np.where(LAT <= 15, 0, P1)
    P1 = P1.astype("int")
    P2 = P1 + 1
    P2 = np.where(LAT >= 75, 4, P2)
    P2 = np.where(LAT <= 15, 0, P2)
    M = (LAT - AVG[P1, 0]) / (AVG[P2, 0] - AVG[P1, 0])
    M = np.where(LAT >= 75, 0, M)
    M = np.where(LAT <= 15, 0, M)

    # Compute average surface tropo values by interpolation
    PAVG = M * (AVG[P2, 1] - AVG[P1, 1]) + AVG[P1, 1]
    TAVG = M * (AVG[P2, 2] - AVG[P1, 2]) + AVG[P1, 2]
    EAVG = M * (AVG[P2, 3] - AVG[P1, 3]) + AVG[P1, 3]
    BETAAVG = M * (AVG[P2, 4] - AVG[P1, 4]) + AVG[P1, 4]
    LAMBDAAVG = M * (AVG[P2, 5] - AVG[P1, 5]) + AVG[P1, 5]

    # Compute variation of average surface tropo values
    PAMP = M * (AMP[P2, 1] - AMP[P1, 1]) + AMP[P1, 1]
    TAMP = M * (AMP[P2, 2] - AMP[P1, 2]) + AMP[P1, 2]
    EAMP = M * (AMP[P2, 3] - AMP[P1, 3]) + AMP[P1, 3]
    BETAAMP = M * (AMP[P2, 4] - AMP[P1, 4]) + AMP[P1, 4]
    LAMBDAAMP = M * (AMP[P2, 5] - AMP[P1, 5]) + AMP[P1, 5]

    # Compute surface tropo values
    P0 = PAVG - PAMP * COSPHS
    T0 = TAVG - TAMP * COSPHS
    E0 = EAVG - EAMP * COSPHS
    BETA = BETAAVG - BETAAMP * COSPHS
    BETA = BETA / 1000
    LAMBDA = LAMBDAAVG - LAMBDAAMP * COSPHS

    # Transform from relative humidity to WVP (IERS Conventions 2003)
    ES = 0.01 * np.exp(1.2378847e-5 * (T0 ** 2) - 1.9121316e-2 * T0 + 3.393711047e1 - 6.3431645e3 * (T0 ** -1))
    FW = 1.00062 + 3.14e-6 * P0 + 5.6e-7 * ((T0 - 273.15) ** 2)
    E0 = (E0 / 100) * ES * FW

    # Compute power value for pressure & water vapour
    EP = 9.80665 / 287.054 / BETA

    # Scale surface values to required height
    T = T0 - BETA * HEIGHTM
    P = P0 * (T / T0) ** EP
    E = E0 * (T / T0) ** (EP * (LAMBDA + 1))

    # Compute the acceleration at the mass center of a vertical column of the atmosphere
    GEOLAT = np.arctan((1 - EXCEN2) * np.tan(LATRAD))
    DGREF = 1 - 2.66e-03 * np.cos(2 * GEOLAT) - 2.8e-07 * HEIGHTM
    GM = 9.784 * DGREF
    DEN = (LAMBDA + 1) * GM

    # Compute mean temperature of the water vapor
    TM = T * (1 - BETA * RD / DEN)

    # Compute zenith hydrostatic delay
    HZD = C1 / DGREF * P

    # Compute zenith wet delay
    WZD = 1e-6 * (K2PRIM + K3 / TM) * RD * E / DEN

    # Compute average NMF(H) coefficient values by interpolation
    A_AVG = M * (ABC_AVG[P2, 1] - ABC_AVG[P1, 1]) + ABC_AVG[P1, 1]
    B_AVG = M * (ABC_AVG[P2, 2] - ABC_AVG[P1, 2]) + ABC_AVG[P1, 2]
    C_AVG = M * (ABC_AVG[P2, 3] - ABC_AVG[P1, 3]) + ABC_AVG[P1, 3]

    # Compute variation of average NMF(H) coefficient values
    A_AMP = M * (ABC_AMP[P2, 1] - ABC_AMP[P1, 1]) + ABC_AMP[P1, 1]
    B_AMP = M * (ABC_AMP[P2, 2] - ABC_AMP[P1, 2]) + ABC_AMP[P1, 2]
    C_AMP = M * (ABC_AMP[P2, 3] - ABC_AMP[P1, 3]) + ABC_AMP[P1, 3]

    # Compute NMF(H) coefficient values
    A = A_AVG - A_AMP * COSPHS
    B = B_AVG - B_AMP * COSPHS
    C = C_AVG - C_AMP * COSPHS

    # Compute sine of elevation angle
    SINE = np.sin(ELEVRAD)

    # Compute NMF(H) value
    ALPHA = B / (SINE + C)
    GAMMA = A / (SINE + ALPHA)
    TOPCON = (1 + A / (1 + B / (1 + C)))
    HMF = TOPCON / (SINE + GAMMA)

    # Compute and apply height correction
    ALPHA = B_HT / (SINE + C_HT)
    GAMMA = A_HT / (SINE + ALPHA)
    HT_CORR_COEF = 1 / SINE - HT_TOPCON / (SINE + GAMMA)
    HT_CORR = HT_CORR_COEF * HEIGHTM / 1000
    HMF = HMF + HT_CORR

    # Compute average NMF(W) coefficient values by interpolation
    A = M * (ABC_W2P0[P2, 1] - ABC_W2P0[P1, 1]) + ABC_W2P0[P1, 1]
    B = M * (ABC_W2P0[P2, 2] - ABC_W2P0[P1, 2]) + ABC_W2P0[P1, 2]
    C = M * (ABC_W2P0[P2, 3] - ABC_W2P0[P1, 3]) + ABC_W2P0[P1, 3]

    # Compute NMF(W) value
    ALPHA = B / (SINE + C)
    GAMMA = A / (SINE + ALPHA)
    TOPCON = (1 + A / (1 + B / (1 + C)))
    WMF = TOPCON / (SINE + GAMMA)

    # Compute total slant delay
    RTROP = HZD * HMF + WZD * WMF

    return RTROP, HZD, HMF, WZD, WMF
