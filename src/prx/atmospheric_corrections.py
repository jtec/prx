import numpy as np
import pandas as pd
import georinex
import logging
from prx.util import deg_2_rad, ecef_2_geodetic, timedelta_2_weeks_and_seconds
import prx.constants as constants


def compute_l1_iono_delay_klobuchar(
    tow_s, gps_a, gps_b, elevation_rad, azimuth_rad, lat_user_rad, lon_user_rad
):
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
    phi_i = np.arcsin(
        np.sin(lat_user_rad) * np.cos(psi)
        + np.cos(lat_user_rad) * np.sin(psi) * np.cos(azimuth_rad)
    )

    # longitude of the Ionospheric Pierce Point
    lambda_i = lon_user_rad + psi * np.sin(azimuth_rad) / np.cos(phi_i)

    # geomagnetic latitude of the Ionospheric Pierce Point
    phi_m = np.arcsin(
        np.sin(phi_i) * np.sin(phi_p)
        + np.cos(phi_i) * np.cos(phi_p) * np.cos(lambda_i - lambda_p)
    )

    # local time at the Ionospheric Pierce Point
    t = np.mod(43200 * lambda_i / np.pi + tow_s, 86400)

    # amplitude of the iono delay
    A_i = (
        gps_a[0]
        + gps_a[1] * phi_m / np.pi
        + gps_a[2] * (phi_m / np.pi) ** 2
        + gps_a[3] * (phi_m / np.pi) ** 3
    )
    A_i = np.where(A_i < 0, 0, A_i)

    # period of iono delay
    P_i = (
        gps_b[0]
        + gps_b[1] * phi_m / np.pi
        + gps_b[2] * (phi_m / np.pi) ** 2
        + gps_b[3] * (phi_m / np.pi) ** 3
    )
    P_i = np.where(P_i < 72000, 72000, P_i)

    # phase of the iono delay
    X_i = 2 * np.pi * (t - 50400) / P_i

    # slant factor
    F = 1 / np.sqrt(1 - (Re / (Re + h) * np.cos(elevation_rad)) ** 2)

    # iono time delay, in m
    iono_correction_l1_m = np.where(
        np.fabs(X_i) < np.pi / 2,
        constants.cGpsSpeedOfLight_mps * (5e-9 + A_i * np.cos(X_i)) * F,
        constants.cGpsSpeedOfLight_mps * 5e-9 * F,
    )

    return iono_correction_l1_m


def add_iono_column(
    flat_obs, rinex_3_ephemerides_files, approximate_receiver_ecef_position_m
):
    # create a dictionary containing the headers of the different NAV files.
    # The keys are the "YYYYDDD" (year and day of year) and are located at
    # [12:19] of the file name using RINEX naming convention
    nav_header_dict = {
        file.name[12:19]: georinex.rinexheader(file)
        for file in rinex_3_ephemerides_files
    }

    idx_all_days = []
    iono_all_days = []
    for file in rinex_3_ephemerides_files:
        # get year and doy from NAV filename
        year = int(file.name[12:16])
        doy = int(file.name[16:19])

        # Selection criteria: time of emission belonging to the day of the current NAV file
        mask = (
            (
                flat_obs.time_of_emission_isagpst
                >= pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            )
            & (
                flat_obs.time_of_emission_isagpst
                < pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy)
            )
            & (flat_obs.observation_type.str.startswith("C"))
        )
        mask_idx = mask.loc[mask].index
        idx_all_days.append(mask_idx)
        if "IONOSPHERIC CORR" in nav_header_dict[f"{year:03d}" + f"{doy:03d}"]:
            logging.info(f"Computing iono delay for {year}-{doy:03d}")
            time_of_emission_weeksecond_isagpst = timedelta_2_weeks_and_seconds(
                flat_obs.loc[mask_idx, "time_of_emission_isagpst"]
                - constants.system_time_scale_rinex_utc_epoch["GPST"]
            )[1].to_numpy()
            [latitude_user_rad, longitude_user_rad, __] = ecef_2_geodetic(
                approximate_receiver_ecef_position_m
            )
            iono_all_days.append(
                compute_l1_iono_delay_klobuchar(
                    time_of_emission_weeksecond_isagpst,
                    nav_header_dict[f"{year:03d}" + f"{doy:03d}"]["IONOSPHERIC CORR"][
                        "GPSA"
                    ],
                    nav_header_dict[f"{year:03d}" + f"{doy:03d}"]["IONOSPHERIC CORR"][
                        "GPSB"
                    ],
                    flat_obs.loc[mask_idx, "elevation_rad"],
                    flat_obs.loc[mask_idx, "azimuth_rad"],
                    latitude_user_rad,
                    longitude_user_rad,
                )
                * (
                    constants.carrier_frequencies_hz()["G"]["L1"][1] ** 2
                    / flat_obs.loc[mask_idx, "carrier_frequency_hz"] ** 2
                )
            )
        else:
            logging.warning(f"Missing iono model parameters for day {doy:03d}")
            iono_all_days.append(np.full(mask_idx.shape, np.nan))
    return np.concatenate(idx_all_days), np.concatenate(iono_all_days)


def compute_tropo_delay_saastamoinen(height, el, lat, humi=0.7):
    """
    code source: rtklib v2.4.3 b33, rtkcmn.c, tropmodel

    INPUTS:
    - height: geodetic height in m. np.ndarray of shape (n,)
    - el in rad. np.ndarray of shape (n,)
    - lat in rad. np.ndarray of shape (n,)
    - humi in ratio. scalar

    OUTPUTS:
    - tropo_tot, tropo_h, tropo_w in m. np.ndarray of shape (n,)
    """
    assert len(height) == len(el) == len(lat)
    assert (-100 <= height).all()
    assert (height <= 1e4).all()
    assert (
        el[~np.isnan(el)] >= -np.deg2rad(5)
    ).all()  # consider slightly negative elevation...

    # standard atmosphere model
    height_new = height.copy()  # to avoid changing input
    height_new[height_new < 0] = 0
    height = height_new

    p = 1013.25 * (1 - 2.2557e-5 * height) ** 5.2568
    t = 15 - 6.5e-3 * height + 273.16
    e_wat = 6.108 * np.exp((17.15 * t - 4684) / (t - 38.45)) * humi

    # zenith angle in rad
    z = np.pi / 2 - el

    # hydrostatic tropospheric delay
    tropo_h = (
        0.0022768 * p / (1 - 0.00266 * np.cos(2 * lat) - 0.28e-6 * height) / np.cos(z)
    )

    # wet tropospheric delay
    tropo_w = 0.002277 / np.cos(z) * (1255 / t + 0.05) * e_wat

    # total tropospheric delay (projected on the slant path)
    tropo_tot = tropo_h + tropo_w

    return tropo_tot, tropo_h, tropo_w


def compute_tropo_delay_unb3m(
    latitude_user_rad, height_user_m, day_of_year, elevation_sat_rad
):
    # This function is the python version of the matlab function UNB3M.m provided in the UNB3m_pack [1]
    #
    # INPUTS: numpy arrays of the same shape are expected
    #
    # OUTPUTS:
    # - tropo_zhd_m  : Hydrostatic zenith delay (m)
    # - tropo_hydrostatic_mapping  : Hydrostatic Niell mapping function
    # - tropo_zwd_m  : Non-hyd. (wet) zenith delay (m)
    # - tropo_wet_mapping  :  Non-hyd. (wet) Niell mapping function
    # - tropo_delay_m : Total slant delay (m)
    #
    # [1] https://gge.ext.unb.ca/Resources/unb3m/unb3m.html

    assert latitude_user_rad.shape == height_user_m.shape, (
        "Input arguments should be arrays of the same shape"
    )
    assert latitude_user_rad.shape == day_of_year.shape, (
        "Input arguments should be arrays of the same shape"
    )
    assert latitude_user_rad.shape == elevation_sat_rad.shape, (
        "Input arguments should be arrays of the same shape"
    )

    # Initialize UNB3m look-up table
    avg = np.array(
        [
            [15.0, 1013.25, 299.65, 75.00, 6.30, 2.77],
            [30.0, 1017.25, 294.15, 80.00, 6.05, 3.15],
            [45.0, 1015.75, 283.15, 76.00, 5.58, 2.57],
            [60.0, 1011.75, 272.15, 77.50, 5.39, 1.81],
            [75.0, 1013.00, 263.65, 82.50, 4.53, 1.55],
        ]
    )
    amp = np.array(
        [
            [15.0, 0.00, 0.00, 0.00, 0.00, 0.00],
            [30.0, -3.75, 7.00, 0.00, 0.25, 0.33],
            [45.0, -2.25, 11.00, -1.00, 0.32, 0.46],
            [60.0, -1.75, 15.00, -2.50, 0.81, 0.74],
            [75.0, -0.50, 14.50, 2.50, 0.62, 0.30],
        ]
    )
    eccentricity = 6.6943799901413e-03
    md = 28.9644
    mw = 18.0152
    k1 = 77.604
    k2 = 64.79
    k3 = 3.776e5
    r = 8314.34
    c1 = 2.2768e-03
    k2_prime = k2 - k1 * (mw / md)
    rd = r / md
    # DTR = 1.745329251994329e-02
    day_of_year_to_rad = 0.31415926535897935601e01 * 2 / 365.25

    # initialize NMF tables
    abc_avg = np.array(
        [
            [15.0, 1.2769934e-3, 2.9153695e-3, 62.610505e-3],
            [30.0, 1.2683230e-3, 2.9152299e-3, 62.837393e-3],
            [45.0, 1.2465397e-3, 2.9288445e-3, 63.721774e-3],
            [60.0, 1.2196049e-3, 2.9022565e-3, 63.824265e-3],
            [75.0, 1.2045996e-3, 2.9024912e-3, 64.258455e-3],
        ]
    )
    abc_amp = np.array(
        [
            [15.0, 0.0, 0.0, 0.0],
            [30.0, 1.2709626e-5, 2.1414979e-5, 9.0128400e-5],
            [45.0, 2.6523662e-5, 3.0160779e-5, 4.3497037e-5],
            [60.0, 3.4000452e-5, 7.2562722e-5, 84.795348e-5],
            [75.0, 4.1202191e-5, 11.723375e-5, 170.37206e-5],
        ]
    )
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    ht_topcon = 1 + a_ht / (1 + b_ht / (1 + c_ht))

    abc_w2p0 = np.array(
        [
            [15.0, 5.8021897e-4, 1.4275268e-3, 4.3472961e-2],
            [30.0, 5.6794847e-4, 1.5138625e-3, 4.6729510e-2],
            [45.0, 5.8118019e-4, 1.4572752e-3, 4.3908931e-2],
            [60.0, 5.9727542e-4, 1.5007428e-3, 4.4626982e-2],
            [75.0, 6.1641693e-4, 1.7599082e-3, 5.4736038e-2],
        ]
    )

    latitude_user_deg = np.rad2deg(latitude_user_rad)

    # Deal with southern hemisphere and yearly variation
    t_doy = day_of_year
    t_doy = np.where(latitude_user_deg < 0, t_doy + 182.625, t_doy)

    cos_phs = np.cos((t_doy - 28) * day_of_year_to_rad)

    # Initialize pointers to lookup table
    lat_abs = np.abs(latitude_user_deg)
    p1 = np.fix((lat_abs - 15) / 15) + 1 - 1
    p1 = np.where(lat_abs >= 75, 4, p1)
    p1 = np.where(lat_abs <= 15, 0, p1)
    p1 = p1.astype("int")
    p2 = p1 + 1
    p2 = np.where(lat_abs >= 75, 4, p2)
    p2 = np.where(lat_abs <= 15, 0, p2)
    m = (lat_abs - avg[p1, 0]) / (avg[p2, 0] - avg[p1, 0])
    m = np.where(lat_abs >= 75, 0, m)
    m = np.where(lat_abs <= 15, 0, m)

    # Compute average surface tropo values by interpolation
    p_avg = m * (avg[p2, 1] - avg[p1, 1]) + avg[p1, 1]
    t_avg = m * (avg[p2, 2] - avg[p1, 2]) + avg[p1, 2]
    e_avg = m * (avg[p2, 3] - avg[p1, 3]) + avg[p1, 3]
    beta_avg = m * (avg[p2, 4] - avg[p1, 4]) + avg[p1, 4]
    lambda_avg = m * (avg[p2, 5] - avg[p1, 5]) + avg[p1, 5]

    # Compute variation of average surface tropo values
    p_amp = m * (amp[p2, 1] - amp[p1, 1]) + amp[p1, 1]
    t_amp = m * (amp[p2, 2] - amp[p1, 2]) + amp[p1, 2]
    e_amp = m * (amp[p2, 3] - amp[p1, 3]) + amp[p1, 3]
    beta_amb = m * (amp[p2, 4] - amp[p1, 4]) + amp[p1, 4]
    lambda_amp = m * (amp[p2, 5] - amp[p1, 5]) + amp[p1, 5]

    # Compute surface tropo values
    p_0 = p_avg - p_amp * cos_phs
    t_0 = t_avg - t_amp * cos_phs
    e_0 = e_avg - e_amp * cos_phs
    beta = beta_avg - beta_amb * cos_phs
    beta = beta / 1000
    LAMBDA = lambda_avg - lambda_amp * cos_phs

    # Transform from relative humidity to WVP (IERS Conventions 2003)
    es = 0.01 * np.exp(
        1.2378847e-5 * (t_0**2)
        - 1.9121316e-2 * t_0
        + 3.393711047e1
        - 6.3431645e3 * (t_0**-1)
    )
    fw = 1.00062 + 3.14e-6 * p_0 + 5.6e-7 * ((t_0 - 273.15) ** 2)
    e_0 = (e_0 / 100) * es * fw

    # Compute power value for pressure & water vapour
    ep = 9.80665 / 287.054 / beta

    # Scale surface values to required height
    t = t_0 - beta * height_user_m
    p = p_0 * (t / t_0) ** ep
    e = e_0 * (t / t_0) ** (ep * (LAMBDA + 1))

    # Compute the acceleration at the mass center of a vertical column of the atmosphere
    geo_lat = np.arctan((1 - eccentricity) * np.tan(latitude_user_rad))
    dgref = 1 - 2.66e-03 * np.cos(2 * geo_lat) - 2.8e-07 * height_user_m
    gm = 9.784 * dgref
    den = (LAMBDA + 1) * gm

    # Compute mean temperature of the water vapor
    t_m = t * (1 - beta * rd / den)

    # Compute zenith hydrostatic delay
    tropo_zhd_m = c1 / dgref * p

    # Compute zenith wet delay
    tropo_zwd_m = 1e-6 * (k2_prime + k3 / t_m) * rd * e / den

    # Compute average NMF(H) coefficient values by interpolation
    a_avg = m * (abc_avg[p2, 1] - abc_avg[p1, 1]) + abc_avg[p1, 1]
    b_avg = m * (abc_avg[p2, 2] - abc_avg[p1, 2]) + abc_avg[p1, 2]
    c_avg = m * (abc_avg[p2, 3] - abc_avg[p1, 3]) + abc_avg[p1, 3]

    # Compute variation of average NMF(H) coefficient values
    a_amp = m * (abc_amp[p2, 1] - abc_amp[p1, 1]) + abc_amp[p1, 1]
    b_amp = m * (abc_amp[p2, 2] - abc_amp[p1, 2]) + abc_amp[p1, 2]
    c_amp = m * (abc_amp[p2, 3] - abc_amp[p1, 3]) + abc_amp[p1, 3]

    # Compute NMF(H) coefficient values
    a = a_avg - a_amp * cos_phs
    b = b_avg - b_amp * cos_phs
    c = c_avg - c_amp * cos_phs

    # Compute sine of elevation angle
    sin_e = np.sin(elevation_sat_rad)

    # Compute NMF(H) value
    alpha = b / (sin_e + c)
    gamma = a / (sin_e + alpha)
    topcon = 1 + a / (1 + b / (1 + c))
    tropo_hydrostatic_mapping = topcon / (sin_e + gamma)

    # Compute and apply height correction
    alpha = b_ht / (sin_e + c_ht)
    gamma = a_ht / (sin_e + alpha)
    ht_corr_coef = 1 / sin_e - ht_topcon / (sin_e + gamma)
    ht_corr = ht_corr_coef * height_user_m / 1000
    tropo_hydrostatic_mapping = tropo_hydrostatic_mapping + ht_corr

    # Compute average NMF(W) coefficient values by interpolation
    a = m * (abc_w2p0[p2, 1] - abc_w2p0[p1, 1]) + abc_w2p0[p1, 1]
    b = m * (abc_w2p0[p2, 2] - abc_w2p0[p1, 2]) + abc_w2p0[p1, 2]
    c = m * (abc_w2p0[p2, 3] - abc_w2p0[p1, 3]) + abc_w2p0[p1, 3]

    # Compute NMF(W) value
    alpha = b / (sin_e + c)
    gamma = a / (sin_e + alpha)
    topcon = 1 + a / (1 + b / (1 + c))
    tropo_wet_mapping = topcon / (sin_e + gamma)

    # Compute total slant delay
    tropo_delay_m = (
        tropo_zhd_m * tropo_hydrostatic_mapping + tropo_zwd_m * tropo_wet_mapping
    )

    return (
        tropo_delay_m,
        tropo_zhd_m,
        tropo_hydrostatic_mapping,
        tropo_zwd_m,
        tropo_wet_mapping,
    )


def compute_tropo_delay(sat_states, flat_obs, receiver_ecef_position_m, model):
    """
    model is either "saastamoinen" or "unb3m"
    """
    [latitude_user_rad, __, height_user_m] = ecef_2_geodetic(receiver_ecef_position_m)
    match model:
        case "saastamoinen":
            tropo_delay_m, _, _ = compute_tropo_delay_saastamoinen(
                height_user_m * np.ones((len(sat_states),)),
                sat_states.elevation_rad.to_numpy(),
                latitude_user_rad * np.ones((len(sat_states),)),
            )
        case "unb3m":
            # We need Timestamps to compute tropo delays
            sat_states = sat_states.merge(
                flat_obs[
                    [
                        "satellite",
                        "time_of_emission_isagpst",
                        "time_of_reception_in_receiver_time",
                    ]
                ].drop_duplicates(),
                on=["satellite", "time_of_emission_isagpst"],
                how="left",
            )
            days_of_year = np.array(
                sat_states["time_of_reception_in_receiver_time"]
                .apply(lambda element: element.timetuple().tm_yday)
                .to_numpy()
            )
            (
                tropo_delay_m,
                __,
                __,
                __,
                __,
            ) = compute_tropo_delay_unb3m(
                latitude_user_rad * np.ones((len(sat_states),)),
                height_user_m * np.ones((len(sat_states),)),
                days_of_year,
                sat_states.elevation_rad.to_numpy(),
            )
    return tropo_delay_m
