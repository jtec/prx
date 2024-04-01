import numpy as np
import pandas as pd
from collections import defaultdict

cGpstUtcEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))
cBdtUtcEpoch = pd.Timestamp(np.datetime64("2006-01-01T00:00:00.000000000"))

cNanoSecondsPerSecond = 1e9
cMicrosecondsPerSecond = 1e6
cSecondsPerDay = 86400
cSecondsPerMinute = 60
cSecondsPerHour = 60 * cSecondsPerMinute
cSecondsPerWeek = 7 * cSecondsPerDay
cNanoSecondsPerWeek = cSecondsPerWeek * cNanoSecondsPerSecond
cMetersPerKilometer = 1e3
cHzPerMhz = 1e6
# WGS84 Geoid constants
cWgs84EarthFlatteningFactor = 1 / 298.257223563
cWgs84EarthSemiMajorAxis_m = 6378137.0
cWgs84EarthEccentricity = np.sqrt(
    2 * cWgs84EarthFlatteningFactor - cWgs84EarthFlatteningFactor ** 2
)
# GPS ICD constants
cGpsPi = 3.1415926535898
cGpsSpeedOfLight_mps = 2.99792458 * 1e8
cGpsMuEarth_m3ps2 = 3.986005e14
cGpsOmegaDotEarth_rps = 7.2921151467e-5
# Beidou ICD constants
cBdsPi = cGpsPi
cBdsSpeedOfLight_mps = cGpsSpeedOfLight_mps
cBdsMuEarth_m3ps2 = 3.986004418e14
cBdsOmegaDotEarth_rps = 7.2921150e-5
cBdsCgcs2000SmiMajorAxis_m = 6378137.0
# Galileo ICD constants
cGalPi = cGpsPi
cGalSpeedOfLight_mps = cGpsSpeedOfLight_mps
cGalMuEarth_m3ps2 = 3.986004418e14
cGalOmegaDotEarth_rps = 7.2921151467e-5
# QZSS ICD constants
cQzssPi = cGpsPi
cQzssSpeedOfLight_mps = cGpsSpeedOfLight_mps
cQzssMuEarth_m3ps2 = 3.986005e14
cQzssOmegaDotEarth_rps = 7.2921151467e-5
# Heuristic: demand micrometer precision in computations involving distances
cPrxPrecision_m = 1e-6
cMaxOrbitalSpeedOfAnyGnssSatellite_mps = 1e4

cPrxJsonTextSequenceFileExtension = "jsonseq"
cPrxCsvFileExtension = "csv"


# From RINEX Version 3.05, 1 December, 2020.
def carrier_frequencies_hz():
    cf = defaultdict(dict)
    # GPS
    # We use at least one frequency slot here for all constellations to accommodate GLONASS.
    cf["G"]["L1"] = {int(1): 1575.42 * cHzPerMhz}
    cf["G"]["L2"] = {int(1): 1227.60 * cHzPerMhz}
    cf["G"]["L5"] = {int(1): 1176.45 * cHzPerMhz}
    # GLONASS FDMA signals
    cf["R"]["L1"] = defaultdict(dict)
    cf["R"]["L2"] = defaultdict(dict)
    for frequency_slot in range(-7, 12 + 1):
        cf["R"]["L1"][frequency_slot] = (1602 + frequency_slot * 9 / 16) * cHzPerMhz
        cf["R"]["L2"][frequency_slot] = (1246 + frequency_slot * 7 / 16) * cHzPerMhz
    # Glonass CDMA signals
    cf["R"]["L4"] = {int(1): 1600.995 * cHzPerMhz}
    cf["R"]["L3"] = {int(1): 1202.025 * cHzPerMhz}
    # Galileo
    cf["E"]["L1"] = {int(1): 1575.42 * cHzPerMhz}
    cf["E"]["L5"] = {int(1): 1176.45 * cHzPerMhz}
    cf["E"]["L7"] = {int(1): 1207.140 * cHzPerMhz}
    cf["E"]["L8"] = {int(1): 1191.795 * cHzPerMhz}
    cf["E"]["L6"] = {int(1): 1278.75 * cHzPerMhz}
    # SBAS
    cf["S"]["L1"] = {int(1): 1575.42 * cHzPerMhz}
    cf["S"]["L5"] = {int(1): 1176.45 * cHzPerMhz}
    # QZSS
    cf["J"]["L1"] = {int(1): 1575.42 * cHzPerMhz}
    cf["J"]["L2"] = {int(1): 1227.60 * cHzPerMhz}
    cf["J"]["L5"] = {int(1): 1176.45 * cHzPerMhz}
    cf["J"]["L6"] = {int(1): 1278.75 * cHzPerMhz}
    # Beidou
    cf["C"]["L1"] = {int(1): 1575.42 * cHzPerMhz}
    cf["C"]["L2"] = {int(1): 1561.098 * cHzPerMhz}
    cf["C"]["L5"] = {int(1): 1176.45 * cHzPerMhz}
    cf["C"]["L7"] = {int(1): 1207.140 * cHzPerMhz}
    cf["C"]["L6"] = {int(1): 1268.52 * cHzPerMhz}
    cf["C"]["L8"] = {int(1): 1191.795 * cHzPerMhz}
    # NavIC/IRNSS
    cf["I"]["L5"] = {int(1): 1176.45 * cHzPerMhz}
    cf["I"]["S"] = {int(1): 2492.028 * cHzPerMhz}
    return dict(cf)


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

system_time_scale_rinex_utc_epoch = {
    # Returns the time at which the system time scale (e.g. in weeks and week seconds) is zero.
    "GPST": cGpstUtcEpoch,
    "SBAST": cGpstUtcEpoch,
    "QZSST": cGpstUtcEpoch,
    "IRNSST": cGpstUtcEpoch,
    # The offset between GST and GPST is removed by RINEX file encoders:
    "GST": cGpstUtcEpoch,
    "BDT": cBdtUtcEpoch,
    # GLONASS time does not use weeks and week seconds, it has no epoch,
    "GLONASST": pd.NaT,
}

gfzrnx_binary = {
    "Windows": "gfzrnx_217_win64.exe",
    "Linux": "gfzrnx_217_lx64",
    "Darwin": "gfzrnx_217_osx_intl64",
}
