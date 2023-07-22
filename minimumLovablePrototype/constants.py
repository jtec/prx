import numpy as np
import pandas as pd
from collections import defaultdict

cGpstUtcEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))
cArbitraryGlonassUtcEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))

cNanoSecondsPerSecond = 1e9
cMicrosecondsPerSecond = 1e6
cSecondsPerDay = 86400
cSecondsPerMinute = 60
cSecondsPerHour = 60 * cSecondsPerMinute
cSecondsPerWeek = 7 * cSecondsPerDay
cNanoSecondsPerWeek = cSecondsPerWeek * cNanoSecondsPerSecond
cHzPerMhz = 1e6
cGpsIcdSpeedOfLight_mps = 2.99792458 * 1e8
cEarthRotationRate_radps = 7.2921151467e-5
cPrxReferenceFrame = {"name": "IGS14", "epoch": "2005.001"}
# Heuristic: demand micrometer precision in computations involving distances
cPrxPrecision_m = 1e-6
cMaxOrbitalSpeed_mps = 1e4

cPrxJsonTextSequenceFileExtension = "jsonseq"
cPrxCsvFileExtension = "csv"

# the 8th and 9th bit of the `data source` parameter in the Galileo navigation message allows to identify the type of message (F/NAV vs I/NAV)
cGalileoFnavDataSourceIndicator = 512


# From RINEX Version 3.05, 1 December, 2020.
def carrier_frequencies_hz():
    cf = defaultdict(dict)
    # GPS
    cf["G"]["L1"] = 1575.42 * cHzPerMhz
    cf["G"]["L2"] = 1227.60 * cHzPerMhz
    cf["G"]["L5"] = 1176.45 * cHzPerMhz
    # GLONASS FDMA signals
    cf["R"]["L1"] = defaultdict(dict)
    cf["R"]["L2"] = defaultdict(dict)
    for frequency_slot in range(-7, 12 + 1):
        cf["R"]["L1"][frequency_slot] = (1602 + frequency_slot * 9 / 16) * cHzPerMhz
        cf["R"]["L2"][frequency_slot] = (1246 + frequency_slot * 7 / 16) * cHzPerMhz
    # Glonass CDMA signals
    cf["R"]["L4"] = 1600.995 * cHzPerMhz
    cf["R"]["L3"] = 1202.025 * cHzPerMhz
    # Galileo
    cf["E"]["L1"] = 1575.42 * cHzPerMhz
    cf["E"]["L5"] = 1176.45 * cHzPerMhz
    cf["E"]["L7"] = 1207.140 * cHzPerMhz
    cf["E"]["L8"] = 1191.795 * cHzPerMhz
    cf["E"]["L6"] = 1278.75 * cHzPerMhz
    # SBAS
    cf["S"]["L1"] = 1575.42 * cHzPerMhz
    cf["S"]["L5"] = 1176.45 * cHzPerMhz
    # QZSS
    cf["J"]["L1"] = 1575.42 * cHzPerMhz
    cf["J"]["L2"] = 1227.60 * cHzPerMhz
    cf["J"]["L5"] = 1176.45 * cHzPerMhz
    cf["J"]["L6"] = 1278.75 * cHzPerMhz
    # Beidou
    cf["C"]["L1"] = 1575.42 * cHzPerMhz
    cf["C"]["L2"] = 1561.098 * cHzPerMhz
    cf["C"]["L5"] = 1176.45 * cHzPerMhz
    cf["C"]["L7"] = 1207.140 * cHzPerMhz
    cf["C"]["L6"] = 1268.52 * cHzPerMhz
    cf["C"]["L8"] = 1191.795 * cHzPerMhz
    # NavIC/IRNSS
    cf["I"]["L5"] = 1176.45 * cHzPerMhz
    cf["I"]["S"] = 2492.028 * cHzPerMhz
    return cf
