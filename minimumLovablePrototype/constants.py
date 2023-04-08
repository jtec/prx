import numpy as np
import pandas as pd

cGpstUtcEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))
cArbitraryGlonassUtcEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))


cNanoSecondsPerSecond = 1e9
cMicrosecondsPerSecond = 1e6
cSecondsPerDay = 86400
cSecondsPerHour = 60 * 60
cSecondsPerWeek = 7 * cSecondsPerDay
cNanoSecondsPerWeek = cSecondsPerWeek * cNanoSecondsPerSecond
cHzPerMhz = 1e6
cGpsIcdSpeedOfLight_mps = 2.99792458 * 1e8
cPrxReferenceFrame = {"name": "IGS14", "epoch": "2005.001"}
# Heuristic: demand micrometer precision in computations involving distances
cPrxPrecision_m = 1e-6
cMaxOrbitalSpeed_mps = 1e4

cPrxJsonTextSequenceFileExtension = "jsonseq"
cPrxCsvFileExtension = "csv"
