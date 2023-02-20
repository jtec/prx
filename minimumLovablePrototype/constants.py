import numpy as np
import pandas as pd

cGpstEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))
cNanoSecondsPerSecond = 1e9
cMicrosecondsPerSecond = 1e6
cSecondsPerDay = 86400
cSecondsPerHour = 60 * 60
cSecondsPerWeek = 7 * cSecondsPerDay
cHzPerMhz = 1e6
cGpsIcdSpeedOfLight_mps = 2.99792458 * 1e8
cPrxReferenceFrame = {"name": "IGS14", "epoch": "2005.001"}

cPrxJsonTextSequenceFileExtension = "jsonseq"
cPrxCsvFileExtension = "csv"
