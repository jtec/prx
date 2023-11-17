import pandas as pd
from pathlib import Path
from prx.rinex_nav import evaluate as rinex_nav_evaluate
from prx import constants
from prx import converters
import cProfile as profile


def benchmark_1():
    rinex_nav_file = converters.compressed_to_uncompressed(
        Path(__file__).parent / "datasets/BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    query_times = {}
    sat_state_query_time_gpst = (
        pd.Timestamp("2022-01-01T01:00:00.000000000") - constants.cGpstUtcEpoch
    )
    # Time-of-transmission is different for each satellite, simulate that here
    # Multiple satellites with ephemerides provided as Kepler orbits
    # Two Beidou GEO (from http://www.csno-tarc.cn/en/system/constellation)
    query_times["C03"] = sat_state_query_time_gpst
    query_times["C05"] = sat_state_query_time_gpst
    # One Beidou IGSO
    query_times["C38"] = sat_state_query_time_gpst
    # One Beidou MEO
    query_times["C30"] = sat_state_query_time_gpst
    # Two GPS
    query_times["G15"] = sat_state_query_time_gpst
    query_times["G12"] = sat_state_query_time_gpst
    # Two Galileo
    query_times["E24"] = sat_state_query_time_gpst
    query_times["E30"] = sat_state_query_time_gpst
    # Two QZSS
    query_times["J02"] = sat_state_query_time_gpst
    query_times["J03"] = sat_state_query_time_gpst

    # Multiple satellites with orbits that require propagation of an initial state
    # Two GLONASS satellites
    # query_times["R04"] = sat_state_query_time_gpst + pd.Timedelta(seconds=20)/1e3
    # query_times["R05"] = sat_state_query_time_gpst + pd.Timedelta(seconds=21)/1e3
    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query_times)


if __name__ == "__main__":
    p = profile.Profile()
    p.runcall(benchmark_1)
    stats_file = Path("benchmark_1.prof").resolve()
    p.dump_stats(stats_file)
    print(f"Call snakeviz {stats_file} to inspect profiling results.")
