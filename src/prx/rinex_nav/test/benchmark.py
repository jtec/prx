import pandas as pd
from pathlib import Path
from prx.rinex_nav import evaluate as rinex_nav_evaluate
from prx import constants
from prx import converters
import cProfile


def generate_query(n_epochs=1):
    rinex_nav_file = converters.compressed_to_uncompressed(
        Path(__file__).parent / "datasets/BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    df = rinex_nav_evaluate.parse_rinex_nav_file(rinex_nav_file)
    df["constellation"] = df["sv"].str[0]
    sats = []
    # Grab ten (or all, if the constellation has less than ten) satellites of each constellation as an approximation
    # of what an open-sky receiver would see
    for _, constellation_df in df.groupby("constellation"):
        constellation_sats = constellation_df["sv"].unique()
        sats.extend(constellation_sats[0 : min(10, len(constellation_sats))])
    # Only Kepler orbits supported:
    sats = [sat for sat in sats if sat[0] not in ["S", "R"]]
    query_template = pd.DataFrame(
        {
            "sv": sats,
            "signal": "C1C",
            "query_time_isagpst": pd.Timestamp("2022-01-01T01:10:00.000000000")
            - constants.cGpstUtcEpoch,
        }
    )
    query = query_template.copy()
    for i in range(1, n_epochs):
        next_query = query_template.copy()
        next_query["query_time_isagpst"] += pd.Timedelta(seconds=i)
        query = pd.concat((query, next_query), axis=0)
    return query


def benchmark(query, rinex_nav_file):
    return rinex_nav_evaluate.compute(rinex_nav_file, query)


if __name__ == "__main__":
    rinex_nav_file = converters.compressed_to_uncompressed(
        Path(__file__).parent / "datasets/BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    query = generate_query(100)
    print(
        f"Profiling ephemeris evaluation with {len(query.sv.unique())} satellites and"
        f" {len(query.query_time_isagpst.unique())} epochs"
    )
    p = cProfile.Profile()
    # Warm up cache: we can expect a navigation file to be cached
    result = benchmark(query, rinex_nav_file)
    p.enable()
    benchmark(query, rinex_nav_file)
    p.disable()
    stats_file = Path("benchmark_1.prof").resolve()
    p.dump_stats(stats_file)
    df = pd.DataFrame(
        p.getstats(),
        columns=["func", "ncalls", "ccalls", "tottime", "cumtime", "callers"],
    ).sort_values(by="tottime", ascending=False)
    print(
        f"Approximate ephemeris evaluation speed: {len(query.query_time_isagpst.unique()) / df.iloc[0, :]['tottime']} epochs per second"
    )
    print(f"To inspect profiling results, call \n snakeviz {stats_file}")
