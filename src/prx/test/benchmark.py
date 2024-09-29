import pandas as pd
from pathlib import Path

from prx import converters
from prx.main import process
import cProfile


def benchmark():
    obs_file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_2024001"
        / "TLSE00FRA_R_20240011200_15M_01S_MO.crx.gz"
    )
    # Move ephemeris file into obs file so that prx finds it and does not attempt to download it

    process(obs_file)


if __name__ == "__main__":
    p = cProfile.Profile()
    # Warm up cached functions, parsers are benchmarked separately.
    benchmark()
    p.enable()
    benchmark()
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
