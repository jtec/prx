import shutil

import pandas as pd
from pathlib import Path

from prx import converters
from prx.main import process
import cProfile


def setup():
    benchmark_dataset_directory = Path(__file__).parent / "benchmark"
    shutil.rmtree(benchmark_dataset_directory, ignore_errors=True)
    benchmark_dataset_directory.mkdir(exist_ok=True, parents=True)
    obs_file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_2024001"
        / "TLSE00FRA_R_20240011200_15M_01S_MO.crx.gz"
    )
    nav_file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_2024001"
        / "BRDC00IGS_R_20240010000_01D_MN.rnx.gz"
    )
    shutil.copy(obs_file, benchmark_dataset_directory)
    shutil.copy(nav_file, benchmark_dataset_directory)
    return benchmark_dataset_directory / obs_file.name


def benchmark(obs_file: Path):
    process(obs_file)


if __name__ == "__main__":
    p = cProfile.Profile()
    obs_file = setup()
    # Warm up cached functions, parsers are benchmarked separately.
    benchmark(obs_file)
    p.enable()
    benchmark(obs_file)
    p.disable()
    stats_file = Path("benchmark_prx.prof").resolve()
    p.dump_stats(stats_file)
    df = pd.DataFrame(
        p.getstats(),
        columns=["func", "ncalls", "ccalls", "tottime", "cumtime", "callers"],
    ).sort_values(by="tottime", ascending=False)
    print(f"Processed {obs_file.name} in {df.iloc[0, :]['tottime']} seconds")
    print(f"To inspect profiling results, call \n snakeviz {stats_file}")
