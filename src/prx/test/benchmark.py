import argparse
import logging

import numpy as np
import pandas as pd
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from prx.main import process
import cProfile
import memray
from prx.rinex_obs.test.benchmark import generate_inputs
from prx.util import configure_logging, disk_cache

logger = logging.getLogger(__name__)


def run_case(case: dict, ram: bool) -> pd.DataFrame:
    obs_file = Path(case["obs_file"])
    # Purge caches, we're going for run time and peak RAM with a file prx has never seen before
    disk_cache.clear()
    p = cProfile.Profile()
    p.enable()
    process(observation_file_path=obs_file)
    p.disable()

    # RAM
    # memray adds overhead, measure peak RAM separately from run time
    peak_ram_mb = np.nan
    if ram:
        memray_output = Path("memray.bin")
        memray_output.unlink(missing_ok=True)
        disk_cache.clear()
        with memray.Tracker(memray_output):
            process(observation_file_path=obs_file)
        reader = memray.FileReader(memray_output)
        metadata = reader.metadata
        peak_ram_mb = metadata.peak_memory / 1024 / 1024
        memray_output.unlink(missing_ok=True)

    # Run time
    stats_file = Path("benchmark_prx.prof").resolve()
    p.dump_stats(stats_file)
    df = (
        pd.DataFrame(
            p.getstats(),
            columns=["func", "ncalls", "ccalls", "tottime", "cumtime", "callers"],
        )
        .sort_values(by="tottime", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"Processed {obs_file.name} in {df.iloc[0, :]['tottime']} seconds")
    df = df[["func", "tottime"]]
    df["function"] = df["func"].apply(lambda x: getattr(x, "co_name", None))
    df["file"] = df["func"].apply(lambda x: getattr(x, "co_filename", None))
    df = df[df["function"].notnull() & df["file"].str.contains("prx/src/prx/")]
    not_interesting = ["timeit_wrapper", "<genexpr>"]
    df = df[~df["function"].isin(not_interesting)]
    df = df.drop(columns=["func"])
    df["obs_epochs"] = case["epochs"]
    df["peak_ram_mb"] = peak_ram_mb
    return df


def main(ram: bool):
    configure_logging("DEBUG")
    cases = generate_inputs(n_steps=10)
    df = pd.concat([run_case(case, ram) for case in cases])
    fig = make_subplots(rows=2, cols=1)
    for (file, function), group in df.groupby(["file", "function"]):
        fig.add_trace(
            go.Scatter(x=group["obs_epochs"], y=group["tottime"], name=function),
            row=1,
            col=1,
        )
    ram = df[["obs_epochs", "peak_ram_mb"]].drop_duplicates().reset_index(drop=True)
    fig.add_trace(
        go.Scatter(
            x=ram["obs_epochs"], y=ram["peak_ram_mb"], name="peak ram", showlegend=False
        ),
        row=2,
        col=1,
    )
    for row in [1, 2]:
        fig.update_xaxes(title_text="#epochs", row=row, col=1)
    fig.update_yaxes(title_text="total time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Peak RAM [Mb]", row=1, col=1)
    fig.update_layout(hoverlabel=dict(namelength=-1))
    fig.update_layout(title=dict(text="prx benchmark"))
    fig.show("browser")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--ram", action="store_true")
    args = arg_parser.parse_args()
    main(ram=args.ram)
