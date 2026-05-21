import shutil
import subprocess
import timeit

import numpy as np
import pandas as pd
from pathlib import Path

from prx import converters, util
from prx.rinex_obs.parser import parse as prx_obs_parse
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import georinex

from prx.util import prx_src_directory


def generate_inputs(n_steps: int = 10, root: Path = None) -> list[dict]:
    if root is None:
        root = Path(__file__).parent
    base_obs_file = converters.anything_to_rinex_3(
        root / "datasets" / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx.gz"
    )
    base_nav_file = converters.anything_to_rinex_3(
        prx_src_directory()
        / "rinex_nav/test/datasets/BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    sweep_dir = base_obs_file.parent / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    times = georinex.obstime3(base_obs_file)
    header = georinex.rinexheader(base_obs_file)
    t_start = pd.Timestamp(np.min(times))
    t_end = pd.Timestamp(np.max(times))
    dt = (t_end - t_start) / n_steps
    cases = []
    for steps in range(1, n_steps, 1):
        duration = dt * steps
        slice_obs_file = (
            sweep_dir
            / f"{base_obs_file.name}_slice_{duration / pd.Timedelta('1h'):.2f}h.rnx"
        )
        cmd = (
            f"gfzrnx -finp {base_obs_file}"
            f" -fout {slice_obs_file}"
            f" -epo_beg {t_start.strftime('%Y-%m-%d_%H%M%S')}"
            f" -d {int(duration / pd.Timedelta('1s'))}"
        )
        if not slice_obs_file.exists():
            process_output = subprocess.run(cmd, shell=True, capture_output=True)
            assert process_output.returncode == 0
            print(f"Created {slice_obs_file}")
        print(f"Adding {slice_obs_file} to the database ...")
        slice_nav_file = slice_obs_file.parent / base_nav_file.name
        if not slice_nav_file.exists():
            shutil.copy(base_nav_file, slice_nav_file)
        cases.append(
            {
                "epochs": (duration / pd.Timedelta("1s")) / float(header["INTERVAL"]),
                "obs_file": slice_obs_file,
                "nav_file": slice_nav_file,
            }
        )
    return cases


def run_parser(cases: list[dict]) -> pd.DataFrame:
    for case in cases:
        for parser in [
            ("prx", prx_obs_parse),
        ]:
            print(f"Processing {case}")
            util.disk_cache.clear()
            case[f"{parser[0]}_parsing_s"] = timeit.timeit(
                lambda: parser[1](case["obs_file"]), number=1
            )
            case[f"{parser[0]}_epochs_per_second"] = (
                case["epochs"] / case[f"{parser[0]}_parsing_s"]
            )
            print(
                f"took {case[f'{parser[0]}_parsing_s']:.2f} s"
                f" ({case[f'{parser[0]}_epochs_per_second']:.2f} epochs/s)"
                f" with {parser[0]}"
            )
    return pd.DataFrame(cases)


if __name__ == "__main__":
    df = run_parser(generate_inputs())
    df["file_size_mbytes"] = (
        df["obs_file"].apply(lambda x: Path(x).stat().st_size) / 1e6
    )
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for col in [col for col in df.columns if "_epochs_per_second" in col]:
        fig.add_trace(
            go.Scatter(
                x=df.file_size_mbytes,
                y=df[col],
                mode="lines+markers",
                name=f"Epochs per second {col.split('_')[0]}",
            ),
            row=1,
            col=1,
        )
    fig.update_layout(title_text="Performance of rinex 3 obs file parsers")
    fig.update_xaxes(title_text="File size [MB]", row=1, col=1)
    fig.update_yaxes(title_text="Epochs per second", range=[0, None], row=1, col=1)
    fig.show()
    ...
