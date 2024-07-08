import subprocess
import timeit

import numpy as np
import pandas as pd
from pathlib import Path
from prx.rinex_obs.parser import parse as prx_parse
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import georinex


def run_obs3_parser(file_path):
    # parsed = gr.load(file_path)
    parsed = janobs(file_path)
    ...


def generate_data():
    base_file = Path(__file__).parent / "datasets" / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx"
    sweep_dir = base_file.parent / "sweep"
    results_file = sweep_dir / "benchmark.csv"
    if results_file.exists():
        return pd.read_csv(results_file)
    sweep_dir.mkdir(exist_ok=True)
    times = georinex.obstime3(base_file)
    header = georinex.rinexheader(base_file)
    t_start = pd.Timestamp(np.min(times))
    t_end = pd.Timestamp(np.max(times))
    n_steps = 10
    dt = (t_end - t_start) / n_steps
    cases = []
    for steps in range(1, n_steps, 1):
        duration = dt * steps
        slice_file = sweep_dir / f"{base_file.name}_slice_{duration / pd.Timedelta('1h'):.2f}h.rnx"
        cmd = (f"gfzrnx_217_osx_intl64 -finp {base_file}"
               f" -fout {slice_file}"
               f" -epo_beg {t_start.strftime('%Y-%m-%d_%H%M%S')}"
               f" -d {int(duration / pd.Timedelta('1s'))}")
        if not slice_file.exists():
            process_output = subprocess.run(cmd, shell=True, capture_output=True)
            assert process_output.returncode == 0
            print(f"Created {slice_file}")
        print(f"Adding {slice_file} to the database ...")
        cases.append({"epochs": (duration / pd.Timedelta('1s')) / float(header['INTERVAL']),
                      "file": slice_file, })
    for case in cases:
        for parser in [georinex.load, prx_parse]:
            print(f"Processing {case}")
            case["georinex_parsing_s"] = timeit.timeit(lambda: georinex.load(case['file']), number=2)
            case["prx_parsing_s"] = timeit.timeit(lambda: prx_parse(case['file']), number=2)
            case['epochs_per_second'] = case['epochs'] / case['parsing_s']
            print(f"took {case['parsing_s']:.2f} s ({case['epochs_per_second']:.2f} epochs/s)")
    df = pd.DataFrame(cases)
    df.to_csv(results_file, index=False)


if __name__ == "__main__":
    df = generate_data()
    df["file_size_mbytes"] = df.file.apply(lambda x: Path(x).stat().st_size) / 1e6
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(x=df.file_size_mbytes, y=df.epochs_per_second, mode='lines+markers', name="Epochs per second"),
        row=1, col=1)
    fig.update_layout(title_text="Performance of the Obs3 parser")
    fig.update_xaxes(title_text="File size [MB]", row=1, col=1)
    fig.update_yaxes(title_text="Epochs per second", range=[0, df.epochs_per_second.max() * 1.1], row=1, col=1)
    fig.show()
    ...
