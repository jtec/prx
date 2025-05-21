import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from prx.rinex_nav.evaluate import parse_rinex_nav_file
from src.prx.rinex_nav.nav_file_discovery import discover_or_download_ephemerides


def main(t1: pd.Timestamp, t2: pd.Timestamp) -> None:
    nav_files = discover_or_download_ephemerides(t1, t2)
    ephemeris_blocks = []
    for path in nav_files:
        block = parse_rinex_nav_file(path)
        # Not using iono model parameters here, removing them from dataframe attributes to
        # enable concatenation
        block.attrs.pop("ionospheric_corr_GPS", None)
        ephemeris_blocks.append(block)
    ephemerides = pd.concat(ephemeris_blocks)
    ephemerides = (
        ephemerides[ephemerides.sv.str.startswith("C")]
        .sort_values(by=["sv"])
        .reset_index(drop=True)
    )
    fig = make_subplots(rows=1, cols=1)

    health2color = {
        0: "green",
        1: "red",
    }
    for (sv, health), group_df in ephemerides.groupby(["sv", "SatH1"]):
        prn = float(sv[1:])
        offset = 0 if health == 0 else 0.1
        fig.add_trace(
            go.Scatter(
                x=group_df["time"],
                y=np.full_like(group_df["time"].to_numpy(), prn, dtype=float) + offset,
                mode="lines+markers",
                marker=dict(
                    color=health2color[health],
                    size=5,
                ),
                customdata=group_df[["TransTime", "ephemeris_hash"]],
                hovertemplate="<br>".join(
                    [
                        "timestamp: %{x}",
                        "PRN: %{y:.0f}",
                        "time of transmission [tow]: %{customdata[0]}",
                        "ephemeris_hash [-]: %{customdata[1]}",
                    ]
                ),
                name=sv,
                legendgroup=sv,
                showlegend=bool(health == 0),
            )
        )
    fig.update_xaxes(title="ephemeris time stamp")
    fig.update_yaxes(title="PRN")
    fig.show()
    pass


if __name__ == "__main__":
    t1 = pd.Timestamp("2025-04-10T00:00:00")
    t2 = pd.Timestamp("2025-04-11T00:00:00")
    main(t1, t2)
