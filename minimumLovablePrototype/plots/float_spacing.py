import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_float_spacing(a, b):
    floats = np.geomspace(a, b, num=100)
    space_between_float64s = np.spacing(floats)
    plt.plot(floats, space_between_float64s)
    plt.grid()
    plt.title(f"Smallest spacing between float64 numbers using np.spacing \n within [{a}, today's GPST in seconds].")
    plt.xlabel("float64 number")
    plt.ylabel("Space")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(Path(__file__).parent.joinpath("float_spacing.png"), dpi=300)


if __name__ == "__main__":
    cGpstUtcEpoch = pd.Timestamp(np.datetime64("1980-01-06T00:00:00.000000000"))
    today_gpst_s = pd.Timestamp.now() - cGpstUtcEpoch
    plot_float_spacing(1e-6, today_gpst_s.total_seconds())
