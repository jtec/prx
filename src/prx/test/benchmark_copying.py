import time
import tracemalloc

import numpy as np
import pandas as pd
import polars as pl
from string import ascii_uppercase


def wraps_polars(df: pd.DataFrame):
    pl_df = pl.from_pandas(df)
    pl_df = pl_df.select(pl.all() * 2)
    return pl_df.to_pandas(
        split_blocks=True,
        self_destruct=True,
    )


def main():
    array_bytes = 1e9
    n_columns = 10
    np_datatype = np.float64
    n_rows = int((array_bytes / n_columns) / 8)
    time.sleep(1)

    pandas_df = pd.DataFrame(
        {
            column_name: np.random.random(n_rows).astype(np_datatype)
            for column_name in ascii_uppercase[:n_columns]
        }
    )
    time.sleep(1)

    pandas_df = wraps_polars(pandas_df)
    time.sleep(1)

    # input(f"Press Enter to exit process ({os.getpid()}) ...")


if __name__ == "__main__":
    tracemalloc.start()
    main()
    time.sleep(1)
    current, peak = tracemalloc.get_traced_memory()
    print(f"tracemalloc current : {current / 10**6} MB")
    print(f"tracemalloc peak: {peak / 10**6} MB")
