# Imports at the top of the file
import polars as pl
from prx.util import (
    timeit,
)


@timeit
def select_ephemerides_polars(df, query):
    # 1. Filter out rows with missing reference times
    df = df.filter(pl.col("ephemeris_reference_time_isagpst").is_not_null())

    # 2. Sort both DataFrames
    query = query.sort("ephemeris_selection_time_isagpst")
    df = df.sort("ephemeris_reference_time_isagpst")

    # 3. Add FNAV/INAV indicator (minimal branching)
    query = query.with_columns(
        pl.when(
            (pl.col("constellation") == "E")
            & (pl.col("signal").cast(pl.Utf8).str.slice(1, 1) == "5")
        )
        .then(pl.lit("fnav"))
        .when(pl.col("constellation") == "E")
        .then(pl.lit("inav"))
        .otherwise(pl.lit(""))
        .alias("fnav_or_inav")
    )

    # 4. Create join_key (needed since Polars join_asof only supports one `by`)
    query = query.with_columns(
        (
            pl.col("constellation")
            + "_"
            + pl.col("sv").cast(pl.Utf8)
            + "_"
            + pl.col("fnav_or_inav")
        ).alias("join_key")
    )
    df = df.with_columns(
        (
            pl.col("constellation")
            + "_"
            + pl.col("sv").cast(pl.Utf8)
            + "_"
            + pl.lit("")
        ).alias("join_key")
    )

    # 5. Merge with asof join (fastest available in Polars)
    merged = query.join_asof(
        df,
        left_on="ephemeris_selection_time_isagpst",
        right_on="ephemeris_reference_time_isagpst",
        by="join_key",
        strategy="backward",
    )

    # 6. Compute time deltas directly
    merged = merged.with_columns(
        [
            (pl.col("query_time_isagpst") - pl.col("ephemeris_reference_time_isagpst"))
            .dt.total_seconds()
            .alias("query_time_wrt_ephemeris_reference_time_s"),
            (pl.col("query_time_isagpst") - pl.col("clock_reference_time_isagpst"))
            .dt.total_seconds()
            .alias("query_time_wrt_clock_reference_time_s"),
        ]
    )

    # 7. Ephemeris validity mask
    merged = merged.with_columns(
        (
            (pl.col("query_time_isagpst") < pl.col("validity_end"))
            & (pl.col("query_time_isagpst") > pl.col("validity_start"))
        ).alias("ephemeris_valid")
    )

    return merged
