from pathlib import Path

from main import process

if __name__ == "__main__":
    process(
        Path(
            "/Users/janbolting/repositories/prx/src/prx/test/datasets/TLSE00FRA_R_20230010100_10S_01S_MO.rnx"
        ),
        output_format="feather",
    )
