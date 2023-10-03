import os
import time
import random
from pathlib import Path

import csv_example


def test_csv_file_generated():
    random.seed(time.time())
    # Leverage the file name argument to generate a file with a unique name and delete it right afterwards:
    test_file_path = (
        Path("__file__")
        .resolve()
        .parent.joinpath(Path(f"tmp_test_file_{random.randint(int(0), int(1e12))}.csv"))
    )
    # Be extra careful and check that the file does not already exist before we try to generate it:
    assert not test_file_path.exists()
    csv_example.generate_example_file(test_file_path)
    assert test_file_path.exists()
    assert os.path.getsize(test_file_path) > 0
    os.remove(test_file_path)
