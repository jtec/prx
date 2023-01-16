import os
import time
import random
from pathlib import Path

import json_example


def test_json_file_generated():
    random.seed(time.time())
    # Leverage the file name argument to generate a file with a unique name and delete it right afterwards:
    test_file_path = Path('__file__').resolve().parent.joinpath(
        Path(f"tmp_test_file_{random.randint(int(0), int(1e12))}.jsonl"))
    # Be extra careful and check that the file does not already exist before we try to generate it:
    assert not test_file_path.exists()
    json_example.generate_example_file(test_file_path)
    assert test_file_path.exists()
    assert os.path.getsize(test_file_path) > 0
    print(f"Removing test file {test_file_path}")
    os.remove(test_file_path)


def test_replace_flat_by_array():
    flat_dict = {"position_x_m": 1, "position_y_m": 2, "position_z_m": 3}
    packed = json_example.replace_flat_by_array(
        flat_dict, "position_m",
        ["position_x_m", "position_y_m", "position_z_m"])
    expected = {"position_m": [1, 2, 3]}
    assert (packed == expected)
