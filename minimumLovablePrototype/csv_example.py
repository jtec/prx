import csv
import os
from pathlib import Path

import example_observations


def generate_example_file(
        file_path=Path('__file__').resolve().parent.joinpath(Path("example.csv"))
) -> Path:
    header, observations = example_observations.generate()
    # write single epoch to csv file
    if file_path.exists():
        print(f"Removed existing CSV example file: {file_path}")
        os.remove(file_path)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(observations[0]))
        writer.writeheader()
        for observation in observations:
            writer.writerow(observation)
    print(f"Generated CSV example file: {file_path}")
    return file_path


if __name__ == "__main__":
    generate_example_file()