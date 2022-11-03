from jsonseq.encode import JSONSeqEncoder
import json
import os
from pathlib import Path

import example_observations


def generate_example_file(
        file_path=Path('__file__').resolve().parent.joinpath(Path("example.jsonl"))
) -> Path:
    header, observations = example_observations.generate()
    # write header and observations to json file
    indent = 2
    if file_path.exists():
        print(f"Removed existing JSONL example file: {file_path}")
        os.remove(file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("\u241E" + json.dumps(header, ensure_ascii=False, indent=indent) + "\n")
        for observation in observations:
            file.write("\u241E" + json.dumps(fields_to_arrays(observation), ensure_ascii=False, indent=indent) + "\n")
    print(f"Generated JSONL example file: {file_path}")
    return file_path


def fields_to_arrays(flat_obs):
    obs = flat_obs.copy()
    obs = replace_flat_by_array(obs, "satellite_position_m",
                                ["satellite_position_x_m",
                                 "satellite_position_y_m",
                                 "satellite_position_z_m"])
    obs = replace_flat_by_array(obs, "satellite_velocity_mps",
                                ["satellite_velocity_x_mps",
                                 "satellite_velocity_y_mps",
                                 "satellite_velocity_z_mps"])
    obs = replace_flat_by_array(obs, "approximate_antenna_position_m",
                                ["approximate_antenna_position_x_m",
                                 "approximate_antenna_position_y_m",
                                 "approximate_antenna_position_z_m"])

    return obs


def replace_flat_by_array(flat_dict, array_field, flat_fields):
    assert(len(flat_fields) > 0)
    dict_with_array = flat_dict.copy()
    dict_with_array[array_field] = []
    for flat_field in flat_fields:
        dict_with_array[array_field].append(flat_dict[flat_field])
        dict_with_array.pop(flat_field)
    return dict_with_array


if __name__ == "__main__":
    generate_example_file()
