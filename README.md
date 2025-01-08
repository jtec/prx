# prx

prx reads a RINEX 3.05 observation file and outputs a CSV file of the following format


```
# {"approximate_receiver_ecef_position_m": [4627852.5264, 119640.514, 4372994.8358], "input_files": [{"name": "TLSE00FRA_R_20240011200_15M_01S_MO.rnx", "murmur3_hash": "bbc2db05fb5d0384ac26e9b9c0236f69"}, {"name": "BRDC00IGS_R_20240010000_01D_MN.rnx", "murmur3_hash": "f7df97043856eb0e57b8e1589c061043"}], "prx_git_commit_id": "ee48104e3d96098a7dc945f7af8ae8484fcd232a", "processing_start_time": "2024-10-02 17:21:08.7285603", "processing_time": "0 days 00:00:19.708024"}
time_of_reception_in_receiver_time,sat_clock_offset_m,sat_clock_drift_mps,sat_pos_x_m,sat_pos_y_m,sat_pos_z_m,sat_vel_x_mps,sat_vel_y_mps,sat_vel_z_mps,ephemeris_hash,frequency_slot,relativistic_clock_effect_m,sagnac_effect_m,tropo_delay_m,sat_code_bias_m,carrier_frequency_hz,iono_delay_m,sat_elevation_deg,sat_azimuth_deg,rnx_obs_identifier,C_obs_m,D_obs_hz,L_obs_cycles,LLI,S_obs_dBHz,constellation,prn
2024-01-01 12:00:00.000000,57153.224736,-0.018397,21838606.222023,36012172.441686,-1479022.231769,-3.184508,4.675864,-0.920381,2245131395643944762,1.000000,-0.668490,-39.902407,10.053189,0.149896,1561098000.000000,-15.044328,13.082414,115.281248,2I,40176280.391000,-14.730000,209208378.723000,,35.400000,C,05
2024-01-01 12:00:00.000000,57153.224736,-0.018397,21838606.222023,36012172.441686,-1479022.231769,-3.184508,4.675864,-0.920381,2245131395643944762,1.000000,-0.668490,-39.902407,10.053189,0.000000,1268520000.000000,-22.784446,13.082414,115.281248,6I,40176273.273000,-11.969000,169998932.830000,,36.900000,C,05
2024-01-01 12:00:00.000000,57153.224736,-0.018397,21838606.222023,36012172.441686,-1479022.231769,-3.184508,4.675864,-0.920381,2245131395643944762,1.000000,-0.668490,-39.902407,10.053189,-2.728111,1207140000.000000,-25.160417,13.082414,115.281248,7I,40176276.934000,-11.391000,161773185.396000,,38.400000,C,05
...
```

In addition to the observation values extracted from the RINEX file, the rows of the CSV contain the corresponding satellite position and speed at the time of transmission as well as relativistic effects, broadcast group delays, broadcast ionosphere delays etc.

## Running prx
From the `prx` repository root, run

```
poetry env use 3.12
poetry install
poetry shell
python src/prx/main.py --observation_file_path <path_to_rinex_file> 
```

You might have to add `<path to prx root>/src/prx` to your `PYTHONPATH` environment variable if you run 
into import errors.

## How we manage python version and dependencies
We use `poetry` to make sure every developer and user of prx runs the same python version
and the same set of dependencies such as `numpy` and `pandas`. Poetry reads `pyproject.toml`, 
resolves dependencies and writes the result to `poetry.lock`, which is the file used when running
`poetry install`.

To install poetry see https://python-poetry.org/docs/#installation

To create the virtual environment, run
`poetry install` in the `prx` repository root. This has to be run every time an update to the `pyproject.toml` and `poetry.lock`
files is done (for example, after a `git pull`)

### Jumping into the virtual environment
Run `poetry shell` to activate the virtual environment in a terminal.

### Using the poetry virtual environment in PyCharm
Run `poetry env info -p` to find the path of the virtual environment, which we'll call `<venv-path>`
Under `File` -> `Settings` -> `Project` -> `Python Interpreter` add `<venv-path>` as python interpreter.
If you run into problems, try `<venv-path>/bin/python` on Linux or `<venv-path>\Scripts\python.exe` on Windows.

### Add packages to poetry
Let's say you wrote some code that uses `import new_package`. To have `new package` added (what you otherwise
 would use `pip` or some other package manager for), run

`poetry add new_package`

Poetry will resolve dependencies - i.e. figure out which is the latest version of `new_package` that is compatible with 
our virtual environment - and add it to `pyproject.toml` and `poetry.lock`.

## Testing
After `poetry shell`

Run `pytest` in the `prx` repository root to run all tests.
Run `pytest -x` to stop after the first test that fails.
Run `pytest -k "my_test"` to run a specific test
Run `pytest ---durations=10` to run all tests and have pytest list the 10 longest running tests.

## Coding style
We use https://google.github.io/styleguide/pyguide.html as our python style guide.

## Acronyms
See the [Rinex 3.05 spec](https://files.igs.org/pub/data/format/rinex305.pdf), page 5, for a list of most acronyms used in the code. Those not covered by the RINEX spec are listed below.

| Acronym      | Long Form|
| ----------- | ----------- |
| PRX      | Preprocessed Rinex       |
