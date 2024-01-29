# prx

Making GNSS positioning a bit more accessible.

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

## Documentation

We write on Confluence:
https://prxproject.atlassian.net/wiki/spaces/PD/pages/262145/What+does+prx+do

## Acronyms
See the [Rinex 3.05 spec](https://files.igs.org/pub/data/format/rinex305.pdf), page 5, for a list of most acronyms used in the code. Those not covered by the RINEX spec are listed below.

| Acronym      | Long Form|
| ----------- | ----------- |
| PRX      | Preprocessed Rinex       |
