# prx

Making GNSS positioning a bit more accessible.

## How we manage python version and dependencies
We use `pipenv` to make sure every developer and user of prx runs the same python version
and the same set of dependencies such as `numpy` and `pandas`.
The environment parameters are defined by the `Pipfile` and `Pipfile.lock` files 
in the repository. 

To install pipenv, if necessary, run `pip install --user pipenv`

To create the virtual environment, run
`pipenv install` in the `prx` repository root. This has to be run every time an update to the `Pipfile` and `Pipfile.lock`
files is done (for example, after a `git fetch`)

### Jumping to the pipenv virtual environment
Run `pipenv shell` to activate the virtual environment in a terminal.

### Using the pipenv virtual environment in PyCharm
Run `pipenv --venv` to find the path of the virtual environment, which we'll call `<venv-path>`
Under `File` -> `Settings` -> `Project` -> `Python Interpreter` add `<venv-path>` as python interpreter.
If you run into problems, try `<venv-path>/bin/python`.

### Add packages to pipenv
Let's say you wrote some code that uses `import new_package`. To have `new package` added to the pipenv (what you otherwise
 would use `pip` or some other package manager for), run

`pipenv install new_package`

The package information will be added to the `Pipenv` and `Pipenv.lock` files. Simply commit 
those changes along with the new code that relies on the new package.

## Testing

Run `pytest` in the `prx` repository root to run all tests.
Run `pytest -x` to stop after the first test that fails.
Run `pytest ---durations=10` to run all tests and have pytest list the 10 longest running tests.

## Coding style
We use https://google.github.io/styleguide/pyguide.html as our python style guide.

## Documentation

We write on Confluence:
https://prxproject.atlassian.net/wiki/spaces/PD/pages/262145/What+does+prx+do

## Known bugs
### ModuleNotFoundError
In case you run in a `ModuleNotFoundError` during execution of `prx`, it is probably linked to an issue with the `PYTHONPATH` environment variable. One quick way to solve this is to create a file named `.env` in the `prx` root folder, with a single line:
> PYTHONPATH=src

When running `pipenv shell`, this will automatically load the `.env` file and add the `prx/src` folder to `PYTHONPATH`.

## Acronyms
See the [Rinex 3.05 spec](https://files.igs.org/pub/data/format/rinex305.pdf), page 5, for a list of most acronyms used in the code. Those not covered by the RINEX spec are listed below.

| Acronym      | Long Form|
| ----------- | ----------- |
| PRX      | Preprocessed Rinex       |
