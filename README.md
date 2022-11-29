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

## Coding style
We use https://google.github.io/styleguide/pyguide.html as our python style guide.

## Documentation

We write on Confluence:
https://prxproject.atlassian.net/wiki/spaces/PD/pages/262145/What+does+prx+do
