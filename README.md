## KYOS Poetry/Python template

### Setup

1. In the terminal, run the project init script:
    ```bash
    python init MODEL_NAME

    # eg: python init coverage_report
    ```
2. In the terminal, update the poetry lockfile:
   ```bash
   poetry update
   ```
3. Update description and authors in `pyproject.toml`. For authors, provide both the name and the 
   KYOS email address. Example: `authors = ["John Doe <doe@kyos.com>"]`.
4. Update `README.md` with a proper model description. This includes:
   - replacing the top header ("KYOS Poetry/Python template") with the model name
   - providing the high-level purpose of the model
   - deleting the "Setup" section ("Development features" and "Deployment settings" 
     sections can be left unchanged)
5. Before any coding, merge all changes from steps 1-4 in a single, dedicated pull request.
6. In a later stage, once the coding is done, the following sections of the `README.md` file 
   should also be filled:
   - List of KYOS servers on which the script is running
   - Setting up the prototype template on the KYOS platform
   - Running the script locally

### List of KYOS servers on which the script is running

This section contains an **exhaustive list** of **all** KYOS servers on which the script is 
running. Please keep this section up to date, so that we know which servers to update in case of 
changes to the script.

List of servers:

- _Please provide the list of KYOS servers here._

### Setting up the prototype template on the KYOS platform

_Please explain how to set up the prototype template on the KYOS platform. This includes:_

- _all the input fields that need to be set up_
- _the content of the `modelsettings.json`, if not empty_

### Running the script locally

_If copy-pasting the content of a job folder is not sufficient to run the script locally, please 
explain how to do it. This includes:_

- _any extra input files (please indicate in which directory those files should be placed)_
- _any extra environment variable_

### Development features

[comment]: <> (- Linting provided by Flake8 with Flakehell)

- Dependency tracking using [Poetry](https://python-poetry.org/)
- Testing setup with [Pytest](https://github.com/pytest-dev/pytest)
- [Github Actions](https://github.com/features/actions) ready for CI/CD
- Docstring linting provided by [Darglint](https://github.com/terrencepreilly/darglint) using the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Formatting provided by [Black](https://github.com/psf/black) and [Isort](https://github.com/timothycrosley/isort)
- Checks dependencies for known security vulnerabilities with [Safety](https://github.com/pyupio/safety)
- All development tasks (lint, format, test, etc) wrapped up in a python CLI by [invoke](https://www.pyinvoke.org/)
- Automated dependency updates with [Dependabot](https://dependabot.com/)

This project is executing CI checks using **GitHub actions**.

It will run `pytest` and `black`.

#### Invoking development tasks

[Invoke](https://www.pyinvoke.org/) is used for development tasks.

Tasks are stored in `tasks.py` and are executable using the `inv[oke]` command line tool.

To see the list of tasks, you can type:

```bash
inv --list
```

For example, before your commits, you should run
```bash
inv format
```
to format your code.

### Deployment settings

Set `DEPLOY` to `1` if the model should be auto-deployed in the platform
> This is for standalone models and not for Prototypes

If `DEPLOY` is set to `0`, the model will be published as an artifact instead.

### Testing

In order to create the tests:
- Add your test case to a new folder './tests/fixture'
- Add the parameters 'testcase, name' in the @pytest.mark.parametrize decorator. This can be done with the following syntax `pytest.param(testcase, name")` testcase should match the numbers in the folders created above. For instance
- run `pytest --snapshot-update` in the terminal

To run test after making a change, run `pytest` in a terminal

To update expected outputs, run `pytest --snapshot-update` in the terminal
