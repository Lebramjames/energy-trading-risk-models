"""
Tasks for maintaining the project.

Execute 'inv[oke] --list' for guidance on using Invoke
"""

import os.path
import pathlib
import platform
import site
import webbrowser

import toml
from invoke import call, task
from invoke.context import Context
from invoke.runners import Result

import constants

# Packages to be included as custom PyInstaller hooks.
HOOK_PACKAGES = ["plotly"]

# Project related paths.
ROOT_DIR = pathlib.Path(__file__).parent
SOURCE_DIR = ROOT_DIR.joinpath("src")
TEST_DIR = ROOT_DIR.joinpath("tests")
MODEL_ENTRY_FILE = ROOT_DIR.joinpath("run.py")

PYTHON_TARGETS = [
    SOURCE_DIR,
    TEST_DIR,
    MODEL_ENTRY_FILE,
    pathlib.Path(__file__),
]
PYTHON_TARGETS_STR = " ".join([str(p) for p in PYTHON_TARGETS])

# Coverage related paths.
COVERAGE_FILE = ROOT_DIR.joinpath(".coverage")
COVERAGE_XML = ROOT_DIR.joinpath("coverage.xml")
COVERAGE_DIR = ROOT_DIR.joinpath("htmlcov")
COVERAGE_REPORT = COVERAGE_DIR.joinpath("index.html")


def _run(c: Context, command: str) -> Result:
    return c.run(command, pty=platform.system() != "Windows")


@task()
def clean_tests(c):
    # type: (Context) -> None
    """Clean up files from testing."""
    _run(c, f"rm -f {COVERAGE_FILE}")
    _run(c, f"rm -f {COVERAGE_XML}")
    _run(c, f"rm -fr {COVERAGE_DIR}")
    _run(c, "rm -fr .pytest_cache")


@task()
def doclint(c):
    # type: (Context) -> None
    """Lint docstrings' descriptions."""
    _run(c, f"poetry run darglint -v 2 {SOURCE_DIR}")


@task()
def safety(c):
    # type: (Context) -> None
    """Checks dependencies for known vulnerabilities."""
    _run(
        c,
        "poetry export --dev --format=requirements.txt --without-hashes | "
        "poetry run safety check --stdin --full-report",
    )


@task(
    name="format",
    help={"check": "Checks if source is formatted without applying changes"},
)
def format_(c, check=False):
    # type: (Context, bool) -> None
    """Format code."""
    isort_options = ["--check-only", "--diff"] if check else []
    _run(c, f"poetry run isort {' '.join(isort_options)} {PYTHON_TARGETS_STR}")

    black_options = ["--diff", "--check"] if check else ["--quiet"]
    _run(c, f"poetry run black {' '.join(black_options)} {PYTHON_TARGETS_STR}")


@task(pre=[doclint, call(format_, check=True)])
def lint(c):
    # type: (Context) -> None
    """Run all linters."""


@task()
def test(c):
    # type: (Context) -> None
    """Run unit tests."""
    _run(c, "poetry run pytest")


@task(help={"html": "Open the coverage report in the web browser"})
def cov(c, html=False):
    # type: (Context, bool) -> None
    """Run unit tests with code coverage."""
    report = "html" if html else "xml"
    _run(c, f"poetry run pytest --cov-report {report} --cov-report term-missing --cov=src tests")
    if html:
        webbrowser.open(COVERAGE_REPORT.as_uri())


@task(
    name="constant",
    help={
        "name": "Name of the constant to retrieve value for.",
        "default": "Default value in case constant is not found (Optional, default: None)",
    },
)
def constant_(c, name, default=None):
    # type: (Context, str, str or None) -> str
    """Get a constant from constants.py file."""
    val = getattr(constants, name, default)

    if val is not None:
        print(val)
        return val
    else:
        print("Constant not found")
        exit(1)


@task()
def pyhook(c):
    # type: (Context) -> str
    """Custom hooks for PyInstaller."""
    absolute_package_path = site.getsitepackages()[0]
    custom_py_imports = " "

    py_project = toml.load("pyproject.toml")
    dependencies = list(py_project["tool"]["poetry"]["dependencies"].keys()) + list(
        py_project["tool"]["poetry"]["group"]["dev"]["dependencies"].keys()
    )

    for i in HOOK_PACKAGES:
        if i in dependencies:
            path = f"{absolute_package_path}/{i}"
            if os.path.exists(path):
                custom_py_imports = f'{custom_py_imports}--add-data="{path}:{i}" '

    print(custom_py_imports)
    return custom_py_imports


@task(
    name="compile",
    help={
        "target_name": "Override the compiled executable filename. (Defaults to MODEL from constants.py)"
    },
)
def compile_(c, target_name=None):
    # type: (Context, str or None) -> None
    """Compile using PyInstaller."""
    hooks = pyhook(c)
    model_name = target_name if target_name is not None else constant_(c, "MODEL")

    compilation_command = f"pyinstaller {hooks} --name {model_name} --onefile run.py"
    _run(c, f"poetry add pyinstaller && {compilation_command}")
