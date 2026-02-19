import os

import pytest
import yaml

import constants
from src.entry import main


def run_testcase(testcase: int, name: str, capsys):
    """
        This function  changes the working directory to the test
        file and runs the main function of the model directly into it.

    Args:
        testcase (int): number of the test case
        name (str): name of the test case
        capsys (fixture): allows access to outputs of tests

    Returns:
        main(): tuple of model results
    """
    test_path = os.path.join(constants.ROOT, 'tests', 'fixtures', f"test_{testcase}")
    os.chdir(test_path)

    with capsys.disabled():
        print(f"Running {constants.MODEL} test case #{testcase} {name}... [{test_path}]")

    return main()


def test_pass():
    assert True == True
