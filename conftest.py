import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def repo_root(pytestconfig):
    # The rootdir is automatically determined by pytest (or set in pytest.ini)
    return Path(pytestconfig.rootdir)
