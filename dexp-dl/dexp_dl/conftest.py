from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest

from dexp_dl.utils.testing.data import nuclei_dexp, nuclei_tiles


@pytest.fixture
def nuclei_tiles_directory(request: SubRequest, tmp_path: Path) -> Path:
    nuclei_tiles(tmp_path, **getattr(request, "param", {}))
    return tmp_path


@pytest.fixture(scope="session")
def nuclei_dexp_directory(request: SubRequest, tmp_path_factory) -> Path:
    ds_path = tmp_path_factory.mktemp("data") / "input.zarr"
    nuclei_dexp(ds_path, **getattr(request, "param", {}))
    return ds_path
