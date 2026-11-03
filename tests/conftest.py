import sys
from pathlib import Path

import pytest
import torch

from tests._test_utils import _checkpoint, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: marks tests that require a CUDA-capable GPU")
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running integration or stability tests"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="GPU test requires CUDA, but CUDA is unavailable.")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def seed_everything():
    set_seed(42)


@pytest.fixture(scope="session")
def torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
