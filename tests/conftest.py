import pytest, torch

def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)
