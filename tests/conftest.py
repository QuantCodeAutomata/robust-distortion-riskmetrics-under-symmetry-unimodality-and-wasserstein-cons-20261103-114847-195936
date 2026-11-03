import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "market_data: requires live API credentials")


@pytest.fixture(scope="session")
def tiny_normal_sample():
    rng = np.random.default_rng(42)
    return rng.standard_normal(50)


@pytest.fixture(scope="session")
def uniform_quantile_fn():
    import numpy as np
    _SQRT3 = np.sqrt(3.0)
    def _q(u):
        return (1.0 - _SQRT3) + 2.0 * _SQRT3 * np.asarray(u)
    return _q
