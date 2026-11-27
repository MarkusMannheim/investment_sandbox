import matplotlib
import pytest

from investment_sandbox.config import load_settings
from investment_sandbox.weather import load_weather

# Figures are drawn to files here, never to a window
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def weather():
    return load_weather()
