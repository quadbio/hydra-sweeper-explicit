"""Shared fixtures for ExplicitSweeper tests."""

from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.hydra_sweeper_explicit import ExplicitSweeper


@pytest.fixture
def mock_config() -> DictConfig:
    """Minimal Hydra config with two combinations."""
    return OmegaConf.create(
        {
            "hydra": {
                "sweeper": {
                    "_target_": "hydra_sweeper_explicit.ExplicitSweeper",
                    "combinations": [
                        {"sampling": "independent"},
                        {"sampling": "ot", "sparsify.mass_threshold": 0.5},
                    ],
                }
            }
        }
    )


@pytest.fixture
def mock_config_with_seeds() -> DictConfig:
    """Hydra config with two combinations and seed expansion."""
    return OmegaConf.create(
        {
            "hydra": {
                "sweeper": {
                    "_target_": "hydra_sweeper_explicit.ExplicitSweeper",
                    "combinations": [
                        {"sampling": "independent"},
                        {"sampling": "ot"},
                    ],
                    "seeds": [42, 43],
                    "seed_key": "seed",
                }
            }
        }
    )


@pytest.fixture
def sweeper_with_mock_launcher() -> ExplicitSweeper:
    """Sweeper with a mocked default launcher that returns indexed results."""
    sweeper = ExplicitSweeper()
    sweeper.launcher = MagicMock()
    sweeper.launcher.launch.side_effect = lambda overrides, **kw: [f"result_{i}" for i in range(len(overrides))]
    return sweeper
