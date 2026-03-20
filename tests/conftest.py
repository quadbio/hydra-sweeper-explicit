"""Shared fixtures for ExplicitSweeper tests."""

from unittest.mock import MagicMock

import pytest

from hydra_plugins.hydra_sweeper_explicit import ExplicitSweeper


@pytest.fixture
def sweeper_with_mock_launcher() -> ExplicitSweeper:
    """Sweeper with a mocked default launcher that returns indexed results."""
    sweeper = ExplicitSweeper()
    sweeper.launcher = MagicMock()
    sweeper.launcher.launch.side_effect = lambda overrides, **kw: [f"result_{i}" for i in range(len(overrides))]
    return sweeper
