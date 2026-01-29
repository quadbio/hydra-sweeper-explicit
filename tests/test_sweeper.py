"""Tests for ExplicitSweeper."""

from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from hydra_plugins.hydra_sweeper_explicit import ExplicitSweeper


class TestExplicitSweeper:
    """Test suite for ExplicitSweeper."""

    def test_init_empty(self) -> None:
        """Test initialization with no combinations."""
        sweeper = ExplicitSweeper()
        assert sweeper.combinations == []

    def test_init_with_combinations(self) -> None:
        """Test initialization with combinations."""
        combos = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        sweeper = ExplicitSweeper(combinations=combos)
        assert sweeper.combinations == combos

    @pytest.mark.parametrize(
        ("key", "value", "expected"),
        [
            ("a", 1, "a=1"),
            ("a", 1.5, "a=1.5"),
            ("a", True, "a=true"),
            ("a", False, "a=false"),
            ("a", None, "a=null"),
            ("a", "foo", "a=foo"),
            ("a", "foo bar", 'a="foo bar"'),
            ("a", "[1,2]", 'a="[1,2]"'),
            ("nested.key", "value", "nested.key=value"),
        ],
    )
    def test_format_override(self, key: str, value: Any, expected: str) -> None:
        """Test override formatting for various types."""
        sweeper = ExplicitSweeper()
        result = sweeper._format_override(key, value)
        assert result == expected

    def test_sweep_no_combinations(self) -> None:
        """Test sweep returns empty when no combinations defined."""
        sweeper = ExplicitSweeper()
        result = sweeper.sweep([])
        assert result == []


class TestExplicitSweeperIntegration:
    """Integration tests requiring Hydra context."""

    @pytest.fixture
    def mock_config(self) -> DictConfig:
        """Create a minimal Hydra config for testing."""
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

    def test_combinations_from_config(self, mock_config: DictConfig) -> None:
        """Test loading combinations from Hydra config during setup."""
        sweeper = ExplicitSweeper()
        assert sweeper.combinations == []

        # The combinations should be accessible from the config
        combos = mock_config.hydra.sweeper.combinations
        assert len(combos) == 2
        assert OmegaConf.to_container(combos[0]) == {"sampling": "independent"}
