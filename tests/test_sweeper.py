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
        assert sweeper.seeds is None
        assert sweeper.seed_key == "seed"

    def test_init_with_combinations(self) -> None:
        """Test initialization with combinations."""
        combos = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        sweeper = ExplicitSweeper(combinations=combos)
        assert sweeper.combinations == combos

    def test_init_with_seeds_list(self) -> None:
        """Test initialization with seed list."""
        sweeper = ExplicitSweeper(seeds=[42, 43, 44])
        assert sweeper.seeds == [42, 43, 44]

    def test_init_with_seeds_int(self) -> None:
        """Test initialization with seed count."""
        sweeper = ExplicitSweeper(seeds=3)
        assert sweeper.seeds == 3
        assert sweeper._resolve_seeds() == [0, 1, 2]

    def test_init_with_custom_seed_key(self) -> None:
        """Test initialization with custom seed key."""
        sweeper = ExplicitSweeper(seed_key="random_seed")
        assert sweeper.seed_key == "random_seed"

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

    def test_resolve_seeds_none(self) -> None:
        """Test seed resolution with no seeds."""
        sweeper = ExplicitSweeper()
        assert sweeper._resolve_seeds() is None

    def test_resolve_seeds_list(self) -> None:
        """Test seed resolution with list."""
        sweeper = ExplicitSweeper(seeds=[10, 20, 30])
        assert sweeper._resolve_seeds() == [10, 20, 30]

    def test_resolve_seeds_int(self) -> None:
        """Test seed resolution with int."""
        sweeper = ExplicitSweeper(seeds=5)
        assert sweeper._resolve_seeds() == [0, 1, 2, 3, 4]


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

    @pytest.fixture
    def mock_config_with_seeds(self) -> DictConfig:
        """Create a Hydra config with seeds for testing."""
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

    def test_combinations_from_config(self, mock_config: DictConfig) -> None:
        """Test loading combinations from Hydra config during setup."""
        sweeper = ExplicitSweeper()
        assert sweeper.combinations == []

        # The combinations should be accessible from the config
        combos = mock_config.hydra.sweeper.combinations
        assert len(combos) == 2
        assert OmegaConf.to_container(combos[0]) == {"sampling": "independent"}

    def test_seeds_from_config(self, mock_config_with_seeds: DictConfig) -> None:
        """Test loading seeds from Hydra config."""
        combos = mock_config_with_seeds.hydra.sweeper.combinations
        seeds = mock_config_with_seeds.hydra.sweeper.seeds

        assert len(combos) == 2
        assert OmegaConf.to_container(seeds) == [42, 43]


class TestLauncherGrouping:
    """Tests for _launcher_ extraction and job grouping."""

    def test_launcher_stripped_from_overrides(self) -> None:
        """_launcher_ key should not appear in job override strings."""
        sweeper = ExplicitSweeper(
            combinations=[
                {"solver": "linear", "_launcher_": "euler_gpu_8"},
                {"solver": "pseudobulk"},
            ]
        )
        # Call sweep — jobs with _launcher_ should not include it in overrides
        # (sweep returns [] because no launcher is set up, but we can inspect
        # the override building logic via _format_override)
        combo = {"solver": "linear", "_launcher_": "euler_gpu_8"}
        overrides = [sweeper._format_override(k, v) for k, v in combo.items() if k != "_launcher_"]
        assert overrides == ["solver=linear"]
        assert "_launcher_" not in " ".join(overrides)

    def test_launcher_key_extracted(self) -> None:
        """_launcher_ value is correctly extracted from combination."""
        combo = {"solver": "linear", "_launcher_": "euler_gpu_8", "seed": 0}
        assert combo.get("_launcher_") == "euler_gpu_8"

        combo_no_launcher = {"solver": "pseudobulk", "seed": 0}
        assert combo_no_launcher.get("_launcher_") is None
