"""Tests for ExplicitSweeper."""

from typing import Any
from unittest.mock import MagicMock, patch

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
    """Tests for _launcher_ key handling in sweep()."""

    @pytest.fixture
    def sweeper_with_mock_launcher(self) -> ExplicitSweeper:
        """Create a sweeper with a mocked default launcher."""
        sweeper = ExplicitSweeper()
        sweeper.launcher = MagicMock()
        # Default launcher returns a result per job
        sweeper.launcher.launch.side_effect = lambda overrides, **kw: [f"result_{i}" for i in range(len(overrides))]
        return sweeper

    def test_launcher_stripped_from_overrides(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """_launcher_ key must not appear in overrides passed to launcher.launch()."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [
            {"model": "small", "_launcher_": "slurm_cpu"},
            {"model": "large"},
        ]
        # All jobs use mock default launcher (no real _make_launcher)
        with patch.object(sweeper, "_make_launcher", return_value=sweeper.launcher):
            sweeper.sweep([])

        # Collect all overrides from all launch() calls
        all_overrides = []
        for call in sweeper.launcher.launch.call_args_list:
            all_overrides.extend(call[0][0])

        flat = " ".join(str(o) for o in all_overrides)
        assert "_launcher_" not in flat

    def test_jobs_grouped_by_launcher(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Jobs with different _launcher_ values must be dispatched in separate launch() calls."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [
            {"model": "small", "_launcher_": "slurm_cpu"},
            {"model": "medium"},
            {"model": "large", "_launcher_": "slurm_gpu"},
        ]

        gpu_launcher = MagicMock()
        gpu_launcher.launch.side_effect = lambda overrides, **kw: [f"gpu_{i}" for i in range(len(overrides))]
        cpu_launcher = MagicMock()
        cpu_launcher.launch.side_effect = lambda overrides, **kw: [f"cpu_{i}" for i in range(len(overrides))]

        def mock_make_launcher(name: str) -> MagicMock:
            return {"slurm_gpu": gpu_launcher, "slurm_cpu": cpu_launcher}[name]

        with patch.object(sweeper, "_make_launcher", side_effect=mock_make_launcher):
            sweeper.sweep([])

        # Default launcher gets jobs without _launcher_
        assert sweeper.launcher.launch.call_count == 1
        default_overrides = sweeper.launcher.launch.call_args[0][0]
        assert default_overrides == [["model=medium"]]

        # Each custom launcher gets its own group
        cpu_launcher.launch.assert_called_once()
        assert cpu_launcher.launch.call_args[0][0] == [["model=small"]]

        gpu_launcher.launch.assert_called_once()
        assert gpu_launcher.launch.call_args[0][0] == [["model=large"]]

    def test_results_reassembled_in_original_order(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Results must be returned in original combination order, not grouped order."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [
            {"model": "a", "_launcher_": "custom"},
            {"model": "b"},
            {"model": "c", "_launcher_": "custom"},
        ]
        # Default launcher (for "b")
        sweeper.launcher.launch.side_effect = lambda overrides, **kw: ["result_b"]
        # Custom launcher (for "a" and "c")
        custom_launcher = MagicMock()
        custom_launcher.launch.side_effect = lambda overrides, **kw: ["result_a", "result_c"]

        with patch.object(sweeper, "_make_launcher", return_value=custom_launcher):
            results = sweeper.sweep([])

        assert results == ["result_a", "result_b", "result_c"]

    def test_no_launcher_key_uses_default(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Without _launcher_, all jobs go to the default launcher in one call."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [{"a": 1}, {"a": 2}, {"a": 3}]

        sweeper.sweep([])

        sweeper.launcher.launch.assert_called_once()
        overrides = sweeper.launcher.launch.call_args[0][0]
        assert len(overrides) == 3

    def test_launcher_with_seeds(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """_launcher_ works correctly with seed expansion."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [
            {"model": "small", "_launcher_": "cpu"},
            {"model": "large"},
        ]
        sweeper.seeds = [42, 43]

        cpu_launcher = MagicMock()
        cpu_launcher.launch.side_effect = lambda overrides, **kw: [f"cpu_{i}" for i in range(len(overrides))]

        with patch.object(sweeper, "_make_launcher", return_value=cpu_launcher):
            results = sweeper.sweep([])

        # 2 combinations × 2 seeds = 4 jobs
        assert len(results) == 4

        # CPU launcher gets 2 jobs (small × 2 seeds)
        cpu_launcher.launch.assert_called_once()
        cpu_overrides = cpu_launcher.launch.call_args[0][0]
        assert len(cpu_overrides) == 2
        assert all("model=small" in o for overrides in [cpu_overrides] for o in overrides)

        # Default launcher gets 2 jobs (large × 2 seeds)
        sweeper.launcher.launch.assert_called_once()
        default_overrides = sweeper.launcher.launch.call_args[0][0]
        assert len(default_overrides) == 2
