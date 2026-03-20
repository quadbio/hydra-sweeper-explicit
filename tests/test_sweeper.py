"""Tests for ExplicitSweeper."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hydra_plugins.hydra_sweeper_explicit import ExplicitSweeper


class TestExplicitSweeper:
    """Unit tests for individual methods."""

    def test_init_defaults(self) -> None:
        sweeper = ExplicitSweeper()
        assert sweeper.combinations == []
        assert sweeper.seeds is None
        assert sweeper.seed_key == "seed"
        assert sweeper.launcher_config_group == "hydra/launcher"

    def test_init_with_combinations(self) -> None:
        combos = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        sweeper = ExplicitSweeper(combinations=combos)
        assert sweeper.combinations == combos

    def test_init_with_seeds_list(self) -> None:
        sweeper = ExplicitSweeper(seeds=[42, 43, 44])
        assert sweeper.seeds == [42, 43, 44]

    def test_init_with_seeds_int(self) -> None:
        sweeper = ExplicitSweeper(seeds=3)
        assert sweeper.seeds == 3

    def test_init_with_custom_seed_key(self) -> None:
        sweeper = ExplicitSweeper(seed_key="random_seed")
        assert sweeper.seed_key == "random_seed"

    def test_init_with_custom_launcher_config_group(self) -> None:
        sweeper = ExplicitSweeper(launcher_config_group="launcher")
        assert sweeper.launcher_config_group == "launcher"

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
        assert ExplicitSweeper()._format_override(key, value) == expected

    @pytest.mark.parametrize(
        ("seeds", "expected"),
        [
            (None, None),
            (5, [0, 1, 2, 3, 4]),
            ([10, 20, 30], [10, 20, 30]),
        ],
    )
    def test_resolve_seeds(self, seeds: list[int] | int | None, expected: list[int] | None) -> None:
        assert ExplicitSweeper(seeds=seeds)._resolve_seeds() == expected


class TestSweep:
    """Tests for sweep() — dispatch, seeds, CLI arguments, launcher grouping."""

    def test_no_combinations(self) -> None:
        assert ExplicitSweeper().sweep([]) == []

    def test_basic_dispatch(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Combinations are forwarded as overrides to the launcher."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

        sweeper.sweep([])

        sweeper.launcher.launch.assert_called_once()
        overrides = sweeper.launcher.launch.call_args[0][0]
        assert overrides == [["a=1", "b=x"], ["a=2", "b=y"]]

    def test_seed_expansion(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Each combination is expanded across all seeds."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [{"model": "a"}, {"model": "b"}]
        sweeper.seeds = [42, 43]

        sweeper.sweep([])

        overrides = sweeper.launcher.launch.call_args[0][0]
        assert len(overrides) == 4
        assert overrides[0] == ["model=a", "seed=42"]
        assert overrides[1] == ["model=a", "seed=43"]
        assert overrides[2] == ["model=b", "seed=42"]
        assert overrides[3] == ["model=b", "seed=43"]

    def test_cli_arguments_appended(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Extra CLI arguments are appended to every job's overrides."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [{"a": 1}]

        sweeper.sweep(["extra=42"])

        overrides = sweeper.launcher.launch.call_args[0][0]
        assert overrides == [["a=1", "extra=42"]]

    def test_launcher_stripped_from_overrides(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """_launcher_ key must not appear in overrides passed to launch()."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [
            {"model": "small", "_launcher_": "slurm_cpu"},
            {"model": "large"},
        ]

        with patch.object(sweeper, "_make_launcher", return_value=sweeper.launcher):
            sweeper.sweep([])

        all_overrides = []
        for call in sweeper.launcher.launch.call_args_list:
            all_overrides.extend(call[0][0])

        flat = " ".join(str(o) for o in all_overrides)
        assert "_launcher_" not in flat

    def test_jobs_grouped_by_launcher(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Different _launcher_ values dispatch to separate launcher instances."""
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

        # Default launcher gets only the untagged job
        assert sweeper.launcher.launch.call_count == 1
        assert sweeper.launcher.launch.call_args[0][0] == [["model=medium"]]

        # Each custom launcher gets its own group
        cpu_launcher.launch.assert_called_once()
        assert cpu_launcher.launch.call_args[0][0] == [["model=small"]]
        gpu_launcher.launch.assert_called_once()
        assert gpu_launcher.launch.call_args[0][0] == [["model=large"]]

    def test_results_reassembled_in_original_order(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Results are returned in combination order, not grouped order."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [
            {"model": "a", "_launcher_": "custom"},
            {"model": "b"},
            {"model": "c", "_launcher_": "custom"},
        ]
        sweeper.launcher.launch.side_effect = lambda overrides, **kw: ["result_b"]
        custom_launcher = MagicMock()
        custom_launcher.launch.side_effect = lambda overrides, **kw: ["result_a", "result_c"]

        with patch.object(sweeper, "_make_launcher", return_value=custom_launcher):
            results = sweeper.sweep([])

        assert results == ["result_a", "result_b", "result_c"]

    def test_all_default_launcher(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """Without _launcher_, all jobs go to the default launcher in one call."""
        sweeper = sweeper_with_mock_launcher
        sweeper.combinations = [{"a": 1}, {"a": 2}, {"a": 3}]

        sweeper.sweep([])

        sweeper.launcher.launch.assert_called_once()
        assert len(sweeper.launcher.launch.call_args[0][0]) == 3

    def test_launcher_with_seeds(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """_launcher_ grouping works with seed expansion."""
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

    def test_app_level_launcher_config_group(self, sweeper_with_mock_launcher: ExplicitSweeper) -> None:
        """launcher_config_group='launcher' puts override in task overrides, not hydra."""
        sweeper = sweeper_with_mock_launcher
        sweeper.launcher_config_group = "launcher"
        sweeper.combinations = [
            {"model": "large", "_launcher_": "euler_gpu_8"},
            {"model": "small"},
        ]

        # Mock _make_launcher to capture how it's called
        gpu_launcher = MagicMock()
        gpu_launcher.launch.side_effect = lambda overrides, **kw: [f"r_{i}" for i in range(len(overrides))]

        with patch.object(sweeper, "_make_launcher", side_effect=lambda name: gpu_launcher) as mock_make:
            sweeper.sweep([])

        mock_make.assert_called_once_with("euler_gpu_8")
        # Verify both launchers were called
        gpu_launcher.launch.assert_called_once()
        sweeper.launcher.launch.assert_called_once()
