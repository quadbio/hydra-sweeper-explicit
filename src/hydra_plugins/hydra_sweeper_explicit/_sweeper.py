"""Explicit Sweeper - runs exact parameter combinations without Cartesian product.

Hydra's BasicSweeper always computes the Cartesian product of all sweep parameters,
which leads to invalid combinations when parameters are dependent (e.g., certain
settings only make sense with a specific model or backend).

This sweeper lets you define explicit combinations:

    hydra:
      sweeper:
        _target_: hydra_sweeper_explicit.ExplicitSweeper
        combinations:
          - {model: small}
          - {model: large, optimizer.lr: 0.001}
          - {model: large, optimizer.lr: 0.01}

Each dict in ``combinations`` becomes one job with exactly those overrides.

Optionally, expand each combination across multiple seeds:

    hydra:
      sweeper:
        seeds: [42, 43, 44]  # or seeds: 5 for seeds 0-4
        seed_key: seed       # parameter name (default: "seed")
        combinations:
          - {model: small}
          - {model: large}

This creates 6 jobs: 2 combinations × 3 seeds.

Per-combination launcher overrides (for heterogeneous compute resources):

    hydra:
      sweeper:
        combinations:
          - {model: large, _launcher_: slurm_gpu}
          - {model: small, _launcher_: slurm_cpu}

Jobs are grouped by ``_launcher_`` and each group is submitted with its own
launcher instance. Combinations without ``_launcher_`` use the default launcher.

Usage:
    python my_app.py --multirun hydra/sweeper=explicit
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, RunMode, TaskFunction
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class ExplicitSweeper(Sweeper):
    """Sweeper that runs explicit parameter combinations (no Cartesian product).

    Unlike BasicSweeper which computes cartesian product of all sweep parameters,
    this sweeper runs exactly the combinations you specify, optionally expanded
    across multiple seeds.

    Parameters
    ----------
    combinations
        List of dicts, where each dict maps parameter paths to values.
        Each dict becomes one job (or N jobs if seeds is specified).
    seeds
        Seeds to run each combination with. Can be:
        - A list of integers: [42, 43, 44]
        - An integer N: runs seeds 0 to N-1
        - None: no seed expansion (default)
    seed_key
        The config key to use for the seed parameter. Default: "seed"
    launcher_config_group
        Hydra config group path for launcher configs used in ``_launcher_``
        overrides. Default: ``"hydra/launcher"`` (standard Hydra pattern).
        Set to ``"launcher"`` if your launcher configs live at
        ``configs/launcher/`` with ``@package _global_``.
    max_batch_size
        Ignored, for compatibility with BasicSweeper config.
    params
        Ignored, for compatibility with BasicSweeper config.

    Example
    -------
    combinations:
      - {model: small}
      - {model: large, optimizer.lr: 0.001}
      - {model: large, optimizer.lr: 0.01}

    This runs exactly 3 jobs, not a Cartesian product.

    With seeds:
      seeds: [42, 43]
      combinations:
        - {model: small}
        - {model: large}

    This runs 4 jobs: 2 combinations × 2 seeds.
    """

    def __init__(
        self,
        combinations: list[dict[str, Any]] | None = None,
        seeds: list[int] | int | None = None,
        seed_key: str = "seed",
        launcher_config_group: str = "hydra/launcher",
        max_batch_size: int | None = None,  # Ignored, for compatibility
        params: dict[str, str] | None = None,  # Ignored, for compatibility
    ) -> None:
        self.combinations = list(combinations) if combinations else []
        self.seeds = seeds
        self.seed_key = seed_key
        self.launcher_config_group = launcher_config_group
        self.config: DictConfig | None = None
        self.launcher: Any = None
        self.hydra_context: HydraContext | None = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        """Set up the sweeper with Hydra context."""
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context,
            task_function=task_function,
            config=config,
        )

        sweeper_cfg = config.hydra.sweeper

        # If combinations weren't passed to constructor, try to read from config
        if not self.combinations:
            if hasattr(sweeper_cfg, "combinations") and sweeper_cfg.combinations:
                # Convert OmegaConf list to regular Python list of dicts
                self.combinations = [OmegaConf.to_container(c, resolve=True) for c in sweeper_cfg.combinations]
                log.info("Loaded %d combinations from config", len(self.combinations))

        # Load seeds from config if not passed to constructor
        if self.seeds is None and hasattr(sweeper_cfg, "seeds") and sweeper_cfg.seeds is not None:
            self.seeds = OmegaConf.to_container(sweeper_cfg.seeds, resolve=True)
            log.info("Loaded seeds from config: %s", self.seeds)

        # Load seed_key from config if available
        if hasattr(sweeper_cfg, "seed_key") and sweeper_cfg.seed_key:
            self.seed_key = sweeper_cfg.seed_key

        # Load launcher_config_group from config if available
        if hasattr(sweeper_cfg, "launcher_config_group") and sweeper_cfg.launcher_config_group:
            self.launcher_config_group = sweeper_cfg.launcher_config_group

    def _resolve_seeds(self) -> list[int] | None:
        """Resolve seeds specification to a list of integers.

        Returns
        -------
        List of seed integers, or None if no seeds specified.
        """
        if self.seeds is None:
            return None
        if isinstance(self.seeds, int):
            return list(range(self.seeds))
        return list(self.seeds)

    def _format_override(self, key: str, value: Any) -> str:
        """Format a single key=value override string."""
        if isinstance(value, bool):
            return f"{key}={str(value).lower()}"
        elif isinstance(value, str):
            # Quote strings that might contain special chars
            if any(c in value for c in " ,[]{}"):
                return f'{key}="{value}"'
            return f"{key}={value}"
        elif value is None:
            return f"{key}=null"
        else:
            return f"{key}={value}"

    def _make_launcher(self, launcher_name: str) -> Any:
        """Instantiate a launcher for a different launcher config group.

        Re-composes the full Hydra config with the launcher override and creates
        a new launcher instance from it. The config group path is controlled by
        ``launcher_config_group`` (default ``"hydra/launcher"``).
        """
        group = self.launcher_config_group
        is_hydra_group = group.startswith("hydra/")

        # Reconstruct original overrides, adding the launcher to the right list
        task_overrides = list(OmegaConf.to_container(self.config.hydra.overrides.task, resolve=False))
        hydra_overrides = list(OmegaConf.to_container(self.config.hydra.overrides.hydra, resolve=False))

        override = f"{group}={launcher_name}"
        if is_hydra_group:
            hydra_overrides = [o for o in hydra_overrides if not o.startswith(f"{group}=")]
            hydra_overrides.append(override)
        else:
            task_overrides = [o for o in task_overrides if not o.startswith(f"{group}=")]
            task_overrides.append(override)

        new_config = self.hydra_context.config_loader.load_configuration(
            config_name=self.config.hydra.job.config_name,
            overrides=task_overrides + hydra_overrides,
            run_mode=RunMode.MULTIRUN,
            from_shell=False,
        )

        return Plugins.instance().instantiate_launcher(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=new_config,
        )

    def sweep(self, arguments: list[str]) -> Any:
        """Execute the sweep with explicit combinations.

        Parameters
        ----------
        arguments
            Additional CLI arguments passed after --multirun

        Returns
        -------
        List of job returns from the launcher
        """
        if not self.combinations:
            log.warning("ExplicitSweeper: No combinations defined, nothing to run")
            return []

        # Resolve seeds
        seeds = self._resolve_seeds()

        # Build job overrides for each combination (optionally × seeds),
        # tracking which launcher each job belongs to.
        # Each entry is (launcher_name | None, [override_strings]).
        tagged_jobs: list[tuple[str | None, list[str]]] = []

        for combo in self.combinations:
            launcher_name = combo.get("_launcher_")
            overrides_base = [self._format_override(k, v) for k, v in combo.items() if k != "_launcher_"]

            if seeds is not None:
                for seed in seeds:
                    overrides = [*overrides_base, self._format_override(self.seed_key, seed), *arguments]
                    tagged_jobs.append((launcher_name, overrides))
            else:
                tagged_jobs.append((launcher_name, [*overrides_base, *arguments]))

        # Log job summary
        if seeds is not None:
            log.info(
                "ExplicitSweeper: %d combinations × %d seeds = %d jobs",
                len(self.combinations),
                len(seeds),
                len(tagged_jobs),
            )
        else:
            log.info("ExplicitSweeper: Launching %d jobs", len(tagged_jobs))

        for i, (_launcher, overrides) in enumerate(tagged_jobs):
            suffix = f" [launcher={_launcher}]" if _launcher else ""
            log.info("Job %d: %s%s", i, " ".join(overrides), suffix)

        # Group jobs by launcher, preserving order of first appearance
        groups: defaultdict[str | None, list[tuple[int, list[str]]]] = defaultdict(list)
        for i, (launcher_name, overrides) in enumerate(tagged_jobs):
            groups[launcher_name].append((i, overrides))

        # Launch each group with its own launcher instance
        launchers: dict[str | None, Any] = {None: self.launcher}
        all_returns: list[Any] = [None] * len(tagged_jobs)
        job_idx = 0

        # Pre-build (launcher, overrides, indices, job_idx) for each group
        launch_tasks: list[tuple[Any, list[list[str]], list[int], int]] = []
        for launcher_name, jobs in groups.items():
            if launcher_name is not None and launcher_name not in launchers:
                log.info("Instantiating launcher: %s", launcher_name)
                launchers[launcher_name] = self._make_launcher(launcher_name)

            launcher = launchers[launcher_name]
            group_overrides = [overrides for _, overrides in jobs]
            group_indices = [idx for idx, _ in jobs]
            launch_tasks.append((launcher, group_overrides, group_indices, job_idx))
            job_idx += len(group_overrides)

        # Submit all groups concurrently so SLURM jobs hit the queue together,
        # then wait for all results in parallel.
        def _launch(task: tuple[Any, list[list[str]], list[int], int]) -> tuple[list[int], list[Any]]:
            launcher, overrides, indices, idx = task
            returns = launcher.launch(overrides, initial_job_idx=idx)
            return indices, list(returns)

        if len(launch_tasks) == 1:
            indices, returns = _launch(launch_tasks[0])
            for idx, ret in zip(indices, returns, strict=True):
                all_returns[idx] = ret
        else:
            with ThreadPoolExecutor(max_workers=len(launch_tasks)) as pool:
                for indices, returns in pool.map(_launch, launch_tasks):
                    for idx, ret in zip(indices, returns, strict=True):
                        all_returns[idx] = ret

        return all_returns
