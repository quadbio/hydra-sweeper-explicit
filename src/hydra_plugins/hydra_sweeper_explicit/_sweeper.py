"""Explicit Sweeper - runs exact parameter combinations without Cartesian product.

Hydra's BasicSweeper always computes the Cartesian product of all sweep parameters,
which leads to invalid combinations when parameters are dependent (e.g., sparsify
settings only make sense when sampling=ot).

This sweeper lets you define explicit combinations:

    hydra:
      sweeper:
        _target_: hydra_sweeper_explicit.ExplicitSweeper
        combinations:
          - {sampling: independent}
          - {sampling: ot, sparsify.mass_threshold: 0.5}
          - {sampling: ot, sparsify.mass_threshold: 0.9}

Each dict in `combinations` becomes one job with exactly those overrides.

Optionally, expand each combination across multiple seeds:

    hydra:
      sweeper:
        seeds: [42, 43, 44]  # or seeds: 5 for seeds 0-4
        seed_key: seed       # parameter name (default: "seed")
        combinations:
          - {sampling: independent}
          - {sampling: ot}

This creates 6 jobs: 2 combinations × 3 seeds.

Usage:
    python run_experiment.py --multirun hydra/sweeper=explicit +sweep=my_sweep
"""

import logging
from typing import Any

from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
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
    max_batch_size
        Ignored, for compatibility with BasicSweeper config.
    params
        Ignored, for compatibility with BasicSweeper config.

    Example
    -------
    combinations:
      - {datamodule: independent}
      - {datamodule: ot, sparsify.mass_threshold: 0.5}
      - {datamodule: ot, sparsify.mass_threshold: 0.9}

    This runs exactly 3 jobs, not a Cartesian product.

    With seeds:
      seeds: [42, 43]
      combinations:
        - {datamodule: independent}
        - {datamodule: ot}

    This runs 4 jobs: 2 combinations × 2 seeds.
    """

    def __init__(
        self,
        combinations: list[dict[str, Any]] | None = None,
        seeds: list[int] | int | None = None,
        seed_key: str = "seed",
        max_batch_size: int | None = None,  # Ignored, for compatibility
        params: dict[str, str] | None = None,  # Ignored, for compatibility
    ) -> None:
        self.combinations = list(combinations) if combinations else []
        self.seeds = seeds
        self.seed_key = seed_key
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

        # Build job overrides for each combination (optionally × seeds)
        job_overrides: list[list[str]] = []

        for combo in self.combinations:
            if seeds is not None:
                # Expand combination across all seeds
                for seed in seeds:
                    overrides = [self._format_override(k, v) for k, v in combo.items()]
                    overrides.append(self._format_override(self.seed_key, seed))
                    overrides.extend(arguments)
                    job_overrides.append(overrides)
            else:
                # Single job for this combination
                overrides = [self._format_override(k, v) for k, v in combo.items()]
                overrides.extend(arguments)
                job_overrides.append(overrides)

        # Log job summary
        if seeds is not None:
            log.info(
                "ExplicitSweeper: %d combinations × %d seeds = %d jobs",
                len(self.combinations),
                len(seeds),
                len(job_overrides),
            )
        else:
            log.info("ExplicitSweeper: Launching %d jobs", len(job_overrides))

        for i, overrides in enumerate(job_overrides):
            log.info("Job %d: %s", i, " ".join(overrides))

        # Launch all jobs
        returns = self.launcher.launch(job_overrides, initial_job_idx=0)
        return returns
