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
    this sweeper runs exactly the combinations you specify.

    Parameters
    ----------
    combinations
        List of dicts, where each dict maps parameter paths to values.
        Each dict becomes one job.

    Example
    -------
    combinations:
      - {datamodule: independent}
      - {datamodule: ot, sparsify.mass_threshold: 0.5}
      - {datamodule: ot, sparsify.mass_threshold: 0.9}

    This runs exactly 3 jobs, not a Cartesian product.
    """

    def __init__(
        self,
        combinations: list[dict[str, Any]] | None = None,
        max_batch_size: int | None = None,  # Ignored, for compatibility with BasicSweeper config
        params: dict[str, str] | None = None,  # Ignored, for compatibility with BasicSweeper config
    ) -> None:
        self.combinations = list(combinations) if combinations else []
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

        # If combinations weren't passed to constructor, try to read from config
        if not self.combinations:
            sweeper_cfg = config.hydra.sweeper
            if hasattr(sweeper_cfg, "combinations") and sweeper_cfg.combinations:
                # Convert OmegaConf list to regular Python list of dicts
                self.combinations = [OmegaConf.to_container(c, resolve=True) for c in sweeper_cfg.combinations]
                log.info("Loaded %d combinations from config", len(self.combinations))

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

        # Build job overrides for each combination
        job_overrides: list[list[str]] = []
        for i, combo in enumerate(self.combinations):
            # Convert dict to override strings
            overrides = [self._format_override(k, v) for k, v in combo.items()]
            # Add any CLI arguments passed after --multirun
            overrides.extend(arguments)
            job_overrides.append(overrides)
            log.info("Job %d: %s", i, " ".join(overrides))

        log.info("ExplicitSweeper: Launching %d jobs", len(job_overrides))

        # Launch all jobs
        returns = self.launcher.launch(job_overrides, initial_job_idx=0)
        return returns
