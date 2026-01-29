"""Hydra sweeper for explicit parameter combinations without Cartesian product."""

from importlib.metadata import version

from ._sweeper import ExplicitSweeper

__all__ = ["ExplicitSweeper"]

__version__ = version("hydra-sweeper-explicit")
