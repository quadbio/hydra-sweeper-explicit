"""SearchPath plugin to add the sweeper's config to Hydra's search path."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ExplicitSweeperSearchPathPlugin(SearchPathPlugin):
    """Add the explicit sweeper config directory to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Add the package's config directory to the search path."""
        search_path.append(
            provider="hydra-sweeper-explicit",
            path="pkg://hydra_plugins.hydra_sweeper_explicit.conf",
        )
