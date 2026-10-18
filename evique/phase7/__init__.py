"""Phase 7 semantic evidence reconstruction.

The package is isolated from the frozen Phase 1-5.5 implementation.  Legacy
modules are imported only by explicit experiment adapters.
"""

from .config import Phase7Config

__all__ = ["Phase7Config"]
