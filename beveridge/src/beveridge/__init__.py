"""Beveridge: the same package as `bev-decisions`, under the models' name.

    import beveridge
    model = beveridge.load()          # Beveridge-Large-C; beveridge.load(beveridge.PRO_MODEL) for Beveridge-Pro-C

Everything comes from `bev` (installed with bev-decisions); `import bev` keeps working.
"""

from bev import *  # noqa: F403
from bev import __all__ as _bev_all
from bev import __version__ as bev_version  # noqa: F401

__version__ = "0.2.5"
__all__ = list(_bev_all)
