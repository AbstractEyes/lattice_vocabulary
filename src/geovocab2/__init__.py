"""
GeoVocab2 - Geometric vocabulary system

NOTE:
The training/model subsystem has moved to the 'geofractal' package.
If `geofractal` is installed, `geovocab2.train` will automatically
alias to the new model system.

If it is not installed, a placeholder warning module will be used.
"""

from . import shapes
from . import fusion
from . import config
from . import data

__all__ = ["shapes", "fusion", "config", "data"]

# ---------------------------------------------------------
# CONDITIONAL TRAIN SUBSYSTEM
# ---------------------------------------------------------
try:
    # If geofractal has installed its alias at geovocab2/train/, use it
    from . import train
    __all__.append("train")

except ImportError:
    # If train does not exist (geofractal not installed), create a warning module
    import types
    import warnings

    train = types.ModuleType("geovocab2.train")

    def _warn_and_fail(*args, **kwargs):
        warnings.warn(
            "\n"
            "⚠️  The `geovocab2.train` module has moved!\n"
            "   Models, fractal layers, and training utilities now live in:\n"
            "       → geofractal\n"
            "\n"
            "   Install it with:\n"
            "       pip install geofractal\n"
            "\n"
            "   Repo:\n"
            "       https://github.com/AbstractEyes/geofractal\n",
            category=ImportWarning,
            stacklevel=2,
        )
        raise ModuleNotFoundError(
            "The `geovocab2.train` module no longer exists in geovocab2.\n"
            "Install geofractal and import from `geofractal` instead."
        )

    # assign placeholder attributes
    train.__dict__.update({
        "__doc__": "Placeholder module for backward compatibility.",
        "__all__": [],
        "_deprecated": True,
        "error": _warn_and_fail,
    })

    # expose the placeholder module
    globals()["train"] = train
    __all__.append("train")
