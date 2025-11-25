"""
Alias namespace: geovocab2.train → geofractal

If the geofractal system is installed, this namespace becomes a
full transparent alias to the geofractal package.

If geofractal is not installed, this module provides a safe
fallback stub with a clear warning directing users to the new repo.
"""

import warnings
import types

try:
    # Try resolving actual geofractal package
    import geofractal as _gf

    # --- ALIAS MODE ACTIVE ---
    # Mirror geofractal’s namespace into geovocab2.train
    globals().update(_gf.__dict__)

    __all__ = getattr(
        _gf, "__all__", [k for k in _gf.__dict__.keys() if not k.startswith("_")]
    )

except ImportError:
    # --- FALLBACK/STUB MODE ---
    warnings.warn(
        "\n"
        "⚠️  The `geovocab2.train` module has moved!\n"
        "   The new model system lives in the 'geofractal' package.\n"
        "\n"
        "   Install it with:\n"
        "       pip install geofractal\n"
        "\n"
        "   Repo:\n"
        "       https://github.com/AbstractEyes/geofractal\n",
        category=ImportWarning,
        stacklevel=2,
    )

    # Build a minimal stub module that raises on usage
    stub = types.ModuleType("geovocab2.train")

    def _stub_error(*args, **kwargs):
        raise ModuleNotFoundError(
            "The `geovocab2.train` module is no longer part of geovocab2.\n"
            "Install geofractal to access model, fractal, and training code:\n"
            "    pip install geofractal\n"
        )

    # Provide minimal public API
    stub.__dict__.update(
        {
            "__doc__": (
                "Placeholder compatibility module. "
                "Install geofractal to enable model access."
            ),
            "__all__": [],
            "error": _stub_error,         # explicit entrypoint
            "available": False,           # let downstream code detect state
        }
    )

    # Expose stub as this module
    globals().update(stub.__dict__)
