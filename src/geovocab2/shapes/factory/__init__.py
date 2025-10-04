# shapes/factory/__init__.py
"""
Factory package - auto-discovers all FactoryBase subclasses
"""

import inspect
import pkgutil
import importlib
from .factory_base import FactoryBase

_all_classes = ['FactoryBase']

# Scan all .py files in current directory
for importer, modname, ispkg in pkgutil.iter_modules(__path__):
    if not ispkg:
        module = importlib.import_module(f".{modname}", package=__name__)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, FactoryBase) and obj is not FactoryBase:
                globals()[name] = obj
                _all_classes.append(name)

__all__ = _all_classes