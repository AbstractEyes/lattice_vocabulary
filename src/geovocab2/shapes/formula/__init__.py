# shapes/formula/__init__.py
import inspect
import pkgutil
import importlib
from pathlib import Path
from .formula_base import FormulaBase

_all_classes = ['FormulaBase']

# Scan subdirectories
for importer, modname, ispkg in pkgutil.iter_modules(__path__):
    if ispkg:  # symbolic, simple, engineering, etc.
        subpackage = importlib.import_module(f".{modname}", package=__name__)
        subpath = Path(subpackage.__file__).parent

        # Scan .py files in subdirectory
        for sub_importer, sub_modname, sub_ispkg in pkgutil.iter_modules([str(subpath)]):
            if not sub_ispkg:
                module = importlib.import_module(f".{modname}.{sub_modname}", package=__name__)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, FormulaBase) and obj is not FormulaBase:
                        globals()[name] = obj
                        _all_classes.append(name)

# print ("all formula imports;", _all_classes)
__all__ = _all_classes