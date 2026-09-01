import importlib
import sys

_mflux = importlib.import_module("mflux")

# mflux resolves some exports lazily to keep heavy libraries off the plain-import chain.
# Copying those eagerly here would undo it, so they are left to the module __getattr__ below.
_LAZY_NAMES = frozenset(getattr(_mflux, "_LAZY_EXPORTS", ()))

for _name in getattr(_mflux, "__all__", ()):
    if _name in _LAZY_NAMES:
        continue
    globals()[_name] = getattr(_mflux, _name)

__all__ = list(getattr(_mflux, "__all__", ()))

for _subpackage in ("callbacks", "cli", "models", "release", "utils"):
    _module = importlib.import_module(f"mflux.{_subpackage}")
    globals()[_subpackage] = _module
    sys.modules[f"{__name__}.{_subpackage}"] = _module
    if _subpackage not in __all__:
        __all__.append(_subpackage)


def __getattr__(name: str):
    return getattr(_mflux, name)
