import importlib
import sys

_mflux = importlib.import_module("mflux")

for _subpackage in ("callbacks", "cli", "models", "release", "utils"):
    sys.modules[f"{__name__}.{_subpackage}"] = importlib.import_module(f"mflux.{_subpackage}")

sys.modules[__name__] = _mflux
