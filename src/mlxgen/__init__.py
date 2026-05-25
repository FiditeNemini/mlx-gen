import importlib
import sys

_mflux = importlib.import_module("mflux")
sys.modules[__name__] = _mflux
